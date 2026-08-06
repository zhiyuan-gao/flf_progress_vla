"""Semantic subtask text, endpoint images, exact progress, and HOLD targets."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .index import FrameRecord


CURRENT_CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
GOAL_CACHE_NAMES = ("left", "right", "wrist")
MOTION_ACTION_KEYS = (
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.base_motion",
)
HOLD_STATE_ACTION_KEYS = (
    "action.gripper_close",
    "action.control_mode",
)


def semantic_prompt(task_description: str, subtask_description: str) -> str:
    task = str(task_description).strip()
    subtask = str(subtask_description).strip()
    if not task or not subtask:
        raise ValueError("task and subtask descriptions must be non-empty")
    return f"Task: {task}\nCurrent subtask: {subtask}"


def deterministic_frame_shift(seed: int, epoch: int, index: int, radius: int) -> int:
    """Reproducible uniform integer shift in ``[-radius, radius]``."""
    if radius <= 0:
        return 0
    payload = f"semantic-progress:{seed}:{epoch}:{index}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return value % (2 * radius + 1) - radius


def exact_progress_with_frame_jitter(
    row: FrameRecord,
    *,
    seed: int,
    epoch: int,
    index: int,
    radius: int,
) -> float:
    shifted = int(
        np.clip(
            row.frame + deterministic_frame_shift(seed, epoch, index, radius),
            row.segment_start,
            row.segment_end,
        )
    )
    duration = max(row.segment_end - row.segment_start, 1)
    return float((shifted - row.segment_start) / duration)


def apply_explicit_hold_padding(
    raw: Mapping[str, Any],
    *,
    valid_steps: int,
    action_horizon: int = 16,
) -> dict[str, Any]:
    """Replace action targets after a subtask boundary with a supervised HOLD.

    RoboCasa's EEF rotation/translation and mobile-base commands are incremental,
    so zero is the native no-motion command.  The gripper and control mode are
    discrete state-like commands and retain their last valid values.
    """
    valid_steps = int(valid_steps)
    if not 1 <= valid_steps <= action_horizon:
        raise ValueError(f"valid_steps must be in [1,{action_horizon}], got {valid_steps}")
    output = dict(raw)
    for key in (*MOTION_ACTION_KEYS, *HOLD_STATE_ACTION_KEYS):
        if key not in raw:
            raise KeyError(f"raw sample is missing HOLD action key {key}")
        values = np.array(raw[key], copy=True)
        if values.shape[0] != action_horizon:
            raise ValueError(f"{key} must have horizon {action_horizon}, got {values.shape}")
        if valid_steps < action_horizon:
            if key in MOTION_ACTION_KEYS:
                values[valid_steps:] = 0
            else:
                values[valid_steps:] = values[valid_steps - 1]
        output[key] = values
    return output


def append_goal_time_step(
    raw: Mapping[str, Any],
    goal_images: Mapping[str, np.ndarray],
    *,
    camera_keys: Sequence[str] = CURRENT_CAMERA_KEYS,
) -> dict[str, Any]:
    """Append endpoint images as a second visual time step for all three cameras."""
    output = dict(raw)
    for key in camera_keys:
        if key not in raw or key not in goal_images:
            raise KeyError(f"missing current or goal image for {key}")
        current = np.asarray(raw[key])
        goal = np.asarray(goal_images[key])
        if goal.ndim == current.ndim - 1:
            goal = np.expand_dims(goal, axis=0)
        if current.shape[0] != 1 or goal.shape[0] != 1 or current.shape[1:] != goal.shape[1:]:
            raise ValueError(
                f"expected matching one-frame current/goal arrays for {key}, "
                f"got {current.shape} and {goal.shape}"
            )
        output[key] = np.concatenate((current, goal), axis=0)
    return output


def pack_goal_views(video: np.ndarray, goal_image_count: int) -> np.ndarray:
    """Pack ``[current 3, goal 3]`` into Goal1 (4 images) or Goal3 (6 images)."""
    array = np.asarray(video)
    if goal_image_count not in {1, 3}:
        raise ValueError("goal_image_count must be 1 or 3")
    is_batched = array.ndim == 6
    if is_batched:
        if array.shape[1:3] != (2, 3):
            raise ValueError(f"expected batched video [B,2,3,H,W,C], got {array.shape}")
        flattened = array.reshape(array.shape[0], 6, *array.shape[3:])
        selected = flattened[:, : 3 + goal_image_count]
        return selected[:, None]
    if array.ndim != 5 or array.shape[:2] != (2, 3):
        raise ValueError(f"expected video [2,3,H,W,C], got {array.shape}")
    flattened = array.reshape(6, *array.shape[2:])
    return flattened[: 3 + goal_image_count][None]


def make_semantic_goal_transform(*, goal_image_count: int):
    """Build the official PandaOmron transform with one view-packing step inserted."""
    from gr00t.data.transform.base import ModalityTransform
    from gr00t.experiment.data_config import PandaOmronDataConfig

    class PackGoalViewsTransform(ModalityTransform):
        apply_to: list[str] = []
        goal_image_count: int

        def apply(self, data: dict[str, Any]) -> dict[str, Any]:
            output = dict(data)
            output["video"] = pack_goal_views(data["video"], self.goal_image_count)
            return output

    transform = PandaOmronDataConfig().transform()
    transform.transforms.insert(
        -1,
        PackGoalViewsTransform(goal_image_count=int(goal_image_count)),
    )
    return transform


class GoalImageStore:
    """Disk-backed, process-local cache of three-view subtask endpoint images."""

    def __init__(self, root: Path | str, *, memory_items: int = 32):
        self.root = Path(root).expanduser().resolve()
        if memory_items < 1:
            raise ValueError("memory_items must be positive")
        self.memory_items = int(memory_items)
        self._memory: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

    def path_for(self, row: FrameRecord) -> Path:
        return (
            self.root
            / row.task
            / f"episode_{row.episode:06d}"
            / f"stage_{row.stage_index:02d}_frame_{row.segment_end:06d}.npz"
        )

    def load(self, row: FrameRecord) -> dict[str, np.ndarray]:
        path = self.path_for(row)
        if path not in self._memory:
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing goal-image cache {path}; run scripts/build_goal_cache.py first"
                )
            with np.load(path, allow_pickle=False) as archive:
                missing = [name for name in GOAL_CACHE_NAMES if name not in archive]
                if missing:
                    raise KeyError(f"{path} is missing arrays {missing}")
                values = {name: np.array(archive[name], copy=True) for name in GOAL_CACHE_NAMES}
            shapes = {value.shape for value in values.values()}
            if len(shapes) != 1 or any(value.ndim != 3 for value in values.values()):
                raise ValueError(f"goal cache {path} must contain matching HWC images")
            self._memory[path] = values
            while len(self._memory) > self.memory_items:
                self._memory.popitem(last=False)
        else:
            self._memory.move_to_end(path)
        cached = self._memory[path]
        return {
            camera: cached[name]
            for camera, name in zip(CURRENT_CAMERA_KEYS, GOAL_CACHE_NAMES, strict=True)
        }

    def save(self, row: FrameRecord, goal_images: Mapping[str, np.ndarray]) -> Path:
        path = self.path_for(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        for camera, name in zip(CURRENT_CAMERA_KEYS, GOAL_CACHE_NAMES, strict=True):
            value = np.asarray(goal_images[camera])
            if value.ndim == 4 and value.shape[0] == 1:
                value = value[0]
            if value.ndim != 3:
                raise ValueError(f"goal image {camera} must be HWC, got {value.shape}")
            arrays[name] = value
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
        self._memory.pop(path, None)
        return path

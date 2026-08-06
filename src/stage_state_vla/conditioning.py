"""Insert stage and video progress into unused GR00T state dimensions.

The RoboCasa state is normalized and padded by GR00T to 64 dimensions before
this function runs.  We deliberately use the first two masked-out dimensions,
so the checkpoint's 64-D state encoder remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConditionSlots:
    stage: int
    progress: int


def find_progress_slot(state_mask: Any) -> int:
    """Return the first unused state dimension for progress-only conditioning."""
    return find_condition_slots(state_mask, required=1).stage


def normalize_stage(stage_index: int | np.ndarray, max_stages: int = 5) -> Any:
    if max_stages < 2:
        raise ValueError("max_stages must be at least 2")
    values = np.asarray(stage_index)
    if np.any(values < 0) or np.any(values >= max_stages):
        raise ValueError(f"stage_index must be in [0, {max_stages - 1}]")
    result = 2.0 * values.astype(np.float32) / float(max_stages - 1) - 1.0
    return float(result) if result.ndim == 0 else result


def normalize_progress(progress: float | np.ndarray) -> Any:
    values = np.asarray(progress, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("progress must be finite")
    result = 2.0 * np.clip(values, 0.0, 1.0) - 1.0
    return float(result) if result.ndim == 0 else result


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def find_condition_slots(state_mask: Any, required: int = 2) -> ConditionSlots:
    mask = _as_numpy(state_mask).astype(bool, copy=False)
    if mask.ndim < 2:
        raise ValueError(f"state_mask must end in [horizon, dim], got {mask.shape}")
    flattened = mask.reshape(-1, mask.shape[-1])
    first = flattened[0]
    if not np.all(flattened == first):
        raise ValueError("all state horizons/batch items must use the same state dimensions")
    valid = np.flatnonzero(first)
    state_dim = int(valid[-1] + 1) if len(valid) else 0
    if len(valid) and not np.array_equal(valid, np.arange(state_dim)):
        raise ValueError("GR00T state_mask must be a contiguous true prefix")
    if state_dim + required > first.size:
        raise ValueError(
            f"need {required} empty condition dimensions after {state_dim}, only {first.size} total"
        )
    return ConditionSlots(stage=state_dim, progress=state_dim + 1)


def find_injected_condition_slots(state_mask: Any) -> ConditionSlots:
    """Locate the last two dimensions after this method has marked them valid."""
    mask = _as_numpy(state_mask).astype(bool, copy=False)
    flattened = mask.reshape(-1, mask.shape[-1])
    first = flattened[0]
    if not np.all(flattened == first):
        raise ValueError("all state horizons/batch items must use the same state dimensions")
    valid = np.flatnonzero(first)
    if len(valid) < 2 or not np.array_equal(valid, np.arange(int(valid[-1]) + 1)):
        raise ValueError("expected a contiguous state prefix containing two condition dimensions")
    return ConditionSlots(stage=int(valid[-2]), progress=int(valid[-1]))


def _broadcast_condition(value: Any, prefix_shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return np.full(prefix_shape, float(array), dtype=np.float32)
    if array.shape == prefix_shape:
        return array
    # A batch-level value is shared across the one-frame state horizon.
    if len(prefix_shape) >= 2 and array.shape == prefix_shape[:-1]:
        return np.broadcast_to(array[..., None], prefix_shape)
    return np.broadcast_to(array, prefix_shape)


def inject_stage_progress(
    sample: dict[str, Any],
    stage_index: int | np.ndarray,
    progress: float | np.ndarray,
    *,
    max_stages: int = 5,
    copy: bool = True,
) -> tuple[dict[str, Any], ConditionSlots]:
    """Return a transformed GR00T sample with two additional valid state dims."""
    if "state" not in sample or "state_mask" not in sample:
        raise KeyError("sample must contain transformed GR00T state and state_mask")
    state = sample["state"]
    mask = sample["state_mask"]
    slots = find_condition_slots(mask)

    if copy:
        if hasattr(state, "clone"):
            state = state.clone()
            mask = mask.clone()
        else:
            state = np.array(state, copy=True)
            mask = np.array(mask, copy=True)

    prefix_shape = tuple(state.shape[:-1])
    stage_values = _broadcast_condition(normalize_stage(stage_index, max_stages), prefix_shape)
    progress_values = _broadcast_condition(normalize_progress(progress), prefix_shape)

    if hasattr(state, "new_tensor"):
        state[..., slots.stage] = state.new_tensor(stage_values)
        state[..., slots.progress] = state.new_tensor(progress_values)
        mask[..., slots.stage] = True
        mask[..., slots.progress] = True
    else:
        state[..., slots.stage] = stage_values
        state[..., slots.progress] = progress_values
        mask[..., slots.stage] = True
        mask[..., slots.progress] = True

    output = dict(sample)
    output["state"] = state
    output["state_mask"] = mask
    return output, slots


def inject_progress(
    sample: dict[str, Any],
    progress: float | np.ndarray,
    *,
    copy: bool = True,
) -> tuple[dict[str, Any], int]:
    """Add only within-subtask progress to the first unused GR00T state dimension."""
    if "state" not in sample or "state_mask" not in sample:
        raise KeyError("sample must contain transformed GR00T state and state_mask")
    state = sample["state"]
    mask = sample["state_mask"]
    slot = find_progress_slot(mask)

    if copy:
        if hasattr(state, "clone"):
            state = state.clone()
            mask = mask.clone()
        else:
            state = np.array(state, copy=True)
            mask = np.array(mask, copy=True)

    prefix_shape = tuple(state.shape[:-1])
    progress_values = _broadcast_condition(normalize_progress(progress), prefix_shape)
    if hasattr(state, "new_tensor"):
        state[..., slot] = state.new_tensor(progress_values)
        mask[..., slot] = True
    else:
        state[..., slot] = progress_values
        mask[..., slot] = True

    output = dict(sample)
    output["state"] = state
    output["state_mask"] = mask
    return output, slot


def overwrite_stage_progress(
    sample: dict[str, Any],
    stage_index: int | np.ndarray,
    progress: float | np.ndarray,
    *,
    max_stages: int = 5,
    copy: bool = True,
) -> dict[str, Any]:
    """Change conditions in a sample that was already injected."""
    state = (
        sample["state"].clone()
        if copy and hasattr(sample["state"], "clone")
        else sample["state"]
    )
    if copy and not hasattr(sample["state"], "clone"):
        state = np.array(sample["state"], copy=True)
    slots = find_injected_condition_slots(sample["state_mask"])
    prefix_shape = tuple(state.shape[:-1])
    stage_values = _broadcast_condition(normalize_stage(stage_index, max_stages), prefix_shape)
    progress_values = _broadcast_condition(normalize_progress(progress), prefix_shape)
    if hasattr(state, "new_tensor"):
        state[..., slots.stage] = state.new_tensor(stage_values)
        state[..., slots.progress] = state.new_tensor(progress_values)
    else:
        state[..., slots.stage] = stage_values
        state[..., slots.progress] = progress_values
    output = dict(sample)
    output["state"] = state
    return output


def apply_episode_tail_mask(
    sample: dict[str, Any], frame: int, episode_length: int, action_horizon: int = 16
) -> dict[str, Any]:
    """Mask GR00T's repeated tail padding without masking subtask boundaries."""
    if "action_mask" not in sample:
        return sample
    valid_steps = max(0, min(action_horizon, int(episode_length) - int(frame)))
    output = dict(sample)
    mask = (
        sample["action_mask"].clone()
        if hasattr(sample["action_mask"], "clone")
        else np.array(sample["action_mask"], copy=True)
    )
    mask[valid_steps:] = False
    output["action_mask"] = mask
    return output

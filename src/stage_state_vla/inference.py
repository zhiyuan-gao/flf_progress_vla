"""Reusable conditioned inference primitives, independent of a simulator API."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .controller import MonotonicStageController
from .index import FrameRecord
from .reference_video import RollingSubsequenceDTWProgressLocalizer
from .semantic_goal import GoalImageStore


ACTION_KEYS = (
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
)
ACTION_WIDTHS = (3, 3, 1, 4, 1)


@dataclass(frozen=True)
class ConditionEstimate:
    stage_index: int
    progress: float
    source: str
    advanced: bool = False
    terminal: bool = False
    details: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConditionSource(Protocol):
    def estimate(
        self, observation: Mapping[str, Any], control_step: int
    ) -> ConditionEstimate: ...


class ReferenceClockConditionSource:
    """Read stage/progress from the reference annotation at an elapsed step.

    This is a diagnostic reference-timeline baseline. It is not a semantic
    simulator-state oracle once a rollout deviates from the demonstration.
    """

    def __init__(self, records: Sequence[FrameRecord], progress_field: str = "grid_progress"):
        if progress_field not in {"grid_progress", "exact_progress"}:
            raise ValueError("progress_field must be grid_progress or exact_progress")
        if not records:
            raise ValueError("reference clock requires non-empty records")
        self.records = sorted(records, key=lambda row: row.frame)
        self.frames = [row.frame for row in self.records]
        self.progress_field = progress_field

    def estimate(
        self, observation: Mapping[str, Any], control_step: int
    ) -> ConditionEstimate:
        del observation
        position = bisect_right(self.frames, int(control_step)) - 1
        position = min(max(position, 0), len(self.records) - 1)
        row = self.records[position]
        return ConditionEstimate(
            stage_index=row.stage_index,
            progress=float(getattr(row, self.progress_field)),
            source="reference_clock",
            terminal=row.stage_index == row.stage_count - 1,
            details={"reference_frame": row.frame, "stage_name": row.stage_name},
        )


class ReferenceVideoConditionSource:
    """Turn current RGB observations and GT stage videos into `(stage, progress)`."""

    def __init__(
        self,
        localizer: RollingSubsequenceDTWProgressLocalizer,
        *,
        num_stages: int,
        completion_threshold: float = 0.9,
        confirmations: int = 2,
    ) -> None:
        self.localizer = localizer
        self.controller = MonotonicStageController(
            num_stages,
            completion_threshold=completion_threshold,
            confirmations=confirmations,
        )
        self._last_sample_step: int | None = None
        self._last_decision = None

    def estimate(
        self, observation: Mapping[str, Any], control_step: int
    ) -> ConditionEstimate:
        localized = self.localizer.localize(
            self.controller.stage_index, observation, control_step
        )
        fresh = localized.sample_step != self._last_sample_step
        if fresh:
            decision = self.controller.update(localized.progress)
            self._last_sample_step = localized.sample_step
            self._last_decision = decision
        else:
            if self._last_decision is None:
                raise RuntimeError("DTW condition source has no previous fresh decision")
            previous = self._last_decision
            decision = type(previous)(
                stage_index=self.controller.stage_index,
                progress=previous.progress,
                advanced=False,
                terminal=previous.terminal,
                completion_streak=previous.completion_streak,
            )
        if decision.advanced:
            self.localizer.reset(decision.stage_index)
            self.localizer.observe(
                decision.stage_index,
                observation,
                control_step,
                force=True,
            )
        return ConditionEstimate(
            stage_index=decision.stage_index,
            progress=decision.progress,
            source="reference_dtw",
            advanced=decision.advanced,
            terminal=decision.terminal,
            details={
                "localized_stage": localized.stage_index,
                "start_index": localized.start_index,
                "end_index": localized.end_index,
                "node_frame": localized.node_frame,
                "path": localized.path,
                "query_steps": localized.query_steps,
                "sample_step": localized.sample_step,
                "fresh_localization": fresh,
                "mean_cost": localized.mean_cost,
                "confidence_margin": localized.confidence_margin,
                "completion_streak": decision.completion_streak,
            },
        )

    def observe(self, observation: Mapping[str, Any], control_step: int) -> bool:
        return self.localizer.observe(
            self.controller.stage_index,
            observation,
            control_step,
        )


def add_observation_horizon(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Match the official RoboCasa GR00T one-frame observation contract."""
    output: dict[str, Any] = {}
    for key, value in observation.items():
        if key.startswith("video.") or key.startswith("state."):
            output[key] = np.expand_dims(np.asarray(value), axis=0)
        elif key.startswith("annotation."):
            output[key] = np.asarray([value])
        else:
            output[key] = value
    return output


def concatenate_action_chunk(action_dict: Mapping[str, Any]) -> np.ndarray:
    missing = [key for key in ACTION_KEYS if key not in action_dict]
    if missing:
        raise KeyError(f"policy action is missing keys: {missing}")
    arrays = [np.asarray(action_dict[key], dtype=np.float32) for key in ACTION_KEYS]
    chunk = np.concatenate(arrays, axis=-1)
    if chunk.ndim != 2 or chunk.shape[1] != sum(ACTION_WIDTHS):
        raise ValueError(f"expected action chunk [horizon,12], got {chunk.shape}")
    return chunk


def split_action(action: np.ndarray) -> dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (sum(ACTION_WIDTHS),):
        raise ValueError(f"expected one 12-D action, got {action.shape}")
    boundaries = np.cumsum((0,) + ACTION_WIDTHS)
    return {
        key: action[boundaries[index] : boundaries[index + 1]].copy()
        for index, key in enumerate(ACTION_KEYS)
    }


class ConditionedChunkPredictor:
    """Bind a condition source to the repository's conditioned GR00T policy."""

    def __init__(self, policy: Any, condition_source: ConditionSource) -> None:
        self.policy = policy
        self.condition_source = condition_source

    def predict(
        self, observation: Mapping[str, Any], control_step: int
    ) -> tuple[np.ndarray, ConditionEstimate]:
        condition = self.condition_source.estimate(observation, control_step)
        action_dict = self.policy.get_action_with_condition(
            add_observation_horizon(observation),
            stage_index=condition.stage_index,
            video_progress=condition.progress,
        )
        return concatenate_action_chunk(action_dict), condition

    def observe(self, observation: Mapping[str, Any], control_step: int) -> bool:
        observe = getattr(self.condition_source, "observe", None)
        if observe is None:
            return False
        return bool(observe(observation, control_step))


class SemanticGoalChunkPredictor:
    """Bind DTW conditions to semantic subtask text and cached endpoint images."""

    def __init__(
        self,
        policy: Any,
        condition_source: ConditionSource,
        records: Sequence[FrameRecord],
        goal_store: GoalImageStore,
    ) -> None:
        self.policy = policy
        self.condition_source = condition_source
        self.goal_store = goal_store
        by_stage: dict[int, FrameRecord] = {}
        for row in records:
            previous = by_stage.get(row.stage_index)
            if previous is not None and (
                previous.segment_start != row.segment_start
                or previous.segment_end != row.segment_end
            ):
                raise ValueError(f"stage {row.stage_index} has multiple semantic segments")
            by_stage[row.stage_index] = row
        if not by_stage or sorted(by_stage) != list(range(max(by_stage) + 1)):
            raise ValueError(f"semantic stages must be contiguous, got {sorted(by_stage)}")
        if any(
            not row.task_description or not row.subtask_description
            for row in by_stage.values()
        ):
            raise ValueError("semantic inference records require task and subtask descriptions")
        self.by_stage = by_stage

    def predict(
        self, observation: Mapping[str, Any], control_step: int
    ) -> tuple[np.ndarray, ConditionEstimate]:
        condition = self.condition_source.estimate(observation, control_step)
        row = self.by_stage[condition.stage_index]
        action_dict = self.policy.get_action_with_semantic_goal(
            add_observation_horizon(observation),
            video_progress=condition.progress,
            task_description=row.task_description,
            subtask_description=row.subtask_description,
            goal_images=self.goal_store.load(row),
        )
        return concatenate_action_chunk(action_dict), condition

    def observe(self, observation: Mapping[str, Any], control_step: int) -> bool:
        observe = getattr(self.condition_source, "observe", None)
        if observe is None:
            return False
        return bool(observe(observation, control_step))

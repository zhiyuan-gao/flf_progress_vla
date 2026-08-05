"""Inference wrapper that conditions GR00T without changing its checkpoint shape."""

from __future__ import annotations

from typing import Any

from .conditioning import inject_stage_progress


def make_conditioned_policy_class():
    """Return a Gr00tPolicy subclass after the external dependency is available."""
    from gr00t.model.policy import Gr00tPolicy

    class StageStateGr00tPolicy(Gr00tPolicy):
        def __init__(self, *args, max_stages: int = 5, **kwargs):
            self.max_stages = int(max_stages)
            self._stage_index = 0
            self._video_progress = 0.0
            super().__init__(*args, **kwargs)

        def set_condition(self, stage_index: int, video_progress: float) -> None:
            self._stage_index = int(stage_index)
            self._video_progress = float(video_progress)

        def apply_transforms(self, obs: dict[str, Any]) -> dict[str, Any]:
            normalized = super().apply_transforms(obs)
            normalized, _ = inject_stage_progress(
                normalized,
                self._stage_index,
                self._video_progress,
                max_stages=self.max_stages,
                copy=False,
            )
            return normalized

        def get_action_with_condition(
            self,
            observations: dict[str, Any],
            *,
            stage_index: int,
            video_progress: float,
        ) -> dict[str, Any]:
            self.set_condition(stage_index, video_progress)
            return super().get_action(observations)

    return StageStateGr00tPolicy

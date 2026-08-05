"""Monotonic subtask controller used by oracle and video-progress rollouts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageDecision:
    stage_index: int
    progress: float
    advanced: bool
    terminal: bool
    completion_streak: int


class MonotonicStageController:
    """Allow hold or +1 only after repeated high-progress observations."""

    def __init__(
        self,
        num_stages: int,
        completion_threshold: float = 0.9,
        confirmations: int = 2,
    ) -> None:
        if num_stages < 1:
            raise ValueError("num_stages must be positive")
        if not 0.0 <= completion_threshold <= 1.0:
            raise ValueError("completion_threshold must be in [0, 1]")
        if confirmations < 1:
            raise ValueError("confirmations must be positive")
        self.num_stages = int(num_stages)
        self.completion_threshold = float(completion_threshold)
        self.confirmations = int(confirmations)
        self.reset()

    def reset(self) -> None:
        self.stage_index = 0
        self._completion_streak = 0

    def update(self, estimated_progress: float) -> StageDecision:
        progress = min(max(float(estimated_progress), 0.0), 1.0)
        if progress >= self.completion_threshold:
            self._completion_streak += 1
        else:
            self._completion_streak = 0

        advanced = False
        terminal = self.stage_index == self.num_stages - 1
        if self._completion_streak >= self.confirmations:
            if terminal:
                self._completion_streak = self.confirmations
            else:
                self.stage_index += 1
                self._completion_streak = 0
                progress = 0.0
                advanced = True
                terminal = self.stage_index == self.num_stages - 1

        return StageDecision(
            stage_index=self.stage_index,
            progress=progress,
            advanced=advanced,
            terminal=terminal,
            completion_streak=self._completion_streak,
        )

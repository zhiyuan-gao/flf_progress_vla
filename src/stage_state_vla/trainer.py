"""Task-balanced GR00T trainer kept inside the standalone method directory."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator

import torch
from torch.utils.data import Sampler


class EqualTaskSampler(Sampler[int]):
    """Draw each task with equal probability, then draw a frame within it."""

    def __init__(self, dataset, *, seed: int = 42, num_samples: int | None = None) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.num_samples = len(dataset) if num_samples is None else int(num_samples)
        self.epoch = 0
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(dataset.records):
            grouped[row.task].append(index)
        if not grouped:
            raise ValueError("no tasks to sample")
        self.tasks = tuple(sorted(grouped))
        self.indices = {task: torch.tensor(grouped[task], dtype=torch.int64) for task in self.tasks}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        task_ids = torch.randint(len(self.tasks), (self.num_samples,), generator=generator)
        output = []
        for task_id in task_ids.tolist():
            candidates = self.indices[self.tasks[task_id]]
            offset = int(torch.randint(len(candidates), (1,), generator=generator))
            output.append(int(candidates[offset]))
        return iter(output)


def make_task_balanced_trainer_class():
    """Create lazily so unit tests do not need GR00T/Transformers imports."""
    from gr00t.experiment.trainer import DualBrainTrainer

    class TaskBalancedTrainer(DualBrainTrainer):
        def _get_train_sampler(self):
            return EqualTaskSampler(self.train_dataset, seed=self.args.seed)

    return TaskBalancedTrainer

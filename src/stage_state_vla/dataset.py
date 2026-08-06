"""GR00T dataset adapter for stage-state continuation samples."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from torch.utils.data import Dataset

from .conditioning import apply_episode_tail_mask, inject_progress, inject_stage_progress
from .index import FrameRecord, read_jsonl
from .semantic_goal import (
    GoalImageStore,
    append_goal_time_step,
    apply_explicit_hold_padding,
    exact_progress_with_frame_jitter,
    make_semantic_goal_transform,
    semantic_prompt,
)


def deterministic_node_shift(seed: int, epoch: int, index: int, radius: int) -> int:
    if radius <= 0:
        return 0
    payload = f"{seed}:{epoch}:{index}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return value % (2 * radius + 1) - radius


class StageStateDataset(Dataset):
    """Select exact episode/frame samples and add the two continuation signals."""

    def __init__(
        self,
        records: Sequence[FrameRecord],
        base_datasets: dict[str, Any],
        *,
        max_stages: int = 5,
        progress_source: str = "grid",
        progress_noise_nodes: int = 0,
        seed: int = 42,
        action_horizon: int = 16,
    ) -> None:
        if progress_source not in {"grid", "exact"}:
            raise ValueError("progress_source must be 'grid' or 'exact'")
        self.records = list(records)
        self.base_datasets = dict(base_datasets)
        self.max_stages = int(max_stages)
        self.progress_source = progress_source
        self.progress_noise_nodes = int(progress_noise_nodes)
        self.seed = int(seed)
        self.action_horizon = int(action_horizon)
        self.epoch = 0
        if not self.records:
            raise ValueError("StageStateDataset requires at least one record")
        unknown = sorted({row.task for row in self.records} - set(self.base_datasets))
        if unknown:
            raise KeyError(f"missing base datasets for tasks: {unknown}")

    @property
    def tag(self) -> str:
        # Compatibility with GR00T's experiment utilities.
        return next(iter(self.base_datasets.values())).tag

    @property
    def metadata(self) -> Any:
        return next(iter(self.base_datasets.values())).metadata

    @property
    def task_counts(self) -> dict[str, int]:
        return dict(Counter(row.task for row in self.records))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        for dataset in self.base_datasets.values():
            dataset.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def progress_for(self, index: int) -> float:
        row = self.records[index]
        if self.progress_source == "exact":
            return row.exact_progress
        node = row.grid_node + deterministic_node_shift(
            self.seed, self.epoch, index, self.progress_noise_nodes
        )
        node = int(np.clip(node, 0, row.grid_nodes - 1))
        return 1.0 if row.grid_nodes == 1 else node / float(row.grid_nodes - 1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        base = self.base_datasets[row.task]
        raw = base.get_step_data(row.episode, row.frame)
        sample = base.transforms(raw)
        sample, _ = inject_stage_progress(
            sample,
            row.stage_index,
            self.progress_for(index),
            max_stages=self.max_stages,
            copy=False,
        )
        return apply_episode_tail_mask(
            sample,
            frame=row.frame,
            episode_length=row.episode_length,
            action_horizon=self.action_horizon,
        )


def create_base_datasets(
    records: Sequence[FrameRecord],
    *,
    training: bool,
    video_backend: str,
    normalization_metadata: Path | None = None,
):
    """Instantiate one independent GR00T transform/dataset per RoboCasa task."""
    try:
        from gr00t.data.dataset import LeRobotSingleDataset
        from gr00t.data.schema import DatasetMetadata
        from gr00t.experiment.data_config import PandaOmronDataConfig
    except (ImportError, TypeError) as exc:
        raise RuntimeError(
            "GR00T could not be imported. Activate the pinned GR00T environment and call "
            "stage_state_vla.bootstrap.activate_gr00t() before constructing datasets."
        ) from exc

    paths: dict[str, Path] = {}
    for row in records:
        path = Path(row.dataset)
        if row.task in paths and paths[row.task] != path:
            raise ValueError(f"task {row.task} maps to multiple datasets")
        paths[row.task] = path

    shared_metadata = None
    if normalization_metadata is not None:
        payload = json.loads(Path(normalization_metadata).read_text(encoding="utf-8"))
        if "new_embodiment" not in payload:
            raise KeyError(f"new_embodiment is missing from {normalization_metadata}")
        shared_metadata = DatasetMetadata.model_validate(payload["new_embodiment"])

    datasets = {}
    for task, path in sorted(paths.items()):
        data_config = PandaOmronDataConfig()
        transforms = data_config.transform()
        transforms.train() if training else transforms.eval()
        dataset = LeRobotSingleDataset(
            dataset_path=path,
            modality_configs=data_config.modality_config(),
            transforms=transforms,
            embodiment_tag="new_embodiment",
            video_backend=video_backend,
        )
        # Continuation must see exactly the normalization used by its parent
        # checkpoint.  Per-task statistics would otherwise leak task identity
        # through scale and shift the action targets between tasks.
        if shared_metadata is not None:
            dataset.set_transforms_metadata(copy.deepcopy(shared_metadata))
        datasets[task] = dataset
    return datasets


def load_stage_dataset(
    index_path: Path,
    *,
    training: bool,
    video_backend: str = "opencv",
    max_stages: int = 5,
    progress_source: str = "grid",
    progress_noise_nodes: int = 0,
    seed: int = 42,
    action_horizon: int = 16,
    normalization_metadata: Path | None = None,
) -> StageStateDataset:
    records = read_jsonl(index_path)
    bases = create_base_datasets(
        records,
        training=training,
        video_backend=video_backend,
        normalization_metadata=normalization_metadata,
    )
    return StageStateDataset(
        records,
        bases,
        max_stages=max_stages,
        progress_source=progress_source,
        progress_noise_nodes=progress_noise_nodes,
        seed=seed,
        action_horizon=action_horizon,
    )


class SemanticGoalDataset(Dataset):
    """Goal-conditioned subtask samples with exact progress and explicit HOLD tails."""

    def __init__(
        self,
        records: Sequence[FrameRecord],
        base_datasets: dict[str, Any],
        transforms: dict[str, Any],
        *,
        goal_store: GoalImageStore,
        goal_image_count: int,
        progress_jitter_frames: int = 4,
        seed: int = 42,
        action_horizon: int = 16,
    ) -> None:
        if goal_image_count not in {1, 3}:
            raise ValueError("goal_image_count must be 1 or 3")
        self.records = list(records)
        self.base_datasets = dict(base_datasets)
        self.transforms = dict(transforms)
        self.goal_store = goal_store
        self.goal_image_count = int(goal_image_count)
        self.progress_jitter_frames = int(progress_jitter_frames)
        self.seed = int(seed)
        self.action_horizon = int(action_horizon)
        self.epoch = 0
        if not self.records:
            raise ValueError("SemanticGoalDataset requires at least one record")
        if any(not row.task_description or not row.subtask_description for row in self.records):
            raise ValueError("semantic-goal indices must contain task and subtask descriptions")
        tasks = {row.task for row in self.records}
        if tasks - self.base_datasets.keys() or tasks - self.transforms.keys():
            raise KeyError("every semantic-goal task needs a base dataset and transform")

    @property
    def tag(self) -> str:
        return next(iter(self.base_datasets.values())).tag

    @property
    def metadata(self) -> Any:
        return next(iter(self.base_datasets.values())).metadata

    @property
    def task_counts(self) -> dict[str, int]:
        return dict(Counter(row.task for row in self.records))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        for dataset in self.base_datasets.values():
            dataset.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def progress_for(self, index: int) -> float:
        return exact_progress_with_frame_jitter(
            self.records[index],
            seed=self.seed,
            epoch=self.epoch,
            index=index,
            radius=self.progress_jitter_frames,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        base = self.base_datasets[row.task]
        raw = base.get_step_data(row.episode, row.frame)
        raw["annotation.human.task_description"] = [
            semantic_prompt(row.task_description, row.subtask_description)
        ]
        raw = append_goal_time_step(raw, self.goal_store.load(row))
        valid_steps = max(
            1,
            min(self.action_horizon, row.segment_end - row.frame + 1),
        )
        raw = apply_explicit_hold_padding(
            raw,
            valid_steps=valid_steps,
            action_horizon=self.action_horizon,
        )
        sample = self.transforms[row.task](raw)
        sample, _ = inject_progress(sample, self.progress_for(index), copy=False)
        return sample


def load_semantic_goal_dataset(
    index_path: Path,
    *,
    training: bool,
    goal_cache_dir: Path,
    goal_image_count: int,
    video_backend: str = "opencv",
    progress_jitter_frames: int = 4,
    seed: int = 42,
    action_horizon: int = 16,
    normalization_metadata: Path | None = None,
) -> SemanticGoalDataset:
    records = read_jsonl(index_path)
    bases = create_base_datasets(
        records,
        training=training,
        video_backend=video_backend,
        normalization_metadata=normalization_metadata,
    )
    metadata = None
    if normalization_metadata is not None:
        from gr00t.data.schema import DatasetMetadata

        payload = json.loads(Path(normalization_metadata).read_text(encoding="utf-8"))
        metadata = DatasetMetadata.model_validate(payload["new_embodiment"])
    transforms: dict[str, Any] = {}
    for task, base in bases.items():
        transform = make_semantic_goal_transform(goal_image_count=goal_image_count)
        transform.train() if training else transform.eval()
        transform.set_metadata(copy.deepcopy(metadata or base.metadata))
        transforms[task] = transform
    return SemanticGoalDataset(
        records,
        bases,
        transforms,
        goal_store=GoalImageStore(goal_cache_dir),
        goal_image_count=goal_image_count,
        progress_jitter_frames=progress_jitter_frames if training else 0,
        seed=seed,
        action_horizon=action_horizon,
    )

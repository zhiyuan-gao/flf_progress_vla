"""Build per-frame continuation samples from fixed RoboCasa episode splits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReferenceGrid:
    frames: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.frames)

    def nearest(self, frame: int) -> tuple[int, float]:
        distances = np.abs(np.asarray(self.frames, dtype=np.int64) - int(frame))
        node = int(np.argmin(distances))
        progress = 1.0 if self.size == 1 else node / float(self.size - 1)
        return node, progress


@dataclass(frozen=True)
class FrameRecord:
    task: str
    split: str
    dataset: str
    episode: int
    episode_length: int
    frame: int
    stage_index: int
    stage_count: int
    stage_name: str
    segment_start: int
    segment_end: int
    exact_progress: float
    grid_node: int
    grid_nodes: int
    grid_progress: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def find_dataset(task_root: Path) -> Path:
    matches = sorted(path.parent.parent for path in task_root.glob("*/lerobot/meta/info.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one dataset under {task_root}, found {matches}")
    return matches[0].resolve()


def load_task_names(dataset: Path) -> dict[int, str]:
    rows = (dataset / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines()
    return {int((row := json.loads(line))["task_index"]): str(row["task"]) for line in rows}


def episode_parquet(dataset: Path, episode: int) -> Path:
    return dataset / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"


def reference_grid(start: int, end: int, stride: int = 8) -> ReferenceGrid:
    if stride < 1 or end < start:
        raise ValueError(f"invalid segment/grid: start={start}, end={end}, stride={stride}")
    frames = list(range(start, end + 1, stride))
    if frames[-1] != end:
        frames.append(end)
    return ReferenceGrid(tuple(frames))


def contiguous_segments(
    stage_values: np.ndarray, subtask_values: np.ndarray
) -> Iterator[tuple[int, int, int, int]]:
    if len(stage_values) != len(subtask_values):
        raise ValueError("stage and subtask arrays must have the same length")
    if len(stage_values) == 0:
        return
    start = 0
    for index in range(1, len(stage_values) + 1):
        boundary = index == len(stage_values)
        if not boundary:
            boundary = (
                stage_values[index] != stage_values[start]
                or subtask_values[index] != subtask_values[start]
            )
        if boundary:
            yield (
                start,
                index - 1,
                int(stage_values[start]),
                int(subtask_values[start]),
            )
            start = index


def build_episode_records(
    *,
    task: str,
    split: str,
    dataset: Path,
    episode: int,
    stage_count: int,
    reference_stride: int = 8,
) -> list[FrameRecord]:
    columns = ["frame_index", "subtask_idx", "annotation.human.subtask_stage"]
    table = pd.read_parquet(episode_parquet(dataset, episode), columns=columns)
    frame_indices = table["frame_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(frame_indices, np.arange(len(table), dtype=np.int64)):
        raise ValueError(f"{task} episode {episode} has non-contiguous frame indices")
    names = load_task_names(dataset)
    stages = table["annotation.human.subtask_stage"].to_numpy(dtype=np.int64)
    subtasks = table["subtask_idx"].to_numpy(dtype=np.int64)
    records: list[FrameRecord] = []
    observed_active: set[int] = set()
    for start, end, stage_id, stage_index in contiguous_segments(stages, subtasks):
        stage_name = names[stage_id]
        if stage_name.strip().lower() in {"done", "task complete"}:
            continue
        if stage_index < 0 or stage_index >= stage_count:
            raise ValueError(
                f"{task} episode {episode}: stage index {stage_index} outside [0,{stage_count})"
            )
        observed_active.add(stage_index)
        grid = reference_grid(start, end, reference_stride)
        duration = max(end - start, 1)
        for frame in range(start, end + 1):
            node, grid_progress = grid.nearest(frame)
            records.append(
                FrameRecord(
                    task=task,
                    split=split,
                    dataset=str(dataset),
                    episode=int(episode),
                    episode_length=len(table),
                    frame=frame,
                    stage_index=stage_index,
                    stage_count=stage_count,
                    stage_name=stage_name,
                    segment_start=start,
                    segment_end=end,
                    exact_progress=float(np.clip((frame - start) / duration, 0.0, 1.0)),
                    grid_node=node,
                    grid_nodes=grid.size,
                    grid_progress=grid_progress,
                )
            )
    expected = set(range(stage_count))
    if observed_active != expected:
        raise ValueError(
            f"{task} episode {episode}: active stages {sorted(observed_active)}, "
            f"expected {sorted(expected)}"
        )
    return records


def read_split_config(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_split_records(
    *,
    data_root: Path,
    split_config: Path,
    tasks: list[str],
    stage_counts: dict[str, int],
    split: str,
    reference_stride: int,
    max_episodes_per_task: int | None = None,
) -> list[FrameRecord]:
    split_data = read_split_config(split_config)
    records: list[FrameRecord] = []
    for task in tasks:
        dataset = find_dataset(data_root / task)
        episodes = list(split_data["tasks"][task]["splits"][split])
        if max_episodes_per_task is not None:
            episodes = episodes[:max_episodes_per_task]
        for episode in episodes:
            records.extend(
                build_episode_records(
                    task=task,
                    split=split,
                    dataset=dataset,
                    episode=int(episode),
                    stage_count=int(stage_counts[task]),
                    reference_stride=reference_stride,
                )
            )
    return records


def write_jsonl(path: Path, records: Iterable[FrameRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def read_jsonl(path: Path) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(FrameRecord(**json.loads(line)))
    return records

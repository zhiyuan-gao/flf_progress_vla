from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stage_state_vla.index import build_episode_records, reference_grid


def test_reference_grid_adds_exact_endpoint():
    grid = reference_grid(10, 29, stride=8)
    assert grid.frames == (10, 18, 26, 29)
    assert grid.nearest(28) == (3, 1.0)


def test_same_stage_name_is_split_by_subtask_index(tmp_path: Path):
    dataset = tmp_path / "lerobot"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "data/chunk-000").mkdir(parents=True)
    tasks = [
        {"task_index": 0, "task": "do the complete task"},
        {"task_index": 1, "task": "complete the first subtask"},
        {"task_index": 2, "task": "complete the second subtask"},
        {"task_index": 5, "task": "done"},
        {"task_index": 8, "task": "execute"},
    ]
    (dataset / "meta/tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in tasks), encoding="utf-8"
    )
    table = pd.DataFrame(
        {
            "frame_index": list(range(7)),
            "subtask_idx": [0, 0, 0, 1, 1, 1, 2],
            "annotation.human.task_description": [0] * 7,
            "annotation.human.subtask": [1, 1, 1, 2, 2, 2, 2],
            "annotation.human.subtask_stage": [8, 8, 8, 8, 8, 8, 5],
        }
    )
    table.to_parquet(dataset / "data/chunk-000/episode_000000.parquet")
    records = build_episode_records(
        task="RinseSinkBasin",
        split="train",
        dataset=dataset,
        episode=0,
        stage_count=2,
        reference_stride=2,
    )
    assert [row.stage_index for row in records] == [0, 0, 0, 1, 1, 1]
    assert records[2].exact_progress == 1.0
    assert records[3].exact_progress == 0.0
    assert records[0].task_description == "do the complete task"
    assert records[0].subtask_description == "complete the first subtask"
    assert records[3].subtask_description == "complete the second subtask"

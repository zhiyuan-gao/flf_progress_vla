#!/usr/bin/env python3
"""Build standalone train/validation frame indices from the fixed split."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


STANDALONE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STANDALONE / "src"))

from stage_state_vla.config import load_config, resolve_paths, validate_external_paths
from stage_state_vla.index import build_split_records, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=STANDALONE / "configs/mvp.json")
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_paths(config)
    validate_external_paths(paths, required=("data_root", "split_config"))
    paths.index_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": str(Path(config["_config_path"])),
        "reference_stride": config["condition"]["reference_stride"],
        "splits": {},
    }
    for split in args.splits:
        records = build_split_records(
            data_root=paths.data_root,
            split_config=paths.split_config,
            tasks=config["tasks"],
            stage_counts=config["stage_counts"],
            split=split,
            reference_stride=int(config["condition"]["reference_stride"]),
            max_episodes_per_task=args.max_episodes_per_task,
        )
        suffix = "" if args.max_episodes_per_task is None else f"_smoke{args.max_episodes_per_task}"
        output = paths.index_dir / f"{split}{suffix}.jsonl"
        write_jsonl(output, records)
        task_counts = Counter(row.task for row in records)
        stage_counts = Counter(f"{row.task}:{row.stage_index}" for row in records)
        summary["splits"][split] = {
            "path": str(output),
            "num_frames": len(records),
            "task_counts": dict(sorted(task_counts.items())),
            "stage_counts": dict(sorted(stage_counts.items())),
        }
        print(f"{split}: {len(records):,} frames -> {output}")
        for task, count in sorted(task_counts.items()):
            print(f"  {task}: {count:,}")
    summary_path = paths.index_dir / "summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

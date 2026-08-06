#!/usr/bin/env python3
"""Decode and cache three-view endpoint images for semantic-goal training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


STANDALONE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STANDALONE / "src"))

from stage_state_vla.bootstrap import activate_gr00t, validate_python_environment
from stage_state_vla.config import load_config, resolve_paths, validate_external_paths
from stage_state_vla.dataset import create_base_datasets
from stage_state_vla.index import FrameRecord, read_jsonl
from stage_state_vla.semantic_goal import CURRENT_CAMERA_KEYS, GoalImageStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=STANDALONE / "configs/semantic_goal1.json",
    )
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument("--indices", nargs="+", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-endpoints", type=int, default=None)
    return parser.parse_args()


def unique_endpoints(rows: list[FrameRecord]) -> list[FrameRecord]:
    unique: dict[tuple[str, int, int, int], FrameRecord] = {}
    for row in rows:
        unique[(row.task, row.episode, row.stage_index, row.segment_end)] = row
    return [unique[key] for key in sorted(unique)]


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_paths(config)
    validate_external_paths(paths, required=("gr00t_root", "data_root", "index_dir"))
    validate_python_environment()
    activate_gr00t(paths.gr00t_root)

    rows: list[FrameRecord] = []
    indices = (
        [path.expanduser().resolve() for path in args.indices]
        if args.indices
        else [paths.index_dir / f"{split}.jsonl" for split in args.splits]
    )
    for index in indices:
        if not index.is_file():
            raise FileNotFoundError(f"build semantic-goal indices first: {index}")
        rows.extend(read_jsonl(index))
    endpoints = unique_endpoints(rows)
    if args.max_endpoints is not None:
        endpoints = endpoints[: int(args.max_endpoints)]

    cache_dir = (paths.repo_root / config["goal"]["cache_dir"]).resolve()
    store = GoalImageStore(cache_dir)
    bases = create_base_datasets(
        endpoints,
        training=False,
        video_backend=config["training"]["video_backend"],
    )
    written = 0
    skipped = 0
    for position, row in enumerate(endpoints, 1):
        path = store.path_for(row)
        if path.is_file() and not args.overwrite:
            skipped += 1
            continue
        raw = bases[row.task].get_step_data(row.episode, row.segment_end)
        store.save(row, {key: raw[key] for key in CURRENT_CAMERA_KEYS})
        written += 1
        if position == 1 or position % 50 == 0 or position == len(endpoints):
            print(f"{position}/{len(endpoints)} endpoints; written={written}, skipped={skipped}")
    print(f"goal cache: {cache_dir}; written={written}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

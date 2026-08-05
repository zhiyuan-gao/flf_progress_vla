#!/usr/bin/env python3
"""Load one real three-view sample and verify its state/action contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


STANDALONE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STANDALONE / "src"))

from stage_state_vla.bootstrap import activate_gr00t, validate_python_environment
from stage_state_vla.conditioning import find_injected_condition_slots
from stage_state_vla.config import load_config, resolve_paths, validate_external_paths
from stage_state_vla.dataset import load_stage_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=STANDALONE / "configs/mvp.json")
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_paths(config)
    validate_external_paths(paths)
    validate_python_environment()
    activate_gr00t(paths.gr00t_root)
    index = args.index or paths.index_dir / "train_smoke1.jsonl"
    dataset = load_stage_dataset(
        index,
        training=True,
        video_backend=config["training"]["video_backend"],
        max_stages=config["condition"]["max_stages"],
        progress_source=config["condition"]["progress_source"],
        progress_noise_nodes=config["condition"]["progress_noise_nodes"],
        seed=config["training"]["seed"],
        action_horizon=config["model"]["action_horizon"],
        normalization_metadata=paths.base_checkpoint / "experiment_cfg/metadata.json",
    )
    row = dataset.records[args.sample_index]
    sample = dataset[args.sample_index]
    slots = find_injected_condition_slots(sample["state_mask"])
    report = {
        "row": row.as_dict(),
        "dataset_length": len(dataset),
        "task_counts": dataset.task_counts,
        "state_shape": list(sample["state"].shape),
        "state_mask_true": int(np.asarray(sample["state_mask"]).sum()),
        "condition_slots": {"stage": slots.stage, "progress": slots.progress},
        "condition_values": {
            "stage": float(sample["state"][0, slots.stage]),
            "progress": float(sample["state"][0, slots.progress]),
        },
        "action_shape": list(sample["action"].shape),
        "valid_action_values": int(np.asarray(sample["action_mask"]).sum()),
        "keys": sorted(sample),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["state_shape"] != [1, 64] or report["action_shape"] != [16, 32]:
        raise RuntimeError("unexpected GR00T tensor contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

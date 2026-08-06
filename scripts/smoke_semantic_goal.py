#!/usr/bin/env python3
"""Load one real Goal1/Goal3 sample and verify the multimodal tensor contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


STANDALONE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STANDALONE / "src"))

from stage_state_vla.bootstrap import activate_gr00t, validate_python_environment
from stage_state_vla.conditioning import find_progress_slot
from stage_state_vla.config import load_config, resolve_paths, validate_external_paths
from stage_state_vla.dataset import load_semantic_goal_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=STANDALONE / "configs/semantic_goal1.json",
    )
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
    dataset = load_semantic_goal_dataset(
        index,
        training=True,
        goal_cache_dir=(paths.repo_root / config["goal"]["cache_dir"]).resolve(),
        goal_image_count=int(config["goal"]["goal_image_count"]),
        video_backend=config["training"]["video_backend"],
        progress_jitter_frames=int(config["condition"]["progress_jitter_frames"]),
        seed=int(config["training"]["seed"]),
        action_horizon=int(config["model"]["action_horizon"]),
        normalization_metadata=paths.base_checkpoint / "experiment_cfg/metadata.json",
    )
    row = dataset.records[args.sample_index]
    sample = dataset[args.sample_index]
    original_mask = np.asarray(sample["state_mask"]).copy()
    original_mask[..., np.flatnonzero(original_mask.reshape(-1, 64)[0])[-1]] = False
    progress_slot = find_progress_slot(original_mask)
    eagle = sample["eagle_content"]
    image_count = len(eagle["image_inputs"])
    report = {
        "variant": config["goal"]["variant"],
        "row": row.as_dict(),
        "state_shape": list(sample["state"].shape),
        "state_mask_true": int(np.asarray(sample["state_mask"]).sum()),
        "progress_slot": progress_slot,
        "progress_value": float(sample["state"][0, progress_slot]),
        "action_shape": list(sample["action"].shape),
        "valid_action_values": int(np.asarray(sample["action_mask"]).sum()),
        "eagle_image_count": image_count,
        "text_list": eagle["text_list"],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    expected_images = 3 + int(config["goal"]["goal_image_count"])
    if report["state_mask_true"] != 21 or image_count != expected_images:
        raise RuntimeError("unexpected semantic-goal multimodal contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

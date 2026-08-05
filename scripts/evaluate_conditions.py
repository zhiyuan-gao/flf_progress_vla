#!/usr/bin/env python3
"""Compare validation flow loss under correct and counterfactual conditions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


STANDALONE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STANDALONE / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from stage_state_vla.bootstrap import activate_gr00t, validate_python_environment
from stage_state_vla.conditioning import overwrite_stage_progress
from stage_state_vla.config import load_config, resolve_paths, validate_external_paths
from stage_state_vla.dataset import load_stage_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=STANDALONE / "configs/mvp.json")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_paths(config)
    validate_external_paths(
        paths,
        required=("gr00t_root", "data_root", "base_checkpoint", "index_dir"),
    )
    validate_python_environment()
    activate_gr00t(paths.gr00t_root)
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from gr00t.model.transforms import DefaultDataCollator
    checkpoint = (args.checkpoint or paths.output_dir).resolve()
    index = (args.index or paths.index_dir / "val.jsonl").resolve()
    dataset = load_stage_dataset(
        index,
        training=False,
        video_backend=config["training"]["video_backend"],
        max_stages=config["condition"]["max_stages"],
        progress_source=config["condition"]["progress_source"],
        progress_noise_nodes=0,
        seed=config["training"]["seed"],
        action_horizon=config["model"]["action_horizon"],
        normalization_metadata=checkpoint / "experiment_cfg/metadata.json",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=DefaultDataCollator(),
    )
    model = GR00T_N1_5.from_pretrained(
        str(checkpoint),
        tune_visual=False,
        tune_llm=False,
        tune_projector=False,
        tune_diffusion_model=False,
    ).to(args.device)
    model.eval()
    totals = {name: 0.0 for name in ("correct", "progress_start", "progress_mid", "shuffled")}
    counts = {name: 0 for name in totals}
    cursor = 0
    rng = np.random.default_rng(config["training"]["seed"])
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        batch_size = int(batch["state"].shape[0])
        rows = dataset.records[cursor : cursor + batch_size]
        cursor += batch_size
        stage = np.asarray([row.stage_index for row in rows], dtype=np.int64)
        progress = np.asarray(
            [dataset.progress_for(cursor - batch_size + i) for i in range(batch_size)]
        )
        permutation = rng.permutation(batch_size)
        variants = {
            "correct": batch,
            "progress_start": overwrite_stage_progress(
                batch, stage, np.zeros(batch_size), max_stages=config["condition"]["max_stages"]
            ),
            "progress_mid": overwrite_stage_progress(
                batch, stage, np.full(batch_size, 0.5), max_stages=config["condition"]["max_stages"]
            ),
            "shuffled": overwrite_stage_progress(
                batch,
                stage[permutation],
                progress[permutation],
                max_stages=config["condition"]["max_stages"],
            ),
        }
        for name, inputs in variants.items():
            # The same seed gives all variants the same flow-matching noise/time.
            torch.manual_seed(config["training"]["seed"] + batch_index)
            torch.cuda.manual_seed_all(config["training"]["seed"] + batch_index)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                loss = float(model(inputs)["loss"].float())
            totals[name] += loss * batch_size
            counts[name] += batch_size
    losses = {name: totals[name] / max(counts[name], 1) for name in totals}
    report = {
        "checkpoint": str(checkpoint),
        "index": str(index),
        "num_samples": counts["correct"],
        "flow_loss": losses,
        "delta_from_correct": {name: value - losses["correct"] for name, value in losses.items()},
        "interpretation": (
            "A condition-aware model should change under shuffled/constant conditions; "
            "lower correct loss is the desired direction."
        ),
    }
    print(json.dumps(report, indent=2))
    output = args.output or checkpoint / "condition_sensitivity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

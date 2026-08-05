#!/usr/bin/env python3
"""Run a balanced four-GPU offline condition-sensitivity evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/mvp.json")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "outputs/mvp_continuation/checkpoint-10000",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=REPO_ROOT / "artifacts/indices/val.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/condition_sensitivity_mvp256/report.json",
    )
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--samples-per-task", type=int, default=64)
    parser.add_argument("--progress-bins", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--signal-threshold", type=float, default=0.01)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_digest(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def progress_bin(progress: float, num_bins: int) -> int:
    return min(int(float(progress) * num_bins), num_bins - 1)


def select_balanced_indices(
    dataset: Any,
    *,
    task: str,
    count: int,
    num_bins: int,
    action_horizon: int,
    seed: int,
) -> list[int]:
    strata: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.records):
        # Keep every target action chunk fully valid so all samples contribute
        # the same number of flow-loss elements.
        if int(row.frame) + int(action_horizon) > int(row.episode_length):
            continue
        progress = float(dataset.progress_for(index))
        strata[(int(row.stage_index), progress_bin(progress, num_bins))].append(index)

    for key, values in strata.items():
        values.sort(
            key=lambda index: stable_digest(
                "sample", seed, task, key, dataset.records[index].episode,
                dataset.records[index].frame,
            )
        )
    if not strata:
        raise RuntimeError(f"no eligible validation samples for {task}")

    selected: list[int] = []
    offsets = {key: 0 for key in strata}
    keys = sorted(strata)
    while len(selected) < count:
        added = False
        for key in keys:
            offset = offsets[key]
            if offset >= len(strata[key]):
                continue
            selected.append(strata[key][offset])
            offsets[key] += 1
            added = True
            if len(selected) == count:
                break
        if not added:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"requested {count} balanced samples for {task}, found {len(selected)}"
        )
    return selected


def build_counterfactuals(
    dataset: Any,
    selected: list[int],
    *,
    task: str,
    action_horizon: int,
    num_bins: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_stage: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.records):
        if int(row.frame) + int(action_horizon) <= int(row.episode_length):
            by_stage[int(row.stage_index)].append(index)
    valid_stages = sorted(by_stage)
    if len(valid_stages) < 2:
        raise RuntimeError(f"{task} needs at least two stages for sensitivity")

    specifications: list[dict[str, Any]] = []
    for index in selected:
        row = dataset.records[index]
        stage = int(row.stage_index)
        progress = float(dataset.progress_for(index))

        progress_candidates = [
            candidate
            for candidate in by_stage[stage]
            if float(dataset.progress_for(candidate)) != progress
        ]
        if not progress_candidates:
            raise RuntimeError(
                f"no alternative progress for {task} stage={stage} index={index}"
            )
        # Prefer a different episode and a substantial within-stage progress
        # displacement, then choose deterministically by hash.
        donor = min(
            progress_candidates,
            key=lambda candidate: (
                dataset.records[candidate].episode == row.episode,
                abs(float(dataset.progress_for(candidate)) - progress) < 0.25,
                stable_digest(
                    "wrong-progress",
                    seed,
                    task,
                    row.episode,
                    row.frame,
                    dataset.records[candidate].episode,
                    dataset.records[candidate].frame,
                ),
            ),
        )
        wrong_progress = float(dataset.progress_for(donor))

        other_stages = [value for value in valid_stages if value != stage]
        wrong_stage = other_stages[
            int(stable_digest("wrong-stage", seed, task, row.episode, row.frame)[:8], 16)
            % len(other_stages)
        ]
        donor_row = dataset.records[donor]
        specifications.append(
            {
                "dataset_index": index,
                "episode": int(row.episode),
                "frame": int(row.frame),
                "stage": stage,
                "progress": progress,
                "progress_bin": progress_bin(progress, num_bins),
                "wrong_progress": wrong_progress,
                "wrong_progress_abs_delta": abs(wrong_progress - progress),
                "progress_donor_episode": int(donor_row.episode),
                "progress_donor_frame": int(donor_row.frame),
                "wrong_stage": int(wrong_stage),
            }
        )
    return specifications


def evaluate_task(settings: dict[str, Any]) -> dict[str, Any]:
    gpu_id = int(settings["gpu_id"])
    task = str(settings["task"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"
    sys.path.insert(0, str(REPO_ROOT / "src"))

    import numpy as np
    import torch

    from stage_state_vla.bootstrap import activate_gr00t, validate_python_environment
    from stage_state_vla.conditioning import overwrite_stage_progress
    from stage_state_vla.config import load_config, resolve_paths, validate_external_paths
    from stage_state_vla.dataset import (
        StageStateDataset,
        create_base_datasets,
    )
    from stage_state_vla.index import read_jsonl

    config = load_config(settings["config"])
    paths = resolve_paths(config)
    validate_external_paths(paths, required=("gr00t_root", "data_root"))
    validate_python_environment()
    activate_gr00t(paths.gr00t_root)

    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from gr00t.model.transforms import DefaultDataCollator

    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    checkpoint = Path(settings["checkpoint"]).resolve()
    all_records = read_jsonl(Path(settings["index"]))
    records = [row for row in all_records if row.task == task]
    if not records:
        raise RuntimeError(f"validation index has no records for {task}")
    condition_cfg = config["condition"]
    model_cfg = config["model"]
    base_datasets = create_base_datasets(
        records,
        training=False,
        video_backend=config["training"]["video_backend"],
        normalization_metadata=checkpoint / "experiment_cfg/metadata.json",
    )
    for base_dataset in base_datasets.values():
        transforms = getattr(base_dataset.transforms, "transforms", None)
        if not transforms or type(transforms[-1]).__name__ != "GR00TTransform":
            raise RuntimeError("expected GR00TTransform to be the final dataset transform")
        # Keep deterministic eval-mode video/state preprocessing, but enable the
        # final packing transform's target-action fields required by flow loss.
        transforms[-1].train()
    dataset = StageStateDataset(
        records,
        base_datasets,
        max_stages=int(condition_cfg["max_stages"]),
        progress_source=condition_cfg["progress_source"],
        progress_noise_nodes=0,
        seed=int(config["training"]["seed"]),
        action_horizon=int(model_cfg["action_horizon"]),
    )
    selected = select_balanced_indices(
        dataset,
        task=task,
        count=int(settings["samples_per_task"]),
        num_bins=int(settings["progress_bins"]),
        action_horizon=int(model_cfg["action_horizon"]),
        seed=int(settings["seed"]),
    )
    specifications = build_counterfactuals(
        dataset,
        selected,
        task=task,
        action_horizon=int(model_cfg["action_horizon"]),
        num_bins=int(settings["progress_bins"]),
        seed=int(settings["seed"]),
    )

    print(
        f"[{task} gpu={gpu_id}] loading {checkpoint} for "
        f"{len(specifications)} samples",
        flush=True,
    )
    model = GR00T_N1_5.from_pretrained(
        str(checkpoint),
        tune_visual=False,
        tune_llm=False,
        tune_projector=False,
        tune_diffusion_model=False,
    ).to(device)
    model.compute_dtype = model_cfg["compute_dtype"]
    model.eval()
    collator = DefaultDataCollator()
    totals = {name: 0.0 for name in ("correct", "wrong_progress", "wrong_stage")}
    batch_results: list[dict[str, Any]] = []
    started = time.time()
    batch_size = int(settings["batch_size"])
    for batch_index, start in enumerate(range(0, len(specifications), batch_size)):
        subset = specifications[start : start + batch_size]
        batch = collator([dataset[row["dataset_index"]] for row in subset])
        stages = np.asarray([row["stage"] for row in subset], dtype=np.int64)
        progresses = np.asarray([row["progress"] for row in subset], dtype=np.float32)
        wrong_progresses = np.asarray(
            [row["wrong_progress"] for row in subset], dtype=np.float32
        )
        wrong_stages = np.asarray(
            [row["wrong_stage"] for row in subset], dtype=np.int64
        )
        variants = {
            "correct": batch,
            "wrong_progress": overwrite_stage_progress(
                batch,
                stages,
                wrong_progresses,
                max_stages=int(condition_cfg["max_stages"]),
            ),
            "wrong_stage": overwrite_stage_progress(
                batch,
                wrong_stages,
                progresses,
                max_stages=int(condition_cfg["max_stages"]),
            ),
        }
        losses: dict[str, float] = {}
        flow_seed = int(
            stable_digest("flow", settings["seed"], task, batch_index)[:8], 16
        ) & 0x7FFFFFFF
        for name, inputs in variants.items():
            torch.manual_seed(flow_seed)
            torch.cuda.manual_seed(flow_seed)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                loss = float(model(inputs)["loss"].float())
            losses[name] = loss
            totals[name] += loss * len(subset)
        batch_results.append(
            {
                "batch_index": batch_index,
                "num_samples": len(subset),
                "flow_seed": flow_seed,
                "flow_loss": losses,
                "delta_from_correct": {
                    name: value - losses["correct"] for name, value in losses.items()
                },
            }
        )
        print(
            f"[{task} gpu={gpu_id}] batch={batch_index + 1}/"
            f"{(len(specifications) + batch_size - 1) // batch_size} "
            f"correct={losses['correct']:.6f} "
            f"wrong_progress={losses['wrong_progress']:.6f} "
            f"wrong_stage={losses['wrong_stage']:.6f}",
            flush=True,
        )
    torch.cuda.synchronize(gpu_id)
    count = len(specifications)
    losses = {name: value / count for name, value in totals.items()}
    relative = {
        name: (value - losses["correct"]) / losses["correct"]
        for name, value in losses.items()
    }
    stage_counts = Counter(row["stage"] for row in specifications)
    bin_counts = Counter(row["progress_bin"] for row in specifications)
    episode_counts = Counter(row["episode"] for row in specifications)
    report = {
        "task": task,
        "gpu_id": gpu_id,
        "num_samples": count,
        "batch_size": batch_size,
        "flow_loss": losses,
        "delta_from_correct": {
            name: value - losses["correct"] for name, value in losses.items()
        },
        "relative_delta_from_correct": relative,
        "selection": {
            "stage_counts": {str(key): value for key, value in sorted(stage_counts.items())},
            "progress_bin_counts": {
                str(key): value for key, value in sorted(bin_counts.items())
            },
            "episode_counts": {
                str(key): value for key, value in sorted(episode_counts.items())
            },
            "mean_wrong_progress_abs_delta": sum(
                row["wrong_progress_abs_delta"] for row in specifications
            )
            / count,
            "samples": specifications,
        },
        "batches": batch_results,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(gpu_id) / 1024**2,
    }
    print(
        f"[{task} gpu={gpu_id}] complete relative deltas: "
        f"progress={relative['wrong_progress']:+.3%} "
        f"stage={relative['wrong_stage']:+.3%}",
        flush=True,
    )
    return report


def classify_signal(relative: dict[str, float], threshold: float) -> dict[str, Any]:
    progress_signal = relative["wrong_progress"] >= threshold
    stage_signal = relative["wrong_stage"] >= threshold
    if progress_signal and stage_signal:
        interpretation = "detectable_stage_and_progress_signal"
    elif progress_signal:
        interpretation = "detectable_progress_signal_only"
    elif stage_signal:
        interpretation = "detectable_stage_signal_only"
    else:
        interpretation = "no_mvp_threshold_signal"
    return {
        "relative_loss_threshold": threshold,
        "progress_signal": progress_signal,
        "stage_signal": stage_signal,
        "interpretation": interpretation,
    }


def main() -> int:
    args = parse_args()
    if args.samples_per_task < 1 or args.progress_bins < 2 or args.batch_size < 1:
        raise ValueError("samples-per-task and batch-size must be positive; progress-bins >= 2")
    if args.num_gpus < 1:
        raise ValueError("num-gpus must be positive")
    config_path = args.config.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    index = args.index.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if not index.is_file():
        raise FileNotFoundError(index)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tasks = list(args.tasks or config["tasks"])
    unknown = sorted(set(tasks) - set(config["tasks"]))
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    if len(tasks) > args.num_gpus:
        raise ValueError("this evaluator requires one GPU per concurrently evaluated task")

    started = time.time()
    settings_by_task = [
        {
            "task": task,
            "gpu_id": index,
            "config": str(config_path),
            "checkpoint": str(checkpoint),
            "index": str(args.index.expanduser().resolve()),
            "samples_per_task": args.samples_per_task,
            "progress_bins": args.progress_bins,
            "batch_size": args.batch_size,
            "seed": args.seed,
        }
        for index, task in enumerate(tasks)
    ]
    task_reports: dict[str, dict[str, Any]] = {}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=context) as executor:
        futures = {
            executor.submit(evaluate_task, settings): settings["task"]
            for settings in settings_by_task
        }
        for future in as_completed(futures):
            task = futures[future]
            task_reports[task] = future.result()
            atomic_write_json(
                output.parent / "tasks" / f"{task}.json", task_reports[task]
            )

    ordered = {task: task_reports[task] for task in tasks}
    total_samples = sum(report["num_samples"] for report in ordered.values())
    overall_losses = {
        name: sum(report["flow_loss"][name] * report["num_samples"] for report in ordered.values())
        / total_samples
        for name in ("correct", "wrong_progress", "wrong_stage")
    }
    overall_relative = {
        name: (value - overall_losses["correct"]) / overall_losses["correct"]
        for name, value in overall_losses.items()
    }
    report = {
        "protocol": "condition_sensitivity_mvp256_v1",
        "created_at": utc_now(),
        "checkpoint": str(checkpoint),
        "validation_index": str(index),
        "seed": args.seed,
        "tasks": tasks,
        "gpu_assignment": {
            settings["task"]: settings["gpu_id"] for settings in settings_by_task
        },
        "samples_per_task": args.samples_per_task,
        "total_samples": total_samples,
        "variants": {
            "correct": "correct stage and correct progress",
            "wrong_progress": "correct stage and within-stage progress from another validation frame",
            "wrong_stage": "different valid stage from the same task and correct progress",
        },
        "controls": (
            "Within each task batch, RGB observations, robot state, language, target action, "
            "flow-matching noise, and diffusion timestep are identical across variants."
        ),
        "overall": {
            "flow_loss": overall_losses,
            "delta_from_correct": {
                name: value - overall_losses["correct"]
                for name, value in overall_losses.items()
            },
            "relative_delta_from_correct": overall_relative,
            "mvp_signal": classify_signal(overall_relative, args.signal_threshold),
        },
        "per_task": ordered,
        "elapsed_seconds": time.time() - started,
        "interpretation_limit": (
            "This offline paired loss diagnostic tests detectable conditional use on held-out "
            "validation samples; it does not by itself establish closed-loop causal benefit."
        ),
    }
    atomic_write_json(output, report)
    print(json.dumps(report["overall"], indent=2), flush=True)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

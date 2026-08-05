#!/usr/bin/env python3
"""Measure online DTW/controller localization on frozen held-out GT videos."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/mvp.json")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=REPO_ROOT / "configs/gt_test50_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs/reference_video_rollout/gt_test50_v1/dtw_gt_replay",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--allow-frozen-test", action="store_true")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(type(value).__name__)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def decode_stride_frames(video_path: Path, last_frame: int, stride: int) -> dict[int, Any]:
    import cv2

    needed = set(range(0, last_frame + 1, stride))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    decoded: dict[int, Any] = {}
    index = 0
    try:
        while index <= last_frame:
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"could not decode frame {index} from {video_path}")
            if index in needed:
                decoded[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        capture.release()
    missing = sorted(needed - set(decoded))
    if missing:
        raise RuntimeError(f"missing requested frames in {video_path}: {missing[:5]}")
    return decoded


def evaluate_episode(job: dict[str, Any]) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    sys.path.insert(0, str(REPO_ROOT / "src"))

    import numpy as np

    from stage_state_vla.config import load_config, resolve_paths
    from stage_state_vla.index import build_episode_records, find_dataset
    from stage_state_vla.inference import ReferenceVideoConditionSource
    from stage_state_vla.reference_video import (
        RGBReferenceEncoder,
        RollingSubsequenceDTWProgressLocalizer,
        SubsequenceDTWLocalizer,
        build_ground_truth_references,
        episode_video_path,
    )

    config = load_config(job["config"])
    paths = resolve_paths(config)
    task = str(job["task"])
    episode = int(job["episode"])
    output_path = Path(job["output_path"])
    condition_cfg = config["condition"]
    localizer_cfg = condition_cfg["localizer"]
    dataset = find_dataset(paths.data_root / task)
    stage_count = int(config["stage_counts"][task])
    records = build_episode_records(
        task=task,
        split="gt_test50_v1",
        dataset=dataset,
        episode=episode,
        stage_count=stage_count,
        reference_stride=int(condition_cfg["reference_stride"]),
    )
    encoder = RGBReferenceEncoder(
        camera_keys=localizer_cfg["camera_keys"],
        spatial_size=int(localizer_cfg["image_size"]),
    )
    references = build_ground_truth_references(
        dataset,
        episode,
        records,
        reference_stride=int(condition_cfg["reference_stride"]),
        encoder=encoder,
    )
    localizer = RollingSubsequenceDTWProgressLocalizer(
        references,
        encoder=encoder,
        query_length=int(localizer_cfg["query_length"]),
        observation_stride=int(localizer_cfg["query_stride"]),
        aligner=SubsequenceDTWLocalizer(
            motion_weight=float(localizer_cfg["motion_weight"]),
            stay_penalty=float(localizer_cfg["stay_penalty"]),
            jump_penalty=float(localizer_cfg["jump_penalty"]),
        ),
    )
    source = ReferenceVideoConditionSource(
        localizer,
        num_stages=stage_count,
        completion_threshold=float(condition_cfg["completion_threshold"]),
        confirmations=int(condition_cfg["completion_confirmations"]),
    )

    record_by_frame = {row.frame: row for row in records}
    last_frame = max(record_by_frame)
    observation_stride = int(localizer_cfg["query_stride"])
    policy_stride = int(config["model"]["action_horizon"])
    camera_key = str(localizer_cfg["camera_keys"][0])
    frames = decode_stride_frames(
        episode_video_path(dataset, episode, camera_key),
        last_frame,
        observation_stride,
    )

    samples: list[dict[str, Any]] = []
    first_gt_step: dict[int, int] = {}
    first_pred_step: dict[int, int] = {}
    for step in range(0, last_frame + 1, observation_stride):
        observation = {camera_key: frames[step]}
        if step:
            source.observe(observation, step)
        if step % policy_stride:
            continue
        gt = record_by_frame[step]
        estimate = source.estimate(observation, step)
        first_gt_step.setdefault(gt.stage_index, step)
        first_pred_step.setdefault(estimate.stage_index, step)
        stage_correct = estimate.stage_index == gt.stage_index
        within_stage_abs_error = (
            abs(float(estimate.progress) - float(gt.grid_progress))
            if stage_correct
            else None
        )
        gt_task_progress = (gt.stage_index + float(gt.grid_progress)) / stage_count
        pred_task_progress = (
            estimate.stage_index + float(estimate.progress)
        ) / stage_count
        task_progress_abs_error = abs(pred_task_progress - gt_task_progress)
        details = estimate.details or {}
        node_frame_error = None
        if int(details.get("localized_stage", -1)) == gt.stage_index:
            gt_node_frame = references[gt.stage_index].node_frames[gt.grid_node]
            node_frame_error = abs(int(details["node_frame"]) - int(gt_node_frame))
        samples.append(
            {
                "control_step": step,
                "gt_stage": gt.stage_index,
                "gt_grid_progress": gt.grid_progress,
                "pred_stage": estimate.stage_index,
                "pred_progress": estimate.progress,
                "stage_correct": stage_correct,
                "within_stage_abs_error": within_stage_abs_error,
                "gt_task_progress": gt_task_progress,
                "pred_task_progress": pred_task_progress,
                "task_progress_abs_error": task_progress_abs_error,
                "node_frame_error": node_frame_error,
                "mean_cost": details.get("mean_cost"),
                "confidence_margin": details.get("confidence_margin"),
                "advanced": estimate.advanced,
            }
        )

    correct_stage_errors = [
        row["within_stage_abs_error"]
        for row in samples
        if row["within_stage_abs_error"] is not None
    ]
    task_errors = [row["task_progress_abs_error"] for row in samples]
    node_errors = [
        row["node_frame_error"]
        for row in samples
        if row["node_frame_error"] is not None
    ]
    transition_deltas = {
        str(stage): (
            first_pred_step[stage] - first_gt_step[stage]
            if stage in first_pred_step and stage in first_gt_step
            else None
        )
        for stage in range(1, stage_count)
    }
    metrics = {
        "protocol": "gt_replay_online_dtw_v1",
        "interpretation": (
            "Self-localization accuracy when the held-out GT video is replayed through "
            "the same online DTW/controller schedule. This is not ground-truth accuracy "
            "for a free policy rollout after it deviates from the demonstration."
        ),
        "task": task,
        "episode": episode,
        "evaluation_split_sha256": job["split_sha256"],
        "reference_stride": int(condition_cfg["reference_stride"]),
        "observation_stride": observation_stride,
        "policy_stride": policy_stride,
        "samples": len(samples),
        "stage_correct": sum(row["stage_correct"] for row in samples),
        "stage_accuracy": sum(row["stage_correct"] for row in samples) / len(samples),
        "within_stage_progress_mae_on_correct_stage": (
            float(np.mean(correct_stage_errors)) if correct_stage_errors else None
        ),
        "within_stage_progress_rmse_on_correct_stage": (
            float(np.sqrt(np.mean(np.square(correct_stage_errors))))
            if correct_stage_errors
            else None
        ),
        "task_progress_mae": float(np.mean(task_errors)),
        "task_progress_rmse": float(np.sqrt(np.mean(np.square(task_errors)))),
        "task_progress_within_0_05": sum(error <= 0.05 for error in task_errors)
        / len(task_errors),
        "task_progress_within_0_10": sum(error <= 0.10 for error in task_errors)
        / len(task_errors),
        "node_frame_mae_on_matching_stage": (
            float(np.mean(node_errors)) if node_errors else None
        ),
        "transition_delta_steps": transition_deltas,
        "sample_records": samples,
    }
    atomic_write_json(output_path, metrics)
    return metrics


def aggregate(rows: list[dict[str, Any]], split_sha256: str) -> dict[str, Any]:
    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        sample_count = sum(row["samples"] for row in group)
        stage_correct = sum(row["stage_correct"] for row in group)
        sample_records = [sample for row in group for sample in row["sample_records"]]
        correct_stage_errors = [
            sample["within_stage_abs_error"]
            for sample in sample_records
            if sample["within_stage_abs_error"] is not None
        ]
        task_errors = [sample["task_progress_abs_error"] for sample in sample_records]
        node_errors = [
            sample["node_frame_error"]
            for sample in sample_records
            if sample["node_frame_error"] is not None
        ]
        transition_errors = [
            abs(delta)
            for row in group
            for delta in row["transition_delta_steps"].values()
            if delta is not None
        ]
        return {
            "episodes": len(group),
            "samples": sample_count,
            "stage_accuracy": stage_correct / sample_count,
            "within_stage_progress_mae_on_correct_stage": (
                sum(correct_stage_errors) / len(correct_stage_errors)
                if correct_stage_errors
                else None
            ),
            "task_progress_mae": sum(task_errors) / len(task_errors),
            "task_progress_rmse": math.sqrt(
                sum(error * error for error in task_errors) / len(task_errors)
            ),
            "task_progress_within_0_05": sum(error <= 0.05 for error in task_errors)
            / len(task_errors),
            "task_progress_within_0_10": sum(error <= 0.10 for error in task_errors)
            / len(task_errors),
            "node_frame_mae_on_matching_stage": (
                sum(node_errors) / len(node_errors) if node_errors else None
            ),
            "transition_abs_error_steps": (
                sum(transition_errors) / len(transition_errors)
                if transition_errors
                else None
            ),
        }

    tasks = sorted({row["task"] for row in rows})
    return {
        "protocol": "gt_replay_online_dtw_v1",
        "evaluation_split_sha256": split_sha256,
        "interpretation": (
            "Held-out GT-video replay localization diagnostic; not free-rollout "
            "semantic progress accuracy."
        ),
        "overall": summarize(rows),
        "per_task": {
            task: summarize([row for row in rows if row["task"] == task])
            for task in tasks
        },
    }


def main() -> int:
    args = parse_args()
    if not args.allow_frozen_test:
        raise PermissionError("pass --allow-frozen-test after explicit approval")
    if args.num_workers < 1:
        raise ValueError("num-workers must be positive")
    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    output_dir = args.output_dir.expanduser().resolve()
    jobs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for task, spec in split_manifest["tasks"].items():
        for episode in spec["episodes"]:
            output_path = output_dir / f"{task}_ep{int(episode):06d}.json"
            if output_path.is_file():
                try:
                    row = json.loads(output_path.read_text(encoding="utf-8"))
                    if (
                        row.get("protocol") == "gt_replay_online_dtw_v1"
                        and row.get("evaluation_split_sha256")
                        == split_manifest["all_episode_sets_sha256"]
                    ):
                        rows.append(row)
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
            jobs.append(
                {
                    "config": str(args.config.expanduser().resolve()),
                    "task": task,
                    "episode": int(episode),
                    "output_path": str(output_path),
                    "split_sha256": split_manifest["all_episode_sets_sha256"],
                }
            )
    print(f"gt_replay queued={len(jobs)} existing={len(rows)}", flush=True)
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(evaluate_episode, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            row = future.result()
            rows.append(row)
            print(
                f"gt_replay {index}/{len(jobs)} {job['task']} "
                f"ep={job['episode']:06d} stage_acc={row['stage_accuracy']:.3f} "
                f"task_progress_mae={row['task_progress_mae']:.4f}",
                flush=True,
            )
    rows.sort(key=lambda row: (row["task"], row["episode"]))
    if len(rows) != 200:
        raise RuntimeError(f"expected 200 metrics rows, got {len(rows)}")
    summary = aggregate(rows, split_manifest["all_episode_sets_sha256"])
    atomic_write_json(output_dir.parent / "dtw_gt_replay_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

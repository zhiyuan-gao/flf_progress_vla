#!/usr/bin/env python3
"""Run the frozen gt_test50_v1 paired-reference evaluation with persistent workers."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_POLICY_MODES = {
    "conditioned_reference_dtw",
    "semantic_goal_reference_dtw",
}


def is_reference_policy_mode(policy_mode: str) -> bool:
    return policy_mode in REFERENCE_POLICY_MODES


def resolve_execute_steps(config: dict[str, Any], override: int | None) -> int:
    horizon = int(config["model"]["action_horizon"])
    configured = int(config.get("action", {}).get("max_execute_steps", horizon))
    execute_steps = int(override or configured)
    if not 1 <= execute_steps <= horizon:
        raise ValueError(f"execute-steps must be in [1, {horizon}]")
    return execute_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/mvp.json")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=REPO_ROOT / "configs/gt_test50_v1.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "outputs/mvp_continuation/checkpoint-10000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/reference_video_rollout/gt_test50_v1/reference_dtw",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--execute-steps", type=int, default=None)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=20260805)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--policy-mode",
        choices=(
            "conditioned_reference_dtw",
            "semantic_goal_reference_dtw",
            "official_unconditioned",
        ),
        default="conditioned_reference_dtw",
        help=(
            "Use the v1 conditioned policy, the semantic-goal policy, or the native "
            "official GR00T policy."
        ),
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--allow-frozen-test", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: Any) -> None:
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def episode_seed(namespace: str, base_seed: int, task: str, episode: int) -> int:
    material = f"{namespace}|policy_seed|{base_seed}|{task}|{episode:06d}".encode()
    return int.from_bytes(sha256(material).digest()[:4], byteorder="big") & 0x7FFFFFFF


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def result_is_complete(
    summary_path: Path,
    *,
    split_sha256: str,
    checkpoint: Path,
) -> bool:
    if not summary_path.is_file():
        return False
    try:
        payload = load_json(summary_path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("evaluation_split_sha256") == split_sha256
        and Path(payload.get("checkpoint", "")).resolve() == checkpoint.resolve()
        and payload.get("termination_reason") in {
            "success",
            "max_steps",
            "environment_termination",
        }
    )


def build_jobs(
    split_manifest: dict[str, Any],
    output_dir: Path,
    checkpoint: Path,
) -> tuple[list[dict[str, Any]], list[Path]]:
    jobs: list[dict[str, Any]] = []
    completed: list[Path] = []
    tasks = list(split_manifest["tasks"])
    split_sha256 = str(split_manifest["all_episode_sets_sha256"])
    # Interleave tasks so early progress covers all four tasks and the dynamic
    # queue naturally balances their different horizons.
    episode_lists = {
        task: [int(value) for value in split_manifest["tasks"][task]["episodes"]]
        for task in tasks
    }
    for index in range(max(len(values) for values in episode_lists.values())):
        for task in tasks:
            values = episode_lists[task]
            if index >= len(values):
                continue
            episode = values[index]
            episode_dir = output_dir / f"{task}_ep{episode:06d}"
            summary_path = episode_dir / "summary.json"
            if result_is_complete(
                summary_path,
                split_sha256=split_sha256,
                checkpoint=checkpoint,
            ):
                completed.append(summary_path)
                continue
            jobs.append(
                {
                    "task": task,
                    "episode": episode,
                    "output_dir": str(episode_dir),
                    "attempt": 1,
                }
            )
    return jobs, completed


def worker_main(
    worker_id: int,
    gpu_id: int,
    job_queue: Any,
    result_queue: Any,
    settings: dict[str, Any],
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MUJOCO_GL"] = "egl"
    # robosuite validates this against the physical IDs listed in
    # CUDA_VISIBLE_DEVICES, while PyTorch sees the selected card as cuda:0.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

    try:
        import gc

        import gymnasium as gym
        import imageio.v2 as imageio
        import numpy as np
        import robocasa  # noqa: F401 - registers robocasa/* Gym environments
        import torch
        from robocasa.utils.dataset_registry_utils import get_task_horizon

        from rollout_reference_video import reference_summary, restore_episode_start
        from stage_state_vla.bootstrap import activate_gr00t, validate_python_environment
        from stage_state_vla.config import (
            load_config,
            resolve_paths,
            validate_external_paths,
        )
        from stage_state_vla.index import build_episode_records, find_dataset
        from stage_state_vla.inference import (
            ConditionedChunkPredictor,
            ReferenceVideoConditionSource,
            SemanticGoalChunkPredictor,
            add_observation_horizon,
            concatenate_action_chunk,
            split_action,
        )
        from stage_state_vla.policy import (
            make_conditioned_policy_class,
            make_semantic_goal_policy_class,
        )
        from stage_state_vla.reference_video import (
            RGBReferenceEncoder,
            RollingSubsequenceDTWProgressLocalizer,
            SubsequenceDTWLocalizer,
            build_ground_truth_references,
        )
        from stage_state_vla.semantic_goal import (
            CURRENT_CAMERA_KEYS,
            GoalImageStore,
            make_semantic_goal_transform,
        )
        from stage_state_vla.dataset import create_base_datasets

        config = load_config(settings["config"])
        paths = resolve_paths(config)
        validate_external_paths(paths, required=("gr00t_root", "data_root"))
        validate_python_environment()
        activate_gr00t(paths.gr00t_root)

        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.policy import Gr00tPolicy

        checkpoint = Path(settings["checkpoint"]).resolve()
        condition_cfg = config["condition"]
        localizer_cfg = condition_cfg["localizer"]
        data_config = DATA_CONFIG_MAP[config["model"]["data_config"]]
        policy_mode = str(settings["policy_mode"])
        reference_mode = is_reference_policy_mode(policy_mode)
        semantic_mode = policy_mode == "semantic_goal_reference_dtw"
        method_family = config.get("method_family")
        if semantic_mode and method_family != "semantic_goal":
            raise ValueError("semantic_goal_reference_dtw requires a semantic-goal config")
        if policy_mode == "conditioned_reference_dtw" and method_family == "semantic_goal":
            raise ValueError(
                "semantic-goal configs require policy-mode semantic_goal_reference_dtw"
            )
        if policy_mode == "official_unconditioned":
            Policy = Gr00tPolicy
        elif semantic_mode:
            Policy = make_semantic_goal_policy_class()
        else:
            Policy = make_conditioned_policy_class()
        torch.cuda.set_device(0)
        print(
            f"[worker={worker_id} gpu={gpu_id}] loading checkpoint {checkpoint}",
            flush=True,
        )
        policy_kwargs = {
            "model_path": str(checkpoint),
            "modality_config": data_config.modality_config(),
            "modality_transform": (
                make_semantic_goal_transform(
                    goal_image_count=int(config["goal"]["goal_image_count"])
                )
                if semantic_mode
                else data_config.transform()
            ),
            "embodiment_tag": config["model"]["embodiment_tag"],
            "denoising_steps": int(settings["denoising_steps"]),
            "device": "cuda:0",
        }
        if policy_mode == "conditioned_reference_dtw":
            policy_kwargs["max_stages"] = int(condition_cfg["max_stages"])
        policy = Policy(
            **policy_kwargs,
        )
        encoder = (
            RGBReferenceEncoder(
                camera_keys=localizer_cfg["camera_keys"],
                spatial_size=int(localizer_cfg["image_size"]),
            )
            if reference_mode
            else None
        )
        goal_store = (
            GoalImageStore((paths.repo_root / config["goal"]["cache_dir"]).resolve())
            if semantic_mode
            else None
        )
        goal_base_datasets: dict[str, Any] = {}
        print(
            f"[worker={worker_id} gpu={gpu_id}] ready mode={policy_mode}",
            flush=True,
        )

        while True:
            job = job_queue.get()
            if job is None:
                break
            task = str(job["task"])
            episode = int(job["episode"])
            output_dir = Path(job["output_dir"]).resolve()
            attempt = int(job["attempt"])
            started = time.time()
            print(
                f"[worker={worker_id} gpu={gpu_id}] start {task} ep={episode:06d} "
                f"attempt={attempt}",
                flush=True,
            )
            try:
                seed = episode_seed(
                    settings["split_namespace"],
                    int(settings["base_seed"]),
                    task,
                    episode,
                )
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                dataset = find_dataset(paths.data_root / task)
                stage_count = int(config["stage_counts"][task])
                references = None
                predictor = None
                if reference_mode:
                    records = build_episode_records(
                        task=task,
                        split=settings["split_namespace"],
                        dataset=dataset,
                        episode=episode,
                        stage_count=stage_count,
                        reference_stride=int(condition_cfg["reference_stride"]),
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
                    condition_source = ReferenceVideoConditionSource(
                        localizer,
                        num_stages=stage_count,
                        completion_threshold=float(condition_cfg["completion_threshold"]),
                        confirmations=int(condition_cfg["completion_confirmations"]),
                    )
                    if semantic_mode:
                        assert goal_store is not None
                        endpoint_rows = {
                            row.stage_index: row for row in records
                        }
                        missing = [
                            row
                            for row in endpoint_rows.values()
                            if not goal_store.path_for(row).is_file()
                        ]
                        if missing:
                            if task not in goal_base_datasets:
                                goal_base_datasets.update(
                                    create_base_datasets(
                                        records,
                                        training=False,
                                        video_backend=config["training"]["video_backend"],
                                    )
                                )
                            base = goal_base_datasets[task]
                            for row in missing:
                                raw = base.get_step_data(row.episode, row.segment_end)
                                goal_store.save(
                                    row,
                                    {key: raw[key] for key in CURRENT_CAMERA_KEYS},
                                )
                        predictor = SemanticGoalChunkPredictor(
                            policy,
                            condition_source,
                            records,
                            goal_store,
                        )
                    else:
                        predictor = ConditionedChunkPredictor(policy, condition_source)
                max_steps = int(get_task_horizon(task))
                run_manifest = {
                    "task": task,
                    "episode_split": settings["split_namespace"],
                    "reference_episode": (
                        episode if reference_mode else None
                    ),
                    "dataset": str(dataset),
                    "policy_mode": policy_mode,
                    "condition_source": (
                        "reference_dtw" if reference_mode else "none"
                    ),
                    "semantic_goal": config.get("goal") if semantic_mode else None,
                    "evaluation_split_sha256": settings["split_sha256"],
                    "checkpoint": str(checkpoint),
                    "reference_stride": int(condition_cfg["reference_stride"]),
                    "query_stride": int(localizer_cfg["query_stride"]),
                    "query_length": int(localizer_cfg["query_length"]),
                    "feature_source": localizer_cfg["feature_source"],
                    "dtw": localizer_cfg,
                    "execute_steps": int(settings["execute_steps"]),
                    "stage_count": stage_count,
                    "references": (
                        reference_summary(references) if references is not None else None
                    ),
                    "max_steps": max_steps,
                    "denoising_steps": int(settings["denoising_steps"]),
                    "seed": seed,
                    "seed_derivation": settings["seed_derivation"],
                    "worker_id": worker_id,
                    "physical_gpu": gpu_id,
                    "attempt": attempt,
                }
                atomic_write_json(output_dir / "run_manifest.json", run_manifest)

                gym_env = None
                writer = None
                conditions: list[dict[str, Any]] = []
                rollout_started = time.time()
                step = 0
                success = False
                termination_reason = "max_steps"
                try:
                    gym_env = gym.make(
                        f"robocasa/{task}", split="target", enable_render=True
                    )
                    observation = restore_episode_start(gym_env, dataset, episode)
                    if not settings["no_video"]:
                        writer = imageio.get_writer(
                            output_dir / "rollout.mp4",
                            format="FFMPEG",
                            fps=20,
                            codec="libx264",
                        )
                    while step < max_steps:
                        if predictor is not None:
                            chunk, condition = predictor.predict(observation, step)
                            conditions.append(
                                {"control_step": step, **condition.as_dict()}
                            )
                        else:
                            action_dict = policy.get_action(
                                add_observation_horizon(observation)
                            )
                            chunk = concatenate_action_chunk(action_dict)
                            condition = None
                        if step == 0 or step % 320 == 0:
                            status = (
                                f"stage={condition.stage_index} "
                                f"progress={condition.progress:.3f}"
                                if condition is not None
                                else "native_official_policy"
                            )
                            print(
                                f"[worker={worker_id} gpu={gpu_id}] {task} "
                                f"ep={episode:06d} step={step}/{max_steps} {status}",
                                flush=True,
                            )
                        for within_chunk in range(
                            min(int(settings["execute_steps"]), max_steps - step)
                        ):
                            observation, _, terminated, truncated, info = gym_env.step(
                                split_action(chunk[within_chunk])
                            )
                            step += 1
                            if predictor is not None:
                                predictor.observe(observation, step)
                            if writer is not None:
                                writer.append_data(
                                    np.asarray(
                                        observation["video.robot0_agentview_left"]
                                    )
                                )
                            success = bool(info.get("success", False))
                            if success:
                                termination_reason = "success"
                                break
                            if terminated or truncated:
                                termination_reason = "environment_termination"
                                break
                        if success or termination_reason == "environment_termination":
                            break
                finally:
                    if writer is not None:
                        writer.close()
                    if gym_env is not None:
                        gym_env.close()

                summary = {
                    **run_manifest,
                    "steps_executed": step,
                    "policy_calls": len(conditions),
                    "success": success,
                    "termination_reason": termination_reason,
                    "rollout_seconds": time.time() - rollout_started,
                    "total_job_seconds": time.time() - started,
                    "final_condition": conditions[-1] if conditions else None,
                    "completed_at": utc_now(),
                }
                if predictor is not None:
                    atomic_write_json(output_dir / "conditions.json", conditions)
                atomic_write_json(output_dir / "summary.json", summary)
                (output_dir / "error.json").unlink(missing_ok=True)
                result_queue.put(
                    {
                        "status": "completed",
                        "job": job,
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                        "policy_success": success,
                        "steps": step,
                        "seconds": time.time() - started,
                    }
                )
                print(
                    f"[worker={worker_id} gpu={gpu_id}] done {task} ep={episode:06d} "
                    f"success={success} steps={step} seconds={time.time()-started:.1f}",
                    flush=True,
                )
            except Exception as exc:
                error = {
                    "task": task,
                    "episode": episode,
                    "attempt": attempt,
                    "worker_id": worker_id,
                    "physical_gpu": gpu_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "failed_at": utc_now(),
                }
                atomic_write_json(output_dir / "error.json", error)
                result_queue.put(
                    {
                        "status": "error",
                        "job": job,
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                        "error": error,
                    }
                )
                print(
                    f"[worker={worker_id} gpu={gpu_id}] ERROR {task} "
                    f"ep={episode:06d}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            finally:
                gc.collect()
                torch.cuda.empty_cache()
    except Exception as exc:
        result_queue.put(
            {
                "status": "worker_fatal",
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise


def main() -> int:
    args = parse_args()
    if not args.allow_frozen_test:
        raise PermissionError(
            "gt_test50_v1 is frozen evaluation data; pass --allow-frozen-test only "
            "after explicit approval"
        )
    if args.num_workers < 1 or args.num_gpus < 1:
        raise ValueError("num-workers and num-gpus must be positive")
    if args.max_attempts < 1:
        raise ValueError("max-attempts must be positive")

    config_path = args.config.expanduser().resolve()
    split_path = args.split_manifest.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    config = load_json(config_path)
    execute_steps = resolve_execute_steps(config, args.execute_steps)
    if (
        args.policy_mode == "semantic_goal_reference_dtw"
        and config.get("method_family") != "semantic_goal"
    ):
        raise ValueError("semantic_goal_reference_dtw requires a semantic-goal config")
    split_manifest = load_json(split_path)
    if split_manifest.get("name") != "gt_test50_v1":
        raise ValueError(f"unexpected frozen split name in {split_path}")
    if any(len(spec["episodes"]) != 50 for spec in split_manifest["tasks"].values()):
        raise ValueError("gt_test50_v1 must contain exactly 50 episodes per task")

    jobs, existing = build_jobs(split_manifest, output_dir, checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_derivation = (
        "uint31(first 4 bytes, big-endian, of SHA-256("
        "'{namespace}|policy_seed|{base_seed}|{task}|{episode:06d}'))"
    )
    settings = {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "execute_steps": execute_steps,
        "denoising_steps": args.denoising_steps,
        "base_seed": args.base_seed,
        "split_namespace": split_manifest["name"],
        "split_sha256": split_manifest["all_episode_sets_sha256"],
        "seed_derivation": seed_derivation,
        "no_video": args.no_video,
        "policy_mode": args.policy_mode,
    }
    batch_manifest = {
        "name": f"gt_test50_v1_{args.policy_mode}",
        "started_at": utc_now(),
        "config": str(config_path),
        "split_manifest": str(split_path),
        "evaluation_split_sha256": split_manifest["all_episode_sets_sha256"],
        "checkpoint": str(checkpoint),
        "policy_mode": args.policy_mode,
        "condition_source": (
            "reference_dtw"
            if is_reference_policy_mode(args.policy_mode)
            else "none"
        ),
        "method_family": config.get("method_family", "ordinal_stage"),
        "semantic_goal": config.get("goal"),
        "output_dir": str(output_dir),
        "num_workers": args.num_workers,
        "num_gpus": args.num_gpus,
        "worker_to_physical_gpu": {
            str(worker_id): worker_id % args.num_gpus
            for worker_id in range(args.num_workers)
        },
        "execute_steps": execute_steps,
        "denoising_steps": args.denoising_steps,
        "base_seed": args.base_seed,
        "seed_derivation": seed_derivation,
        "video": not args.no_video,
        "total_episodes": sum(
            len(spec["episodes"]) for spec in split_manifest["tasks"].values()
        ),
        "already_complete_at_start": len(existing),
        "queued_at_start": len(jobs),
    }
    atomic_write_json(output_dir.parent / "batch_manifest.json", batch_manifest)
    print(json.dumps(batch_manifest, indent=2), flush=True)
    if not jobs:
        print("All frozen evaluation episodes are already complete.", flush=True)
        return 0

    context = mp.get_context("spawn")
    job_queue = context.Queue()
    result_queue = context.Queue()
    for job in jobs:
        job_queue.put(job)

    workers: list[mp.Process] = []
    for worker_id in range(args.num_workers):
        gpu_id = worker_id % args.num_gpus
        process = context.Process(
            target=worker_main,
            args=(worker_id, gpu_id, job_queue, result_queue, settings),
            name=f"gt-test50-worker-{worker_id}-gpu-{gpu_id}",
        )
        process.start()
        workers.append(process)

    target = len(jobs)
    terminal = 0
    completed = 0
    policy_successes = 0
    terminal_errors: list[dict[str, Any]] = []
    started = time.time()
    try:
        while terminal < target:
            try:
                result = result_queue.get(timeout=30)
            except queue.Empty:
                dead = [worker for worker in workers if not worker.is_alive()]
                if dead:
                    raise RuntimeError(
                        "workers exited before reporting their active jobs: "
                        + ", ".join(
                            f"{worker.name}(exit={worker.exitcode})" for worker in dead
                        )
                    )
                continue

            status = result["status"]
            if status == "worker_fatal":
                raise RuntimeError(
                    f"worker {result['worker_id']} fatal error: {result['error']}"
                )
            if status == "completed":
                terminal += 1
                completed += 1
                policy_successes += int(result["policy_success"])
            elif status == "error":
                job = result["job"]
                if int(job["attempt"]) < args.max_attempts:
                    retry = {**job, "attempt": int(job["attempt"]) + 1}
                    job_queue.put(retry)
                    print(
                        f"retrying {job['task']} ep={int(job['episode']):06d} "
                        f"attempt={retry['attempt']}",
                        flush=True,
                    )
                else:
                    terminal += 1
                    terminal_errors.append(result["error"])
            else:
                raise RuntimeError(f"unknown worker result status {status!r}")

            progress = {
                "updated_at": utc_now(),
                "target_jobs_this_run": target,
                "terminal_jobs_this_run": terminal,
                "completed_jobs_this_run": completed,
                "policy_successes_this_run": policy_successes,
                "terminal_errors_this_run": len(terminal_errors),
                "already_complete_at_start": len(existing),
                "elapsed_seconds": time.time() - started,
            }
            atomic_write_json(output_dir.parent / "batch_progress.json", progress)
            print(
                f"[batch] terminal={terminal}/{target} completed={completed} "
                f"policy_successes={policy_successes} errors={len(terminal_errors)}",
                flush=True,
            )
    finally:
        for _ in workers:
            job_queue.put(None)
        for worker in workers:
            worker.join(timeout=30)
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(timeout=10)

    summaries = sorted(output_dir.glob("*_ep*/summary.json"))
    summary_payloads = [load_json(path) for path in summaries]
    batch_summary = {
        **batch_manifest,
        "completed_at": utc_now(),
        "elapsed_seconds": time.time() - started,
        "completed_episodes": len(summary_payloads),
        "successful_episodes": sum(bool(row["success"]) for row in summary_payloads),
        "success_rate": (
            sum(bool(row["success"]) for row in summary_payloads) / len(summary_payloads)
            if summary_payloads
            else None
        ),
        "per_task": {
            task: {
                "episodes": sum(row["task"] == task for row in summary_payloads),
                "successes": sum(
                    row["task"] == task and bool(row["success"])
                    for row in summary_payloads
                ),
            }
            for task in split_manifest["tasks"]
        },
        "terminal_errors": terminal_errors,
    }
    for stats in batch_summary["per_task"].values():
        stats["success_rate"] = (
            stats["successes"] / stats["episodes"] if stats["episodes"] else None
        )
    atomic_write_json(output_dir.parent / "batch_summary.json", batch_summary)
    print(json.dumps(batch_summary, indent=2), flush=True)
    return 1 if terminal_errors else 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

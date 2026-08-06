#!/usr/bin/env python3
"""Run conditioned GR00T from an exact RoboCasa episode with a GT reference video."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any


STANDALONE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STANDALONE / "src"))

import numpy as np

from stage_state_vla.bootstrap import activate_gr00t, validate_python_environment
from stage_state_vla.config import load_config, resolve_paths, validate_external_paths
from stage_state_vla.index import (
    build_episode_records,
    find_dataset,
    read_split_config,
)
from stage_state_vla.inference import (
    ConditionedChunkPredictor,
    ReferenceClockConditionSource,
    ReferenceVideoConditionSource,
    SemanticGoalChunkPredictor,
    split_action,
)
from stage_state_vla.policy import make_conditioned_policy_class, make_semantic_goal_policy_class
from stage_state_vla.reference_video import (
    RGBReferenceEncoder,
    RollingSubsequenceDTWProgressLocalizer,
    SubsequenceDTWLocalizer,
    build_ground_truth_references,
)
from stage_state_vla.semantic_goal import GoalImageStore, make_semantic_goal_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=STANDALONE / "configs/mvp.json")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--task", default="PreSoakPan")
    parser.add_argument(
        "--episode-split",
        choices=("train", "val", "test"),
        default="val",
        help="Fixed repository split used to select/validate the reference episode.",
    )
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument(
        "--condition-source",
        choices=("reference_dtw", "reference_clock"),
        default="reference_dtw",
    )
    parser.add_argument("--execute-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--spatial-size", type=int, default=None)
    parser.add_argument("--query-length", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-locked-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def select_episode(args: argparse.Namespace, split_config: Path) -> int:
    payload = read_split_config(split_config)
    if args.task not in payload["tasks"]:
        raise ValueError(f"task {args.task!r} is absent from {split_config}")
    allowed = [int(value) for value in payload["tasks"][args.task]["splits"][args.episode_split]]
    episode = allowed[0] if args.episode is None else int(args.episode)
    if episode not in allowed:
        raise ValueError(
            f"episode {episode} is not in the fixed {args.episode_split} split for {args.task}"
        )
    if args.episode_split == "test" and not args.allow_locked_test:
        raise PermissionError(
            "the test split is locked; pass --allow-locked-test only for an explicitly approved "
            "final evaluation"
        )
    return episode


def restore_episode_start(gym_env: Any, dataset: Path, episode: int) -> dict[str, Any]:
    """Restore the demonstration XML and MuJoCo state before free rollout."""
    import robosuite

    gym_env.reset()
    wrapper = gym_env.unwrapped
    episode_dir = dataset / "extras" / f"episode_{episode:06d}"
    with np.load(episode_dir / "states.npz") as states_file:
        initial_state = states_file["states"][0]
    with gzip.open(episode_dir / "model.xml.gz", "rt", encoding="utf-8") as xml_file:
        model_xml = xml_file.read()
    ep_meta = json.loads((episode_dir / "ep_meta.json").read_text(encoding="utf-8"))

    env = wrapper.env
    if hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(ep_meta)
    elif hasattr(env, "set_ep_meta"):
        env.set_ep_meta(ep_meta)
    env.reset()
    robosuite_minor = int(robosuite.__version__.split(".")[1])
    if robosuite_minor <= 3:
        from robosuite.utils.mjcf_utils import postprocess_model_xml

        model_xml = postprocess_model_xml(model_xml)
    else:
        model_xml = env.edit_model_xml(model_xml)
    env.reset_from_xml_string(model_xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(initial_state)
    env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()
    raw = wrapper.env._get_observations(force_update=True)
    return wrapper.get_observation(raw)


def reference_summary(references: dict[int, Any]) -> dict[str, Any]:
    return {
        str(stage): {
            "stage_name": reference.stage_name,
            "segment": [reference.segment_start, reference.segment_end],
            "num_nodes": len(reference.node_frames),
            "first_node": reference.node_frames[0],
            "last_node": reference.node_frames[-1],
            "feature_dim": int(reference.features.shape[1]),
        }
        for stage, reference in references.items()
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_paths(config)
    validate_external_paths(
        paths,
        required=("gr00t_root", "data_root", "split_config"),
    )
    if args.task not in config["tasks"]:
        raise ValueError(f"task must be one of {config['tasks']}, got {args.task!r}")
    semantic_goal = config.get("method_family") == "semantic_goal"
    configured_execute_steps = int(
        config.get("action", {}).get("max_execute_steps", config["model"]["action_horizon"])
    )
    execute_steps = int(args.execute_steps or configured_execute_steps)
    if not 1 <= execute_steps <= int(config["model"]["action_horizon"]):
        raise ValueError(
            f"execute-steps must be in [1, {config['model']['action_horizon']}]"
        )
    episode = select_episode(args, paths.split_config)
    dataset = find_dataset(paths.data_root / args.task)
    condition_cfg = config["condition"]
    localizer_cfg = condition_cfg["localizer"]
    spatial_size = int(
        localizer_cfg["image_size"] if args.spatial_size is None else args.spatial_size
    )
    query_length = int(
        localizer_cfg["query_length"] if args.query_length is None else args.query_length
    )
    if int(localizer_cfg["query_stride"]) != int(condition_cfg["reference_stride"]):
        raise ValueError("the validated DTW protocol requires query_stride=reference_stride")
    stage_count = int(config["stage_counts"][args.task])
    records = build_episode_records(
        task=args.task,
        split=args.episode_split,
        dataset=dataset,
        episode=episode,
        stage_count=stage_count,
        reference_stride=int(condition_cfg["reference_stride"]),
    )

    encoder = RGBReferenceEncoder(
        camera_keys=localizer_cfg["camera_keys"],
        spatial_size=spatial_size,
    )
    references: dict[int, Any] = {}
    if args.condition_source == "reference_dtw":
        references = build_ground_truth_references(
            dataset,
            episode,
            records,
            reference_stride=int(condition_cfg["reference_stride"]),
            encoder=encoder,
        )

    default_output_root = paths.repo_root / "outputs/reference_video_rollout"
    if semantic_goal:
        default_output_root = default_output_root / config["goal"]["variant"]
    output_dir = (
        args.output_dir
        or default_output_root / args.condition_source / f"{args.task}_ep{episode:06d}"
    ).resolve()
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite and not args.prepare_only:
        raise FileExistsError(
            f"rollout already exists: {summary_path}; pass --overwrite to replace"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "task": args.task,
        "episode_split": args.episode_split,
        "reference_episode": episode,
        "dataset": dataset,
        "condition_source": args.condition_source,
        "reference_stride": int(condition_cfg["reference_stride"]),
        "query_stride": int(localizer_cfg["query_stride"]),
        "query_length": query_length,
        "feature_source": localizer_cfg["feature_source"],
        "dtw": localizer_cfg,
        "execute_steps": execute_steps,
        "semantic_goal": config.get("goal"),
        "stage_count": stage_count,
        "references": reference_summary(references),
        "prepare_only": args.prepare_only,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    if args.prepare_only:
        print(json.dumps(manifest, indent=2, default=json_default))
        return 0

    checkpoint = (args.checkpoint or paths.output_dir).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"conditioned checkpoint does not exist: {checkpoint}; "
            "train it first or pass --checkpoint"
        )
    validate_python_environment()
    activate_gr00t(paths.gr00t_root)

    import gymnasium as gym
    import imageio.v2 as imageio
    import robocasa  # noqa: F401 - registers robocasa/* Gym environments
    import torch
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_config = DATA_CONFIG_MAP[config["model"]["data_config"]]
    Policy = make_semantic_goal_policy_class() if semantic_goal else make_conditioned_policy_class()
    modality_transform = (
        make_semantic_goal_transform(goal_image_count=int(config["goal"]["goal_image_count"]))
        if semantic_goal
        else data_config.transform()
    )
    print(f"Loading conditioned checkpoint: {checkpoint}", flush=True)
    policy = Policy(
        model_path=str(checkpoint),
        modality_config=data_config.modality_config(),
        modality_transform=modality_transform,
        embodiment_tag=config["model"]["embodiment_tag"],
        denoising_steps=args.denoising_steps,
        device=args.device,
        **({} if semantic_goal else {"max_stages": int(condition_cfg["max_stages"])}),
    )

    if args.condition_source == "reference_dtw":
        localizer = RollingSubsequenceDTWProgressLocalizer(
            references,
            encoder=encoder,
            query_length=query_length,
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
    else:
        progress_field = (
            "exact_progress" if condition_cfg.get("progress_source") == "exact" else "grid_progress"
        )
        condition_source = ReferenceClockConditionSource(records, progress_field=progress_field)
    if semantic_goal:
        goal_store = GoalImageStore((paths.repo_root / config["goal"]["cache_dir"]).resolve())
        predictor = SemanticGoalChunkPredictor(policy, condition_source, records, goal_store)
    else:
        predictor = ConditionedChunkPredictor(policy, condition_source)

    max_steps = int(args.max_steps or get_task_horizon(args.task))
    manifest.update(
        {
            "checkpoint": checkpoint,
            "max_steps": max_steps,
            "denoising_steps": args.denoising_steps,
            "seed": args.seed,
            "device": args.device,
        }
    )
    write_json(output_dir / "run_manifest.json", manifest)

    gym_env = gym.make(f"robocasa/{args.task}", split="target", enable_render=True)
    observation = restore_episode_start(gym_env, dataset, episode)
    writer = None
    if not args.no_video:
        writer = imageio.get_writer(
            output_dir / "rollout.mp4",
            format="FFMPEG",
            fps=20,
            codec="libx264",
        )
    conditions: list[dict[str, Any]] = []
    started = time.time()
    step = 0
    success = False
    termination_reason = "max_steps"
    try:
        while step < max_steps:
            chunk, condition = predictor.predict(observation, step)
            conditions.append({"control_step": step, **condition.as_dict()})
            print(
                f"step={step:04d}/{max_steps} stage={condition.stage_index} "
                f"progress={condition.progress:.3f} advanced={condition.advanced}",
                flush=True,
            )
            for within_chunk in range(min(execute_steps, max_steps - step)):
                observation, _, terminated, truncated, info = gym_env.step(
                    split_action(chunk[within_chunk])
                )
                step += 1
                predictor.observe(observation, step)
                if writer is not None:
                    writer.append_data(
                        np.asarray(observation["video.robot0_agentview_left"])
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
        gym_env.close()

    summary = {
        **manifest,
        "steps_executed": step,
        "policy_calls": len(conditions),
        "success": success,
        "termination_reason": termination_reason,
        "elapsed_seconds": time.time() - started,
        "final_condition": conditions[-1] if conditions else None,
    }
    write_json(output_dir / "conditions.json", conditions)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

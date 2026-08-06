#!/usr/bin/env python3
"""Continue GR00T training with ordinal stage and continuous video progress."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


STANDALONE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STANDALONE / "src"))

import numpy as np
import torch
from transformers import TrainingArguments

from stage_state_vla.bootstrap import activate_gr00t, validate_python_environment
from stage_state_vla.config import load_config, resolve_paths, validate_external_paths
from stage_state_vla.dataset import load_semantic_goal_dataset, load_stage_dataset
from stage_state_vla.trainer import make_task_balanced_trainer_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=STANDALONE / "configs/mvp.json")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--per-device-batch-size", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-index", type=Path, default=None)
    parser.add_argument("--val-index", type=Path, default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run the requested steps without writing final checkpoint weights.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def allow_local_rng_state_for_resume() -> None:
    """Allow the NumPy RNG types written by HF Trainer in our local checkpoint."""
    from numpy._core.multiarray import _reconstruct

    torch.serialization.add_safe_globals(
        [_reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))]
    )


def close_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main() -> int:
    args = parse_args()
    if args.resume:
        allow_local_rng_state_for_resume()
    config = load_config(args.config)
    paths = resolve_paths(config)
    validate_external_paths(
        paths,
        required=(
            "gr00t_root",
            "data_root",
            "split_config",
            "base_checkpoint",
            "index_dir",
        ),
    )
    validate_python_environment()
    activate_gr00t(paths.gr00t_root)
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from gr00t.model.transforms import DefaultDataCollator
    from gr00t.utils.experiment import (
        CheckpointFormatCallback,
        safe_save_model_for_hf_trainer,
    )
    train_cfg = config["training"]
    condition_cfg = config["condition"]
    model_cfg = config["model"]
    max_steps = int(args.max_steps or train_cfg["max_steps"])
    output_dir = (args.output_dir or paths.output_dir).resolve()
    train_index = (args.train_index or paths.index_dir / "train.jsonl").resolve()
    val_index = (args.val_index or paths.index_dir / "val.jsonl").resolve()
    if not train_index.is_file():
        raise FileNotFoundError(f"build the train index first: {train_index}")
    if not args.no_eval and not val_index.is_file():
        raise FileNotFoundError(f"build the validation index first: {val_index}")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_batch = int(args.per_device_batch_size or train_cfg["per_device_batch_size"])
    global_batch = int(args.global_batch_size or train_cfg["global_batch_size"])
    denominator = world_size * local_batch
    if global_batch % denominator:
        raise ValueError(
            f"global batch {global_batch} must divide world_size*local_batch={denominator}"
        )
    grad_accum = global_batch // denominator

    semantic_goal = config.get("method_family") == "semantic_goal"
    normalization_metadata = paths.base_checkpoint / "experiment_cfg/metadata.json"
    if semantic_goal:
        goal_cache_dir = (paths.repo_root / config["goal"]["cache_dir"]).resolve()
        train_dataset = load_semantic_goal_dataset(
            train_index,
            training=True,
            goal_cache_dir=goal_cache_dir,
            goal_image_count=int(config["goal"]["goal_image_count"]),
            video_backend=train_cfg["video_backend"],
            progress_jitter_frames=int(condition_cfg["progress_jitter_frames"]),
            seed=train_cfg["seed"],
            action_horizon=model_cfg["action_horizon"],
            normalization_metadata=normalization_metadata,
        )
    else:
        train_dataset = load_stage_dataset(
            train_index,
            training=True,
            video_backend=train_cfg["video_backend"],
            max_stages=condition_cfg["max_stages"],
            progress_source=condition_cfg["progress_source"],
            progress_noise_nodes=condition_cfg["progress_noise_nodes"],
            seed=train_cfg["seed"],
            action_horizon=model_cfg["action_horizon"],
            normalization_metadata=normalization_metadata,
        )
    val_dataset = None
    if not args.no_eval:
        if semantic_goal:
            val_dataset = load_semantic_goal_dataset(
                val_index,
                training=False,
                goal_cache_dir=goal_cache_dir,
                goal_image_count=int(config["goal"]["goal_image_count"]),
                video_backend=train_cfg["video_backend"],
                progress_jitter_frames=0,
                seed=train_cfg["seed"],
                action_horizon=model_cfg["action_horizon"],
                normalization_metadata=normalization_metadata,
            )
        else:
            val_dataset = load_stage_dataset(
                val_index,
                training=False,
                video_backend=train_cfg["video_backend"],
                max_stages=condition_cfg["max_stages"],
                progress_source=condition_cfg["progress_source"],
                progress_noise_nodes=0,
                seed=train_cfg["seed"],
                action_horizon=model_cfg["action_horizon"],
                normalization_metadata=normalization_metadata,
            )

    model = GR00T_N1_5.from_pretrained(
        str(paths.base_checkpoint),
        tune_visual=model_cfg["tune_visual"],
        tune_llm=model_cfg["tune_llm"],
        tune_projector=model_cfg["tune_projector"],
        tune_diffusion_model=model_cfg["tune_diffusion_model"],
    )
    model.compute_dtype = model_cfg["compute_dtype"]
    model.config.compute_dtype = model_cfg["compute_dtype"]
    if model.action_head.config.max_state_dim != 64:
        raise RuntimeError("this method requires the native 64-D GR00T state input")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_experiment_cfg = paths.base_checkpoint / "experiment_cfg"
    target_experiment_cfg = output_dir / "experiment_cfg"
    if source_experiment_cfg.is_dir():
        shutil.copytree(source_experiment_cfg, target_experiment_cfg, dirs_exist_ok=True)
    run_contract = {
        "method": config["method"],
        "base_checkpoint": str(paths.base_checkpoint),
        "train_index": str(train_index),
        "val_index": None if args.no_eval else str(val_index),
        "max_steps": max_steps,
        "world_size": world_size,
        "per_device_batch_size": local_batch,
        "gradient_accumulation_steps": grad_accum,
        "effective_global_batch_size": denominator * grad_accum,
        "periodic_eval": not args.no_eval,
        "task_sampling": "uniform task then uniform frame",
        "condition_slots": (
            "first unused dimension is within-subtask progress in native state[64]"
            if semantic_goal
            else "first two unused dimensions in native state[64]"
        ),
        "normalization_metadata": str(
            paths.base_checkpoint / "experiment_cfg/metadata.json"
        ),
        "condition": condition_cfg,
        "goal": config.get("goal"),
        "action": config.get("action"),
        "model": model_cfg,
    }
    run_config_name = (
        "semantic_goal_run_config.json" if semantic_goal else "stage_state_run_config.json"
    )
    (output_dir / run_config_name).write_text(
        json.dumps(run_contract, indent=2) + "\n", encoding="utf-8"
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(run_contract, indent=2), flush=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        remove_unused_columns=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=local_batch,
        per_device_eval_batch_size=local_batch,
        gradient_accumulation_steps=grad_accum,
        dataloader_num_workers=int(train_cfg["dataloader_num_workers"]),
        dataloader_pin_memory=False,
        dataloader_persistent_workers=int(train_cfg["dataloader_num_workers"]) > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        lr_scheduler_type="cosine",
        logging_steps=int(train_cfg["logging_steps"]),
        max_steps=max_steps,
        eval_strategy="no" if args.no_eval else "steps",
        eval_steps=int(train_cfg["eval_steps"]),
        save_strategy="no" if args.smoke_only else "steps",
        save_steps=int(train_cfg["save_steps"]),
        save_total_limit=int(train_cfg["save_total_limit"]),
        report_to="tensorboard",
        seed=int(train_cfg["seed"]),
        do_eval=not args.no_eval,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
    )
    Trainer = make_task_balanced_trainer_class()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DefaultDataCollator(),
        compute_dtype=torch.bfloat16,
    )
    if target_experiment_cfg.is_dir():
        trainer.add_callback(
            CheckpointFormatCallback(
                run_name=output_dir.name,
                exp_cfg_dir=target_experiment_cfg,
            )
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"trainable={trainable:,}/{total:,}; task_counts={train_dataset.task_counts}; "
        f"effective_global_batch={denominator * grad_accum}",
        flush=True,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    if args.smoke_only:
        close_distributed()
        return 0
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=str(output_dir))
    close_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

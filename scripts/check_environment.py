#!/usr/bin/env python3
"""Validate the external resources and software versions without training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stage_state_vla.bootstrap import (
    EXPECTED_GR00T_COMMIT,
    activate_gr00t,
    validate_python_environment,
)
from stage_state_vla.config import load_config, resolve_paths, validate_external_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/mvp.json")
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_paths(config)
    validate_external_paths(paths)
    validate_python_environment()
    activate_gr00t(paths.gr00t_root)
    import gr00t  # noqa: F401
    import torch

    head = git_head(paths.gr00t_root)
    if head is not None and head != EXPECTED_GR00T_COMMIT:
        raise RuntimeError(
            f"GR00T checkout is {head}, expected pinned commit {EXPECTED_GR00T_COMMIT}"
        )
    metadata = paths.base_checkpoint / "experiment_cfg/metadata.json"
    if not metadata.is_file():
        raise FileNotFoundError(f"checkpoint normalization metadata is missing: {metadata}")
    report = {
        "repo_root": str(paths.repo_root),
        "python": sys.version.split()[0],
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "torchvision",
                "transformers",
                "accelerate",
                "numpy",
                "pandas",
                "pyarrow",
            )
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        },
        "resources": {
            "gr00t_root": str(paths.gr00t_root),
            "gr00t_commit": head,
            "data_root": str(paths.data_root),
            "base_checkpoint": str(paths.base_checkpoint),
            "split_config": str(paths.split_config),
        },
        "status": "ok",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

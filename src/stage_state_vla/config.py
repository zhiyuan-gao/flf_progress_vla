"""Portable configuration and external-resource path resolution."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_UNEXPANDED_VARIABLE = re.compile(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    gr00t_root: Path
    data_root: Path
    split_config: Path
    base_checkpoint: Path
    index_dir: Path
    output_dir: Path


def load_dotenv(path: Path, *, override: bool = False) -> None:
    """Load the small KEY=VALUE subset used by this repository."""
    if not path.is_file():
        return
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value[:1] in {'"', "'"}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"{path}:{line_number}: unmatched quote")
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value


def load_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    repo_root = config_path.parent.parent
    load_dotenv(repo_root / ".env")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    config["_repo_root"] = str(repo_root)
    return config


def _resolve_path(repo_root: Path, value: str, key: str) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(value))
    unresolved = _UNEXPANDED_VARIABLE.findall(expanded)
    if unresolved:
        names = ", ".join(unresolved)
        raise EnvironmentError(
            f"Cannot resolve {key}={value!r}; set {names} in the environment "
            f"or {repo_root / '.env'}"
        )
    path = Path(expanded)
    return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()


def resolve_paths(config: dict[str, Any]) -> ProjectPaths:
    repo_root = Path(config["_repo_root"])
    resources = config["resources"]
    return ProjectPaths(
        repo_root=repo_root,
        gr00t_root=_resolve_path(repo_root, resources["gr00t_root"], "gr00t_root"),
        data_root=_resolve_path(repo_root, resources["data_root"], "data_root"),
        split_config=_resolve_path(repo_root, resources["split_config"], "split_config"),
        base_checkpoint=_resolve_path(
            repo_root, resources["base_checkpoint"], "base_checkpoint"
        ),
        index_dir=_resolve_path(repo_root, config["index_dir"], "index_dir"),
        output_dir=_resolve_path(repo_root, config["output_dir"], "output_dir"),
    )


def validate_external_paths(
    paths: ProjectPaths,
    *,
    required: Iterable[str] = ("gr00t_root", "data_root", "split_config", "base_checkpoint"),
) -> None:
    available = {
        "repo_root": paths.repo_root,
        "gr00t_root": paths.gr00t_root,
        "data_root": paths.data_root,
        "split_config": paths.split_config,
        "base_checkpoint": paths.base_checkpoint,
        "index_dir": paths.index_dir,
        "output_dir": paths.output_dir,
    }
    unknown = sorted(set(required) - set(available))
    if unknown:
        raise KeyError(f"unknown path checks requested: {unknown}")
    missing = [f"{name}={available[name]}" for name in required if not available[name].exists()]
    if missing:
        raise FileNotFoundError("Missing required paths: " + ", ".join(missing))

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stage_state_vla.config import load_config, resolve_paths


def write_config(repo: Path) -> Path:
    path = repo / "configs/test.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "resources": {
                    "gr00t_root": "${TEST_GR00T_ROOT}",
                    "data_root": "${TEST_DATA_ROOT}",
                    "split_config": "configs/split.json",
                    "base_checkpoint": "${TEST_CHECKPOINT}",
                },
                "index_dir": "artifacts/indices",
                "output_dir": "outputs/run",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_paths_are_repo_relative_or_environment_driven(tmp_path: Path, monkeypatch):
    config_path = write_config(tmp_path)
    monkeypatch.setenv("TEST_GR00T_ROOT", str(tmp_path / "external/gr00t"))
    monkeypatch.setenv("TEST_DATA_ROOT", str(tmp_path / "external/data"))
    monkeypatch.setenv("TEST_CHECKPOINT", str(tmp_path / "external/checkpoint"))
    paths = resolve_paths(load_config(config_path))
    assert paths.repo_root == tmp_path
    assert paths.split_config == tmp_path / "configs/split.json"
    assert paths.index_dir == tmp_path / "artifacts/indices"
    assert paths.gr00t_root == tmp_path / "external/gr00t"


def test_missing_environment_variable_is_actionable(tmp_path: Path, monkeypatch):
    config_path = write_config(tmp_path)
    for key in ("TEST_GR00T_ROOT", "TEST_DATA_ROOT", "TEST_CHECKPOINT"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(EnvironmentError, match="TEST_GR00T_ROOT"):
        resolve_paths(load_config(config_path))


def test_semantic_goal_variants_share_training_contract(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("GR00T_ROOT", "/external/gr00t")
    monkeypatch.setenv("ROBOCASA365_DATA_ROOT", "/external/data")
    monkeypatch.setenv("GR00T_BASE_CHECKPOINT", "/external/checkpoint-60000")
    goal1 = load_config(root / "configs/semantic_goal1.json")
    goal3 = load_config(root / "configs/semantic_goal3.json")
    assert goal1["goal"]["goal_image_count"] == 1
    assert goal3["goal"]["goal_image_count"] == 3
    assert goal1["condition"] == goal3["condition"]
    assert goal1["action"] == goal3["action"]
    assert goal1["training"]["global_batch_size"] == 128
    assert goal3["training"]["global_batch_size"] == 128

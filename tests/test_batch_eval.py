from __future__ import annotations

import pytest

from scripts.run_gt_test50_batch import (
    is_reference_policy_mode,
    resolve_execute_steps,
)


def config(*, horizon: int = 16, execute_steps: int | None = 8):
    payload = {"model": {"action_horizon": horizon}}
    if execute_steps is not None:
        payload["action"] = {"max_execute_steps": execute_steps}
    return payload


def test_semantic_policy_is_a_reference_mode():
    assert is_reference_policy_mode("conditioned_reference_dtw")
    assert is_reference_policy_mode("semantic_goal_reference_dtw")
    assert not is_reference_policy_mode("official_unconditioned")


def test_execute_steps_uses_semantic_config_and_validates_override():
    assert resolve_execute_steps(config(), None) == 8
    assert resolve_execute_steps(config(execute_steps=None), None) == 16
    assert resolve_execute_steps(config(), 4) == 4
    with pytest.raises(ValueError):
        resolve_execute_steps(config(), 17)

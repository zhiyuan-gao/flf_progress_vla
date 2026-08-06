from __future__ import annotations

import numpy as np
import pytest

from stage_state_vla.conditioning import (
    apply_episode_tail_mask,
    find_injected_condition_slots,
    inject_progress,
    inject_stage_progress,
    normalize_stage,
    overwrite_stage_progress,
)


def base_sample(real_dims: int = 20):
    state = np.zeros((1, 64), dtype=np.float32)
    mask = np.zeros((1, 64), dtype=bool)
    mask[:, :real_dims] = True
    return {"state": state, "state_mask": mask, "action_mask": np.ones((16, 32), bool)}


def test_injection_uses_first_empty_slots_without_resizing():
    sample, slots = inject_stage_progress(base_sample(), stage_index=2, progress=0.75)
    assert sample["state"].shape == (1, 64)
    assert slots.stage == 20 and slots.progress == 21
    np.testing.assert_allclose(sample["state"][0, 20:22], [0.0, 0.5])
    assert sample["state_mask"].sum() == 22
    assert find_injected_condition_slots(sample["state_mask"]) == slots


def test_overwrite_changes_only_condition_values():
    sample, _ = inject_stage_progress(base_sample(), stage_index=0, progress=0.0)
    changed = overwrite_stage_progress(sample, stage_index=4, progress=1.0)
    np.testing.assert_allclose(changed["state"][0, 20:22], [1.0, 1.0])
    np.testing.assert_allclose(changed["state"][..., :20], sample["state"][..., :20])


def test_progress_only_uses_one_empty_state_dimension():
    sample, slot = inject_progress(base_sample(), progress=0.75)
    assert slot == 20
    assert sample["state_mask"].sum() == 21
    assert sample["state"][0, slot] == pytest.approx(0.5)


def test_stage_range_and_episode_tail_mask():
    assert normalize_stage(0) == pytest.approx(-1.0)
    assert normalize_stage(4) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        normalize_stage(5)
    sample = apply_episode_tail_mask(base_sample(), frame=97, episode_length=100)
    assert sample["action_mask"][:3].all()
    assert not sample["action_mask"][3:].any()

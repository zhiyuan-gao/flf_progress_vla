from __future__ import annotations

from pathlib import Path

import numpy as np

from stage_state_vla.index import FrameRecord
from stage_state_vla.semantic_goal import (
    CURRENT_CAMERA_KEYS,
    GoalImageStore,
    append_goal_time_step,
    apply_explicit_hold_padding,
    exact_progress_with_frame_jitter,
    pack_goal_views,
    semantic_prompt,
)


def record() -> FrameRecord:
    return FrameRecord(
        task="Task",
        split="train",
        dataset="/dataset",
        episode=7,
        episode_length=100,
        frame=15,
        stage_index=1,
        stage_count=3,
        stage_name="execute",
        segment_start=10,
        segment_end=20,
        exact_progress=0.5,
        grid_node=1,
        grid_nodes=3,
        grid_progress=0.5,
        task_description="do everything",
        subtask_description="finish this part",
    )


def action_raw() -> dict[str, np.ndarray]:
    return {
        "action.end_effector_position": np.full((16, 3), 3.0),
        "action.end_effector_rotation": np.full((16, 3), 4.0),
        "action.gripper_close": np.arange(16, dtype=np.float32)[:, None],
        "action.base_motion": np.full((16, 4), 5.0),
        "action.control_mode": np.arange(16, dtype=np.float32)[:, None] + 100,
    }


def test_prompt_exact_progress_jitter_and_hold_padding():
    assert semantic_prompt("whole task", "one subtask") == (
        "Task: whole task\nCurrent subtask: one subtask"
    )
    row = record()
    progress = exact_progress_with_frame_jitter(
        row, seed=42, epoch=0, index=3, radius=4
    )
    assert 0.1 <= progress <= 0.9
    assert progress == exact_progress_with_frame_jitter(
        row, seed=42, epoch=0, index=3, radius=4
    )

    padded = apply_explicit_hold_padding(action_raw(), valid_steps=3)
    np.testing.assert_allclose(padded["action.end_effector_position"][:3], 3.0)
    np.testing.assert_allclose(padded["action.end_effector_position"][3:], 0.0)
    np.testing.assert_allclose(padded["action.end_effector_rotation"][3:], 0.0)
    np.testing.assert_allclose(padded["action.base_motion"][3:], 0.0)
    np.testing.assert_allclose(padded["action.gripper_close"][3:], 2.0)
    np.testing.assert_allclose(padded["action.control_mode"][3:], 102.0)


def test_goal_frames_are_packed_in_fixed_goal1_and_goal3_order():
    raw = {
        key: np.full((1, 2, 2, 3), index, dtype=np.uint8)
        for index, key in enumerate(CURRENT_CAMERA_KEYS)
    }
    goals = {
        key: np.full((2, 2, 3), index + 3, dtype=np.uint8)
        for index, key in enumerate(CURRENT_CAMERA_KEYS)
    }
    appended = append_goal_time_step(raw, goals)
    video = np.stack([appended[key] for key in CURRENT_CAMERA_KEYS], axis=1)
    assert video.shape == (2, 3, 2, 2, 3)
    goal1 = pack_goal_views(video, 1)
    goal3 = pack_goal_views(video, 3)
    assert goal1.shape == (1, 4, 2, 2, 3)
    assert goal3.shape == (1, 6, 2, 2, 3)
    assert goal1[:, :, 0, 0, 0].tolist() == [[0, 1, 2, 3]]
    assert goal3[:, :, 0, 0, 0].tolist() == [[0, 1, 2, 3, 4, 5]]

    batched = pack_goal_views(video[None], 1)
    assert batched.shape == (1, 1, 4, 2, 2, 3)


def test_goal_image_store_round_trip(tmp_path: Path):
    store = GoalImageStore(tmp_path)
    goals = {
        key: np.full((4, 4, 3), index, dtype=np.uint8)
        for index, key in enumerate(CURRENT_CAMERA_KEYS)
    }
    path = store.save(record(), goals)
    assert path.is_file()
    loaded = store.load(record())
    for key in CURRENT_CAMERA_KEYS:
        np.testing.assert_array_equal(loaded[key], goals[key])

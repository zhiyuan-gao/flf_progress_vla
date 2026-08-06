from __future__ import annotations

import numpy as np

from stage_state_vla.index import FrameRecord
from stage_state_vla.inference import (
    ConditionedChunkPredictor,
    ReferenceClockConditionSource,
    ReferenceVideoConditionSource,
    SemanticGoalChunkPredictor,
    add_observation_horizon,
    concatenate_action_chunk,
    split_action,
)
from stage_state_vla.reference_video import DTWProgressLocalization


def record(frame: int, stage: int, progress: float) -> FrameRecord:
    return FrameRecord(
        task="Task",
        split="val",
        dataset="/dataset",
        episode=1,
        episode_length=20,
        frame=frame,
        stage_index=stage,
        stage_count=2,
        stage_name=f"stage-{stage}",
        segment_start=0 if stage == 0 else 10,
        segment_end=9 if stage == 0 else 19,
        exact_progress=progress,
        grid_node=round(progress * 2),
        grid_nodes=3,
        grid_progress=progress,
        task_description="do the complete task",
        subtask_description=f"complete stage {stage}",
    )


def test_reference_clock_and_observation_horizon():
    source = ReferenceClockConditionSource(
        [record(0, 0, 0.0), record(8, 0, 1.0), record(10, 1, 0.0)]
    )
    estimate = source.estimate({}, control_step=9)
    assert estimate.stage_index == 0 and estimate.progress == 1.0

    observation = {
        "video.cam": np.zeros((8, 8, 3), dtype=np.uint8),
        "state.arm": np.zeros(3, dtype=np.float32),
        "annotation.human.task_description": "do it",
    }
    expanded = add_observation_horizon(observation)
    assert expanded["video.cam"].shape == (1, 8, 8, 3)
    assert expanded["state.arm"].shape == (1, 3)
    assert expanded["annotation.human.task_description"].shape == (1,)


def test_action_conversion_and_conditioned_predictor():
    action_dict = {
        "action.end_effector_position": np.zeros((16, 3)),
        "action.end_effector_rotation": np.ones((16, 3)),
        "action.gripper_close": np.zeros((16, 1)),
        "action.base_motion": np.ones((16, 4)),
        "action.control_mode": np.zeros((16, 1)),
    }
    chunk = concatenate_action_chunk(action_dict)
    assert chunk.shape == (16, 12)
    assert set(split_action(chunk[0])) == set(action_dict)

    class Policy:
        def get_action_with_condition(self, observations, *, stage_index, video_progress):
            assert observations["state.arm"].shape == (1, 2)
            assert stage_index == 0 and video_progress == 0.0
            return action_dict

    predictor = ConditionedChunkPredictor(
        Policy(), ReferenceClockConditionSource([record(0, 0, 0.0)])
    )
    predicted, condition = predictor.predict({"state.arm": np.zeros(2)}, 0)
    assert predicted.shape == (16, 12)
    assert condition.source == "reference_clock"


def test_semantic_predictor_passes_text_goal_and_progress():
    action_dict = {
        "action.end_effector_position": np.zeros((16, 3)),
        "action.end_effector_rotation": np.zeros((16, 3)),
        "action.gripper_close": np.zeros((16, 1)),
        "action.base_motion": np.zeros((16, 4)),
        "action.control_mode": np.zeros((16, 1)),
    }

    class Policy:
        def get_action_with_semantic_goal(self, observations, **condition):
            assert observations["state.arm"].shape == (1, 2)
            assert condition["video_progress"] == 0.0
            assert condition["task_description"] == "do the complete task"
            assert condition["subtask_description"] == "complete stage 0"
            assert set(condition["goal_images"]) == {"video.goal"}
            return action_dict

    class Store:
        def load(self, row):
            assert row.stage_index == 0
            return {"video.goal": np.zeros((2, 2, 3), dtype=np.uint8)}

    rows = [record(0, 0, 0.0), record(1, 0, 0.1), record(10, 1, 0.0)]
    predictor = SemanticGoalChunkPredictor(
        Policy(), ReferenceClockConditionSource(rows), rows, Store()
    )
    chunk, condition = predictor.predict({"state.arm": np.zeros(2)}, 0)
    assert chunk.shape == (16, 12)
    assert condition.stage_index == 0


def test_repeated_policy_call_does_not_reuse_one_dtw_sample_as_two_confirmations():
    class Localizer:
        def __init__(self):
            self.sample_step = 8

        def localize(self, stage_index, observation, control_step):
            return DTWProgressLocalization(
                stage_index=stage_index,
                start_index=0,
                end_index=9,
                node_frame=72,
                progress=0.95,
                path=(9,),
                query_steps=1,
                sample_step=self.sample_step,
                mean_cost=0.0,
                confidence_margin=1.0,
            )

        def reset(self, stage_index):
            pass

        def observe(self, stage_index, observation, control_step, force=False):
            return True

    localizer = Localizer()
    source = ReferenceVideoConditionSource(localizer, num_stages=2, confirmations=2)
    first = source.estimate({}, 8)
    repeated = source.estimate({}, 8)
    assert first.details["completion_streak"] == 1
    assert repeated.details["completion_streak"] == 1
    assert not repeated.advanced
    localizer.sample_step = 16
    advanced = source.estimate({}, 16)
    assert advanced.advanced and advanced.stage_index == 1

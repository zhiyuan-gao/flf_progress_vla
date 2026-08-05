from __future__ import annotations

import numpy as np
import pytest
import torch

from stage_state_vla.inference import ReferenceVideoConditionSource
from stage_state_vla.reference_video import (
    RGBReferenceEncoder,
    RollingSubsequenceDTWProgressLocalizer,
    StageReference,
    SubsequenceDTWLocalizer,
)


class FeatureEncoder:
    def encode(self, observation):
        return np.asarray(observation["feature"], dtype=np.float32)


def stage_reference(stage: int, features: np.ndarray) -> StageReference:
    return StageReference(
        stage_index=stage,
        stage_name=f"stage-{stage}",
        segment_start=stage * 10,
        segment_end=stage * 10 + len(features) - 1,
        node_frames=tuple(range(stage * 10, stage * 10 + len(features))),
        features=features.astype(np.float32),
    )


def test_rgb_encoder_contract():
    encoder = RGBReferenceEncoder(camera_keys=("video.a", "video.b"), spatial_size=4)
    observation = {
        "video.a": np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3),
        "video.b": np.full((8, 8, 3), 127, dtype=np.uint8),
    }
    feature = encoder.encode(observation)
    assert feature.shape == (2 * 4 * 4 * 3,)
    assert np.isfinite(feature).all()
    assert 0.0 <= float(feature.min()) <= float(feature.max()) <= 1.0


@pytest.mark.parametrize(
    "path",
    [
        (2, 3, 4, 5, 6, 7, 8, 9),
        (2, 2, 3, 3, 4, 4, 5, 5),
        (1, 3, 4, 6, 7, 9, 10, 12),
        (1, 1, 2, 5, 5, 7, 8, 11),
        (2, 2, 2, 2, 5, 8, 11, 13),
    ],
)
def test_subsequence_dtw_recovers_validated_warp_styles(path):
    reference = torch.eye(16, dtype=torch.float32)
    query = reference[torch.tensor(path)]
    result = SubsequenceDTWLocalizer().localize(reference, query)
    assert result.path == path
    assert result.end_index == path[-1]
    assert result.mean_cost == pytest.approx(0.0)


def test_video_condition_source_advances_only_after_confirmation():
    features = np.eye(2, dtype=np.float32)
    references = {
        0: stage_reference(0, features),
        1: stage_reference(1, features),
    }
    localizer = RollingSubsequenceDTWProgressLocalizer(
        references,
        encoder=FeatureEncoder(),
        query_length=8,
        observation_stride=8,
    )
    source = ReferenceVideoConditionSource(
        localizer,
        num_stages=2,
        completion_threshold=0.9,
        confirmations=2,
    )
    first = source.estimate({"feature": features[1]}, 0)
    assert first.stage_index == 0 and not first.advanced and first.progress == 1.0
    second = source.estimate({"feature": features[1]}, 16)
    assert second.stage_index == 1 and second.advanced and second.progress == 0.0


def test_rolling_dtw_samples_at_stride_and_never_rewinds():
    features = np.eye(6, dtype=np.float32)
    localizer = RollingSubsequenceDTWProgressLocalizer(
        {0: stage_reference(0, features)},
        encoder=FeatureEncoder(),
        query_length=3,
        observation_stride=8,
    )
    assert localizer.observe(0, {"feature": features[1]}, 1) is False
    assert localizer.observe(0, {"feature": features[1]}, 8) is True
    assert localizer.observe(0, {"feature": features[1]}, 8) is False
    first = localizer.localize(0, {"feature": features[3]}, 16)
    assert first.path == (1, 3)
    second = localizer.localize(0, {"feature": features[2]}, 24)
    assert second.end_index >= first.end_index
    assert second.query_steps == 3

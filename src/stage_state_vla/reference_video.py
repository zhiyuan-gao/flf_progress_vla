"""Ground-truth reference-video loading and subsequence-DTW localization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .index import FrameRecord, reference_grid


CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
DTW_CAMERA_KEYS = ("video.robot0_agentview_left",)


@dataclass(frozen=True)
class StageReference:
    """Visual reference nodes for one active subtask."""

    stage_index: int
    stage_name: str
    segment_start: int
    segment_end: int
    node_frames: tuple[int, ...]
    features: np.ndarray

    def __post_init__(self) -> None:
        features = np.asarray(self.features)
        if features.ndim != 2:
            raise ValueError(f"features must have shape [nodes, dim], got {features.shape}")
        if len(self.node_frames) != len(features) or len(features) < 1:
            raise ValueError("node_frames and features must have the same non-zero length")
        if tuple(sorted(self.node_frames)) != self.node_frames:
            raise ValueError("node_frames must be sorted")


class RGBReferenceEncoder:
    """RGB descriptor used by the previously validated DTW protocol.

    The validated setting is one ``agentview_left`` image resized to 32x32 with
    OpenCV INTER_AREA and flattened in RGB order. Feature normalization happens
    inside the DTW localizer, exactly as in the original experiment.
    """

    def __init__(
        self,
        camera_keys: Sequence[str] = DTW_CAMERA_KEYS,
        spatial_size: int = 32,
    ) -> None:
        if spatial_size < 2:
            raise ValueError("spatial_size must be at least 2")
        if not camera_keys:
            raise ValueError("at least one camera key is required")
        self.camera_keys = tuple(camera_keys)
        self.spatial_size = int(spatial_size)

    def encode(self, observation: Mapping[str, Any]) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - rollout environment dependency
            raise RuntimeError("OpenCV is required for RGB reference features") from exc

        descriptors: list[np.ndarray] = []
        for key in self.camera_keys:
            if key not in observation:
                raise KeyError(f"observation is missing reference camera {key!r}")
            image = np.asarray(observation[key])
            if image.ndim == 4:
                image = image[-1]
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{key} must be an RGB image [H,W,3], got {image.shape}")
            resized = cv2.resize(
                image,
                (self.spatial_size, self.spatial_size),
                interpolation=cv2.INTER_AREA,
            )
            descriptors.append(resized.astype(np.float32).reshape(-1) / 255.0)
        return np.concatenate(descriptors).astype(np.float32, copy=False)


def episode_video_path(dataset: Path, episode: int, camera_key: str) -> Path:
    prefix = "video."
    if not camera_key.startswith(prefix):
        raise ValueError(f"camera key must start with {prefix!r}: {camera_key}")
    name = camera_key[len(prefix) :]
    path = (
        Path(dataset)
        / f"videos/chunk-{int(episode) // 1000:03d}"
        / f"observation.images.{name}"
        / f"episode_{int(episode):06d}.mp4"
    )
    if not path.is_file():
        raise FileNotFoundError(f"reference video is missing: {path}")
    return path


def _read_selected_frames(path: Path, frame_indices: Sequence[int]) -> dict[int, np.ndarray]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised in the rollout environment
        raise RuntimeError("OpenCV is required to decode reference videos") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open reference video: {path}")
    frames: dict[int, np.ndarray] = {}
    try:
        for index in sorted(set(int(value) for value in frame_indices)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"could not read frame {index} from {path}")
            frames[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    return frames


def _unique_segments(records: Sequence[FrameRecord]) -> list[FrameRecord]:
    by_stage: dict[int, FrameRecord] = {}
    for row in records:
        previous = by_stage.get(row.stage_index)
        if previous is not None and (
            previous.segment_start != row.segment_start
            or previous.segment_end != row.segment_end
        ):
            raise ValueError(
                f"stage {row.stage_index} has multiple disjoint segments; "
                "the MVP expects one reference video per ordinal stage"
            )
        by_stage[row.stage_index] = row
    if not by_stage:
        raise ValueError("cannot build references from an empty episode")
    expected = list(range(max(by_stage) + 1))
    if sorted(by_stage) != expected:
        raise ValueError(f"reference stages must be contiguous, got {sorted(by_stage)}")
    return [by_stage[index] for index in expected]


def build_ground_truth_references(
    dataset: Path,
    episode: int,
    records: Sequence[FrameRecord],
    *,
    reference_stride: int = 8,
    encoder: RGBReferenceEncoder | None = None,
) -> dict[int, StageReference]:
    """Decode one demonstration and create visual nodes for every active stage."""
    encoder = encoder or RGBReferenceEncoder()
    segments = _unique_segments(records)
    grids = {
        row.stage_index: reference_grid(row.segment_start, row.segment_end, reference_stride)
        for row in segments
    }
    required_frames = sorted({frame for grid in grids.values() for frame in grid.frames})
    decoded = {
        key: _read_selected_frames(episode_video_path(dataset, episode, key), required_frames)
        for key in encoder.camera_keys
    }

    references: dict[int, StageReference] = {}
    for row in segments:
        node_frames = grids[row.stage_index].frames
        features = np.stack(
            [
                encoder.encode({key: decoded[key][frame] for key in encoder.camera_keys})
                for frame in node_frames
            ]
        )
        references[row.stage_index] = StageReference(
            stage_index=row.stage_index,
            stage_name=row.stage_name,
            segment_start=row.segment_start,
            segment_end=row.segment_end,
            node_frames=node_frames,
            features=features,
        )
    return references


@dataclass(frozen=True)
class SubsequenceDTWResult:
    start_index: int
    end_index: int
    progress: float
    path: tuple[int, ...]
    mean_cost: float
    confidence_margin: float


class SubsequenceDTWLocalizer:
    """Align a query to any monotonic reference subsequence without fixed speed.

    The zero-penalty configuration is the protocol that recovered every tested
    uniform, stay, skip, mixed, and stall-then-fast path in the earlier held-out
    GT-video experiment.
    """

    def __init__(
        self,
        motion_weight: float = 0.0,
        stay_penalty: float = 0.0,
        jump_penalty: float = 0.0,
    ) -> None:
        if min(motion_weight, stay_penalty, jump_penalty) < 0:
            raise ValueError("alignment weights and penalties must be non-negative")
        self.motion_weight = float(motion_weight)
        self.stay_penalty = float(stay_penalty)
        self.jump_penalty = float(jump_penalty)

    @torch.inference_mode()
    def localize(
        self,
        reference_features: torch.Tensor,
        query_features: torch.Tensor,
        *,
        min_end_index: int = 0,
    ) -> SubsequenceDTWResult:
        if reference_features.ndim != 2 or query_features.ndim != 2:
            raise ValueError("reference and query must have [T, D] shapes")
        if reference_features.shape[1] != query_features.shape[1]:
            raise ValueError("reference and query feature dimensions differ")
        if len(reference_features) < 1 or len(query_features) < 1:
            raise ValueError("reference and query must be non-empty")
        if not 0 <= min_end_index < len(reference_features):
            raise ValueError("min_end_index is outside the reference")

        reference = F.normalize(reference_features.detach().float(), dim=-1)
        query = F.normalize(query_features.detach().float(), dim=-1).to(reference.device)
        appearance_cost = 1.0 - query @ reference.T
        query_steps, reference_steps = appearance_cost.shape
        cumulative = torch.full_like(appearance_cost, torch.inf)
        backpointer = torch.full(
            (query_steps, reference_steps),
            -1,
            dtype=torch.long,
            device=reference.device,
        )
        # Subsequence initialization: the query may begin at any reference node.
        cumulative[0] = appearance_cost[0]

        previous_index = torch.arange(reference_steps, device=reference.device)[:, None]
        current_index = torch.arange(reference_steps, device=reference.device)[None, :]
        advance = current_index - previous_index
        valid_transition = advance >= 0
        transition = torch.where(
            advance == 0,
            torch.full_like(advance, self.stay_penalty, dtype=torch.float32),
            self.jump_penalty * torch.clamp(advance.float() - 1.0, min=0.0).square(),
        )
        reference_gram = reference @ reference.T
        reference_delta_norm = torch.sqrt(
            torch.clamp(2.0 - 2.0 * reference_gram, min=0.0)
        )

        for query_index in range(1, query_steps):
            query_delta = query[query_index] - query[query_index - 1]
            query_norm = torch.linalg.vector_norm(query_delta)
            query_unit = query_delta / torch.clamp(query_norm, min=1e-8)
            projected = reference @ query_unit
            motion_similarity = (projected[None, :] - projected[:, None]) / torch.clamp(
                reference_delta_norm, min=1e-8
            )
            both_static = (query_norm <= 1e-8) & (reference_delta_norm <= 1e-8)
            one_static = (query_norm <= 1e-8) ^ (reference_delta_norm <= 1e-8)
            motion_cost = 1.0 - motion_similarity
            motion_cost = torch.where(both_static, torch.zeros_like(motion_cost), motion_cost)
            motion_cost = torch.where(one_static, torch.ones_like(motion_cost), motion_cost)
            candidate = (
                cumulative[query_index - 1, :, None]
                + transition
                + self.motion_weight * motion_cost
            )
            candidate = candidate.masked_fill(~valid_transition, torch.inf)
            best_value, best_previous = torch.min(candidate, dim=0)
            cumulative[query_index] = appearance_cost[query_index] + best_value
            backpointer[query_index] = best_previous

        allowed_endpoint_cost = cumulative[-1, min_end_index:]
        end = min_end_index + int(torch.argmin(allowed_endpoint_cost).item())
        if len(allowed_endpoint_cost) > 1:
            two = torch.topk(allowed_endpoint_cost, k=2, largest=False).values
            confidence = float(((two[1] - two[0]) / query_steps).item())
        else:
            confidence = float("inf")
        path = [end]
        current = end
        for query_index in range(query_steps - 1, 0, -1):
            current = int(backpointer[query_index, current].item())
            path.append(current)
        path.reverse()
        progress = 1.0 if reference_steps == 1 else end / float(reference_steps - 1)
        return SubsequenceDTWResult(
            start_index=path[0],
            end_index=end,
            progress=progress,
            path=tuple(path),
            mean_cost=float((cumulative[-1, end] / query_steps).item()),
            confidence_margin=confidence,
        )


@dataclass(frozen=True)
class DTWProgressLocalization:
    stage_index: int
    start_index: int
    end_index: int
    node_frame: int
    progress: float
    path: tuple[int, ...]
    query_steps: int
    sample_step: int
    mean_cost: float
    confidence_margin: float


class RollingSubsequenceDTWProgressLocalizer:
    """Collect stride-spaced rollout frames and localize their rolling subsequence."""

    def __init__(
        self,
        references: Mapping[int, StageReference],
        *,
        encoder: RGBReferenceEncoder | None = None,
        query_length: int = 8,
        observation_stride: int = 8,
        aligner: SubsequenceDTWLocalizer | None = None,
    ) -> None:
        if not references:
            raise ValueError("at least one stage reference is required")
        if query_length < 1 or observation_stride < 1:
            raise ValueError("query_length and observation_stride must be positive")
        self.references = dict(references)
        self.encoder = encoder or RGBReferenceEncoder()
        self.query_length = int(query_length)
        self.observation_stride = int(observation_stride)
        self.aligner = aligner or SubsequenceDTWLocalizer()
        self._history: dict[int, list[np.ndarray]] = {
            stage: [] for stage in self.references
        }
        self._sample_steps: dict[int, list[int]] = {
            stage: [] for stage in self.references
        }
        self._last_end = {stage: 0 for stage in self.references}

    def reset(self, stage_index: int | None = None) -> None:
        stages = list(self.references) if stage_index is None else [stage_index]
        for stage in stages:
            if stage not in self.references:
                raise KeyError(f"no reference for stage {stage}")
            self._history[stage] = []
            self._sample_steps[stage] = []
            self._last_end[stage] = 0

    def observe(
        self,
        stage_index: int,
        observation: Mapping[str, Any],
        control_step: int,
        *,
        force: bool = False,
    ) -> bool:
        if stage_index not in self.references:
            raise KeyError(f"no reference for stage {stage_index}")
        step = int(control_step)
        if not force and step % self.observation_stride:
            return False
        steps = self._sample_steps[stage_index]
        if steps and step == steps[-1]:
            return False
        if steps and step < steps[-1]:
            raise ValueError("control steps must be observed in nondecreasing order")
        self._history[stage_index].append(self.encoder.encode(observation))
        steps.append(step)
        if len(steps) > self.query_length:
            del steps[0]
            del self._history[stage_index][0]
        return True

    def localize(
        self,
        stage_index: int,
        observation: Mapping[str, Any],
        control_step: int,
    ) -> DTWProgressLocalization:
        self.observe(stage_index, observation, control_step)
        if not self._history[stage_index]:
            # Normal rollout starts at step zero. This fallback only bootstraps
            # callers whose first request is off the configured sampling grid.
            self.observe(stage_index, observation, control_step, force=True)
        reference = self.references[stage_index]
        query = np.stack(self._history[stage_index])
        result = self.aligner.localize(
            torch.from_numpy(reference.features),
            torch.from_numpy(query),
            min_end_index=self._last_end[stage_index],
        )
        self._last_end[stage_index] = result.end_index
        return DTWProgressLocalization(
            stage_index=stage_index,
            start_index=result.start_index,
            end_index=result.end_index,
            node_frame=reference.node_frames[result.end_index],
            progress=result.progress,
            path=result.path,
            query_steps=len(query),
            sample_step=self._sample_steps[stage_index][-1],
            mean_cost=result.mean_cost,
            confidence_margin=result.confidence_margin,
        )

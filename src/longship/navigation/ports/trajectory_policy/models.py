"""Transport-neutral models for executor-side trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType

from longship.navigation.common import TimePoint
from longship.navigation.map_engine.models import (
    AnchorId,
    NodeId,
    ResourceDescriptor,
    ResourceId,
    SegmentId,
    SnapshotId,
)


TrajectoryCandidateId = NewType("TrajectoryCandidateId", str)


class TrajectoryPolicyErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    CONTEXT_NOT_READY = "context_not_ready"
    CONTEXT_STALE = "context_stale"
    GOAL_UNAVAILABLE = "goal_unavailable"
    PROFILE_MISMATCH = "profile_mismatch"
    INFERENCE_FAILED = "inference_failed"


@dataclass(frozen=True, slots=True)
class VisualGoalTrajectoryRequest:
    """One active route step bound to an immutable visual target resource."""

    snapshot_id: SnapshotId
    segment_id: SegmentId
    source_node_id: NodeId
    target_node_id: NodeId
    target_anchor_id: AnchorId
    goal_resource: ResourceDescriptor
    requested_at: TimePoint
    max_observation_age_s: float
    num_candidates: int
    sampling_seed: int | None
    expected_image_profile_id: str
    expected_model_artifact_id: str
    expected_model_artifact_digest: str


@dataclass(frozen=True, slots=True)
class PolicyNativeWaypoint:
    """One raw planar waypoint with no platform unit or timing claim."""

    step_index: int
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class TrajectoryCandidate:
    candidate_id: TrajectoryCandidateId
    waypoints: tuple[PolicyNativeWaypoint, ...]


@dataclass(frozen=True, slots=True)
class TrajectoryCandidateSet:
    """Raw candidates tied to the exact route step and observation context."""

    snapshot_id: SnapshotId
    segment_id: SegmentId
    source_node_id: NodeId
    target_node_id: NodeId
    target_anchor_id: AnchorId
    goal_resource_id: ResourceId
    observation_time: TimePoint
    produced_at: TimePoint
    temporal_distance: float
    coordinate_frame: str
    coordinate_units: str
    sampling_seed: int | None
    candidates: tuple[TrajectoryCandidate, ...]
    policy_id: str
    image_profile_id: str
    model_artifact_id: str
    model_artifact_digest: str

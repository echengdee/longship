"""Internal policy boundary for visual topological localization.

The contract carries map identities and learned temporal-distance results
only. Camera messages, decoded tensors, PyTorch objects, and model-specific
preprocessing remain behind the policy implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from longship.navigation.common import TimePoint
from longship.navigation.map_engine.models import (
    AnchorId,
    NodeId,
    ResourceDescriptor,
    ResourceId,
    SnapshotId,
)


class VisualPolicyErrorCode(str, Enum):
    CONTEXT_NOT_READY = "context_not_ready"
    CONTEXT_STALE = "context_stale"
    PROFILE_MISMATCH = "profile_mismatch"
    GOAL_UNAVAILABLE = "goal_unavailable"
    INVALID_REQUEST = "invalid_request"
    INFERENCE_FAILED = "inference_failed"


class VisualPolicyError(RuntimeError):
    """Structured visual-policy failure consumed by Localization Engine."""

    def __init__(
        self,
        code: VisualPolicyErrorCode,
        message: str,
        *,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class VisualGoalDistanceRequest:
    """One distance-head comparison against a map visual anchor."""

    snapshot_id: SnapshotId
    target_node_id: NodeId
    target_anchor_id: AnchorId
    goal_resource: ResourceDescriptor
    requested_at: TimePoint
    max_observation_age_s: float
    expected_image_profile_id: str
    expected_model_artifact_id: str
    expected_model_artifact_digest: str


@dataclass(frozen=True, slots=True)
class VisualGoalDistanceMeasurement:
    """Model output without interpreting distance as confidence or metric."""

    snapshot_id: SnapshotId
    target_node_id: NodeId
    target_anchor_id: AnchorId
    goal_resource_id: ResourceId
    observation_time: TimePoint
    produced_at: TimePoint
    temporal_distance: float
    policy_id: str
    image_profile_id: str
    model_artifact_id: str
    model_artifact_digest: str


@dataclass(frozen=True, slots=True)
class VisualGoalCandidate:
    """One map-pinned visual goal included in a local candidate search."""

    target_node_id: NodeId
    target_anchor_id: AnchorId
    goal_resource: ResourceDescriptor


@dataclass(frozen=True, slots=True)
class VisualGoalDistanceBatchRequest:
    """Compares one observation context against ordered local map goals."""

    snapshot_id: SnapshotId
    candidates: tuple[VisualGoalCandidate, ...]
    requested_at: TimePoint
    max_observation_age_s: float
    expected_image_profile_id: str
    expected_model_artifact_id: str
    expected_model_artifact_digest: str


@dataclass(frozen=True, slots=True)
class VisualGoalCandidateDistance:
    """One candidate identity and its uninterpreted model distance."""

    target_node_id: NodeId
    target_anchor_id: AnchorId
    goal_resource_id: ResourceId
    temporal_distance: float


@dataclass(frozen=True, slots=True)
class VisualGoalDistanceBatchMeasurement:
    """Distances for one candidate set evaluated on the same context."""

    snapshot_id: SnapshotId
    candidate_distances: tuple[VisualGoalCandidateDistance, ...]
    observation_time: TimePoint
    produced_at: TimePoint
    policy_id: str
    image_profile_id: str
    model_artifact_id: str
    model_artifact_digest: str


@runtime_checkable
class VisualGoalDistancePolicy(Protocol):
    """Policy plugin capability required by visual Localization Engine."""

    async def compare_goal(
        self,
        request: VisualGoalDistanceRequest,
    ) -> VisualGoalDistanceMeasurement:
        """Compares the latest complete observation context with one goal."""
        ...


@runtime_checkable
class VisualGoalDistanceBatchPolicy(Protocol):
    """Batch policy capability used by robust local-map localization."""

    async def compare_goals(
        self,
        request: VisualGoalDistanceBatchRequest,
    ) -> VisualGoalDistanceBatchMeasurement:
        """Evaluates all candidates against one immutable observation."""
        ...

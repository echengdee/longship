"""Domain models for the Harness Local Trajectory Engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType

from longship.navigation.common import TimePoint
from longship.navigation.localization_engine.models import BeliefRevision
from longship.navigation.map_engine.models import (
    AnchorId,
    NodeId,
    ResourceId,
    SegmentId,
    SnapshotId,
)
from longship.navigation.planning_engine.models import RouteId


LocalTrajectoryStreamId = NewType("LocalTrajectoryStreamId", str)
LocalTrajectoryId = NewType("LocalTrajectoryId", str)


class LocalTrajectoryState(str, Enum):
    """State of the latest Harness trajectory publication."""

    INITIALIZING = "initializing"
    HOLDING = "holding"
    ACTIVE = "active"
    ROUTE_COMPLETED = "route_completed"
    FAULTED = "faulted"
    STOPPED = "stopped"


class LocalTrajectoryHoldReason(str, Enum):
    """Why no trajectory is currently safe to publish as active."""

    LOCATION_UNUSABLE = "location_unusable"
    ROUTE_POSITION_UNRESOLVED = "route_position_unresolved"
    OBSERVATION_CONTEXT_NOT_READY = "observation_context_not_ready"
    OBSERVATION_STALE = "observation_stale"
    GOAL_UNAVAILABLE = "goal_unavailable"
    POLICY_UNAVAILABLE = "policy_unavailable"
    STALE_POLICY_RESULT = "stale_policy_result"
    SERVICE_STOPPED = "service_stopped"


class LocalTrajectoryUpdateOutcome(str, Enum):
    UPDATED = "updated"
    STREAM_RESET = "stream_reset"
    TIMED_OUT = "timed_out"


class LocalTrajectoryStreamErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    STREAM_UNAVAILABLE = "stream_unavailable"
    INTERNAL_FAILURE = "internal_failure"


@dataclass(frozen=True, slots=True)
class LocalTrajectoryRevision:
    stream_id: LocalTrajectoryStreamId
    sequence: int


@dataclass(frozen=True, slots=True)
class LocalTrajectoryWaypoint:
    """One selected waypoint; units are declared by its trajectory."""

    step_index: int
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class LocalTrajectory:
    """One full policy trajectory selected by the Harness."""

    trajectory_id: LocalTrajectoryId
    source_candidate_id: str
    waypoints: tuple[LocalTrajectoryWaypoint, ...]
    coordinate_frame: str
    coordinate_units: str
    selection_policy_id: str
    source_candidate_index: int
    source_candidate_count: int
    sampling_seed: int | None
    temporal_distance: float
    policy_id: str
    image_profile_id: str
    model_artifact_id: str
    model_artifact_digest: str


@dataclass(frozen=True, slots=True)
class LocalTrajectoryPublication:
    """Latest immutable trajectory or an explicit non-motion state.

    An ``ACTIVE`` publication contains one full trajectory and all route,
    localization, target-resource, and observation identities needed to reject
    stale data outside the Harness. Other states never imply permission to
    move and therefore carry no trajectory.
    """

    revision: LocalTrajectoryRevision
    route_id: RouteId
    snapshot_id: SnapshotId
    state: LocalTrajectoryState
    published_at: TimePoint
    belief_revision: BeliefRevision | None = None
    traversal_sequence: int | None = None
    segment_id: SegmentId | None = None
    source_node_id: NodeId | None = None
    target_node_id: NodeId | None = None
    target_anchor_id: AnchorId | None = None
    goal_resource_id: ResourceId | None = None
    observation_time: TimePoint | None = None
    generated_at: TimePoint | None = None
    valid_until: TimePoint | None = None
    trajectory: LocalTrajectory | None = None
    hold_reason: LocalTrajectoryHoldReason | None = None
    detail_code: str | None = None


@dataclass(frozen=True, slots=True)
class WaitForLocalTrajectoryRequest:
    after_revision: LocalTrajectoryRevision
    timeout_s: float | None = None


@dataclass(frozen=True, slots=True)
class LocalTrajectoryUpdateResult:
    outcome: LocalTrajectoryUpdateOutcome
    publication: LocalTrajectoryPublication

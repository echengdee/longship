"""Public data models for Localization Engine v0.1.

The engine is a system-level, continuously running state-estimation service.
These models are transport-neutral and do not expose sensor messages,
localization algorithms, ROS 2 types, or inference backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType, TypeAlias

from longship.navigation.common import Pose3D, TimePoint
from longship.navigation.map_engine.models import (
    NodeId,
    PlaceId,
    SegmentId,
    SnapshotId,
)


BeliefStreamId = NewType("BeliefStreamId", str)
HypothesisId = NewType("HypothesisId", str)
RelocalizationId = NewType("RelocalizationId", str)


class LocalizationCapability(str, Enum):
    TOPOLOGICAL_LOCATION = "topological_location"
    FIXED_START_TOPOLOGICAL_TRACKING = (
        "fixed_start_topological_tracking"
    )
    SEMANTIC_PLACE = "semantic_place"
    METRIC_POSE = "metric_pose"
    MULTI_HYPOTHESIS = "multi_hypothesis"
    SOURCE_HEALTH = "source_health"


class LocalizationStatus(str, Enum):
    """Quality/state of the current location estimate."""

    INITIALIZING = "initializing"
    TRACKING = "tracking"
    DEGRADED = "degraded"
    AMBIGUOUS = "ambiguous"
    LOST = "lost"
    UNAVAILABLE = "unavailable"


class LocalizationEngineState(str, Enum):
    """Operational lifecycle of the continuously running engine."""

    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAULTED = "faulted"
    STOPPED = "stopped"


class LocalizationSourceState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


class BeliefUpdateOutcome(str, Enum):
    UPDATED = "updated"
    STREAM_RESET = "stream_reset"
    TIMED_OUT = "timed_out"


class RelocalizationDisposition(str, Enum):
    ACCEPTED = "accepted"
    ALREADY_RUNNING = "already_running"
    UNAVAILABLE = "unavailable"


class LocalizationErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    ENGINE_NOT_RUNNING = "engine_not_running"
    ENGINE_FAULTED = "engine_faulted"
    INCOMPATIBLE_SNAPSHOT = "incompatible_snapshot"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    BACKEND_UNAVAILABLE = "backend_unavailable"


@dataclass(frozen=True, slots=True)
class BeliefRevision:
    """Cursor for one publication in a localization output stream.

    ``stream_id`` changes whenever the engine restarts or changes its bound map
    snapshot. ``sequence`` is strictly increasing within one stream.
    """

    stream_id: BeliefStreamId
    sequence: int


@dataclass(frozen=True, slots=True)
class NodeLocation:
    node_id: NodeId


@dataclass(frozen=True, slots=True)
class SegmentLocation:
    segment_id: SegmentId
    progress: float | None = None


TopologicalLocation: TypeAlias = NodeLocation | SegmentLocation


@dataclass(frozen=True, slots=True)
class MetricPoseEstimate:
    """Robot pose in a map-defined metric frame.

    ``covariance_6x6`` is row-major over x, y, z, roll, pitch, yaw. It may be
    omitted when the source cannot provide a meaningful covariance.
    """

    pose: Pose3D
    covariance_6x6: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class LocalizationHint:
    """Optional prior evidence, never a navigation-goal bias."""

    topological_candidates: tuple[TopologicalLocation, ...] = ()
    place_candidates: tuple[PlaceId, ...] = ()
    metric_pose: MetricPoseEstimate | None = None
    confidence: float | None = None
    observed_at: TimePoint | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class LocalizationSourceHealth:
    source_id: str
    state: LocalizationSourceState
    last_observation_at: TimePoint | None = None
    detail_code: str | None = None


@dataclass(frozen=True, slots=True)
class LocationHypothesis:
    hypothesis_id: HypothesisId
    topological_location: TopologicalLocation | None = None
    semantic_places: tuple[PlaceId, ...] = ()
    metric_pose: MetricPoseEstimate | None = None
    weight: float | None = None


@dataclass(frozen=True, slots=True)
class LocationBelief:
    """One immutable publication of the continuously maintained belief."""

    snapshot_id: SnapshotId
    revision: BeliefRevision
    estimate_time: TimePoint
    published_at: TimePoint
    status: LocalizationStatus
    confidence: float | None
    hypotheses: tuple[LocationHypothesis, ...] = ()
    source_health: tuple[LocalizationSourceHealth, ...] = ()


@dataclass(frozen=True, slots=True)
class WaitForUpdateRequest:
    after_revision: BeliefRevision
    timeout_s: float | None = None


@dataclass(frozen=True, slots=True)
class BeliefUpdateResult:
    """New belief, stream reset, or normal timeout of a long-poll request."""

    outcome: BeliefUpdateOutcome
    belief: LocationBelief


@dataclass(frozen=True, slots=True)
class RelocalizationRequest:
    hint: LocalizationHint | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RelocalizationAcceptance:
    relocalization_id: RelocalizationId | None
    disposition: RelocalizationDisposition
    accepted_at: TimePoint | None = None
    detail_code: str | None = None


@dataclass(frozen=True, slots=True)
class LocalizationEngineStatus:
    """Operational status, distinct from location-estimate quality."""

    state: LocalizationEngineState
    snapshot_id: SnapshotId | None
    stream_id: BeliefStreamId | None
    capabilities: frozenset[LocalizationCapability]
    latest_sequence: int | None
    last_update_at: TimePoint | None
    active_relocalization_id: RelocalizationId | None = None
    detail_code: str | None = None

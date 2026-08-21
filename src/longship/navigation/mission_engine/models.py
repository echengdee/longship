"""Public data models for Mission Engine v0.1.

Mission Engine is the only task-level orchestrator inside Navigation Harness.
It owns mission lifecycle, recovery budgets, target resolution, and final
success verification while depending only on the public contracts of Map,
Localization, Planning, and the external RouteExecutionPort.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NewType

from longship.navigation.common import NavigationExecutionContext, TimePoint
from longship.navigation.localization_engine.models import BeliefRevision
from longship.navigation.map_engine.models import (
    AnchorId,
    MapSelector,
    NodeId,
    PlaceId,
    SegmentId,
    SnapshotId,
)
from longship.navigation.planning_engine.models import (
    NoRouteReason,
    PlanningContext,
    RouteConstraints,
    RouteId,
    RoutePreferences,
)
from longship.navigation.ports.route_execution.models import (
    RouteCommandId,
    RouteExecutionFailureReason,
    RouteExecutionLimits,
    RouteExecutionState,
    RouteSubmissionRejectionReason,
)


NavigationMissionId = NewType("NavigationMissionId", str)
NavigationMissionControlCommandId = NewType(
    "NavigationMissionControlCommandId",
    str,
)


class MissionTargetKind(str, Enum):
    """How a caller identifies the logical navigation destination."""

    PLACE_ID = "place_id"
    NODE_ID = "node_id"
    ANCHOR_ID = "anchor_id"
    PLACE_QUERY = "place_query"


class TargetAmbiguityPolicy(str, Enum):
    """Policy for multiple place matches returned by Map Engine."""

    REQUIRE_UNIQUE = "require_unique"
    SELECT_TOP_RANKED = "select_top_ranked"


class TargetResolutionBasis(str, Enum):
    DIRECT_PLACE = "direct_place"
    DIRECT_NODE = "direct_node"
    DIRECT_ANCHOR = "direct_anchor"
    RANKED_PLACE_QUERY = "ranked_place_query"


class MissionState(str, Enum):
    ACCEPTED = "accepted"
    RESOLVING_MAP = "resolving_map"
    RESOLVING_TARGET = "resolving_target"
    WAITING_FOR_LOCALIZATION = "waiting_for_localization"
    PLANNING = "planning"
    DISPATCHING = "dispatching"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    PAUSING = "pausing"
    PAUSED = "paused"
    RESUMING = "resuming"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MissionSubmissionOutcome(str, Enum):
    ACCEPTED = "accepted"
    ALREADY_EXISTS = "already_exists"
    REJECTED = "rejected"


class MissionSubmissionRejectionReason(str, Enum):
    ENGINE_BUSY = "engine_busy"


class MissionUpdateOutcome(str, Enum):
    UPDATED = "updated"
    TERMINAL = "terminal"
    TIMED_OUT = "timed_out"


class MissionControlAction(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"


class MissionControlDisposition(str, Enum):
    ACCEPTED = "accepted"
    ALREADY_SATISFIED = "already_satisfied"
    REJECTED = "rejected"


class MissionControlRejectionReason(str, Enum):
    INVALID_STATE = "invalid_state"
    TERMINAL_MISSION = "terminal_mission"
    DEPENDENCY_REJECTED = "dependency_rejected"


class MissionFailureReason(str, Enum):
    MAP_UNAVAILABLE = "map_unavailable"
    MAP_CONTEXT_CHANGED = "map_context_changed"
    TARGET_NOT_FOUND = "target_not_found"
    TARGET_AMBIGUOUS = "target_ambiguous"
    TARGET_UNSUPPORTED = "target_unsupported"
    LOCALIZATION_UNAVAILABLE = "localization_unavailable"
    LOCALIZATION_TIMEOUT = "localization_timeout"
    NO_ROUTE = "no_route"
    ROUTE_SUBMISSION_REJECTED = "route_submission_rejected"
    ROUTE_EXECUTION_FAILED = "route_execution_failed"
    ROUTE_EXECUTION_CANCELLED = "route_execution_cancelled"
    GOAL_VERIFICATION_FAILED = "goal_verification_failed"
    RECOVERY_BUDGET_EXHAUSTED = "recovery_budget_exhausted"
    MISSION_TIMEOUT = "mission_timeout"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    INTERNAL_FAILURE = "internal_failure"


class GoalEvidenceKind(str, Enum):
    TARGET_NODE = "target_node"
    TARGET_PLACE = "target_place"
    COMPLETION_ANCHOR = "completion_anchor"


class MissionErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    MISSION_NOT_FOUND = "mission_not_found"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    ENGINE_UNAVAILABLE = "engine_unavailable"
    DEPENDENCY_FAILURE = "dependency_failure"
    INTERNAL_FAILURE = "internal_failure"


@dataclass(frozen=True, slots=True)
class MissionTargetSpec:
    """Caller-facing destination specification.

    ``value`` contains an id for direct selectors or free text for
    ``PLACE_QUERY``.  The pair ``kind + value`` is the primary selector.
    """

    target_ref: str
    kind: MissionTargetKind
    value: str
    required_tags: frozenset[str] = frozenset()
    allowed_place_kinds: tuple[str, ...] = ()
    ambiguity_policy: TargetAmbiguityPolicy = (
        TargetAmbiguityPolicy.REQUIRE_UNIQUE
    )


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    """Version-pinned destination owned by Mission Engine.

    v0.1 always projects a destination to one or more candidate graph nodes so
    it can be passed to Planning Engine without exposing target interpretation
    to the planner.
    """

    target_ref: str
    snapshot_id: SnapshotId
    basis: TargetResolutionBasis
    candidate_node_ids: tuple[NodeId, ...]
    place_id: PlaceId | None = None
    source_anchor_id: AnchorId | None = None
    completion_anchor_ids: tuple[AnchorId, ...] = ()


@dataclass(frozen=True, slots=True)
class MissionBudget:
    """Finite task-level recovery and runtime limits."""

    mission_timeout_s: float | None = None
    localization_wait_timeout_s: float = 10.0
    max_relocalization_attempts: int = 1
    max_planning_attempts: int = 3
    max_route_submissions: int = 3
    max_same_route_retries: int = 0


@dataclass(frozen=True, slots=True)
class MissionSuccessCriteria:
    """Conditions checked by Mission Engine after route completion."""

    min_localization_confidence: float | None = None
    allow_degraded_localization: bool = True
    require_target_node: bool = True
    require_target_place: bool = False
    require_completion_anchor: bool = False


@dataclass(frozen=True, slots=True)
class NavigationMissionRequest:
    """Idempotent request to start one navigation mission."""

    mission_id: NavigationMissionId
    execution_context: NavigationExecutionContext
    requested_at: TimePoint
    map_selector: MapSelector
    target: MissionTargetSpec
    route_constraints: RouteConstraints = field(
        default_factory=RouteConstraints
    )
    route_preferences: RoutePreferences = field(
        default_factory=RoutePreferences
    )
    initial_planning_context: PlanningContext = field(
        default_factory=PlanningContext
    )
    execution_limits: RouteExecutionLimits = field(
        default_factory=RouteExecutionLimits
    )
    success_criteria: MissionSuccessCriteria = field(
        default_factory=MissionSuccessCriteria
    )
    budget: MissionBudget = field(default_factory=MissionBudget)


@dataclass(frozen=True, slots=True)
class MissionRevision:
    """Monotonic cursor for one mission's externally visible state."""

    mission_id: NavigationMissionId
    sequence: int


@dataclass(frozen=True, slots=True)
class MissionProgress:
    planning_attempts: int = 0
    relocalization_attempts: int = 0
    route_submissions: int = 0
    same_route_retries: int = 0
    unavailable_segment_ids: frozenset[SegmentId] = frozenset()


@dataclass(frozen=True, slots=True)
class GoalVerificationEvidence:
    kind: GoalEvidenceKind
    belief_revision: BeliefRevision | None = None
    node_id: NodeId | None = None
    place_id: PlaceId | None = None
    completion_anchor_id: AnchorId | None = None


@dataclass(frozen=True, slots=True)
class MissionCompletion:
    completed_at: TimePoint
    evidence: tuple[GoalVerificationEvidence, ...]
    final_belief_revision: BeliefRevision | None = None
    route_id: RouteId | None = None
    route_command_id: RouteCommandId | None = None


@dataclass(frozen=True, slots=True)
class MissionFailure:
    reason: MissionFailureReason
    failed_at: TimePoint
    retryable_new_mission: bool = False
    no_route_reason: NoRouteReason | None = None
    route_submission_reason: RouteSubmissionRejectionReason | None = None
    route_execution_reason: RouteExecutionFailureReason | None = None
    related_segment_ids: tuple[SegmentId, ...] = ()
    detail_code: str | None = None


@dataclass(frozen=True, slots=True)
class MissionStatus:
    """Immutable Mission Engine publication.

    The status may cache identifiers and the latest external execution state,
    but ownership of live route execution remains outside the Harness.
    """

    mission_id: NavigationMissionId
    revision: MissionRevision
    state: MissionState
    created_at: TimePoint
    updated_at: TimePoint
    progress: MissionProgress
    snapshot_id: SnapshotId | None = None
    resolved_target: ResolvedTarget | None = None
    latest_belief_revision: BeliefRevision | None = None
    latest_route_id: RouteId | None = None
    latest_route_command_id: RouteCommandId | None = None
    latest_execution_state: RouteExecutionState | None = None
    completion: MissionCompletion | None = None
    failure: MissionFailure | None = None
    cancellation_reason: str | None = None
    detail_code: str | None = None


@dataclass(frozen=True, slots=True)
class MissionSubmissionRejection:
    reason: MissionSubmissionRejectionReason
    active_mission_id: NavigationMissionId | None = None
    detail_code: str | None = None


@dataclass(frozen=True, slots=True)
class MissionSubmissionResult:
    mission_id: NavigationMissionId
    outcome: MissionSubmissionOutcome
    decided_at: TimePoint
    status: MissionStatus | None = None
    rejection: MissionSubmissionRejection | None = None


@dataclass(frozen=True, slots=True)
class MissionUpdateRequest:
    after_revision: MissionRevision
    timeout_s: float | None = None


@dataclass(frozen=True, slots=True)
class MissionUpdateResult:
    outcome: MissionUpdateOutcome
    status: MissionStatus


@dataclass(frozen=True, slots=True)
class MissionControlRequest:
    control_command_id: NavigationMissionControlCommandId
    mission_id: NavigationMissionId
    requested_at: TimePoint
    action: MissionControlAction
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class MissionControlResult:
    control_command_id: NavigationMissionControlCommandId
    mission_id: NavigationMissionId
    disposition: MissionControlDisposition
    decided_at: TimePoint
    status: MissionStatus
    rejection_reason: MissionControlRejectionReason | None = None
    detail_code: str | None = None

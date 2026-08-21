"""Transport-neutral models for the external RouteExecutionPort v0.1.

The Navigation Harness owns missions and immutable route plans.  An external
route-execution system owns the live command state, local motion loop, safety
integration, and platform feedback.  These models describe only the boundary
between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NewType

from longship.navigation.common import TimePoint
from longship.navigation.localization_engine.models import BeliefRevision
from longship.navigation.map_engine.models import (
    AnchorId,
    MapSnapshot,
    NodeId,
    SegmentId,
    SnapshotId,
)
from longship.navigation.planning_engine.models import RouteId, RoutePlan


RouteCommandId = NewType("RouteCommandId", str)
RouteControlCommandId = NewType("RouteControlCommandId", str)


class RouteExecutionState(str, Enum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    RESUMING = "resuming"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RouteSubmissionOutcome(str, Enum):
    ACCEPTED = "accepted"
    ALREADY_EXISTS = "already_exists"
    REJECTED = "rejected"


class RouteSubmissionRejectionReason(str, Enum):
    EXECUTOR_BUSY = "executor_busy"
    SNAPSHOT_MISMATCH = "snapshot_mismatch"
    ROUTE_INVALID = "route_invalid"
    ROUTE_STALE = "route_stale"
    START_PRECONDITION_FAILED = "start_precondition_failed"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    RESOURCE_UNAVAILABLE = "resource_unavailable"


class RouteControlAction(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"


class RouteControlDisposition(str, Enum):
    ACCEPTED = "accepted"
    ALREADY_SATISFIED = "already_satisfied"
    REJECTED = "rejected"


class RouteControlRejectionReason(str, Enum):
    INVALID_STATE = "invalid_state"
    TERMINAL_COMMAND = "terminal_command"
    ACTION_NOT_ALLOWED = "action_not_allowed"


class RouteExecutionUpdateOutcome(str, Enum):
    UPDATED = "updated"
    TERMINAL = "terminal"
    TIMED_OUT = "timed_out"


class RouteExecutionFailureReason(str, Enum):
    BLOCKED = "blocked"
    NO_PROGRESS = "no_progress"
    OFF_ROUTE = "off_route"
    LOCALIZATION_UNAVAILABLE = "localization_unavailable"
    MAP_SNAPSHOT_CHANGED = "map_snapshot_changed"
    PLAN_INVALIDATED = "plan_invalidated"
    EXECUTION_STRATEGY_FAILED = "execution_strategy_failed"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    PLATFORM_REJECTED = "platform_rejected"
    PLATFORM_FAULT = "platform_fault"
    SAFETY_STOPPED = "safety_stopped"
    EXECUTION_TIMEOUT = "execution_timeout"
    GOAL_NOT_REACHED = "goal_not_reached"


class RouteExecutionPortErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    COMMAND_NOT_FOUND = "command_not_found"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    PORT_UNAVAILABLE = "port_unavailable"
    EXTERNAL_EXECUTOR_UNAVAILABLE = "external_executor_unavailable"
    TRANSPORT_FAILURE = "transport_failure"
    INTERNAL_FAILURE = "internal_failure"


@dataclass(frozen=True, slots=True)
class RouteExecutionLimits:
    """Mission-provided limits; never a replacement for platform safety."""

    max_speed_mps: float | None = None
    route_timeout_s: float | None = None
    segment_timeout_s: float | None = None
    no_progress_timeout_s: float | None = None


@dataclass(frozen=True, slots=True)
class RouteCommand:
    """Submit one authorized immutable route to an external executor."""

    command_id: RouteCommandId
    mission_ref: str
    skill_call_id: str
    resource_lease_id: str
    cancellation_epoch: int
    expected_state_version: int
    issued_at: TimePoint
    valid_until: TimePoint
    snapshot: MapSnapshot
    route_plan: RoutePlan
    limits: RouteExecutionLimits = field(default_factory=RouteExecutionLimits)


@dataclass(frozen=True, slots=True)
class RouteSubmissionRejection:
    reason: RouteSubmissionRejectionReason
    detail_code: str | None = None
    conflicting_command_id: RouteCommandId | None = None


@dataclass(frozen=True, slots=True)
class RouteExecutionRevision:
    """Monotonic cursor within one external command's status stream."""

    command_id: RouteCommandId
    sequence: int


@dataclass(frozen=True, slots=True)
class RouteExecutionProgress:
    completed_traversal_count: int
    total_traversal_count: int
    active_traversal_sequence: int | None = None
    active_segment_id: SegmentId | None = None
    segment_progress: float | None = None
    route_progress: float | None = None
    latest_belief_revision: BeliefRevision | None = None
    last_progress_at: TimePoint | None = None


@dataclass(frozen=True, slots=True)
class RouteExecutionCompletion:
    completed_at: TimePoint
    goal_node_id: NodeId
    final_belief_revision: BeliefRevision | None = None
    completion_anchor_id: AnchorId | None = None


@dataclass(frozen=True, slots=True)
class RouteExecutionFailure:
    reason: RouteExecutionFailureReason
    failed_at: TimePoint
    active_segment_id: SegmentId | None = None
    related_segment_ids: tuple[SegmentId, ...] = ()
    last_belief_revision: BeliefRevision | None = None
    retryable_same_route: bool = False
    replan_recommended: bool = False
    detail_code: str | None = None


@dataclass(frozen=True, slots=True)
class RouteExecutionStatus:
    """Immutable publication owned by the external execution system."""

    command_id: RouteCommandId
    route_id: RouteId
    snapshot_id: SnapshotId
    revision: RouteExecutionRevision
    state: RouteExecutionState
    created_at: TimePoint
    updated_at: TimePoint
    progress: RouteExecutionProgress
    completion: RouteExecutionCompletion | None = None
    failure: RouteExecutionFailure | None = None
    cancellation_reason: str | None = None
    detail_code: str | None = None


@dataclass(frozen=True, slots=True)
class RouteSubmissionResult:
    command_id: RouteCommandId
    route_id: RouteId
    outcome: RouteSubmissionOutcome
    decided_at: TimePoint
    status: RouteExecutionStatus | None = None
    rejection: RouteSubmissionRejection | None = None


@dataclass(frozen=True, slots=True)
class RouteExecutionUpdateRequest:
    after_revision: RouteExecutionRevision
    timeout_s: float | None = None


@dataclass(frozen=True, slots=True)
class RouteExecutionUpdateResult:
    outcome: RouteExecutionUpdateOutcome
    status: RouteExecutionStatus


@dataclass(frozen=True, slots=True)
class RouteControlRequest:
    control_command_id: RouteControlCommandId
    route_command_id: RouteCommandId
    skill_call_id: str
    resource_lease_id: str
    cancellation_epoch: int
    requested_at: TimePoint
    action: RouteControlAction
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RouteControlResult:
    control_command_id: RouteControlCommandId
    route_command_id: RouteCommandId
    disposition: RouteControlDisposition
    decided_at: TimePoint
    status: RouteExecutionStatus
    rejection_reason: RouteControlRejectionReason | None = None
    detail_code: str | None = None

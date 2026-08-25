"""Public data models for Planning Engine v0.1.

Planning is a request/response capability.  The engine consumes immutable map
and localization publications and returns either one immutable global route or
a normal business outcome explaining why no route was produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, NewType

from longship.navigation.common import TimePoint
from longship.navigation.localization_engine.models import (
    BeliefRevision,
    HypothesisId,
    LocationBelief,
    TopologicalLocation,
)
from longship.navigation.map_engine.models import (
    AnchorId,
    MapSnapshot,
    NodeId,
    PlaceId,
    SegmentId,
    SnapshotId,
)


PlanningRequestId = NewType("PlanningRequestId", str)
RouteId = NewType("RouteId", str)


class RouteObjective(str, Enum):
    SHORTEST_DISTANCE = "shortest_distance"
    FASTEST = "fastest"
    BALANCED = "balanced"


class PlanningOutcome(str, Enum):
    ROUTE_FOUND = "route_found"
    ALREADY_AT_GOAL = "already_at_goal"
    NO_ROUTE = "no_route"


class NoRouteReason(str, Enum):
    LOCATION_UNUSABLE = "location_unusable"
    START_UNRESOLVED = "start_unresolved"
    START_AMBIGUOUS = "start_ambiguous"
    TARGET_UNRESOLVED = "target_unresolved"
    TARGET_UNREACHABLE = "target_unreachable"
    CONSTRAINTS_UNSATISFIABLE = "constraints_unsatisfiable"
    CAPABILITY_MISMATCH = "capability_mismatch"
    MAP_DATA_INCOMPLETE = "map_data_incomplete"


class PlanningErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    SNAPSHOT_MISMATCH = "snapshot_mismatch"
    MAP_UNAVAILABLE = "map_unavailable"
    ENGINE_UNAVAILABLE = "engine_unavailable"
    INTERNAL_FAILURE = "internal_failure"


@dataclass(frozen=True, slots=True)
class PlanningTarget:
    """Planner-facing projection of a Mission Engine ResolvedTarget.

    The Mission Engine owns target interpretation and ambiguity resolution.
    Planning only receives one logical target with one or more acceptable graph
    nodes.  It may choose the cheapest reachable candidate node.
    """

    target_ref: str
    candidate_node_ids: tuple[NodeId, ...]
    place_id: PlaceId | None = None
    completion_anchor_ids: tuple[AnchorId, ...] = ()


@dataclass(frozen=True, slots=True)
class RouteConstraints:
    """Hard constraints; violating any item makes a route invalid."""

    robot_capabilities: frozenset[str] = frozenset()
    forbidden_segment_ids: frozenset[SegmentId] = frozenset()
    forbidden_node_ids: frozenset[NodeId] = frozenset()
    forbidden_segment_tags: frozenset[str] = frozenset()
    max_total_distance_m: float | None = None
    max_total_duration_s: float | None = None


@dataclass(frozen=True, slots=True)
class RoutePreferences:
    """Soft preferences used to rank otherwise valid routes."""

    objective: RouteObjective = RouteObjective.BALANCED
    preferred_segment_tags: frozenset[str] = frozenset()
    avoided_segment_tags: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class PlanningContext:
    """Task-scoped transient knowledge supplied by Mission Engine.

    This data is not written back to Map Engine.  It normally comes from prior
    execution feedback, operator restrictions, or temporary platform state.
    """

    unavailable_segment_ids: frozenset[SegmentId] = frozenset()
    unavailable_node_ids: frozenset[NodeId] = frozenset()
    extra_segment_costs: Mapping[SegmentId, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RoutePlanningRequest:
    request_id: PlanningRequestId
    requested_at: TimePoint
    snapshot: MapSnapshot
    location_belief: LocationBelief
    target: PlanningTarget
    constraints: RouteConstraints = field(default_factory=RouteConstraints)
    preferences: RoutePreferences = field(default_factory=RoutePreferences)
    context: PlanningContext = field(default_factory=PlanningContext)


@dataclass(frozen=True, slots=True)
class PlannedStart:
    """Localization hypothesis and graph attachment selected for planning."""

    belief_revision: BeliefRevision
    hypothesis_id: HypothesisId
    topological_location: TopologicalLocation


@dataclass(frozen=True, slots=True)
class PlannedGoal:
    target_ref: str
    node_id: NodeId
    place_id: PlaceId | None = None
    completion_anchor_ids: tuple[AnchorId, ...] = ()


@dataclass(frozen=True, slots=True)
class PlannedTraversal:
    """One forward traversal of a directed map segment.

    ``entry_progress`` and ``exit_progress`` are normalized to [0, 1].  They
    support starting partway through a segment without creating a synthetic
    map node.  v0.1 only permits forward progress.
    """

    sequence: int
    segment_id: SegmentId
    source_node_id: NodeId
    target_node_id: NodeId
    entry_progress: float = 0.0
    exit_progress: float = 1.0
    estimated_distance_m: float | None = None
    estimated_duration_s: float | None = None
    incremental_cost: float | None = None


@dataclass(frozen=True, slots=True)
class RouteEstimate:
    total_distance_m: float | None = None
    total_duration_s: float | None = None
    total_cost: float | None = None


@dataclass(frozen=True, slots=True)
class PlanningProvenance:
    planner_id: str
    planner_version: str
    cost_model_id: str | None = None
    decision_digest: str | None = None


@dataclass(frozen=True, slots=True)
class RoutePlan:
    """Immutable, globally ordered topological route."""

    route_id: RouteId
    request_id: PlanningRequestId
    snapshot_id: SnapshotId
    created_at: TimePoint
    start: PlannedStart
    goal: PlannedGoal
    traversals: tuple[PlannedTraversal, ...]
    estimate: RouteEstimate
    provenance: PlanningProvenance


@dataclass(frozen=True, slots=True)
class PlanningFailure:
    reason: NoRouteReason
    detail_code: str | None = None
    related_node_ids: tuple[NodeId, ...] = ()
    related_segment_ids: tuple[SegmentId, ...] = ()


@dataclass(frozen=True, slots=True)
class RoutePlanningResult:
    """Normal result of a planning request.

    Exactly one of these semantic shapes is valid:
    - ROUTE_FOUND: ``route_plan`` is present and ``failure`` is absent.
    - ALREADY_AT_GOAL: selected start/goal are present, route is absent.
    - NO_ROUTE: ``failure`` is present and route is absent.
    """

    request_id: PlanningRequestId
    snapshot_id: SnapshotId
    planned_at: TimePoint
    outcome: PlanningOutcome
    selected_start: PlannedStart | None = None
    selected_goal: PlannedGoal | None = None
    route_plan: RoutePlan | None = None
    failure: PlanningFailure | None = None

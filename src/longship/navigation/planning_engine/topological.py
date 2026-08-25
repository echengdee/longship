"""Generic directed-topology implementation of the Planning Engine."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import math

from longship.navigation.localization_engine.models import (
    LocationHypothesis,
    LocalizationStatus,
    NodeLocation,
)
from longship.navigation.map_engine.interface import MapEngine
from longship.navigation.map_engine.models import (
    NodeId,
    SegmentDescriptor,
    TopologyQuery,
)

from .interface import PlanningEngineError
from .models import (
    NoRouteReason,
    PlannedGoal,
    PlannedStart,
    PlannedTraversal,
    PlanningErrorCode,
    PlanningFailure,
    PlanningOutcome,
    PlanningProvenance,
    RouteEstimate,
    RouteId,
    RouteObjective,
    RoutePlan,
    RoutePlanningRequest,
    RoutePlanningResult,
)


@dataclass(frozen=True, slots=True)
class _Path:
    cost: float
    segments: tuple[SegmentDescriptor, ...]


class TopologicalPlanningEngine:
    """Plans deterministic shortest paths over directed Map topology."""

    def __init__(
        self,
        map_engine: MapEngine,
        *,
        planner_id: str = "longship.topological",
        planner_version: str = "0.1",
    ) -> None:
        if not planner_id.strip() or not planner_version.strip():
            raise ValueError("planner identity must not be empty")
        self._map_engine = map_engine
        self._planner_id = planner_id
        self._planner_version = planner_version

    async def plan_route(
        self,
        request: RoutePlanningRequest,
    ) -> RoutePlanningResult:
        """Returns one directed route bound to the request snapshot."""

        self._validate_request(request)
        start_resolution = _resolve_start(request)
        if isinstance(start_resolution, PlanningFailure):
            return self._failure_result(request, start_resolution)
        start_hypothesis, start_location = start_resolution

        topology = await self._map_engine.query_topology(
            request.snapshot,
            TopologyQuery(),
        )
        known_nodes = {node.node_id for node in topology.nodes}
        if start_location.node_id not in known_nodes:
            return self._failure_result(
                request,
                PlanningFailure(
                    reason=NoRouteReason.START_UNRESOLVED,
                    related_node_ids=(start_location.node_id,),
                ),
            )
        goal_candidates = tuple(
            node_id
            for node_id in request.target.candidate_node_ids
            if node_id in known_nodes
            and node_id not in request.constraints.forbidden_node_ids
            and node_id not in request.context.unavailable_node_ids
        )
        if not goal_candidates:
            return self._failure_result(
                request,
                PlanningFailure(
                    reason=NoRouteReason.TARGET_UNRESOLVED,
                    related_node_ids=request.target.candidate_node_ids,
                ),
            )

        planned_start = PlannedStart(
            belief_revision=request.location_belief.revision,
            hypothesis_id=start_hypothesis.hypothesis_id,
            topological_location=start_location,
        )
        if start_location.node_id in goal_candidates:
            planned_goal = self._planned_goal(
                request,
                start_location.node_id,
            )
            return RoutePlanningResult(
                request_id=request.request_id,
                snapshot_id=request.snapshot.snapshot_id,
                planned_at=request.requested_at,
                outcome=PlanningOutcome.ALREADY_AT_GOAL,
                selected_start=planned_start,
                selected_goal=planned_goal,
            )

        adjacency = self._build_adjacency(request, topology.segments)
        path, goal_node_id = self._shortest_path(
            request,
            start_location.node_id,
            goal_candidates,
            adjacency,
        )
        if path is None or goal_node_id is None:
            return self._failure_result(
                request,
                PlanningFailure(
                    reason=NoRouteReason.TARGET_UNREACHABLE,
                    related_node_ids=goal_candidates,
                ),
            )
        constraint_failure = self._check_path_limits(request, path.segments)
        if constraint_failure is not None:
            return self._failure_result(request, constraint_failure)

        planned_goal = self._planned_goal(request, goal_node_id)
        traversals = tuple(
            PlannedTraversal(
                sequence=index,
                segment_id=segment.segment_id,
                source_node_id=segment.source_node_id,
                target_node_id=segment.target_node_id,
                estimated_distance_m=segment.length_m,
                estimated_duration_s=segment.nominal_duration_s,
                incremental_cost=self._segment_cost(request, segment),
            )
            for index, segment in enumerate(path.segments)
        )
        route_id = _route_id(
            request,
            start_location.node_id,
            goal_node_id,
            path.segments,
        )
        route_plan = RoutePlan(
            route_id=route_id,
            request_id=request.request_id,
            snapshot_id=request.snapshot.snapshot_id,
            created_at=request.requested_at,
            start=planned_start,
            goal=planned_goal,
            traversals=traversals,
            estimate=_estimate(path.segments, path.cost),
            provenance=PlanningProvenance(
                planner_id=self._planner_id,
                planner_version=self._planner_version,
                cost_model_id=request.preferences.objective.value,
                decision_digest=str(route_id).split(":")[-1],
            ),
        )
        return RoutePlanningResult(
            request_id=request.request_id,
            snapshot_id=request.snapshot.snapshot_id,
            planned_at=request.requested_at,
            outcome=PlanningOutcome.ROUTE_FOUND,
            selected_start=planned_start,
            selected_goal=planned_goal,
            route_plan=route_plan,
        )

    def _validate_request(self, request: RoutePlanningRequest) -> None:
        if request.snapshot.snapshot_id != request.location_belief.snapshot_id:
            raise PlanningEngineError(
                PlanningErrorCode.SNAPSHOT_MISMATCH,
                "planning snapshot and location belief do not match",
            )
        if not request.target.target_ref.strip():
            raise PlanningEngineError(
                PlanningErrorCode.INVALID_REQUEST,
                "planning target_ref must not be empty",
            )
        if not request.target.candidate_node_ids:
            raise PlanningEngineError(
                PlanningErrorCode.INVALID_REQUEST,
                "planning target must contain at least one candidate node",
            )
        for segment_id, cost in request.context.extra_segment_costs.items():
            if not math.isfinite(cost) or cost < 0.0:
                raise PlanningEngineError(
                    PlanningErrorCode.INVALID_REQUEST,
                    f"extra cost for {segment_id} must be finite and "
                    "non-negative",
                )
        limits = (
            request.constraints.max_total_distance_m,
            request.constraints.max_total_duration_s,
        )
        if any(
            limit is not None
            and (not math.isfinite(limit) or limit < 0.0)
            for limit in limits
        ):
            raise PlanningEngineError(
                PlanningErrorCode.INVALID_REQUEST,
                "route limits must be finite and non-negative",
            )

    def _build_adjacency(
        self,
        request: RoutePlanningRequest,
        segments: tuple[SegmentDescriptor, ...],
    ) -> dict[NodeId, tuple[SegmentDescriptor, ...]]:
        adjacency: dict[NodeId, list[SegmentDescriptor]] = {}
        for segment in segments:
            if not self._segment_allowed(request, segment):
                continue
            adjacency.setdefault(segment.source_node_id, []).append(segment)
        return {
            node_id: tuple(
                sorted(values, key=lambda item: str(item.segment_id))
            )
            for node_id, values in adjacency.items()
        }

    def _segment_allowed(
        self,
        request: RoutePlanningRequest,
        segment: SegmentDescriptor,
    ) -> bool:
        constraints = request.constraints
        context = request.context
        if segment.segment_id in constraints.forbidden_segment_ids:
            return False
        if segment.segment_id in context.unavailable_segment_ids:
            return False
        if segment.target_node_id in constraints.forbidden_node_ids:
            return False
        if segment.target_node_id in context.unavailable_node_ids:
            return False
        if segment.tags & constraints.forbidden_segment_tags:
            return False
        return segment.required_capabilities.issubset(
            constraints.robot_capabilities
        )

    def _shortest_path(
        self,
        request: RoutePlanningRequest,
        start: NodeId,
        goals: tuple[NodeId, ...],
        adjacency: dict[NodeId, tuple[SegmentDescriptor, ...]],
    ) -> tuple[_Path | None, NodeId | None]:
        goal_set = set(goals)
        queue: list[
            tuple[float, str, NodeId, tuple[SegmentDescriptor, ...]]
        ] = [(0.0, str(start), start, ())]
        best_cost: dict[NodeId, float] = {start: 0.0}
        while queue:
            cost, _, node_id, path = heapq.heappop(queue)
            if cost > best_cost.get(node_id, math.inf) + 1.0e-12:
                continue
            if node_id in goal_set:
                return _Path(cost=cost, segments=path), node_id
            for segment in adjacency.get(node_id, ()):
                edge_cost = self._segment_cost(request, segment)
                new_cost = cost + edge_cost
                target = segment.target_node_id
                if new_cost >= best_cost.get(target, math.inf) - 1.0e-12:
                    continue
                best_cost[target] = new_cost
                heapq.heappush(
                    queue,
                    (
                        new_cost,
                        str(target),
                        target,
                        path + (segment,),
                    ),
                )
        return None, None

    def _segment_cost(
        self,
        request: RoutePlanningRequest,
        segment: SegmentDescriptor,
    ) -> float:
        objective = request.preferences.objective
        if objective == RouteObjective.SHORTEST_DISTANCE:
            cost = segment.length_m if segment.length_m is not None else 1.0
        elif objective == RouteObjective.FASTEST:
            cost = _duration_cost(segment)
        else:
            distance = segment.length_m if segment.length_m is not None else 1.0
            duration = _duration_cost(segment)
            cost = 0.5 * distance + 0.5 * duration
        if segment.tags & request.preferences.preferred_segment_tags:
            cost *= 0.9
        if segment.tags & request.preferences.avoided_segment_tags:
            cost *= 1.25
        cost += request.context.extra_segment_costs.get(segment.segment_id, 0.0)
        return max(cost, 1.0e-9)

    def _check_path_limits(
        self,
        request: RoutePlanningRequest,
        segments: tuple[SegmentDescriptor, ...],
    ) -> PlanningFailure | None:
        distances = tuple(segment.length_m for segment in segments)
        durations = tuple(segment.nominal_duration_s for segment in segments)
        max_distance = request.constraints.max_total_distance_m
        if max_distance is not None and any(
            value is None for value in distances
        ):
            return PlanningFailure(
                reason=NoRouteReason.MAP_DATA_INCOMPLETE,
                detail_code="route_distance_is_unavailable",
                related_segment_ids=tuple(
                    segment.segment_id for segment in segments
                ),
            )
        if (
            max_distance is not None
            and all(value is not None for value in distances)
            and sum(value for value in distances if value is not None)
            > max_distance
        ):
            return PlanningFailure(
                reason=NoRouteReason.CONSTRAINTS_UNSATISFIABLE,
                detail_code="maximum_route_distance_exceeded",
                related_segment_ids=tuple(
                    segment.segment_id for segment in segments
                ),
            )
        max_duration = request.constraints.max_total_duration_s
        if max_duration is not None and any(
            value is None for value in durations
        ):
            return PlanningFailure(
                reason=NoRouteReason.MAP_DATA_INCOMPLETE,
                detail_code="route_duration_is_unavailable",
                related_segment_ids=tuple(
                    segment.segment_id for segment in segments
                ),
            )
        if (
            max_duration is not None
            and all(value is not None for value in durations)
            and sum(value for value in durations if value is not None)
            > max_duration
        ):
            return PlanningFailure(
                reason=NoRouteReason.CONSTRAINTS_UNSATISFIABLE,
                detail_code="maximum_route_duration_exceeded",
                related_segment_ids=tuple(
                    segment.segment_id for segment in segments
                ),
            )
        return None

    def _planned_goal(
        self,
        request: RoutePlanningRequest,
        node_id: NodeId,
    ) -> PlannedGoal:
        return PlannedGoal(
            target_ref=request.target.target_ref,
            node_id=node_id,
            place_id=request.target.place_id,
            completion_anchor_ids=request.target.completion_anchor_ids,
        )

    def _failure_result(
        self,
        request: RoutePlanningRequest,
        failure: PlanningFailure,
    ) -> RoutePlanningResult:
        return RoutePlanningResult(
            request_id=request.request_id,
            snapshot_id=request.snapshot.snapshot_id,
            planned_at=request.requested_at,
            outcome=PlanningOutcome.NO_ROUTE,
            failure=failure,
        )


def _resolve_start(
    request: RoutePlanningRequest,
) -> tuple[LocationHypothesis, NodeLocation] | PlanningFailure:
    belief = request.location_belief
    if belief.status not in (
        LocalizationStatus.TRACKING,
        LocalizationStatus.DEGRADED,
    ):
        return PlanningFailure(reason=NoRouteReason.LOCATION_UNUSABLE)
    candidates = tuple(
        (hypothesis, hypothesis.topological_location)
        for hypothesis in belief.hypotheses
        if isinstance(hypothesis.topological_location, NodeLocation)
    )
    if not candidates:
        return PlanningFailure(reason=NoRouteReason.START_UNRESOLVED)
    if len(candidates) != 1:
        return PlanningFailure(reason=NoRouteReason.START_AMBIGUOUS)
    return candidates[0]


def _duration_cost(segment: SegmentDescriptor) -> float:
    if segment.nominal_duration_s is not None:
        return segment.nominal_duration_s
    if (
        segment.length_m is not None
        and segment.speed_hint_mps is not None
        and segment.speed_hint_mps > 0.0
    ):
        return segment.length_m / segment.speed_hint_mps
    return 1.0


def _estimate(
    segments: tuple[SegmentDescriptor, ...],
    total_cost: float,
) -> RouteEstimate:
    distances = tuple(segment.length_m for segment in segments)
    durations = tuple(segment.nominal_duration_s for segment in segments)
    return RouteEstimate(
        total_distance_m=(
            sum(value for value in distances if value is not None)
            if all(value is not None for value in distances)
            else None
        ),
        total_duration_s=(
            sum(value for value in durations if value is not None)
            if all(value is not None for value in durations)
            else None
        ),
        total_cost=total_cost,
    )


def _route_id(
    request: RoutePlanningRequest,
    start: NodeId,
    goal: NodeId,
    segments: tuple[SegmentDescriptor, ...],
) -> RouteId:
    identity = "|".join(
        (
            str(request.snapshot.snapshot_id),
            str(request.request_id),
            str(start),
            str(goal),
            *(str(segment.segment_id) for segment in segments),
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return RouteId(f"{request.request_id}:route:{digest}")

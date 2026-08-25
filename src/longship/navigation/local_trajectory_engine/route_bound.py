"""Route-bound Local Trajectory Engine implementation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math

from longship.navigation.common import TimePoint, TimeSource
from longship.navigation.local_trajectory_engine.models import (
    LocalTrajectory,
    LocalTrajectoryHoldReason,
    LocalTrajectoryId,
    LocalTrajectoryPublication,
    LocalTrajectoryRevision,
    LocalTrajectoryState,
    LocalTrajectoryStreamId,
    LocalTrajectoryUpdateOutcome,
    LocalTrajectoryUpdateResult,
    LocalTrajectoryWaypoint,
    WaitForLocalTrajectoryRequest,
)
from longship.navigation.localization_engine.interface import LocalizationEngine
from longship.navigation.localization_engine.models import (
    LocationBelief,
    LocalizationStatus,
    NodeLocation,
    SegmentLocation,
)
from longship.navigation.map_engine.interface import MapEngine
from longship.navigation.map_engine.models import (
    AnchorDescriptor,
    AnchorKind,
    AnchorPurpose,
    AnchorQuery,
    MapEntityKind,
    MapEntityRef,
    MapSnapshot,
    NodeId,
    ResourceDescriptor,
    ResourceId,
    ResourceKind,
    SegmentDescriptor,
    TopologyQuery,
)
from longship.navigation.planning_engine.models import (
    PlannedTraversal,
    RoutePlan,
)
from longship.navigation.ports.trajectory_policy import (
    TrajectoryCandidateSet,
    TrajectoryPolicyError,
    TrajectoryPolicyErrorCode,
    VisualGoalTrajectoryPolicy,
    VisualGoalTrajectoryRequest,
)


@dataclass(frozen=True, slots=True)
class LocalTrajectoryEngineConfig:
    """Policy selection, freshness, and compatibility constraints."""

    image_profile_id: str
    model_artifact_id: str
    model_artifact_digest: str
    time_source: TimeSource
    num_candidates: int = 8
    selected_candidate_index: int = 0
    selection_policy_id: str = "first_candidate.v1"
    sampling_seed_base: int = 0
    max_observation_age_s: float = 0.5
    publication_validity_s: float = 0.5
    allow_degraded_localization: bool = False

    def validate(self) -> None:
        text_fields = (
            self.image_profile_id,
            self.model_artifact_id,
            self.selection_policy_id,
        )
        if not all(value.strip() for value in text_fields):
            raise ValueError("route trajectory identities must not be empty")
        if not isinstance(self.time_source, TimeSource):
            raise ValueError("local trajectory time source must provide now()")
        _canonical_sha256(self.model_artifact_digest)
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if not 0 <= self.selected_candidate_index < self.num_candidates:
            raise ValueError("selected_candidate_index is unavailable")
        if self.sampling_seed_base < 0:
            raise ValueError("sampling_seed_base must be non-negative")
        durations = (
            self.max_observation_age_s,
            self.publication_validity_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in durations):
            raise ValueError("trajectory freshness durations must be positive")


@dataclass(frozen=True, slots=True)
class _GoalBinding:
    node_id: NodeId
    anchor: AnchorDescriptor
    resource: ResourceDescriptor


@dataclass(frozen=True, slots=True)
class _ActiveTraversal:
    traversal: PlannedTraversal
    segment: SegmentDescriptor


@dataclass(frozen=True, slots=True)
class _RoutePosition:
    active: _ActiveTraversal | None = None
    completed: bool = False
    hold_reason: LocalTrajectoryHoldReason | None = None
    detail_code: str | None = None


class RouteBoundLocalTrajectoryEngine:
    """Selects one full trajectory for the active RoutePlan traversal.

    This component owns no camera, controller, robot transport, or motion
    command. It consumes the latest localization belief and an injected visual
    trajectory policy, then publishes an immutable stream for consumers outside
    Navigation Harness.
    """

    def __init__(
        self,
        *,
        route_plan: RoutePlan,
        snapshot: MapSnapshot,
        localization_engine: LocalizationEngine,
        trajectory_policy: VisualGoalTrajectoryPolicy,
        active_traversals: tuple[_ActiveTraversal, ...],
        goal_bindings: dict[NodeId, _GoalBinding],
        stream_id: LocalTrajectoryStreamId,
        started_at: TimePoint,
        config: LocalTrajectoryEngineConfig,
    ) -> None:
        self._route_plan = route_plan
        self._snapshot = snapshot
        self._localization_engine = localization_engine
        self._trajectory_policy = trajectory_policy
        self._active_traversals = active_traversals
        self._goal_bindings = goal_bindings
        self._stream_id = stream_id
        self._config = config
        self._sequence = 0
        self._minimum_traversal_sequence = 0
        self._stopped = False
        self._faulted = False
        self._tick_lock = asyncio.Lock()
        self._condition = asyncio.Condition()
        self._latest = LocalTrajectoryPublication(
            revision=LocalTrajectoryRevision(
                stream_id=stream_id,
                sequence=0,
            ),
            route_id=route_plan.route_id,
            snapshot_id=snapshot.snapshot_id,
            state=LocalTrajectoryState.INITIALIZING,
            published_at=started_at,
            detail_code="waiting_for_route_position",
        )

    @classmethod
    async def create(
        cls,
        *,
        map_engine: MapEngine,
        snapshot: MapSnapshot,
        route_plan: RoutePlan,
        localization_engine: LocalizationEngine,
        trajectory_policy: VisualGoalTrajectoryPolicy,
        stream_id: LocalTrajectoryStreamId,
        started_at: TimePoint,
        config: LocalTrajectoryEngineConfig,
    ) -> RouteBoundLocalTrajectoryEngine:
        """Validates and pins every route segment and visual target."""

        config.validate()
        _validate_route_plan(route_plan, snapshot)
        segments = await _load_route_segments(
            map_engine,
            snapshot,
            route_plan,
        )
        bindings = await _load_goal_bindings(
            map_engine,
            snapshot,
            route_plan,
        )
        return cls(
            route_plan=route_plan,
            snapshot=snapshot,
            localization_engine=localization_engine,
            trajectory_policy=trajectory_policy,
            active_traversals=tuple(
                _ActiveTraversal(traversal=traversal, segment=segment)
                for traversal, segment in zip(
                    route_plan.traversals,
                    segments,
                    strict=True,
                )
            ),
            goal_bindings=bindings,
            stream_id=stream_id,
            started_at=started_at,
            config=config,
        )

    def get_latest(self) -> LocalTrajectoryPublication:
        return self._latest

    async def wait_for_update(
        self,
        request: WaitForLocalTrajectoryRequest,
    ) -> LocalTrajectoryUpdateResult:
        if request.timeout_s is not None and (
            not math.isfinite(request.timeout_s) or request.timeout_s < 0.0
        ):
            raise ValueError("timeout_s must be finite and non-negative")
        if request.after_revision.stream_id != self._stream_id:
            return LocalTrajectoryUpdateResult(
                outcome=LocalTrajectoryUpdateOutcome.STREAM_RESET,
                publication=self._latest,
            )
        if self._latest.revision.sequence > request.after_revision.sequence:
            return LocalTrajectoryUpdateResult(
                outcome=LocalTrajectoryUpdateOutcome.UPDATED,
                publication=self._latest,
            )

        async def wait_until_updated() -> None:
            async with self._condition:
                await self._condition.wait_for(
                    lambda: self._latest.revision.sequence
                    > request.after_revision.sequence
                )

        try:
            if request.timeout_s is None:
                await wait_until_updated()
            else:
                await asyncio.wait_for(
                    wait_until_updated(),
                    timeout=request.timeout_s,
                )
        except TimeoutError:
            return LocalTrajectoryUpdateResult(
                outcome=LocalTrajectoryUpdateOutcome.TIMED_OUT,
                publication=self._latest,
            )
        return LocalTrajectoryUpdateResult(
            outcome=LocalTrajectoryUpdateOutcome.UPDATED,
            publication=self._latest,
        )

    async def tick(self, now: TimePoint) -> LocalTrajectoryPublication:
        """Publishes one active trajectory or an explicit hold/completion."""

        async with self._tick_lock:
            if self._stopped or self._faulted:
                return self._latest
            belief = self._localization_engine.get_belief()
            position = self._resolve_route_position(belief)
            if position.completed:
                return await self._publish(
                    now=now,
                    state=LocalTrajectoryState.ROUTE_COMPLETED,
                    belief=belief,
                    detail_code="route_goal_node_confirmed",
                )
            if position.active is None:
                return await self._publish(
                    now=now,
                    state=LocalTrajectoryState.HOLDING,
                    belief=belief,
                    hold_reason=position.hold_reason,
                    detail_code=position.detail_code,
                )

            active = position.active
            binding = self._goal_bindings[active.traversal.target_node_id]
            try:
                candidates = (
                    await self._trajectory_policy.generate_trajectories(
                        self._request(now, active, binding)
                    )
                )
            except TrajectoryPolicyError as error:
                published_at = self._completion_time(now)
                return await self._publish(
                    now=published_at,
                    state=LocalTrajectoryState.HOLDING,
                    belief=belief,
                    active=active,
                    binding=binding,
                    hold_reason=_policy_hold_reason(error.code),
                    detail_code=f"trajectory_policy:{error.code.value}",
                )

            published_at = self._completion_time(now)
            latest_belief = self._localization_engine.get_belief()
            latest_position = self._resolve_route_position(latest_belief)
            if latest_position.completed:
                return await self._publish(
                    now=published_at,
                    state=LocalTrajectoryState.ROUTE_COMPLETED,
                    belief=latest_belief,
                    detail_code="route_goal_node_confirmed",
                )
            if (
                latest_position.active is None
                or latest_position.active.traversal.sequence
                != active.traversal.sequence
            ):
                return await self._publish(
                    now=published_at,
                    state=LocalTrajectoryState.HOLDING,
                    belief=latest_belief,
                    hold_reason=LocalTrajectoryHoldReason.STALE_POLICY_RESULT,
                    detail_code="active_traversal_changed_during_inference",
                )
            self._validate_candidate_set(
                candidates,
                active,
                binding,
                requested_at=now,
            )
            trajectory = self._select_trajectory(candidates)
            if published_at.nanoseconds < candidates.produced_at.nanoseconds:
                raise RuntimeError(
                    "local trajectory publication predates policy result"
                )
            valid_until = TimePoint(
                clock_id=candidates.produced_at.clock_id,
                nanoseconds=(
                    candidates.produced_at.nanoseconds
                    + round(self._config.publication_validity_s * 1e9)
                ),
            )
            if valid_until.nanoseconds <= published_at.nanoseconds:
                return await self._publish(
                    now=published_at,
                    state=LocalTrajectoryState.HOLDING,
                    belief=latest_belief,
                    active=active,
                    binding=binding,
                    hold_reason=LocalTrajectoryHoldReason.OBSERVATION_STALE,
                    detail_code="trajectory_expired_before_publication",
                )
            return await self._publish(
                now=published_at,
                state=LocalTrajectoryState.ACTIVE,
                belief=latest_belief,
                active=active,
                binding=binding,
                candidates=candidates,
                valid_until=valid_until,
                trajectory=trajectory,
                detail_code="trajectory_active",
            )

    async def stop(self, now: TimePoint) -> LocalTrajectoryPublication:
        async with self._tick_lock:
            if self._stopped or self._faulted:
                return self._latest
            self._stopped = True
            return await self._publish(
                now=now,
                state=LocalTrajectoryState.STOPPED,
                hold_reason=LocalTrajectoryHoldReason.SERVICE_STOPPED,
                detail_code="local_trajectory_engine_stopped",
            )

    async def fault(
        self,
        now: TimePoint,
        detail_code: str,
    ) -> LocalTrajectoryPublication:
        async with self._tick_lock:
            if self._stopped or self._faulted:
                return self._latest
            self._faulted = True
            return await self._publish(
                now=now,
                state=LocalTrajectoryState.FAULTED,
                detail_code=detail_code,
            )

    def _completion_time(self, requested_at: TimePoint) -> TimePoint:
        """Returns a post-inference publication time in the request clock."""

        published_at = self._config.time_source.now()
        if published_at.clock_id != requested_at.clock_id:
            raise RuntimeError("local trajectory time source changed clock")
        if published_at.nanoseconds < requested_at.nanoseconds:
            raise RuntimeError(
                "local trajectory publication predates its request"
            )
        return published_at

    def _resolve_route_position(self, belief: LocationBelief) -> _RoutePosition:
        if belief.snapshot_id != self._snapshot.snapshot_id:
            return _RoutePosition(
                hold_reason=LocalTrajectoryHoldReason.LOCATION_UNUSABLE,
                detail_code="localization_snapshot_mismatch",
            )
        allowed_statuses = {LocalizationStatus.TRACKING}
        if self._config.allow_degraded_localization:
            allowed_statuses.add(LocalizationStatus.DEGRADED)
        if belief.status not in allowed_statuses:
            return _RoutePosition(
                hold_reason=LocalTrajectoryHoldReason.LOCATION_UNUSABLE,
                detail_code=f"localization_{belief.status.value}",
            )
        locations = tuple(
            hypothesis.topological_location
            for hypothesis in belief.hypotheses
            if hypothesis.topological_location is not None
        )
        if len(locations) != 1:
            return _RoutePosition(
                hold_reason=(
                    LocalTrajectoryHoldReason.ROUTE_POSITION_UNRESOLVED
                ),
                detail_code="route_requires_one_location_hypothesis",
            )
        location = locations[0]
        if isinstance(location, NodeLocation):
            if location.node_id == self._route_plan.goal.node_id:
                return _RoutePosition(completed=True)
            matches = tuple(
                active
                for active in self._active_traversals
                if active.traversal.source_node_id == location.node_id
            )
        elif isinstance(location, SegmentLocation):
            matches = tuple(
                active
                for active in self._active_traversals
                if active.traversal.segment_id == location.segment_id
            )
        else:
            matches = ()
        if len(matches) != 1:
            return _RoutePosition(
                hold_reason=(
                    LocalTrajectoryHoldReason.ROUTE_POSITION_UNRESOLVED
                ),
                detail_code="location_is_not_on_route",
            )
        active = matches[0]
        if active.traversal.sequence < self._minimum_traversal_sequence:
            return _RoutePosition(
                hold_reason=(
                    LocalTrajectoryHoldReason.ROUTE_POSITION_UNRESOLVED
                ),
                detail_code="route_position_regressed",
            )
        self._minimum_traversal_sequence = active.traversal.sequence
        return _RoutePosition(active=active)

    def _request(
        self,
        now: TimePoint,
        active: _ActiveTraversal,
        binding: _GoalBinding,
    ) -> VisualGoalTrajectoryRequest:
        return VisualGoalTrajectoryRequest(
            snapshot_id=self._snapshot.snapshot_id,
            segment_id=active.traversal.segment_id,
            source_node_id=active.traversal.source_node_id,
            target_node_id=active.traversal.target_node_id,
            target_anchor_id=binding.anchor.anchor_id,
            goal_resource=binding.resource,
            requested_at=now,
            max_observation_age_s=self._config.max_observation_age_s,
            num_candidates=self._config.num_candidates,
            sampling_seed=self._config.sampling_seed_base + self._sequence,
            expected_image_profile_id=self._config.image_profile_id,
            expected_model_artifact_id=self._config.model_artifact_id,
            expected_model_artifact_digest=self._config.model_artifact_digest,
        )

    def _validate_candidate_set(
        self,
        candidates: TrajectoryCandidateSet,
        active: _ActiveTraversal,
        binding: _GoalBinding,
        *,
        requested_at: TimePoint,
    ) -> None:
        expected = (
            self._snapshot.snapshot_id,
            active.traversal.segment_id,
            active.traversal.source_node_id,
            active.traversal.target_node_id,
            binding.anchor.anchor_id,
            binding.resource.resource_id,
        )
        actual = (
            candidates.snapshot_id,
            candidates.segment_id,
            candidates.source_node_id,
            candidates.target_node_id,
            candidates.target_anchor_id,
            candidates.goal_resource_id,
        )
        if actual != expected:
            raise RuntimeError("trajectory policy changed route identity")
        if len(candidates.candidates) != self._config.num_candidates:
            raise RuntimeError("trajectory policy changed candidate count")
        expected_policy_identity = (
            self._config.image_profile_id,
            self._config.model_artifact_id,
            _canonical_sha256(self._config.model_artifact_digest),
        )
        actual_policy_identity = (
            candidates.image_profile_id,
            candidates.model_artifact_id,
            _canonical_sha256(candidates.model_artifact_digest),
        )
        if actual_policy_identity != expected_policy_identity:
            raise RuntimeError(
                "trajectory policy changed compatibility identity"
            )
        expected_seed = self._config.sampling_seed_base + self._sequence
        if candidates.sampling_seed != expected_seed:
            raise RuntimeError("trajectory policy changed the sampling seed")
        if not candidates.policy_id.strip():
            raise RuntimeError("trajectory policy id must not be empty")
        if (
            not candidates.coordinate_frame.strip()
            or not candidates.coordinate_units.strip()
        ):
            raise RuntimeError("trajectory coordinate labels must not be empty")
        if not math.isfinite(candidates.temporal_distance):
            raise RuntimeError("trajectory temporal distance must be finite")
        if candidates.observation_time.clock_id != requested_at.clock_id:
            raise RuntimeError("trajectory observation clock changed")
        if candidates.produced_at.clock_id != requested_at.clock_id:
            raise RuntimeError("trajectory production clock changed")
        if candidates.produced_at.nanoseconds < requested_at.nanoseconds:
            raise RuntimeError("trajectory production predates its request")
        if (
            candidates.observation_time.nanoseconds
            > candidates.produced_at.nanoseconds
        ):
            raise RuntimeError("trajectory production predates its observation")
        observation_age_ns = (
            requested_at.nanoseconds
            - candidates.observation_time.nanoseconds
        )
        maximum_age_ns = round(self._config.max_observation_age_s * 1e9)
        if observation_age_ns < 0 or observation_age_ns > maximum_age_ns:
            raise RuntimeError(
                "trajectory observation is outside freshness window"
            )
        for candidate in candidates.candidates:
            if not candidate.waypoints:
                raise RuntimeError("trajectory candidate contains no waypoints")
            for expected_step, waypoint in enumerate(candidate.waypoints):
                if waypoint.step_index != expected_step:
                    raise RuntimeError(
                        "trajectory waypoint indices must be contiguous"
                    )
                if (
                    not math.isfinite(waypoint.x)
                    or not math.isfinite(waypoint.y)
                ):
                    raise RuntimeError("trajectory waypoint must be finite")

    def _select_trajectory(
        self,
        candidates: TrajectoryCandidateSet,
    ) -> LocalTrajectory:
        selected = candidates.candidates[
            self._config.selected_candidate_index
        ]
        if not selected.waypoints:
            raise RuntimeError("selected trajectory contains no waypoints")
        return LocalTrajectory(
            trajectory_id=LocalTrajectoryId(
                f"{self._stream_id}:{self._sequence + 1}:trajectory"
            ),
            source_candidate_id=str(selected.candidate_id),
            waypoints=tuple(
                LocalTrajectoryWaypoint(
                    step_index=waypoint.step_index,
                    x=waypoint.x,
                    y=waypoint.y,
                )
                for waypoint in selected.waypoints
            ),
            coordinate_frame=candidates.coordinate_frame,
            coordinate_units=candidates.coordinate_units,
            selection_policy_id=self._config.selection_policy_id,
            source_candidate_index=self._config.selected_candidate_index,
            source_candidate_count=len(candidates.candidates),
            sampling_seed=candidates.sampling_seed,
            temporal_distance=candidates.temporal_distance,
            policy_id=candidates.policy_id,
            image_profile_id=candidates.image_profile_id,
            model_artifact_id=candidates.model_artifact_id,
            model_artifact_digest=candidates.model_artifact_digest,
        )

    async def _publish(
        self,
        *,
        now: TimePoint,
        state: LocalTrajectoryState,
        belief: LocationBelief | None = None,
        active: _ActiveTraversal | None = None,
        binding: _GoalBinding | None = None,
        candidates: TrajectoryCandidateSet | None = None,
        valid_until: TimePoint | None = None,
        trajectory: LocalTrajectory | None = None,
        hold_reason: LocalTrajectoryHoldReason | None = None,
        detail_code: str | None = None,
    ) -> LocalTrajectoryPublication:
        self._sequence += 1
        publication = LocalTrajectoryPublication(
            revision=LocalTrajectoryRevision(
                stream_id=self._stream_id,
                sequence=self._sequence,
            ),
            route_id=self._route_plan.route_id,
            snapshot_id=self._snapshot.snapshot_id,
            state=state,
            published_at=now,
            belief_revision=(None if belief is None else belief.revision),
            traversal_sequence=(
                None if active is None else active.traversal.sequence
            ),
            segment_id=(
                None if active is None else active.traversal.segment_id
            ),
            source_node_id=(
                None if active is None else active.traversal.source_node_id
            ),
            target_node_id=(
                None if active is None else active.traversal.target_node_id
            ),
            target_anchor_id=(
                None if binding is None else binding.anchor.anchor_id
            ),
            goal_resource_id=(
                None if binding is None else binding.resource.resource_id
            ),
            observation_time=(
                None if candidates is None else candidates.observation_time
            ),
            generated_at=(
                None if candidates is None else candidates.produced_at
            ),
            valid_until=valid_until,
            trajectory=trajectory,
            hold_reason=hold_reason,
            detail_code=detail_code,
        )
        _validate_publication(publication)
        self._latest = publication
        async with self._condition:
            self._condition.notify_all()
        return publication


def _validate_route_plan(route_plan: RoutePlan, snapshot: MapSnapshot) -> None:
    if route_plan.snapshot_id != snapshot.snapshot_id:
        raise ValueError("RoutePlan snapshot does not match MapSnapshot")
    previous_target: NodeId | None = None
    source_node_ids: set[NodeId] = set()
    for index, traversal in enumerate(route_plan.traversals):
        if traversal.sequence != index:
            raise ValueError("RoutePlan traversal sequence must be contiguous")
        if not (
            0.0
            <= traversal.entry_progress
            < traversal.exit_progress
            <= 1.0
        ):
            raise ValueError("RoutePlan traversal progress is invalid")
        if (
            previous_target is not None
            and traversal.source_node_id != previous_target
        ):
            raise ValueError(
                "RoutePlan traversals must form one directed chain"
            )
        if traversal.source_node_id in source_node_ids:
            raise ValueError(
                "RoutePlan cannot revisit a source node in v0.1"
            )
        source_node_ids.add(traversal.source_node_id)
        previous_target = traversal.target_node_id
    if route_plan.traversals:
        start = route_plan.start.topological_location
        first = route_plan.traversals[0]
        if (
            isinstance(start, NodeLocation)
            and start.node_id != first.source_node_id
        ):
            raise ValueError(
                "RoutePlan start does not match its first traversal"
            )
        if (
            isinstance(start, SegmentLocation)
            and start.segment_id != first.segment_id
        ):
            raise ValueError(
                "RoutePlan start does not match its first traversal"
            )
    if (
        route_plan.traversals
        and route_plan.traversals[-1].target_node_id
        != route_plan.goal.node_id
    ):
        raise ValueError("RoutePlan does not terminate at its planned goal")
    if not route_plan.traversals:
        start = route_plan.start.topological_location
        if not isinstance(start, NodeLocation) or (
            start.node_id != route_plan.goal.node_id
        ):
            raise ValueError("empty RoutePlan must already start at its goal")


async def _load_route_segments(
    map_engine: MapEngine,
    snapshot: MapSnapshot,
    route_plan: RoutePlan,
) -> tuple[SegmentDescriptor, ...]:
    result = await map_engine.query_topology(
        snapshot,
        TopologyQuery(
            segment_ids=tuple(
                traversal.segment_id for traversal in route_plan.traversals
            )
        ),
    )
    if result.missing_segment_ids:
        raise ValueError("RoutePlan references missing Map segments")
    indexed = {segment.segment_id: segment for segment in result.segments}
    segments = []
    for traversal in route_plan.traversals:
        segment = indexed.get(traversal.segment_id)
        if segment is None:
            raise ValueError("RoutePlan segment is disabled or unavailable")
        if (
            segment.source_node_id != traversal.source_node_id
            or segment.target_node_id != traversal.target_node_id
        ):
            raise ValueError("RoutePlan traversal changed Map segment identity")
        segments.append(segment)
    return tuple(segments)


async def _load_goal_bindings(
    map_engine: MapEngine,
    snapshot: MapSnapshot,
    route_plan: RoutePlan,
) -> dict[NodeId, _GoalBinding]:
    target_nodes = tuple(
        dict.fromkeys(
            traversal.target_node_id for traversal in route_plan.traversals
        )
    )
    if not target_nodes:
        return {}
    anchors = await map_engine.query_anchors(
        snapshot,
        AnchorQuery(
            attached_to=tuple(
                MapEntityRef(kind=MapEntityKind.NODE, entity_id=str(node_id))
                for node_id in target_nodes
            ),
            kinds=frozenset({AnchorKind.VISUAL}),
            purposes=frozenset({AnchorPurpose.TARGET}),
            limit=max(200, len(target_nodes) * 4),
        ),
    )
    anchors_by_node: dict[NodeId, list[AnchorDescriptor]] = {}
    for anchor in anchors.anchors:
        if anchor.attached_to.kind != MapEntityKind.NODE:
            continue
        node_id = NodeId(anchor.attached_to.entity_id)
        anchors_by_node.setdefault(node_id, []).append(anchor)
    selected: dict[NodeId, AnchorDescriptor] = {}
    resource_ids: list[ResourceId] = []
    for node_id in target_nodes:
        candidates = anchors_by_node.get(node_id, [])
        if len(candidates) != 1:
            raise ValueError(
                f"route target {node_id} requires one visual TARGET anchor"
            )
        anchor = candidates[0]
        if len(anchor.resource_ids) != 1:
            raise ValueError(
                f"route target {node_id} requires one goal resource"
            )
        selected[node_id] = anchor
        resource_ids.append(anchor.resource_ids[0])
    resources = await map_engine.resolve_resources(
        snapshot,
        tuple(resource_ids),
    )
    if resources.missing_resource_ids:
        raise ValueError("route target goal resource is missing")
    resources_by_id = {
        resource.resource_id: resource for resource in resources.resources
    }
    bindings = {}
    for node_id, anchor in selected.items():
        resource = resources_by_id.get(anchor.resource_ids[0])
        if resource is None or resource.kind != ResourceKind.IMAGE:
            raise ValueError("route target resource must be one image")
        bindings[node_id] = _GoalBinding(
            node_id=node_id,
            anchor=anchor,
            resource=resource,
        )
    return bindings


def _policy_hold_reason(
    code: TrajectoryPolicyErrorCode,
) -> LocalTrajectoryHoldReason:
    if code == TrajectoryPolicyErrorCode.CONTEXT_NOT_READY:
        return LocalTrajectoryHoldReason.OBSERVATION_CONTEXT_NOT_READY
    if code == TrajectoryPolicyErrorCode.CONTEXT_STALE:
        return LocalTrajectoryHoldReason.OBSERVATION_STALE
    if code == TrajectoryPolicyErrorCode.GOAL_UNAVAILABLE:
        return LocalTrajectoryHoldReason.GOAL_UNAVAILABLE
    return LocalTrajectoryHoldReason.POLICY_UNAVAILABLE


def _validate_publication(publication: LocalTrajectoryPublication) -> None:
    if publication.state == LocalTrajectoryState.ACTIVE:
        required = (
            publication.belief_revision,
            publication.traversal_sequence,
            publication.segment_id,
            publication.source_node_id,
            publication.target_node_id,
            publication.target_anchor_id,
            publication.goal_resource_id,
            publication.observation_time,
            publication.generated_at,
            publication.valid_until,
            publication.trajectory,
        )
        if any(value is None for value in required):
            raise RuntimeError("active trajectory publication is incomplete")
        if publication.hold_reason is not None:
            raise RuntimeError("active trajectory publication cannot hold")
        assert publication.observation_time is not None
        assert publication.generated_at is not None
        assert publication.valid_until is not None
        clocks = {
            publication.published_at.clock_id,
            publication.observation_time.clock_id,
            publication.generated_at.clock_id,
            publication.valid_until.clock_id,
        }
        if len(clocks) != 1:
            raise RuntimeError("active trajectory clocks must match")
        if publication.valid_until.nanoseconds <= (
            publication.published_at.nanoseconds
        ):
            raise RuntimeError("active trajectory must expire in the future")
    elif publication.trajectory is not None:
        raise RuntimeError("non-active publication cannot contain a trajectory")
    if (
        publication.state == LocalTrajectoryState.HOLDING
        and publication.hold_reason is None
    ):
        raise RuntimeError("holding publication requires a reason")


def _canonical_sha256(value: str) -> str:
    digest = value.removeprefix("sha256:").casefold()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("model_artifact_digest must be a SHA-256 digest")
    return f"sha256:{digest}"

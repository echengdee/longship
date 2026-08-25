"""Tests for the route-bound Local Trajectory Engine."""

from __future__ import annotations

import asyncio
import unittest

from longship.navigation.common import TimePoint
from longship.navigation.local_trajectory_engine import (
    LocalTrajectoryEngine,
    LocalTrajectoryEngineConfig,
    LocalTrajectoryRevision,
    LocalTrajectoryState,
    LocalTrajectoryStream,
    LocalTrajectoryStreamId,
    LocalTrajectoryUpdateOutcome,
    RouteBoundLocalTrajectoryEngine,
    WaitForLocalTrajectoryRequest,
)
from longship.navigation.localization_engine.models import (
    BeliefRevision,
    BeliefStreamId,
    HypothesisId,
    LocationBelief,
    LocationHypothesis,
    LocalizationStatus,
    NodeLocation,
)
from longship.navigation.map_engine.models import (
    AnchorDescriptor,
    AnchorId,
    AnchorKind,
    AnchorPurpose,
    MapCapability,
    MapEntityKind,
    MapEntityRef,
    MapId,
    MapSnapshot,
    MapVersion,
    NodeId,
    ResourceDescriptor,
    ResourceId,
    ResourceKind,
    SegmentDescriptor,
    SegmentId,
    SnapshotId,
    TopologyNode,
)
from longship.navigation.map_engine.static import StaticMap, StaticMapEngine
from longship.navigation.planning_engine.models import (
    PlannedGoal,
    PlannedStart,
    PlannedTraversal,
    PlanningProvenance,
    PlanningRequestId,
    RouteEstimate,
    RouteId,
    RoutePlan,
)
from longship.navigation.ports.trajectory_policy import (
    PolicyNativeWaypoint,
    TrajectoryCandidate,
    TrajectoryCandidateId,
    TrajectoryCandidateSet,
    VisualGoalTrajectoryRequest,
)


_MODEL_DIGEST = "sha256:" + "1" * 64


class _TimeSource:
    def __init__(self, now: TimePoint) -> None:
        self.now_value = now

    def now(self) -> TimePoint:
        return self.now_value

    def set(self, now: TimePoint) -> None:
        self.now_value = now


class _MutableLocalization:
    def __init__(self, belief: LocationBelief) -> None:
        self.belief = belief

    def get_belief(self) -> LocationBelief:
        return self.belief

    async def wait_for_update(self, request: object) -> object:
        raise NotImplementedError

    async def request_relocalization(self, request: object) -> object:
        raise NotImplementedError

    def get_status(self) -> object:
        raise NotImplementedError


class _FakeTrajectoryPolicy:
    def __init__(self) -> None:
        self.requests: list[VisualGoalTrajectoryRequest] = []
        self.started: asyncio.Event | None = None
        self.release: asyncio.Event | None = None

    async def generate_trajectories(
        self,
        request: VisualGoalTrajectoryRequest,
    ) -> TrajectoryCandidateSet:
        self.requests.append(request)
        if self.started is not None and self.release is not None:
            self.started.set()
            await self.release.wait()
        candidates = tuple(
            TrajectoryCandidate(
                candidate_id=TrajectoryCandidateId(
                    f"{request.segment_id}:sample-{candidate_index:04d}"
                ),
                waypoints=tuple(
                    PolicyNativeWaypoint(
                        step_index=step_index,
                        x=float(step_index + 1),
                        y=float(candidate_index),
                    )
                    for step_index in range(8)
                ),
            )
            for candidate_index in range(request.num_candidates)
        )
        return TrajectoryCandidateSet(
            snapshot_id=request.snapshot_id,
            segment_id=request.segment_id,
            source_node_id=request.source_node_id,
            target_node_id=request.target_node_id,
            target_anchor_id=request.target_anchor_id,
            goal_resource_id=request.goal_resource.resource_id,
            observation_time=request.requested_at,
            produced_at=request.requested_at,
            temporal_distance=5.0,
            coordinate_frame="nomad.policy_native.robot_frame.v1",
            coordinate_units="nomad.policy_native.v1",
            sampling_seed=request.sampling_seed,
            candidates=candidates,
            policy_id="nomad",
            image_profile_id=request.expected_image_profile_id,
            model_artifact_id=request.expected_model_artifact_id,
            model_artifact_digest=request.expected_model_artifact_digest,
        )


class RouteBoundLocalTrajectoryEngineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.now = TimePoint(clock_id="camera", nanoseconds=1_000_000_000)
        self.snapshot = _snapshot(self.now)
        self.map_engine = _map_engine(self.snapshot)
        self.localization = _MutableLocalization(
            _belief(
                self.snapshot.snapshot_id,
                self.now,
                status=LocalizationStatus.INITIALIZING,
                node_id=None,
                sequence=0,
            )
        )
        self.policy = _FakeTrajectoryPolicy()
        self.time_source = _TimeSource(_time(1))
        self.engine = await RouteBoundLocalTrajectoryEngine.create(
            map_engine=self.map_engine,
            snapshot=self.snapshot,
            route_plan=_route_plan(self.snapshot.snapshot_id, self.now),
            localization_engine=self.localization,
            trajectory_policy=self.policy,
            stream_id=LocalTrajectoryStreamId("trajectory-stream"),
            started_at=self.now,
            config=LocalTrajectoryEngineConfig(
                image_profile_id="profile",
                model_artifact_id="nomad.pth",
                model_artifact_digest=_MODEL_DIGEST,
                time_source=self.time_source,
            ),
        )

    async def test_publishes_full_sample_zero_and_advances_segments(
        self,
    ) -> None:
        self.assertIsInstance(self.engine, LocalTrajectoryStream)
        self.assertIsInstance(self.engine, LocalTrajectoryEngine)
        hold = await self.engine.tick(_time(2))
        self.assertEqual(hold.state, LocalTrajectoryState.HOLDING)

        self.localization.belief = _belief(
            self.snapshot.snapshot_id,
            _time(3),
            status=LocalizationStatus.TRACKING,
            node_id=NodeId("node-0"),
            sequence=1,
        )
        self.time_source.set(_time(3))
        first = await self.engine.tick(_time(3))

        self.assertEqual(first.state, LocalTrajectoryState.ACTIVE)
        self.assertEqual(first.traversal_sequence, 0)
        self.assertEqual(first.segment_id, SegmentId("edge-0"))
        assert first.trajectory is not None
        self.assertEqual(first.trajectory.source_candidate_index, 0)
        self.assertEqual(first.trajectory.source_candidate_count, 8)
        self.assertEqual(
            first.trajectory.source_candidate_id,
            "edge-0:sample-0000",
        )
        self.assertEqual(len(first.trajectory.waypoints), 8)
        self.assertTrue(
            all(waypoint.y == 0.0 for waypoint in first.trajectory.waypoints)
        )

        self.localization.belief = _belief(
            self.snapshot.snapshot_id,
            _time(4),
            status=LocalizationStatus.TRACKING,
            node_id=NodeId("node-1"),
            sequence=2,
        )
        self.time_source.set(_time(4))
        second = await self.engine.tick(_time(4))

        self.assertEqual(second.state, LocalTrajectoryState.ACTIVE)
        self.assertEqual(second.traversal_sequence, 1)
        self.assertEqual(second.segment_id, SegmentId("edge-1"))
        self.assertEqual(second.goal_resource_id, ResourceId("node-2:image"))
        assert second.trajectory is not None
        self.assertNotEqual(
            first.trajectory.trajectory_id,
            second.trajectory.trajectory_id,
        )

        self.localization.belief = _belief(
            self.snapshot.snapshot_id,
            _time(5),
            status=LocalizationStatus.TRACKING,
            node_id=NodeId("node-2"),
            sequence=3,
        )
        completed = await self.engine.tick(_time(5))

        self.assertEqual(
            completed.state,
            LocalTrajectoryState.ROUTE_COMPLETED,
        )
        self.assertIsNone(completed.trajectory)

    async def test_wait_for_update_reports_reset_and_timeout(self) -> None:
        reset = await self.engine.wait_for_update(
            WaitForLocalTrajectoryRequest(
                after_revision=LocalTrajectoryRevision(
                    stream_id=LocalTrajectoryStreamId("another-stream"),
                    sequence=0,
                ),
            )
        )
        self.assertEqual(
            reset.outcome,
            LocalTrajectoryUpdateOutcome.STREAM_RESET,
        )

        current = self.engine.get_latest().revision
        timed_out = await self.engine.wait_for_update(
            WaitForLocalTrajectoryRequest(
                after_revision=current,
                timeout_s=0.0,
            )
        )
        self.assertEqual(
            timed_out.outcome,
            LocalTrajectoryUpdateOutcome.TIMED_OUT,
        )

    async def test_fault_is_terminal(self) -> None:
        fault = await self.engine.fault(_time(2), "test_fault")
        after_fault = await self.engine.tick(_time(3))

        self.assertEqual(fault.state, LocalTrajectoryState.FAULTED)
        self.assertEqual(after_fault, fault)

    async def test_holds_trajectory_that_expired_during_inference(self) -> None:
        self.localization.belief = _belief(
            self.snapshot.snapshot_id,
            _time(3),
            status=LocalizationStatus.TRACKING,
            node_id=NodeId("node-0"),
            sequence=1,
        )
        self.time_source.set(_time(4))

        publication = await self.engine.tick(_time(3))

        self.assertEqual(publication.state, LocalTrajectoryState.HOLDING)
        self.assertEqual(
            publication.detail_code,
            "trajectory_expired_before_publication",
        )

    async def test_active_publication_uses_latest_same_segment_belief(
        self,
    ) -> None:
        self.localization.belief = _belief(
            self.snapshot.snapshot_id,
            _time(3),
            status=LocalizationStatus.TRACKING,
            node_id=NodeId("node-0"),
            sequence=1,
        )
        self.policy.started = asyncio.Event()
        self.policy.release = asyncio.Event()
        self.time_source.set(_time(3))
        tick = asyncio.create_task(self.engine.tick(_time(3)))
        await self.policy.started.wait()
        self.localization.belief = _belief(
            self.snapshot.snapshot_id,
            _time(3),
            status=LocalizationStatus.TRACKING,
            node_id=NodeId("node-0"),
            sequence=2,
        )
        self.policy.release.set()

        publication = await tick

        self.assertEqual(publication.state, LocalTrajectoryState.ACTIVE)
        assert publication.belief_revision is not None
        self.assertEqual(publication.belief_revision.sequence, 2)


def _time(seconds: int) -> TimePoint:
    return TimePoint(clock_id="camera", nanoseconds=seconds * 1_000_000_000)


def _snapshot(now: TimePoint) -> MapSnapshot:
    return MapSnapshot(
        snapshot_id=SnapshotId("snapshot"),
        map_id=MapId("map"),
        version=MapVersion("v1"),
        schema_version="test.v1",
        content_digest="sha256:map",
        published_at=now,
        map_frame=None,
        capabilities=frozenset(
            {
                MapCapability.TOPOLOGY,
                MapCapability.VISUAL_ANCHORS,
                MapCapability.RESOURCE_REFERENCES,
            }
        ),
    )


def _map_engine(snapshot: MapSnapshot) -> StaticMapEngine:
    nodes = tuple(
        TopologyNode(node_id=NodeId(f"node-{index}"))
        for index in range(3)
    )
    segments = tuple(
        SegmentDescriptor(
            segment_id=SegmentId(f"edge-{index}"),
            source_node_id=NodeId(f"node-{index}"),
            target_node_id=NodeId(f"node-{index + 1}"),
        )
        for index in range(2)
    )
    anchors = tuple(
        AnchorDescriptor(
            anchor_id=AnchorId(f"node-{index}:target"),
            kind=AnchorKind.VISUAL,
            purposes=frozenset({AnchorPurpose.TARGET}),
            attached_to=MapEntityRef(
                kind=MapEntityKind.NODE,
                entity_id=f"node-{index}",
            ),
            resource_ids=(ResourceId(f"node-{index}:image"),),
        )
        for index in (1, 2)
    )
    resources = tuple(
        ResourceDescriptor(
            resource_id=ResourceId(f"node-{index}:image"),
            kind=ResourceKind.IMAGE,
            locator=f"/tmp/node-{index}.png",
        )
        for index in (1, 2)
    )
    return StaticMapEngine(
        StaticMap(
            snapshot=snapshot,
            nodes=nodes,
            segments=segments,
            anchors=anchors,
            resources=resources,
        )
    )


def _belief(
    snapshot_id: SnapshotId,
    now: TimePoint,
    *,
    status: LocalizationStatus,
    node_id: NodeId | None,
    sequence: int,
) -> LocationBelief:
    hypotheses = ()
    if node_id is not None:
        hypotheses = (
            LocationHypothesis(
                hypothesis_id=HypothesisId(f"hypothesis:{node_id}"),
                topological_location=NodeLocation(node_id=node_id),
            ),
        )
    return LocationBelief(
        snapshot_id=snapshot_id,
        revision=BeliefRevision(
            stream_id=BeliefStreamId("belief"),
            sequence=sequence,
        ),
        estimate_time=now,
        published_at=now,
        status=status,
        confidence=None,
        hypotheses=hypotheses,
    )


def _route_plan(snapshot_id: SnapshotId, now: TimePoint) -> RoutePlan:
    start_location = NodeLocation(NodeId("node-0"))
    return RoutePlan(
        route_id=RouteId("route"),
        request_id=PlanningRequestId("request"),
        snapshot_id=snapshot_id,
        created_at=now,
        start=PlannedStart(
            belief_revision=BeliefRevision(
                stream_id=BeliefStreamId("belief"),
                sequence=0,
            ),
            hypothesis_id=HypothesisId("hypothesis:node-0"),
            topological_location=start_location,
        ),
        goal=PlannedGoal(target_ref="goal", node_id=NodeId("node-2")),
        traversals=tuple(
            PlannedTraversal(
                sequence=index,
                segment_id=SegmentId(f"edge-{index}"),
                source_node_id=NodeId(f"node-{index}"),
                target_node_id=NodeId(f"node-{index + 1}"),
            )
            for index in range(2)
        ),
        estimate=RouteEstimate(),
        provenance=PlanningProvenance(
            planner_id="test",
            planner_version="1",
        ),
    )


if __name__ == "__main__":
    unittest.main()

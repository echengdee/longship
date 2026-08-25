"""Tests for the generic directed-topology planning implementation."""

from __future__ import annotations

import unittest

from longship.navigation.common import TimePoint
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
    MapCapability,
    MapId,
    MapSnapshot,
    MapVersion,
    NodeId,
    SegmentDescriptor,
    SegmentId,
    SnapshotId,
    TopologyNode,
)
from longship.navigation.map_engine.static import StaticMap, StaticMapEngine
from longship.navigation.planning_engine.interface import PlanningEngineError
from longship.navigation.planning_engine.models import (
    NoRouteReason,
    PlanningOutcome,
    PlanningRequestId,
    PlanningTarget,
    RouteConstraints,
    RouteObjective,
    RoutePlanningRequest,
    RoutePreferences,
)
from longship.navigation.planning_engine.topological import (
    TopologicalPlanningEngine,
)


class TopologicalPlanningEngineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.now = TimePoint(clock_id="camera", nanoseconds=1_000_000_000)
        self.snapshot = MapSnapshot(
            snapshot_id=SnapshotId("snapshot"),
            map_id=MapId("map"),
            version=MapVersion("v1"),
            schema_version="test.v1",
            content_digest="sha256:map",
            published_at=self.now,
            map_frame=None,
            capabilities=frozenset({MapCapability.TOPOLOGY}),
        )
        nodes = tuple(
            TopologyNode(node_id=NodeId(f"node-{index}"))
            for index in range(3)
        )
        segments = (
            SegmentDescriptor(
                segment_id=SegmentId("edge-direct"),
                source_node_id=NodeId("node-0"),
                target_node_id=NodeId("node-2"),
                length_m=10.0,
            ),
            SegmentDescriptor(
                segment_id=SegmentId("edge-0"),
                source_node_id=NodeId("node-0"),
                target_node_id=NodeId("node-1"),
                length_m=2.0,
            ),
            SegmentDescriptor(
                segment_id=SegmentId("edge-1"),
                source_node_id=NodeId("node-1"),
                target_node_id=NodeId("node-2"),
                length_m=2.0,
            ),
        )
        self.map_engine = StaticMapEngine(
            StaticMap(
                snapshot=self.snapshot,
                nodes=nodes,
                segments=segments,
            )
        )
        self.engine = TopologicalPlanningEngine(self.map_engine)

    async def test_builds_an_immutable_shortest_route_plan(self) -> None:
        result = await self.engine.plan_route(self._request())

        self.assertEqual(result.outcome, PlanningOutcome.ROUTE_FOUND)
        self.assertIsNotNone(result.route_plan)
        route = result.route_plan
        assert route is not None
        self.assertEqual(route.snapshot_id, self.snapshot.snapshot_id)
        self.assertEqual(
            tuple(item.segment_id for item in route.traversals),
            (SegmentId("edge-0"), SegmentId("edge-1")),
        )
        self.assertEqual(
            tuple(item.sequence for item in route.traversals),
            (0, 1),
        )
        self.assertEqual(route.estimate.total_distance_m, 4.0)
        self.assertEqual(route.goal.node_id, NodeId("node-2"))

    async def test_respects_forbidden_segments(self) -> None:
        result = await self.engine.plan_route(
            self._request(
                constraints=RouteConstraints(
                    forbidden_segment_ids=frozenset({SegmentId("edge-0")}),
                )
            )
        )

        route = result.route_plan
        assert route is not None
        self.assertEqual(
            tuple(item.segment_id for item in route.traversals),
            (SegmentId("edge-direct"),),
        )

    async def test_requires_map_metadata_for_a_duration_limit(self) -> None:
        result = await self.engine.plan_route(
            self._request(
                constraints=RouteConstraints(max_total_duration_s=5.0)
            )
        )

        self.assertEqual(result.outcome, PlanningOutcome.NO_ROUTE)
        assert result.failure is not None
        self.assertEqual(
            result.failure.reason,
            NoRouteReason.MAP_DATA_INCOMPLETE,
        )

    async def test_rejects_a_negative_route_limit(self) -> None:
        with self.assertRaises(PlanningEngineError):
            await self.engine.plan_route(
                self._request(
                    constraints=RouteConstraints(
                        max_total_distance_m=-1.0
                    )
                )
            )

    def _request(
        self,
        *,
        constraints: RouteConstraints = RouteConstraints(),
    ) -> RoutePlanningRequest:
        belief = LocationBelief(
            snapshot_id=self.snapshot.snapshot_id,
            revision=BeliefRevision(
                stream_id=BeliefStreamId("belief"),
                sequence=3,
            ),
            estimate_time=self.now,
            published_at=self.now,
            status=LocalizationStatus.TRACKING,
            confidence=1.0,
            hypotheses=(
                LocationHypothesis(
                    hypothesis_id=HypothesisId("node-0"),
                    topological_location=NodeLocation(NodeId("node-0")),
                ),
            ),
        )
        return RoutePlanningRequest(
            request_id=PlanningRequestId("request"),
            requested_at=self.now,
            snapshot=self.snapshot,
            location_belief=belief,
            target=PlanningTarget(
                target_ref="final-node",
                candidate_node_ids=(NodeId("node-2"),),
            ),
            constraints=constraints,
            preferences=RoutePreferences(
                objective=RouteObjective.SHORTEST_DISTANCE
            ),
        )


if __name__ == "__main__":
    unittest.main()

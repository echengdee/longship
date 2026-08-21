"""Tests for the generic immutable Map Engine implementation."""

from __future__ import annotations

import unittest

from longship.navigation.common import TimePoint
from longship.navigation.map_engine.interface import MapEngineError
from longship.navigation.map_engine.models import (
    MapCapability,
    MapErrorCode,
    MapId,
    MapSelector,
    MapSnapshot,
    MapVersion,
    NodeId,
    SegmentDescriptor,
    SegmentId,
    SnapshotId,
    TopologyNode,
    TopologyQuery,
)
from longship.navigation.map_engine.static import StaticMap, StaticMapEngine


class StaticMapEngineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.snapshot = MapSnapshot(
            snapshot_id=SnapshotId("snapshot-1"),
            map_id=MapId("test-map"),
            version=MapVersion("v1"),
            schema_version="test.v1",
            content_digest="sha256:content",
            published_at=TimePoint(clock_id="unix", nanoseconds=0),
            map_frame=None,
            capabilities=frozenset({MapCapability.TOPOLOGY}),
        )
        nodes = tuple(
            TopologyNode(node_id=NodeId(f"node-{index}"))
            for index in range(3)
        )
        segments = (
            SegmentDescriptor(
                segment_id=SegmentId("edge-0"),
                source_node_id=NodeId("node-0"),
                target_node_id=NodeId("node-1"),
            ),
            SegmentDescriptor(
                segment_id=SegmentId("edge-1"),
                source_node_id=NodeId("node-1"),
                target_node_id=NodeId("node-2"),
            ),
        )
        self.engine = StaticMapEngine(
            StaticMap(
                snapshot=self.snapshot,
                nodes=nodes,
                segments=segments,
            )
        )

    async def test_resolves_matching_snapshot(self) -> None:
        result = await self.engine.get_snapshot(
            MapSelector(
                map_id=MapId("test-map"),
                required_capabilities=frozenset({MapCapability.TOPOLOGY}),
            )
        )

        self.assertEqual(result, self.snapshot)

    async def test_rejects_missing_capability(self) -> None:
        with self.assertRaises(MapEngineError) as context:
            await self.engine.get_snapshot(
                MapSelector(
                    map_id=MapId("test-map"),
                    required_capabilities=frozenset(
                        {MapCapability.VISUAL_ANCHORS}
                    ),
                )
            )

        self.assertEqual(
            context.exception.code,
            MapErrorCode.CAPABILITY_UNAVAILABLE,
        )

    async def test_expands_topology_around_a_node(self) -> None:
        result = await self.engine.query_topology(
            self.snapshot,
            TopologyQuery(
                node_ids=(NodeId("node-1"),),
                expand_hops=1,
            ),
        )

        self.assertEqual(
            tuple(node.node_id for node in result.nodes),
            (NodeId("node-0"), NodeId("node-1"), NodeId("node-2")),
        )
        self.assertEqual(
            tuple(segment.segment_id for segment in result.segments),
            (SegmentId("edge-0"), SegmentId("edge-1")),
        )

    def test_rejects_a_segment_with_a_missing_node(self) -> None:
        with self.assertRaisesRegex(ValueError, "references missing node"):
            StaticMapEngine(
                StaticMap(
                    snapshot=self.snapshot,
                    nodes=(TopologyNode(node_id=NodeId("node-0")),),
                    segments=(
                        SegmentDescriptor(
                            segment_id=SegmentId("edge-0"),
                            source_node_id=NodeId("node-0"),
                            target_node_id=NodeId("node-missing"),
                        ),
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()

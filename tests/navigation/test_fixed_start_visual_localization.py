"""Tests for fixed-start visual topological localization."""

from __future__ import annotations

from collections import deque
import unittest

from longship.navigation.common import TimePoint
from longship.navigation.localization_engine.fixed_start_visual import (
    FIXED_START_NODE_ID,
    FixedStartVisualLocalizationEngine,
    FixedStartVisualPhase,
    FixedStartVisualTrackingProfile,
)
from longship.navigation.localization_engine.interface import (
    LocalizationEngine,
    LocalizationEngineError,
)
from longship.navigation.localization_engine.models import (
    BeliefStreamId,
    BeliefUpdateOutcome,
    LocalizationStatus,
    NodeLocation,
    WaitForUpdateRequest,
)
from longship.navigation.localization_engine.visual_policy import (
    VisualGoalCandidateDistance,
    VisualGoalDistanceBatchMeasurement,
    VisualGoalDistanceBatchRequest,
    VisualPolicyError,
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


_MODEL_DIGEST = "sha256:" + "1" * 64
_IMAGE_PROFILE = "nomad.direct-resize.v1"


class _ScriptedVisualPolicy:
    def __init__(
        self,
        outcomes: list[dict[NodeId, float] | VisualPolicyError],
    ) -> None:
        self._outcomes = deque(outcomes)
        self.requests: list[VisualGoalDistanceBatchRequest] = []

    async def compare_goals(
        self,
        request: VisualGoalDistanceBatchRequest,
    ) -> VisualGoalDistanceBatchMeasurement:
        self.requests.append(request)
        outcome = self._outcomes.popleft()
        if isinstance(outcome, VisualPolicyError):
            raise outcome
        return VisualGoalDistanceBatchMeasurement(
            snapshot_id=request.snapshot_id,
            candidate_distances=tuple(
                VisualGoalCandidateDistance(
                    target_node_id=candidate.target_node_id,
                    target_anchor_id=candidate.target_anchor_id,
                    goal_resource_id=candidate.goal_resource.resource_id,
                    temporal_distance=outcome[candidate.target_node_id],
                )
                for candidate in request.candidates
            ),
            observation_time=request.requested_at,
            produced_at=request.requested_at,
            policy_id="fake-policy",
            image_profile_id=request.expected_image_profile_id,
            model_artifact_id=request.expected_model_artifact_id,
            model_artifact_digest=request.expected_model_artifact_digest,
        )


class FixedStartVisualLocalizationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.snapshot = MapSnapshot(
            snapshot_id=SnapshotId("visual-map:snapshot"),
            map_id=MapId("visual-map"),
            version=MapVersion("v1"),
            schema_version="test.visual.v1",
            content_digest="sha256:map",
            published_at=self._time(0),
            map_frame=None,
            capabilities=frozenset(
                {
                    MapCapability.TOPOLOGY,
                    MapCapability.VISUAL_ANCHORS,
                    MapCapability.RESOURCE_REFERENCES,
                }
            ),
        )
        self.map_engine = self._map_engine()
        self.profile = FixedStartVisualTrackingProfile(
            image_profile_id=_IMAGE_PROFILE,
        )

    @staticmethod
    def _time(seconds: int) -> TimePoint:
        return TimePoint(clock_id="camera", nanoseconds=seconds * 1_000_000_000)

    def _map_engine(
        self,
        *,
        image_profile: str = _IMAGE_PROFILE,
        node_offset: int = 0,
        node_count: int = 3,
    ) -> StaticMapEngine:
        nodes = []
        anchors = []
        resources = []
        for index in range(node_count):
            node_id = NodeId(f"node-{index + node_offset:04d}")
            anchor_id = AnchorId(f"{node_id}:visual")
            resource_id = ResourceId(f"{node_id}:goal-image")
            nodes.append(
                TopologyNode(
                    node_id=node_id,
                    anchor_ids=(anchor_id,),
                )
            )
            anchors.append(
                AnchorDescriptor(
                    anchor_id=anchor_id,
                    kind=AnchorKind.VISUAL,
                    purposes=frozenset(
                        {
                            AnchorPurpose.LOCALIZATION,
                            AnchorPurpose.TARGET,
                        }
                    ),
                    attached_to=MapEntityRef(
                        kind=MapEntityKind.NODE,
                        entity_id=str(node_id),
                    ),
                    resource_ids=(resource_id,),
                )
            )
            resources.append(
                ResourceDescriptor(
                    resource_id=resource_id,
                    kind=ResourceKind.IMAGE,
                    locator=f"opaque://goal/{index}",
                    content_digest=f"sha256:{index:064x}",
                    attributes={
                        "image_profile_id": image_profile,
                        "model_artifact_id": "nomad.pth",
                        "model_artifact_digest": _MODEL_DIGEST,
                    },
                )
            )
        segments = tuple(
            SegmentDescriptor(
                segment_id=SegmentId(f"edge-{index:04d}"),
                source_node_id=NodeId(
                    f"node-{index + node_offset:04d}"
                ),
                target_node_id=NodeId(
                    f"node-{index + node_offset + 1:04d}"
                ),
            )
            for index in range(node_count - 1)
        )
        return StaticMapEngine(
            StaticMap(
                snapshot=self.snapshot,
                nodes=tuple(nodes),
                segments=segments,
                anchors=tuple(anchors),
                resources=tuple(resources),
            )
        )

    async def _create(
        self,
        outcomes: list[dict[NodeId, float] | VisualPolicyError],
        *,
        map_engine: StaticMapEngine | None = None,
    ) -> tuple[
        FixedStartVisualLocalizationEngine,
        _ScriptedVisualPolicy,
    ]:
        policy = _ScriptedVisualPolicy(outcomes)
        engine = await FixedStartVisualLocalizationEngine.create(
            map_engine=map_engine or self.map_engine,
            snapshot=self.snapshot,
            policy=policy,
            profile=self.profile,
            stream_id=BeliefStreamId("stream-1"),
            started_at=self._time(0),
        )
        return engine, policy

    @staticmethod
    def _distances(*values: float, offset: int = 0) -> dict[NodeId, float]:
        return {
            NodeId(f"node-{index + offset:04d}"): value
            for index, value in enumerate(values)
        }

    async def test_verifies_start_and_advances_exactly_one_node(self) -> None:
        engine, policy = await self._create(
            [
                self._distances(2.0, 10.0, 15.0),
                self._distances(2.0, 10.0, 15.0),
                self._distances(8.0, 2.0, 10.0),
                self._distances(8.0, 2.0, offset=1),
            ]
        )
        self.assertIsInstance(engine, LocalizationEngine)

        first = await engine.tick(self._time(1))
        self.assertEqual(first.status, LocalizationStatus.INITIALIZING)
        started = await engine.tick(self._time(2))
        self.assertEqual(
            started.hypotheses[0].topological_location,
            NodeLocation(node_id=NodeId("node-0000")),
        )
        self.assertEqual(
            engine.get_tracking_state().phase,
            FixedStartVisualPhase.SEARCHING_NEXT,
        )

        middle = await engine.tick(self._time(3))
        self.assertEqual(
            middle.hypotheses[0].topological_location,
            NodeLocation(node_id=NodeId("node-0001")),
        )
        self.assertEqual(
            engine.get_tracking_state().phase,
            FixedStartVisualPhase.SEARCHING_NEXT,
        )

        final = await engine.tick(self._time(4))
        self.assertEqual(
            final.hypotheses[0].topological_location,
            NodeLocation(node_id=NodeId("node-0002")),
        )
        self.assertEqual(
            engine.get_tracking_state().phase,
            FixedStartVisualPhase.AT_FINAL_NODE,
        )
        self.assertEqual(
            tuple(
                tuple(
                    candidate.target_node_id
                    for candidate in request.candidates
                )
                for request in policy.requests
            ),
            (
                (
                    FIXED_START_NODE_ID,
                    NodeId("node-0001"),
                    NodeId("node-0002"),
                ),
                (
                    FIXED_START_NODE_ID,
                    NodeId("node-0001"),
                    NodeId("node-0002"),
                ),
                (
                    FIXED_START_NODE_ID,
                    NodeId("node-0001"),
                    NodeId("node-0002"),
                ),
                (NodeId("node-0001"), NodeId("node-0002")),
            ),
        )

        update = await engine.wait_for_update(
            WaitForUpdateRequest(after_revision=started.revision, timeout_s=0.0)
        )
        self.assertEqual(update.outcome, BeliefUpdateOutcome.UPDATED)
        self.assertEqual(update.belief, final)

    async def test_relative_window_advances_without_close_crossing(self) -> None:
        engine, _ = await self._create(
            [
                self._distances(2.0, 10.0, 15.0),
                self._distances(2.0, 10.0, 15.0),
                self._distances(10.0, 5.0, 8.0),
                self._distances(11.0, 4.5, 8.0),
            ]
        )
        await engine.tick(self._time(1))
        await engine.tick(self._time(2))

        first_vote = await engine.tick(self._time(3))
        self.assertEqual(
            first_vote.hypotheses[0].topological_location,
            NodeLocation(node_id=NodeId("node-0000")),
        )
        self.assertEqual(engine.get_tracking_state().relative_count, 1)

        advanced = await engine.tick(self._time(4))
        self.assertEqual(
            advanced.hypotheses[0].topological_location,
            NodeLocation(node_id=NodeId("node-0001")),
        )
        self.assertEqual(
            engine.get_tracking_state().phase,
            FixedStartVisualPhase.SEARCHING_NEXT,
        )

    async def test_later_close_candidate_triggers_relocalization(self) -> None:
        engine, _ = await self._create(
            [
                self._distances(2.0, 10.0, 15.0),
                self._distances(2.0, 10.0, 15.0),
                self._distances(10.0, 8.0, 2.0),
                self._distances(10.0, 8.0, 2.0),
                self._distances(10.0, 8.0, 2.0),
            ]
        )
        await engine.tick(self._time(1))
        await engine.tick(self._time(2))

        still_tracking = await engine.tick(self._time(3))
        self.assertEqual(still_tracking.status, LocalizationStatus.TRACKING)
        lost = await engine.tick(self._time(4))
        self.assertEqual(lost.status, LocalizationStatus.LOST)
        recovered = await engine.tick(self._time(5))

        self.assertEqual(
            recovered.hypotheses[0].topological_location,
            NodeLocation(node_id=NodeId("node-0002")),
        )
        self.assertEqual(
            engine.get_status().detail_code,
            "final_node_relocalized",
        )

    async def test_relocalizes_in_expanded_monotonic_window(self) -> None:
        map_engine = self._map_engine(node_count=6)
        engine, policy = await self._create(
            [
                self._distances(2.0, 10.0, 15.0),
                self._distances(2.0, 10.0, 15.0),
                self._distances(19.0, 19.0, 19.0),
                self._distances(19.0, 19.0, 19.0),
                self._distances(19.0, 19.0, 19.0),
                self._distances(19.0, 19.0, 19.0, 19.0, 2.0, 8.0),
                self._distances(19.0, 19.0, 19.0, 19.0, 2.0, 8.0),
            ],
            map_engine=map_engine,
        )
        await engine.tick(self._time(1))
        await engine.tick(self._time(2))
        await engine.tick(self._time(3))
        await engine.tick(self._time(4))
        lost = await engine.tick(self._time(5))

        self.assertEqual(lost.status, LocalizationStatus.LOST)
        self.assertFalse(lost.hypotheses)
        self.assertEqual(
            engine.get_tracking_state().phase,
            FixedStartVisualPhase.LOCALIZATION_LOST,
        )

        still_lost = await engine.tick(self._time(6))
        self.assertEqual(still_lost.status, LocalizationStatus.LOST)
        recovered = await engine.tick(self._time(7))

        self.assertEqual(
            recovered.hypotheses[0].topological_location,
            NodeLocation(node_id=NodeId("node-0004")),
        )
        self.assertEqual(recovered.status, LocalizationStatus.TRACKING)
        self.assertEqual(engine.get_status().detail_code, "visual_relocalized")
        self.assertEqual(
            len(policy.requests[-1].candidates),
            6,
        )

    async def test_rejects_incompatible_goal_image_profile(self) -> None:
        incompatible_map = self._map_engine(image_profile="other-profile")

        with self.assertRaisesRegex(
            LocalizationEngineError,
            "incompatible image profile",
        ):
            await self._create([], map_engine=incompatible_map)

    async def test_rejects_a_chain_without_node_zero(self) -> None:
        map_without_node_zero = self._map_engine(node_offset=1)

        with self.assertRaisesRegex(
            LocalizationEngineError,
            "fixed start node is missing: node-0000",
        ):
            await self._create([], map_engine=map_without_node_zero)


if __name__ == "__main__":
    unittest.main()

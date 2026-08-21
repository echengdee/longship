"""End-to-end test for producer, continuous tick, and belief consumer."""

from __future__ import annotations

import asyncio
import unittest

from longship.navigation.common import TimePoint
from longship.navigation.localization_engine.fixed_start_visual import (
    FixedStartVisualLocalizationEngine,
    FixedStartVisualPhase,
    FixedStartVisualTrackingProfile,
)
from longship.navigation.localization_engine.models import (
    BeliefStreamId,
    BeliefUpdateOutcome,
    NodeLocation,
    WaitForUpdateRequest,
)
from longship.navigation.localization_engine.service import (
    ContinuousLocalizationService,
    LocalizationServiceConfig,
    MonotonicTimeSource,
)
from longship.navigation.localization_engine.visual_policy import (
    VisualGoalCandidateDistance,
    VisualGoalDistanceBatchMeasurement,
    VisualGoalDistanceBatchRequest,
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
from longship.navigation.runtime import (
    LocalizationObservationProducerState,
    LocalizationObservationProducerStatus,
    LocalizationRuntime,
    LocalizationRuntimeState,
)


_CLOCK_ID = "camera"
_IMAGE_PROFILE_ID = "nomad.direct-resize.v1"
_MODEL_ARTIFACT_ID = "nomad.pth"
_MODEL_DIGEST = "sha256:" + "1" * 64


class _DistanceObservationProducer:
    def __init__(
        self,
        queue: asyncio.Queue[dict[NodeId, float]],
        distances: tuple[dict[NodeId, float], ...],
    ) -> None:
        self._queue = queue
        self._distances = distances
        self.started = False
        self.stopped = False
        self._state = LocalizationObservationProducerState.CREATED
        self._terminated = asyncio.Event()

    def get_status(self) -> LocalizationObservationProducerStatus:
        return LocalizationObservationProducerStatus(
            state=self._state,
            detail_code=self._state.value,
            last_error=None,
        )

    async def start(self) -> None:
        self.started = True
        self._state = LocalizationObservationProducerState.RUNNING
        for distance in self._distances:
            self._queue.put_nowait(distance)

    async def stop(self) -> None:
        self.stopped = True
        self._state = LocalizationObservationProducerState.STOPPED
        self._terminated.set()

    async def wait_stopped(
        self,
        timeout_s: float | None = None,
    ) -> LocalizationObservationProducerStatus:
        if timeout_s is None:
            await self._terminated.wait()
        else:
            await asyncio.wait_for(self._terminated.wait(), timeout_s)
        return self.get_status()


class _ObservationDrivenVisualPolicy:
    def __init__(
        self,
        queue: asyncio.Queue[dict[NodeId, float]],
    ) -> None:
        self._queue = queue

    async def compare_goals(
        self,
        request: VisualGoalDistanceBatchRequest,
    ) -> VisualGoalDistanceBatchMeasurement:
        distances = await self._queue.get()
        return VisualGoalDistanceBatchMeasurement(
            snapshot_id=request.snapshot_id,
            candidate_distances=tuple(
                VisualGoalCandidateDistance(
                    target_node_id=candidate.target_node_id,
                    target_anchor_id=candidate.target_anchor_id,
                    goal_resource_id=candidate.goal_resource.resource_id,
                    temporal_distance=distances[candidate.target_node_id],
                )
                for candidate in request.candidates
            ),
            observation_time=request.requested_at,
            produced_at=request.requested_at,
            policy_id="observation-driven-test-policy",
            image_profile_id=request.expected_image_profile_id,
            model_artifact_id=request.expected_model_artifact_id,
            model_artifact_digest=request.expected_model_artifact_digest,
        )


class LocalizationRuntimeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_drives_fixed_start_beliefs_to_the_final_node(self) -> None:
        map_engine, snapshot = _create_map()
        distances: asyncio.Queue[dict[NodeId, float]] = asyncio.Queue()
        producer = _DistanceObservationProducer(
            distances,
            (
                _distances(2.0, 10.0, 15.0),
                _distances(2.0, 10.0, 15.0),
                _distances(8.0, 2.0, 10.0),
                _distances(8.0, 2.0, offset=1),
            ),
        )
        clock = MonotonicTimeSource(clock_id=_CLOCK_ID)
        engine = await FixedStartVisualLocalizationEngine.create(
            map_engine=map_engine,
            snapshot=snapshot,
            policy=_ObservationDrivenVisualPolicy(distances),
            profile=FixedStartVisualTrackingProfile(
                image_profile_id=_IMAGE_PROFILE_ID,
            ),
            stream_id=BeliefStreamId("runtime-integration"),
            started_at=clock.now(),
        )
        service = ContinuousLocalizationService(
            engine=engine,
            time_source=clock,
            config=LocalizationServiceConfig(
                tick_period_s=0.005,
                stop_timeout_s=0.5,
            ),
        )
        runtime = LocalizationRuntime(
            observation_producer=producer,
            localization_service=service,
        )

        confirmed_nodes = []
        belief = engine.get_belief()
        await runtime.start()
        try:
            for _ in range(4):
                update = await engine.wait_for_update(
                    WaitForUpdateRequest(
                        after_revision=belief.revision,
                        timeout_s=0.5,
                    )
                )
                self.assertEqual(
                    update.outcome,
                    BeliefUpdateOutcome.UPDATED,
                )
                belief = update.belief
                if belief.hypotheses:
                    location = belief.hypotheses[0].topological_location
                    self.assertIsInstance(location, NodeLocation)
                    node_id = location.node_id
                    if not confirmed_nodes or confirmed_nodes[-1] != node_id:
                        confirmed_nodes.append(node_id)
        finally:
            await runtime.stop()

        self.assertEqual(
            confirmed_nodes,
            [
                NodeId("node-0000"),
                NodeId("node-0001"),
                NodeId("node-0002"),
            ],
        )
        self.assertEqual(
            engine.get_tracking_state().phase,
            FixedStartVisualPhase.AT_FINAL_NODE,
        )
        self.assertTrue(producer.started)
        self.assertTrue(producer.stopped)
        self.assertEqual(
            runtime.get_status().state,
            LocalizationRuntimeState.STOPPED,
        )
        self.assertGreaterEqual(service.get_status().ticks_completed, 4)


def _distances(*values: float, offset: int = 0) -> dict[NodeId, float]:
    return {
        NodeId(f"node-{index + offset:04d}"): value
        for index, value in enumerate(values)
    }


def _create_map() -> tuple[StaticMapEngine, MapSnapshot]:
    snapshot = MapSnapshot(
        snapshot_id=SnapshotId("runtime-map:snapshot"),
        map_id=MapId("runtime-map"),
        version=MapVersion("v1"),
        schema_version="test.visual.v1",
        content_digest="sha256:runtime-map",
        published_at=TimePoint(clock_id="unix", nanoseconds=0),
        map_frame=None,
        capabilities=frozenset(
            {
                MapCapability.TOPOLOGY,
                MapCapability.VISUAL_ANCHORS,
                MapCapability.RESOURCE_REFERENCES,
            }
        ),
    )
    nodes = []
    anchors = []
    resources = []
    for index in range(3):
        node_id = NodeId(f"node-{index:04d}")
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
                    "image_profile_id": _IMAGE_PROFILE_ID,
                    "model_artifact_id": _MODEL_ARTIFACT_ID,
                    "model_artifact_digest": _MODEL_DIGEST,
                },
            )
        )
    segments = tuple(
        SegmentDescriptor(
            segment_id=SegmentId(f"edge-{index:04d}"),
            source_node_id=NodeId(f"node-{index:04d}"),
            target_node_id=NodeId(f"node-{index + 1:04d}"),
        )
        for index in range(2)
    )
    return (
        StaticMapEngine(
            StaticMap(
                snapshot=snapshot,
                nodes=tuple(nodes),
                segments=segments,
                anchors=tuple(anchors),
                resources=tuple(resources),
            )
        ),
        snapshot,
    )


if __name__ == "__main__":
    unittest.main()

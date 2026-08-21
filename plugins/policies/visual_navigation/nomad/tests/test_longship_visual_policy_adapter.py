"""Tests for the NoMaD-to-Localization policy adapter."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from longship_adapter import (
    DecodedImage,
    LocalFileGoalImageLoader,
    NomadVisualGoalDistancePolicy,
    NomadVisualPolicyConfig,
)
from longship.navigation.common import TimePoint
from longship.navigation.localization_engine.visual_policy import (
    VisualGoalCandidate,
    VisualGoalDistanceBatchPolicy,
    VisualGoalDistanceBatchRequest,
    VisualGoalDistancePolicy,
    VisualGoalDistanceRequest,
    VisualPolicyError,
    VisualPolicyErrorCode,
)
from longship.navigation.map_engine.models import (
    AnchorId,
    NodeId,
    ResourceDescriptor,
    ResourceId,
    ResourceKind,
    SnapshotId,
)


_MODEL_DIGEST = "sha256:" + "2" * 64


class _TimeSource:
    def now(self) -> TimePoint:
        return TimePoint(clock_id="camera", nanoseconds=10_200_000_000)


class _FakeGoalLoader:
    def __init__(self) -> None:
        self.loaded: list[ResourceId] = []

    def load(self, resource: ResourceDescriptor) -> DecodedImage:
        self.loaded.append(resource.resource_id)
        return DecodedImage(
            image="decoded-goal",
            layout="hwc",
            channel_order="rgb",
            value_range="byte",
        )


@dataclass
class _SessionFailure(RuntimeError):
    code: str
    retryable: bool

    def __str__(self) -> str:
        return self.code


class _FakeDistanceSession:
    def __init__(self) -> None:
        self.failure: Exception | None = None
        self.observations = []
        self.predictions = []

    def append_observation(
        self,
        image: object,
        timestamp_s: float,
        **representation: str,
    ) -> None:
        self.observations.append((image, timestamp_s, representation))

    def clear_observations(self) -> None:
        self.observations.clear()

    def predict_goal_distance(
        self,
        goal_image: object,
        **arguments: object,
    ) -> object:
        if self.failure is not None:
            raise self.failure
        self.predictions.append((goal_image, arguments))
        return SimpleNamespace(
            temporal_distance=2.25,
            observation_timestamp_s=10.0,
        )

    def predict_goal_distances(
        self,
        goal_images: tuple[object, ...],
        **arguments: object,
    ) -> object:
        if self.failure is not None:
            raise self.failure
        self.predictions.append((goal_images, arguments))
        return SimpleNamespace(
            temporal_distances=tuple(
                2.25 + index for index in range(len(goal_images))
            ),
            observation_timestamp_s=10.0,
        )


class NomadVisualPolicyAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.session = _FakeDistanceSession()
        self.loader = _FakeGoalLoader()
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="nomad-test",
        )
        self.addCleanup(self.executor.shutdown)
        self.policy = NomadVisualGoalDistancePolicy(
            session=self.session,
            goal_image_loader=self.loader,
            inference_executor=self.executor,
            config=NomadVisualPolicyConfig(
                policy_id="nomad-distance-v1",
                image_profile_id="nomad-profile-v1",
                model_artifact_id="nomad.pth",
                model_artifact_digest=_MODEL_DIGEST,
                observation_clock_id="camera",
                time_source=_TimeSource(),
            ),
        )
        self.resource = ResourceDescriptor(
            resource_id=ResourceId("node-0000:goal-image"),
            kind=ResourceKind.IMAGE,
            locator="opaque://goal/0",
            content_digest="sha256:" + "3" * 64,
            attributes={
                "image_profile_id": "nomad-profile-v1",
                "model_artifact_id": "nomad.pth",
                "model_artifact_digest": _MODEL_DIGEST,
            },
        )

    def _request(self, **overrides: object) -> VisualGoalDistanceRequest:
        values = {
            "snapshot_id": SnapshotId("snapshot-1"),
            "target_node_id": NodeId("node-0000"),
            "target_anchor_id": AnchorId("node-0000:visual"),
            "goal_resource": self.resource,
            "requested_at": TimePoint(
                clock_id="camera",
                nanoseconds=10_100_000_000,
            ),
            "max_observation_age_s": 0.2,
            "expected_image_profile_id": "nomad-profile-v1",
            "expected_model_artifact_id": "nomad.pth",
            "expected_model_artifact_digest": _MODEL_DIGEST,
        }
        values.update(overrides)
        return VisualGoalDistanceRequest(**values)

    async def test_maps_goal_resource_to_distance_measurement(self) -> None:
        self.assertIsInstance(self.policy, VisualGoalDistancePolicy)
        self.assertIsInstance(self.policy, VisualGoalDistanceBatchPolicy)
        self.policy.submit_observation(
            "decoded-camera-frame",
            10.0,
            layout="hwc",
            channel_order="bgr",
            value_range="byte",
        )

        measurement = await self.policy.compare_goal(self._request())

        self.assertEqual(measurement.temporal_distance, 2.25)
        self.assertEqual(
            measurement.goal_resource_id,
            self.resource.resource_id,
        )
        self.assertEqual(measurement.observation_time.nanoseconds, 10_000_000_000)
        self.assertEqual(measurement.produced_at.nanoseconds, 10_200_000_000)
        self.assertEqual(self.loader.loaded, [self.resource.resource_id])
        self.assertEqual(
            self.session.predictions[0][0],
            ("decoded-goal",),
        )
        self.assertEqual(len(self.session.observations), 1)

    async def test_batches_local_goal_candidates_on_one_context(self) -> None:
        second_resource = ResourceDescriptor(
            resource_id=ResourceId("node-0001:goal-image"),
            kind=ResourceKind.IMAGE,
            locator="opaque://goal/1",
            content_digest="sha256:" + "4" * 64,
            attributes=self.resource.attributes,
        )
        request = VisualGoalDistanceBatchRequest(
            snapshot_id=SnapshotId("snapshot-1"),
            candidates=(
                VisualGoalCandidate(
                    target_node_id=NodeId("node-0000"),
                    target_anchor_id=AnchorId("node-0000:visual"),
                    goal_resource=self.resource,
                ),
                VisualGoalCandidate(
                    target_node_id=NodeId("node-0001"),
                    target_anchor_id=AnchorId("node-0001:visual"),
                    goal_resource=second_resource,
                ),
            ),
            requested_at=TimePoint(
                clock_id="camera",
                nanoseconds=10_100_000_000,
            ),
            max_observation_age_s=0.2,
            expected_image_profile_id="nomad-profile-v1",
            expected_model_artifact_id="nomad.pth",
            expected_model_artifact_digest=_MODEL_DIGEST,
        )

        measurement = await self.policy.compare_goals(request)

        self.assertEqual(
            tuple(
                candidate.temporal_distance
                for candidate in measurement.candidate_distances
            ),
            (2.25, 3.25),
        )
        self.assertEqual(
            self.loader.loaded,
            [self.resource.resource_id, second_resource.resource_id],
        )
        self.assertEqual(
            self.session.predictions[0][0],
            ("decoded-goal", "decoded-goal"),
        )

    async def test_rejects_map_and_policy_profile_mismatch(self) -> None:
        with self.assertRaises(VisualPolicyError) as context:
            await self.policy.compare_goal(
                self._request(expected_image_profile_id="other-profile")
            )

        self.assertEqual(
            context.exception.code,
            VisualPolicyErrorCode.PROFILE_MISMATCH,
        )
        self.assertFalse(self.loader.loaded)

    async def test_translates_context_not_ready_as_retryable(self) -> None:
        self.session.failure = _SessionFailure(
            code="context_not_ready",
            retryable=True,
        )

        with self.assertRaises(VisualPolicyError) as context:
            await self.policy.compare_goal(self._request())

        self.assertEqual(
            context.exception.code,
            VisualPolicyErrorCode.CONTEXT_NOT_READY,
        )
        self.assertTrue(context.exception.retryable)

    @unittest.skipUnless(
        importlib.util.find_spec("PIL") and importlib.util.find_spec("torch"),
        "local image adapter requires Pillow and PyTorch",
    )
    def test_local_loader_verifies_and_decodes_a_pinned_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "goal.png"
            image_bytes = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lE"
                "QVQYV2NgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
            )
            image_path.write_bytes(image_bytes)
            digest = "sha256:" + hashlib.sha256(image_bytes).hexdigest()
            resource = ResourceDescriptor(
                resource_id=ResourceId("goal-image"),
                kind=ResourceKind.IMAGE,
                locator=str(image_path),
                content_digest=digest,
                size_bytes=len(image_bytes),
            )

            decoded = LocalFileGoalImageLoader((root,)).load(resource)

            self.assertEqual(tuple(decoded.image.shape), (1, 1, 3))
            self.assertEqual(decoded.layout, "hwc")


if __name__ == "__main__":
    unittest.main()

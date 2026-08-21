"""Tests for the NoMaD route-step trajectory adapter."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest

import torch

from longship.navigation.common import TimePoint
from longship.navigation.map_engine.models import (
    AnchorId,
    NodeId,
    ResourceDescriptor,
    ResourceId,
    ResourceKind,
    SegmentId,
    SnapshotId,
)
from longship.navigation.ports.trajectory_policy import (
    TrajectoryPolicyError,
    TrajectoryPolicyErrorCode,
    VisualGoalTrajectoryPolicy,
    VisualGoalTrajectoryRequest,
)
from longship_adapter import (
    DecodedImage,
    NomadTrajectoryPolicyConfig,
    NomadVisualGoalTrajectoryPolicy,
)


_MODEL_DIGEST = "sha256:" + "2" * 64


class _TimeSource:
    def now(self) -> TimePoint:
        return TimePoint(clock_id="camera", nanoseconds=10_200_000_000)


class _FakeGoalLoader:
    def load(self, resource: ResourceDescriptor) -> DecodedImage:
        return DecodedImage(
            image="decoded-goal",
            layout="hwc",
            channel_order="rgb",
            value_range="byte",
        )


class _FakeTrajectorySession:
    def predict_goal_trajectories(
        self,
        goal_image: object,
        **arguments: object,
    ) -> object:
        assert goal_image == "decoded-goal"
        assert arguments["num_candidates"] == 2
        return SimpleNamespace(
            temporal_distance=4.5,
            trajectories=torch.tensor(
                (
                    ((1.0, 0.1), (2.0, 0.2)),
                    ((1.1, -0.1), (2.1, -0.2)),
                )
            ),
            observation_timestamp_s=10.0,
            sampling_seed=7,
        )


class NomadTrajectoryPolicyAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(self.executor.shutdown)
        self.policy = NomadVisualGoalTrajectoryPolicy(
            session=_FakeTrajectorySession(),
            goal_image_loader=_FakeGoalLoader(),
            inference_executor=self.executor,
            config=NomadTrajectoryPolicyConfig(
                policy_id="nomad-trajectory-v1",
                image_profile_id="nomad-profile-v1",
                model_artifact_id="nomad.pth",
                model_artifact_digest=_MODEL_DIGEST,
                observation_clock_id="camera",
                time_source=_TimeSource(),
            ),
        )
        self.resource = ResourceDescriptor(
            resource_id=ResourceId("node-0001:goal-image"),
            kind=ResourceKind.IMAGE,
            locator="opaque://goal/1",
            content_digest="sha256:" + "3" * 64,
            attributes={
                "image_profile_id": "nomad-profile-v1",
                "model_artifact_id": "nomad.pth",
                "model_artifact_digest": _MODEL_DIGEST,
            },
        )

    def _request(self, **overrides: object) -> VisualGoalTrajectoryRequest:
        values = {
            "snapshot_id": SnapshotId("snapshot-1"),
            "segment_id": SegmentId("edge-0000"),
            "source_node_id": NodeId("node-0000"),
            "target_node_id": NodeId("node-0001"),
            "target_anchor_id": AnchorId("node-0001:visual"),
            "goal_resource": self.resource,
            "requested_at": TimePoint(
                clock_id="camera",
                nanoseconds=10_100_000_000,
            ),
            "max_observation_age_s": 0.2,
            "num_candidates": 2,
            "sampling_seed": 7,
            "expected_image_profile_id": "nomad-profile-v1",
            "expected_model_artifact_id": "nomad.pth",
            "expected_model_artifact_digest": _MODEL_DIGEST,
        }
        values.update(overrides)
        return VisualGoalTrajectoryRequest(**values)

    async def test_preserves_every_raw_candidate_and_route_identity(self) -> None:
        self.assertIsInstance(self.policy, VisualGoalTrajectoryPolicy)

        result = await self.policy.generate_trajectories(self._request())

        self.assertEqual(result.segment_id, SegmentId("edge-0000"))
        self.assertEqual(result.source_node_id, NodeId("node-0000"))
        self.assertEqual(result.target_node_id, NodeId("node-0001"))
        self.assertEqual(result.temporal_distance, 4.5)
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.candidates[0].waypoints[1].x, 2.0)
        self.assertAlmostEqual(result.candidates[1].waypoints[1].y, -0.2)
        self.assertEqual(result.coordinate_units, "nomad.policy_native.v1")
        self.assertEqual(result.produced_at.nanoseconds, 10_200_000_000)

    async def test_rejects_profile_mismatch(self) -> None:
        with self.assertRaises(TrajectoryPolicyError) as captured:
            await self.policy.generate_trajectories(
                self._request(expected_image_profile_id="other-profile")
            )

        self.assertEqual(
            captured.exception.code,
            TrajectoryPolicyErrorCode.PROFILE_MISMATCH,
        )


if __name__ == "__main__":
    unittest.main()

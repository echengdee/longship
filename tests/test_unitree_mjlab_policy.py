from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from longship.artifacts import VerifiedArtifact

from longship.policies import (
    G1_29DOF_JOINT_NAMES,
    GuardedPolicyProvider,
    OnnxRuntimeVectorRunner,
    PolicyCandidateRejected,
    PolicyError,
    PolicyRequest,
    UNITREE_G1_29DOF_ACTION_DIM,
    UNITREE_G1_29DOF_OBSERVATION_DIM,
    UNITREE_G1_POLICY_ARTIFACT_ID,
    UNITREE_G1_POLICY_SHA256,
    UNITREE_G1_POLICY_SIZE_BYTES,
    UnitreeG1VelocityBackend,
    UnitreeG1VelocityCommand,
    UnitreeG1VelocityObservation,
    unitree_g1_gait_phase,
    unitree_g1_velocity_guard_profile,
)


class FakeRunner:
    def __init__(self, output_length: int = UNITREE_G1_29DOF_ACTION_DIM) -> None:
        self.output_length = output_length
        self.observations: list[tuple[float, ...]] = []

    def infer(self, observation: tuple[float, ...]) -> tuple[float, ...]:
        self.observations.append(observation)
        return (0.0,) * self.output_length


def observation() -> UnitreeG1VelocityObservation:
    joints = (0.0,) * UNITREE_G1_29DOF_ACTION_DIM
    return UnitreeG1VelocityObservation(
        joint_names=G1_29DOF_JOINT_NAMES,
        base_angular_velocity=(0.0, 0.0, 0.0),
        projected_gravity=(0.0, 0.0, -1.0),
        command=UnitreeG1VelocityCommand(0.2, 0.0, 0.1),
        gait_phase=(0.0, 1.0),
        joint_position_relative=joints,
        joint_velocity_relative=joints,
        last_action=joints,
    )


def request(value: UnitreeG1VelocityObservation) -> PolicyRequest:
    return PolicyRequest(
        call_id="unitree-call",
        model_binding_id="unitree-binding",
        lease_id="whole-body-lease",
        lease_epoch=2,
        observation_version=3,
        deadline_monotonic=20.1,
        max_action_horizon_ms=20,
        resource_scope=("whole_body_motion",),
        payload={"observation": value},
    )


class UnitreeObservationTests(unittest.TestCase):
    def test_builds_official_98_value_order(self) -> None:
        value = observation().vector()
        self.assertEqual(len(value), UNITREE_G1_29DOF_OBSERVATION_DIM)
        self.assertEqual(value[6:9], (0.2, 0.0, 0.1))

    def test_velocity_profile_and_joint_dimensions_are_bounded(self) -> None:
        with self.assertRaises(ValueError):
            UnitreeG1VelocityCommand(1.1, 0.0, 0.0)
        with self.assertRaises(ValueError):
            UnitreeG1VelocityObservation(
                joint_names=G1_29DOF_JOINT_NAMES,
                base_angular_velocity=(0.0, 0.0, 0.0),
                projected_gravity=(0.0, 0.0, -1.0),
                command=UnitreeG1VelocityCommand(0.0, 0.0, 0.0),
                gait_phase=(0.0, 1.0),
                joint_position_relative=(0.0,) * 28,
                joint_velocity_relative=(0.0,) * 29,
                last_action=(0.0,) * 29,
            )
        with self.assertRaisesRegex(ValueError, "joint_names"):
            replace(observation(), joint_names=tuple(reversed(G1_29DOF_JOINT_NAMES)))

    def test_gait_phase_is_zero_at_rest_and_periodic_while_moving(self) -> None:
        rest = UnitreeG1VelocityCommand(0.0, 0.0, 0.0)
        moving = UnitreeG1VelocityCommand(0.2, 0.0, 0.0)
        self.assertEqual(unitree_g1_gait_phase(0.2, rest), (0.0, 0.0))
        phase = unitree_g1_gait_phase(0.15, moving)
        self.assertAlmostEqual(phase[0], 1.0)
        self.assertAlmostEqual(phase[1], 0.0)


class UnitreePolicyBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_candidate_without_target_side_effects(self) -> None:
        runner = FakeRunner()
        backend = UnitreeG1VelocityBackend(runner, clock=lambda: 20.0)
        provider = GuardedPolicyProvider(
            backend,
            unitree_g1_velocity_guard_profile(
                minimum_action_value=-1.0,
                maximum_action_value=1.0,
            ),
            lease_is_current=lambda value: True,
            clock=lambda: 20.0,
        )
        result = await provider.infer(request(observation()))
        self.assertEqual(len(result.frames[0].values), UNITREE_G1_29DOF_ACTION_DIM)
        self.assertEqual(result.resource_scope, ("whole_body_motion",))
        self.assertEqual(len(runner.observations[0]), 98)

    async def test_invalid_model_output_fails_closed(self) -> None:
        backend = UnitreeG1VelocityBackend(FakeRunner(28), clock=lambda: 20.0)
        with self.assertRaises(PolicyError):
            await backend.infer(request(observation()))

    async def test_target_qualified_bounds_fail_closed(self) -> None:
        runner = FakeRunner()
        runner.infer = lambda value: (1.1,) + (0.0,) * 28  # type: ignore[method-assign]
        provider = GuardedPolicyProvider(
            UnitreeG1VelocityBackend(runner, clock=lambda: 20.0),
            unitree_g1_velocity_guard_profile(
                minimum_action_value=-1.0,
                maximum_action_value=1.0,
            ),
            lease_is_current=lambda value: True,
            clock=lambda: 20.0,
        )
        with self.assertRaisesRegex(PolicyCandidateRejected, "bound"):
            await provider.infer(request(observation()))

    async def test_requires_whole_body_motion_lease(self) -> None:
        backend = UnitreeG1VelocityBackend(FakeRunner(), clock=lambda: 20.0)
        invalid = PolicyRequest(
            call_id="unitree-call",
            model_binding_id="unitree-binding",
            lease_id="legs-only",
            lease_epoch=2,
            observation_version=3,
            deadline_monotonic=20.1,
            max_action_horizon_ms=20,
            resource_scope=("legs",),
            payload={"observation": observation()},
        )
        with self.assertRaisesRegex(PolicyError, "whole_body_motion"):
            await backend.infer(invalid)


class UnitreeOnnxLoaderTests(unittest.TestCase):
    def test_rejects_stale_or_forged_verified_path_before_onnx_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.onnx"
            path.write_bytes(b"not the pinned policy")
            file_stat = path.stat()
            forged = VerifiedArtifact(
                artifact_id=UNITREE_G1_POLICY_ARTIFACT_ID,
                path=path,
                sha256=UNITREE_G1_POLICY_SHA256,
                size_bytes=UNITREE_G1_POLICY_SIZE_BYTES,
                device=file_stat.st_dev,
                inode=file_stat.st_ino,
            )
            with self.assertRaisesRegex(PolicyError, "identity changed"):
                OnnxRuntimeVectorRunner(forged)


if __name__ == "__main__":
    unittest.main()

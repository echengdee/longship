from __future__ import annotations

import asyncio
import threading
import unittest
from dataclasses import replace

from longship.policies import (
    G1_29DOF_JOINT_NAMES,
    GuardedPolicyProvider,
    HOLOSOMA_G1_ACTION_DIM,
    HOLOSOMA_G1_ACTION_SCALE,
    HOLOSOMA_G1_OBSERVATION_DIM,
    HolosomaG1LocomotionBackend,
    HolosomaG1LocomotionObservation,
    PolicyError,
    PolicyRequest,
    holosoma_g1_guard_profile,
)


class FakeRunner:
    def __init__(self, output_length: int = HOLOSOMA_G1_ACTION_DIM) -> None:
        self.output_length = output_length
        self.observations: list[tuple[float, ...]] = []

    def infer(self, observation: tuple[float, ...]) -> tuple[float, ...]:
        self.observations.append(observation)
        return (0.0,) * self.output_length


def observation() -> HolosomaG1LocomotionObservation:
    joints = (0.0,) * HOLOSOMA_G1_ACTION_DIM
    return HolosomaG1LocomotionObservation(
        joint_names=G1_29DOF_JOINT_NAMES,
        last_action=joints,
        base_angular_velocity=(0.0, 0.0, 0.0),
        command_angular_velocity=(0.1,),
        command_linear_velocity=(0.2, -0.1),
        cosine_phase=(1.0, 1.0),
        joint_position_relative=joints,
        joint_velocity=joints,
        projected_gravity=(0.0, 0.0, -1.0),
        sine_phase=(0.0, 0.0),
    )


def request(value: HolosomaG1LocomotionObservation) -> PolicyRequest:
    return PolicyRequest(
        call_id="holosoma-call",
        model_binding_id="holosoma-binding",
        lease_id="whole-body-lease",
        lease_epoch=2,
        observation_version=4,
        deadline_monotonic=30.1,
        max_action_horizon_ms=20,
        resource_scope=("whole_body_motion",),
        payload={"observation": value},
    )


class HolosomaObservationTests(unittest.TestCase):
    def test_builds_official_100_value_order(self) -> None:
        value = observation().vector()
        self.assertEqual(len(value), HOLOSOMA_G1_OBSERVATION_DIM)
        self.assertEqual(value[29:32], (0.0, 0.0, 0.0))
        self.assertEqual(value[32:35], (0.1, 0.2, -0.1))
        self.assertEqual(value[35:37], (1.0, 1.0))
        self.assertEqual(value[95:100], (0.0, 0.0, -1.0, 0.0, 0.0))
        self.assertEqual(HOLOSOMA_G1_ACTION_SCALE, 0.25)

    def test_applies_exported_velocity_scales_before_inference(self) -> None:
        joints = (0.0,) * HOLOSOMA_G1_ACTION_DIM
        value = HolosomaG1LocomotionObservation(
            joint_names=G1_29DOF_JOINT_NAMES,
            last_action=joints,
            base_angular_velocity=(4.0, 8.0, 12.0),
            command_angular_velocity=(0.0,),
            command_linear_velocity=(0.0, 0.0),
            cosine_phase=(1.0, 1.0),
            joint_position_relative=joints,
            joint_velocity=(2.0,) * HOLOSOMA_G1_ACTION_DIM,
            projected_gravity=(0.0, 0.0, -1.0),
            sine_phase=(0.0, 0.0),
        ).vector()
        self.assertEqual(value[29:32], (1.0, 2.0, 3.0))
        self.assertEqual(value[66:95], (0.1,) * HOLOSOMA_G1_ACTION_DIM)

    def test_rejects_invalid_phase_and_dimensions(self) -> None:
        joints = (0.0,) * HOLOSOMA_G1_ACTION_DIM
        with self.assertRaisesRegex(ValueError, "unit circle"):
            HolosomaG1LocomotionObservation(
                joint_names=G1_29DOF_JOINT_NAMES,
                last_action=joints,
                base_angular_velocity=(0.0, 0.0, 0.0),
                command_angular_velocity=(0.0,),
                command_linear_velocity=(0.0, 0.0),
                cosine_phase=(0.0, 0.0),
                joint_position_relative=joints,
                joint_velocity=joints,
                projected_gravity=(0.0, 0.0, -1.0),
                sine_phase=(0.0, 0.0),
            )
        with self.assertRaisesRegex(ValueError, "joint_names"):
            replace(observation(), joint_names=tuple(reversed(G1_29DOF_JOINT_NAMES)))


class HolosomaBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_guarded_candidate_without_target_side_effects(self) -> None:
        runner = FakeRunner()
        provider = GuardedPolicyProvider(
            HolosomaG1LocomotionBackend(runner, clock=lambda: 30.0),
            holosoma_g1_guard_profile(
                minimum_action_value=-1.0,
                maximum_action_value=1.0,
            ),
            lease_is_current=lambda value: True,
            clock=lambda: 30.0,
        )
        result = await provider.infer(request(observation()))
        self.assertEqual(len(result.frames[0].values), HOLOSOMA_G1_ACTION_DIM)
        self.assertEqual(result.resource_scope, ("whole_body_motion",))
        self.assertEqual(len(runner.observations[0]), HOLOSOMA_G1_OBSERVATION_DIM)

    async def test_bad_output_and_missing_whole_body_lease_fail_closed(self) -> None:
        backend = HolosomaG1LocomotionBackend(FakeRunner(28), clock=lambda: 30.0)
        with self.assertRaises(PolicyError):
            await backend.infer(request(observation()))

        invalid = PolicyRequest(
            call_id="holosoma-call",
            model_binding_id="holosoma-binding",
            lease_id="legs-only",
            lease_epoch=2,
            observation_version=4,
            deadline_monotonic=30.1,
            max_action_horizon_ms=20,
            resource_scope=("legs",),
            payload={"observation": observation()},
        )
        with self.assertRaisesRegex(PolicyError, "whole_body_motion"):
            await HolosomaG1LocomotionBackend(FakeRunner()).infer(invalid)

    async def test_cancelled_native_inference_remains_single_flight(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class BlockingRunner:
            def infer(self, value: tuple[float, ...]) -> tuple[float, ...]:
                started.set()
                release.wait(timeout=1.0)
                return (0.0,) * HOLOSOMA_G1_ACTION_DIM

        backend = HolosomaG1LocomotionBackend(BlockingRunner(), clock=lambda: 30.0)
        task = asyncio.create_task(backend.infer(request(observation())))
        try:
            while not started.is_set():
                await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            with self.assertRaisesRegex(PolicyError, "previous call"):
                await backend.infer(request(observation()))
        finally:
            release.set()
            backend.close(wait=True)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from longship.artifacts import VerifiedArtifact
from longship.policies import (
    UNITREE_RL_GYM_G1_ACTION_DIM,
    UNITREE_RL_GYM_G1_ACTION_SPACE,
    UNITREE_RL_GYM_G1_OBSERVATION_DIM,
    UNITREE_RL_GYM_G1_POLICY_ARTIFACT_ID,
    UNITREE_RL_GYM_G1_POLICY_SHA256,
    UNITREE_RL_GYM_G1_POLICY_SIZE_BYTES,
    UNITREE_RL_GYM_G1_RESOURCE_SCOPE,
    PolicyActionFrame,
    PolicyCandidate,
    PolicyCandidateRejected,
    PolicyError,
    PolicyRequest,
    UnitreeRLGymG1Observation,
    UnitreeRLGymTorchScriptRunner,
    guard_candidate,
    unitree_rl_gym_g1_guard_profile,
)


def observation() -> UnitreeRLGymG1Observation:
    joints = tuple(float(index) for index in range(UNITREE_RL_GYM_G1_ACTION_DIM))
    return UnitreeRLGymG1Observation(
        base_angular_velocity_scaled=(0.1, 0.2, 0.3),
        projected_gravity=(0.0, 0.0, -1.0),
        velocity_command_scaled=(0.4, 0.0, -0.1),
        joint_position_relative_scaled=joints,
        joint_velocity_scaled=tuple(-value for value in joints),
        last_action=(0.0,) * UNITREE_RL_GYM_G1_ACTION_DIM,
        gait_phase=(0.0, 1.0),
    )


class UnitreeRLGymObservationTests(unittest.TestCase):
    def test_builds_named_47_value_order(self) -> None:
        vector = observation().vector()

        self.assertEqual(len(vector), UNITREE_RL_GYM_G1_OBSERVATION_DIM)
        self.assertEqual(
            vector[:9],
            (0.1, 0.2, 0.3, 0.0, 0.0, -1.0, 0.4, 0.0, -0.1),
        )
        self.assertEqual(vector[-2:], (0.0, 1.0))

    def test_rejects_invalid_dimensions_and_phase(self) -> None:
        with self.assertRaisesRegex(ValueError, "joint_velocity_scaled"):
            UnitreeRLGymG1Observation(
                base_angular_velocity_scaled=(0.0, 0.0, 0.0),
                projected_gravity=(0.0, 0.0, -1.0),
                velocity_command_scaled=(0.0, 0.0, 0.0),
                joint_position_relative_scaled=(0.0,) * 12,
                joint_velocity_scaled=(0.0,) * 11,
                last_action=(0.0,) * 12,
                gait_phase=(0.0, 1.0),
            )
        with self.assertRaisesRegex(ValueError, "unit circle"):
            value = observation()
            UnitreeRLGymG1Observation(
                base_angular_velocity_scaled=value.base_angular_velocity_scaled,
                projected_gravity=value.projected_gravity,
                velocity_command_scaled=value.velocity_command_scaled,
                joint_position_relative_scaled=value.joint_position_relative_scaled,
                joint_velocity_scaled=value.joint_velocity_scaled,
                last_action=value.last_action,
                gait_phase=(0.0, 0.0),
            )


class UnitreeRLGymGuardTests(unittest.TestCase):
    def test_target_bound_rejects_untrusted_raw_action(self) -> None:
        request = PolicyRequest(
            call_id="call",
            model_binding_id="binding",
            lease_id="lease",
            lease_epoch=1,
            observation_version=2,
            deadline_monotonic=1.02,
            max_action_horizon_ms=20,
            resource_scope=UNITREE_RL_GYM_G1_RESOURCE_SCOPE,
            payload={"observation": observation()},
        )
        candidate = PolicyCandidate(
            call_id="call",
            model_binding_id="binding",
            lease_id="lease",
            lease_epoch=1,
            observation_version=2,
            generated_at_monotonic=1.0,
            expires_at_monotonic=1.02,
            action_space_id=UNITREE_RL_GYM_G1_ACTION_SPACE,
            resource_scope=UNITREE_RL_GYM_G1_RESOURCE_SCOPE,
            frames=(
                PolicyActionFrame(
                    0,
                    (10.1,) + (0.0,) * (UNITREE_RL_GYM_G1_ACTION_DIM - 1),
                ),
            ),
        )
        profile = unitree_rl_gym_g1_guard_profile(
            policy_step_ms=20,
            maximum_absolute_action=10.0,
        )

        with self.assertRaisesRegex(PolicyCandidateRejected, "bound"):
            guard_candidate(request, candidate, profile, now_monotonic=1.0)


class UnitreeRLGymLoaderTests(unittest.TestCase):
    def test_rejects_forged_verified_path_before_torch_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.pt"
            path.write_bytes(b"not the pinned policy")
            file_stat = path.stat()
            forged = VerifiedArtifact(
                artifact_id=UNITREE_RL_GYM_G1_POLICY_ARTIFACT_ID,
                path=path,
                sha256=UNITREE_RL_GYM_G1_POLICY_SHA256,
                size_bytes=UNITREE_RL_GYM_G1_POLICY_SIZE_BYTES,
                device=file_stat.st_dev,
                inode=file_stat.st_ino,
            )

            with self.assertRaisesRegex(PolicyError, "identity changed"):
                UnitreeRLGymTorchScriptRunner(forged)


if __name__ == "__main__":
    unittest.main()

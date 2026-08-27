from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from longship.rl.sim2sim.dds import G1_29DOF_JOINTS
from longship.rl.sim2sim.php_pipeline import PhpOnnxPolicy, command_vector


class PhpSim2SimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        model_dir = root / "third_party/php-parkour.github.io/mujoco_wasm/public"
        policy_path = model_dir / "2026-01-17_09-51-30_student-new-loco-old-skill_student.onnx"
        depth_path = model_dir / "2026-01-17_09-51-30_student-new-loco-old-skill_depth_backbone.onnx"
        if not policy_path.is_file() or not depth_path.is_file():
            raise unittest.SkipTest("PHP release ONNX artifacts are not installed")
        cls.policy = PhpOnnxPolicy(
            policy_path,
            depth_path,
            "cpu",
        )

    def setUp(self) -> None:
        self.policy.reset()

    def test_released_metadata_maps_every_g1_joint_roundtrip(self) -> None:
        dds = np.arange(29, dtype=np.float64)
        restored = self.policy.policy_to_dds_vector(self.policy.dds_to_policy(dds))
        self.assertTrue(np.array_equal(restored, dds))
        self.assertEqual(set(self.policy.joint_names), set(G1_29DOF_JOINTS))

    def test_discrete_command_matches_released_low_and_high_banks(self) -> None:
        self.assertEqual(np.flatnonzero(command_vector("stop", high_speed=True)).item(), 0)
        self.assertEqual(np.flatnonzero(command_vector("forward", high_speed=False)).item(), 1)
        self.assertEqual(np.flatnonzero(command_vector("forward", high_speed=True)).item(), 6)
        self.assertEqual(np.flatnonzero(command_vector("right", high_speed=True)).item(), 10)

    def test_depth_and_student_policy_produce_finite_joint_targets(self) -> None:
        target = self.policy.infer(
            torso_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            base_angular_velocity=(0.0, 0.0, 0.0),
            joint_position=self.policy.policy_to_dds_vector(self.policy.default_q),
            joint_velocity=np.zeros(29),
            command=command_vector("stop"),
            depth=np.full((270, 480), 3.0, dtype=np.float32),
        )
        self.assertEqual(target.shape, (29,))
        self.assertTrue(np.all(np.isfinite(target)))
        self.assertEqual(len(self.policy.depth_latents), 1)


if __name__ == "__main__":
    unittest.main()

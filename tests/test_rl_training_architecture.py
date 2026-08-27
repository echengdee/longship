from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from longship.rl.config import ExperimentConfig
from longship.rl.experiments import bundled_experiment_path
from longship.rl.training.runner import ExperimentRunner


ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


def minimal_experiment(backend: dict) -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "schema_version": "longship.rl-experiment.v1",
            "name": "backend_plan",
            "seed": 7,
            "model": {"type": "ActorCriticPolicy"},
            "training": {"backend": backend, "trainer": {"type": "PPO"}},
            "environment": {"robot": "unitree_g1_29dof", "task": "test"},
        }
    )


class TrainingArchitectureTests(unittest.TestCase):
    def test_hiking_recipe_loads_and_instinctlab_plan_is_shell_free(self) -> None:
        experiment = ExperimentConfig.load(
            bundled_experiment_path("hiking_g1_parkour.yaml")
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            plan = ExperimentRunner(workspace=ROOT).plan(experiment, output)
        self.assertEqual(plan.backend, "instinctlab")
        self.assertEqual(plan.cwd, ROOT / "third_party/InstinctLab")
        self.assertIn("scripts/instinct_rl/train.py", plan.argv)
        self.assertIn("--task=Instinct-Parkour-Target-Amp-G1-v0", plan.argv)
        self.assertIn("--seed=42", plan.argv)
        self.assertFalse(output.exists())

    def test_all_upstream_backends_have_a_plan(self) -> None:
        cases = (
            (
                {"type": "HoloSomaBackend"},
                "holosoma",
                "src/holosoma/holosoma/train_agent.py",
            ),
            (
                {"type": "SonicBackend", "num_envs": 16},
                "sonic",
                "gear_sonic/train_agent_trl.py",
            ),
            (
                {"type": "InstinctLabBackend", "num_envs": 16},
                "instinctlab",
                "scripts/instinct_rl/train.py",
            ),
            (
                {
                    "type": "MimicLiteBackend",
                    "motion_config": "g1/climb_turn_sit_71cm",
                    "num_envs": 8,
                    "total_iters": 2,
                },
                "mimiclite",
                "scripts/train.py",
            ),
        )
        runner = ExperimentRunner(workspace=ROOT)
        for backend_config, expected_backend, entrypoint in cases:
            with self.subTest(backend=expected_backend):
                plan = runner.plan(
                    minimal_experiment(backend_config), ROOT / "outputs" / "test-plan"
                )
                self.assertEqual(plan.backend, expected_backend)
                self.assertTrue(any(entrypoint in value for value in plan.argv))
                self.assertIn("7", " ".join(plan.argv))

    def test_mimiclite_71cm_recipe_is_a_single_motion_platform_plan(self) -> None:
        experiment = ExperimentConfig.load(
            bundled_experiment_path("mimiclite_g1_71cm_climb_turn_sit.yaml")
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            plan = ExperimentRunner(workspace=ROOT).plan(experiment, output)
        command = " ".join(plan.argv)
        self.assertEqual(plan.backend, "mimiclite")
        self.assertIn("task/motion=g1/climb_turn_sit_71cm", command)
        self.assertIn("task.terrain=box71", command)
        self.assertIn("task.num_envs=32", command)
        self.assertIn("total_iters=4000", command)
        self.assertIn(f"hydra.run.dir={output / 'upstream'}", command)
        self.assertEqual(plan.environment["HF_HUB_OFFLINE"], "0")
        self.assertFalse(output.exists())

    @unittest.skipUnless(HAS_TORCH, "PyTorch is an optional rl-train dependency")
    def test_hiking_policy_builds_from_yaml_and_runs_forward(self) -> None:
        import torch

        from longship.rl.builder import build_model

        experiment = ExperimentConfig.load(
            bundled_experiment_path("hiking_g1_parkour.yaml")
        )
        policy = build_model(experiment.values["model"])
        outputs = policy(
            {
                "depth": torch.zeros(2, 8, 18, 32),
                "proprio": torch.zeros(2, 768),
            }
        )
        self.assertEqual(outputs["mean"].shape, (2, 29))
        self.assertEqual(outputs["log_std"].shape, (2, 29))
        self.assertEqual(outputs["value"].shape, (2, 1))


if __name__ == "__main__":
    unittest.main()

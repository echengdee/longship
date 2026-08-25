from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from longship.rl.builder import build_model
from longship.rl.config import ExperimentConfig, ExperimentConfigError
from longship.rl.registry import ComponentRegistry, RegistryError
from longship.rl.training.runner import ExperimentRunner


@dataclass
class Part:
    values: dict[str, Any]

    def __init__(self, **values: Any) -> None:
        self.values = values


@dataclass
class Policy:
    encoder: Part
    backbone: Part
    actor_decoder: Part
    critic_decoder: Part


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[Mapping[str, Any], Path]] = []

    def train(self, experiment: Mapping[str, Any], output_dir: Path) -> Path:
        self.calls.append((experiment, output_dir))
        return output_dir / "checkpoints" / "last.pt"


def model_registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    registry.register("encoder", "ProprioceptionEncoder")(Part)
    registry.register("backbone", "MLPBackbone")(Part)
    registry.register("decoder", "GaussianActorDecoder")(Part)
    registry.register("decoder", "ValueDecoder")(Part)
    registry.register("policy", "ActorCriticPolicy")(Policy)
    return registry


def experiment_values() -> dict[str, Any]:
    return {
        "schema_version": "longship.rl-experiment.v1",
        "name": "g1_locomotion_ppo",
        "seed": 42,
        "model": {
            "type": "ActorCriticPolicy",
            "encoder": {
                "type": "ProprioceptionEncoder",
                "history_steps": 5,
            },
            "backbone": {
                "type": "MLPBackbone",
                "hidden_dims": [512, 256, 128],
            },
            "actor_decoder": {
                "type": "GaussianActorDecoder",
                "action_dim": 29,
            },
            "critic_decoder": {"type": "ValueDecoder", "output_dim": 1},
        },
        "training": {
            "backend": {"type": "HoloSomaBackend"},
            "trainer": {"type": "PPO"},
        },
        "environment": {"robot": "unitree_g1_29dof", "task": "velocity_tracking"},
    }


class RLPlatformTests(unittest.TestCase):
    def test_reference_experiment_validates(self) -> None:
        experiment = ExperimentConfig.from_mapping(experiment_values())
        self.assertEqual(experiment.name, "g1_locomotion_ppo")
        self.assertEqual(experiment.values["training"]["trainer"]["type"], "PPO")

    def test_model_is_built_from_typed_slots(self) -> None:
        experiment = ExperimentConfig.from_mapping(experiment_values())
        model = build_model(experiment.values["model"], registry=model_registry())
        self.assertIsInstance(model, Policy)
        self.assertEqual(model.backbone.values["hidden_dims"], [512, 256, 128])
        self.assertEqual(model.actor_decoder.values["action_dim"], 29)

    def test_unknown_component_reports_registered_names(self) -> None:
        with self.assertRaisesRegex(RegistryError, "registered: none"):
            ComponentRegistry().create("encoder", {"type": "MissingEncoder"})

    def test_unknown_experiment_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(ExperimentConfigError, "unknown fields"):
            ExperimentConfig.from_mapping(
                {
                    "schema_version": "longship.rl-experiment.v1",
                    "name": "bad",
                    "model": {"type": "Policy"},
                    "training": {
                        "backend": {"type": "Backend"},
                        "trainer": {"type": "PPO"},
                    },
                    "environment": {"robot": "g1", "task": "walk"},
                    "sim2sim": {},
                }
            )

    def test_runner_writes_resolved_config_before_dispatch(self) -> None:
        experiment = ExperimentConfig.from_mapping(experiment_values())
        registry = ComponentRegistry()
        backend = RecordingBackend()
        registry.register("training_backend", "HoloSomaBackend")(lambda **_: backend)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run-001"
            checkpoint = ExperimentRunner(registry).run(experiment, output)
            self.assertTrue((output / "resolved.yaml").is_file())
            self.assertEqual(checkpoint, output / "checkpoints" / "last.pt")
            self.assertEqual(len(backend.calls), 1)
            with self.assertRaises(FileExistsError):
                ExperimentRunner(registry).run(experiment, output)


if __name__ == "__main__":
    unittest.main()

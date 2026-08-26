from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from longship.simulation.follow_person import FollowSimulationScenario
from longship.skills.follow_person.config import FollowProfile


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "plugins/targets/mujoco_follow_person/runner.py"


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "longship_mujoco_follow_runner", MODULE_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load MuJoCo FollowPerson plugin")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class MujocoFollowPluginTests(unittest.TestCase):
    def test_generated_model_uses_bounded_unique_obstacles(self) -> None:
        module = _load_runner()

        model = module._model_xml((0.2, 0.3, 0.4))

        self.assertIn('model="longship_follow_person_proxy"', model)
        self.assertEqual(model.count('mocap="true"'), 4)
        self.assertEqual(model.count('name="barrier_'), 3)

    def test_model_rejects_unbounded_obstacle_count(self) -> None:
        module = _load_runner()

        with self.assertRaises(ValueError):
            module._model_xml(tuple(0.1 for _ in range(257)))

        with self.assertRaises(ValueError):
            module._model_xml((0.0,))

    def test_simulator_target_enforces_session_owner(self) -> None:
        module = _load_runner()
        target = module.MujocoVelocityTarget()

        self.assertTrue(target.acquire("session-a", 1).accepted)
        self.assertFalse(target.acquire("session-b", 2).accepted)

    def test_parser_exposes_longship_interactive_stack(self) -> None:
        module = _load_runner()

        options = module._parser().parse_args(
            [
                "--profile",
                "profile.json",
                "--scenario",
                "scenario.json",
                "--stack",
            ]
        )

        self.assertTrue(options.stack)

    @unittest.skipUnless(
        importlib.util.find_spec("mujoco"), "optional MuJoCo dependency is absent"
    )
    def test_optional_physics_acceptance_passes(self) -> None:
        module = _load_runner()
        profile = FollowProfile.load(
            ROOT / "scenarios/follow_person/profile.v0.json"
        )
        scenario = FollowSimulationScenario.load(
            ROOT / "scenarios/follow_person/closed_loop.v0.json"
        )

        report = module.run_mujoco_system(
            profile,
            scenario,
            instruction="Jackie，跟着我走",
            show_viewer=False,
            real_time=False,
            keep_viewer=False,
        )

        self.assertTrue(report.passed)
        self.assertEqual(report.physical_contact_steps, 0)
        self.assertEqual(report.accepted_skill_calls, 1)


if __name__ == "__main__":
    unittest.main()

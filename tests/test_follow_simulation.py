from __future__ import annotations

import unittest
from pathlib import Path

from longship.simulation.follow_person import FollowSimulationScenario, run_simulation
from longship.skills.follow_person.config import FollowProfile


ROOT = Path(__file__).resolve().parents[1]


class FollowSimulationTests(unittest.TestCase):
    def test_public_closed_loop_passes_declared_gates(self) -> None:
        profile = FollowProfile.load(
            ROOT / "scenarios/follow_person/profile.v0.json"
        )
        scenario = FollowSimulationScenario.load(
            ROOT / "scenarios/follow_person/closed_loop.v0.json"
        )

        report = run_simulation(profile, scenario)

        self.assertTrue(report.passed, report.checks)
        self.assertEqual(report.unsafe_forward_commands, 0)


if __name__ == "__main__":
    unittest.main()

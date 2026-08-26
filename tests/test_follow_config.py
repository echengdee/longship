from __future__ import annotations

import json
import unittest
from pathlib import Path

from longship.skills.follow_person.config import (
    FollowConfigError,
    FollowProfile,
    FollowQualification,
)
from longship.simulation.follow_person import FollowSimulationScenario


ROOT = Path(__file__).resolve().parents[1]


class FollowConfigTests(unittest.TestCase):
    def test_public_profile_and_scenario_load(self) -> None:
        profile = FollowProfile.load(
            ROOT / "scenarios/follow_person/profile.v0.json"
        )
        scenario = FollowSimulationScenario.load(
            ROOT / "scenarios/follow_person/closed_loop.v0.json"
        )

        self.assertEqual(profile.profile_id, "follow-person-public-low-speed-v0")
        self.assertAlmostEqual(profile.control_period_s, scenario.step_s)

    def test_profile_rejects_unknown_actuator_configuration(self) -> None:
        path = ROOT / "scenarios/follow_person/profile.v0.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["raw_velocity_topic"] = "/cmd_vel"

        with self.assertRaises(FollowConfigError):
            FollowProfile.from_mapping(value)

    def test_checked_in_hardware_qualification_is_deliberately_disabled(self) -> None:
        qualification = FollowQualification.load(
            ROOT / "scenarios/follow_person/qualification.g1.example.json"
        )

        self.assertFalse(qualification.approved)
        self.assertEqual(qualification.expires_at_unix_s, 0)


if __name__ == "__main__":
    unittest.main()

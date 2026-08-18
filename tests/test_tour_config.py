from __future__ import annotations

import json
import unittest
from pathlib import Path

from longship.tour.models import TourConfigError, TourPlan


ROOT = Path(__file__).resolve().parents[1]


class TourConfigTests(unittest.TestCase):
    def test_public_scenario_loads(self) -> None:
        plan = TourPlan.load(ROOT / "scenarios/voice_tour/tour.zh-CN.json")
        self.assertEqual(plan.schema_version, "longship.voice-tour.v0")
        self.assertGreaterEqual(len(plan.stops), 2)

    def test_unknown_fields_are_rejected(self) -> None:
        path = ROOT / "scenarios/voice_tour/tour.zh-CN.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["raw_velocity"] = [1, 0, 0]
        with self.assertRaises(TourConfigError):
            TourPlan.from_mapping(value)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path

from longship.contracts.skills.follow_person import ObstaclePoint
from longship.safety.follow_obstacle import ForwardObstacleGuard
from longship.skills.follow_person.config import FollowProfile
from longship.skills.follow_person.planner import LocalFollowPlanner


ROOT = Path(__file__).resolve().parents[1]


class FollowPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = FollowProfile.load(
            ROOT / "scenarios/follow_person/profile.v0.json"
        )
        self.planner = LocalFollowPlanner(
            self.profile.planner, self.profile.control
        )

    def test_local_grid_route_detours_around_supported_obstacle(self) -> None:
        plan = self.planner.plan(
            (3.0, 0.0),
            (ObstaclePoint(0.8, 0.0, 0.2),),
            standoff_m=1.5,
            goal_tolerance_m=0.1,
        )

        self.assertFalse(plan.blocked)
        self.assertTrue(any(abs(point[1]) > 0.3 for point in plan.path_robot_xy_m))
        self.assertNotEqual(plan.yaw_rate_radps, 0.0)

    def test_raw_clearance_guard_is_independent_of_planner_route(self) -> None:
        plan = self.planner.plan(
            (3.0, 0.5),
            (),
            standoff_m=1.5,
            goal_tolerance_m=0.1,
        )
        guarded = ForwardObstacleGuard(self.profile.safety).apply(
            plan,
            raw_clearance_m=0.3,
            current_forward_mps=0.1,
        )

        self.assertTrue(guarded.blocked)
        self.assertEqual((guarded.forward_mps, guarded.yaw_rate_radps), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()

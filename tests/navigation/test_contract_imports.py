"""Minimal packaging and dependency-direction smoke tests."""

import unittest

from longship.navigation.localization_engine.fixed_start_visual import (
    FixedStartVisualLocalizationEngine,
)
from longship.navigation.localization_engine.interface import (
    LocalizationEngine,
)
from longship.navigation.localization_engine.visual_policy import (
    VisualGoalDistanceBatchPolicy,
    VisualGoalDistancePolicy,
)
from longship.navigation.local_trajectory_engine import (
    LocalTrajectoryEngine,
    LocalTrajectoryStream,
    RouteBoundLocalTrajectoryEngine,
)
from longship.navigation.map_engine.interface import MapEngine
from longship.navigation.mission_engine.interface import NavigationMissionEngine
from longship.navigation.planning_engine.interface import PlanningEngine
from longship.navigation.ports.route_execution.interface import (
    RouteExecutionPort,
)
from longship.navigation.runtime import (
    LocalizationDrivenLocalTrajectoryService,
    LocalizationObservationCompletionPolicy,
    LocalizationObservationProducer,
    LocalizationObservationProducerState,
    LocalizationObservationProducerStatus,
    LocalizationRuntime,
    LocalizationRuntimeResource,
    LocalizationTickService,
)
from longship.navigation.skill import NavigateToSkill


class ContractImportTests(unittest.TestCase):
    def test_public_protocols_are_importable(self) -> None:
        self.assertIsNotNone(MapEngine)
        self.assertIsNotNone(LocalizationEngine)
        self.assertIsNotNone(FixedStartVisualLocalizationEngine)
        self.assertIsNotNone(VisualGoalDistanceBatchPolicy)
        self.assertIsNotNone(VisualGoalDistancePolicy)
        self.assertIsNotNone(PlanningEngine)
        self.assertIsNotNone(NavigationMissionEngine)
        self.assertIsNotNone(RouteExecutionPort)
        self.assertIsNotNone(LocalTrajectoryEngine)
        self.assertIsNotNone(LocalTrajectoryStream)
        self.assertIsNotNone(LocalizationObservationProducer)
        self.assertIsNotNone(LocalizationObservationProducerState)
        self.assertIsNotNone(LocalizationObservationProducerStatus)
        self.assertIsNotNone(LocalizationObservationCompletionPolicy)
        self.assertIsNotNone(LocalizationTickService)
        self.assertIsNotNone(LocalizationRuntimeResource)
        self.assertIsNotNone(LocalizationRuntime)
        self.assertIsNotNone(LocalizationDrivenLocalTrajectoryService)
        self.assertIsNotNone(RouteBoundLocalTrajectoryEngine)
        self.assertIsNotNone(NavigateToSkill)


if __name__ == "__main__":
    unittest.main()

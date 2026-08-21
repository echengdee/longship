"""Longship-facing Skill contract for the navigation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, TypeAlias, runtime_checkable

from longship.navigation.common import NavigationExecutionContext, TimePoint
from longship.navigation.map_engine.models import MapSelector
from longship.navigation.mission_engine.models import (
    MissionBudget,
    MissionCompletion,
    MissionFailure,
    MissionSuccessCriteria,
    MissionTargetSpec,
    NavigationMissionId,
)
from longship.navigation.planning_engine.models import (
    PlanningContext,
    RouteConstraints,
    RoutePreferences,
)
from longship.navigation.ports.route_execution.models import (
    RouteExecutionLimits,
)


class NavigateToOutcome(str, Enum):
    """Terminal outcome reported to the Longship Runtime."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


SkillExecutionContext: TypeAlias = NavigationExecutionContext


@dataclass(frozen=True, slots=True)
class NavigateToRequest:
    """Semantic input accepted by the navigation Skill."""

    requested_at: TimePoint
    map_selector: MapSelector
    target: MissionTargetSpec
    route_constraints: RouteConstraints = field(
        default_factory=RouteConstraints
    )
    route_preferences: RoutePreferences = field(
        default_factory=RoutePreferences
    )
    initial_planning_context: PlanningContext = field(
        default_factory=PlanningContext
    )
    execution_limits: RouteExecutionLimits = field(
        default_factory=RouteExecutionLimits
    )
    success_criteria: MissionSuccessCriteria = field(
        default_factory=MissionSuccessCriteria
    )
    budget: MissionBudget = field(default_factory=MissionBudget)


@dataclass(frozen=True, slots=True)
class NavigateToResult:
    """Terminal navigation result returned to the Longship Runtime."""

    skill_call_id: str
    navigation_mission_id: NavigationMissionId
    outcome: NavigateToOutcome
    decided_at: TimePoint
    completion: MissionCompletion | None = None
    failure: MissionFailure | None = None
    cancellation_reason: str | None = None


@runtime_checkable
class NavigateToSkill(Protocol):
    """Thin Longship-facing facade over NavigationMissionEngine.

    Runtime schedules and cancels this Skill. The implementation delegates
    navigation sequencing to the harness and must never bypass Runtime resource
    ownership, command arbitration, or Safety.
    """

    async def execute(
        self,
        request: NavigateToRequest,
        context: SkillExecutionContext,
    ) -> NavigateToResult:
        """Execute one bounded navigation operation."""
        ...

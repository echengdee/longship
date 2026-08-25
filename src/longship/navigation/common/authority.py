"""Runtime authority conveyed into one navigation operation."""

from __future__ import annotations

from dataclasses import dataclass

from .time import TimePoint


@dataclass(frozen=True, slots=True)
class NavigationExecutionContext:
    """Longship Runtime authority and correlation data for navigation.

    The context records authority already granted by Runtime. It never permits
    the navigation harness to allocate or extend a resource lease for itself.
    """

    longship_mission_id: str
    task_id: str
    skill_call_id: str
    correlation_id: str
    resource_lease_id: str
    cancellation_epoch: int
    state_version: int
    plan_version: int
    deadline: TimePoint | None = None

"""Public Mission Engine facade for the Longship navigation harness."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    MissionControlRequest,
    MissionControlResult,
    MissionErrorCode,
    NavigationMissionId,
    MissionStatus,
    MissionSubmissionResult,
    MissionUpdateRequest,
    MissionUpdateResult,
    NavigationMissionRequest,
)


class NavigationMissionEngineError(RuntimeError):
    """Contract, addressing, dependency, or infrastructure failure.

    Normal mission outcomes such as target ambiguity, no route, execution
    failure, or exhausted recovery budget are published in ``MissionStatus``.
    """

    def __init__(
        self,
        code: MissionErrorCode,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@runtime_checkable
class NavigationMissionEngine(Protocol):
    """Navigation-operation orchestrator used inside one navigate Skill."""

    async def submit_mission(
        self,
        request: NavigationMissionRequest,
    ) -> MissionSubmissionResult:
        """Accept or reject a mission without waiting for its completion."""
        ...

    def get_status(
        self,
        mission_id: NavigationMissionId,
    ) -> MissionStatus:
        """Return the latest immutable status publication."""
        ...

    async def wait_for_update(
        self,
        request: MissionUpdateRequest,
    ) -> MissionUpdateResult:
        """Wait for a newer publication, a terminal state, or timeout."""
        ...

    async def control(
        self,
        request: MissionControlRequest,
    ) -> MissionControlResult:
        """Request task-level pause, resume, or graceful cancellation."""
        ...

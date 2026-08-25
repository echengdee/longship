"""Executor-side strategy boundary for route-step trajectory candidates.

This is not part of the mission-facing ``RouteExecutionPort``. Candidates are
policy-native model outputs, never platform commands.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    TrajectoryCandidateSet,
    TrajectoryPolicyErrorCode,
    VisualGoalTrajectoryRequest,
)


class TrajectoryPolicyError(RuntimeError):
    """Structured policy, context, goal, or inference failure."""

    def __init__(
        self,
        code: TrajectoryPolicyErrorCode,
        message: str,
        *,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@runtime_checkable
class VisualGoalTrajectoryPolicy(Protocol):
    """Generates candidates; selection and command conversion stay downstream."""

    async def generate_trajectories(
        self,
        request: VisualGoalTrajectoryRequest,
    ) -> TrajectoryCandidateSet: ...

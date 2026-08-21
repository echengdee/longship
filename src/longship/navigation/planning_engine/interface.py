"""Public Planning Engine facade for the Longship navigation harness."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    PlanningErrorCode,
    RoutePlanningRequest,
    RoutePlanningResult,
)


class PlanningEngineError(RuntimeError):
    """Structured contract or infrastructure failure.

    Normal outcomes such as an unreachable target or an ambiguous start are
    returned as ``RoutePlanningResult`` and are not raised as exceptions.
    """

    def __init__(
        self,
        code: PlanningErrorCode,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@runtime_checkable
class PlanningEngine(Protocol):
    """Stateless request/response facade for global route planning."""

    async def plan_route(
        self,
        request: RoutePlanningRequest,
    ) -> RoutePlanningResult:
        """Produce one immutable route or a normal no-route outcome."""
        ...

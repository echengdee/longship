"""External route-execution port required by the navigation harness.

This module defines a dependency contract only.  The concrete executor and
transport adapter are supplied by the integrating system.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    RouteCommand,
    RouteCommandId,
    RouteControlRequest,
    RouteControlResult,
    RouteExecutionPortErrorCode,
    RouteExecutionStatus,
    RouteExecutionUpdateRequest,
    RouteExecutionUpdateResult,
    RouteSubmissionResult,
)


class RouteExecutionPortError(RuntimeError):
    """Contract, addressing, transport, or external-service failure."""

    def __init__(
        self,
        code: RouteExecutionPortErrorCode,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@runtime_checkable
class RouteExecutionPort(Protocol):
    """Mission-facing contract implemented by an external route executor."""

    async def submit_route(
        self,
        command: RouteCommand,
    ) -> RouteSubmissionResult:
        """Submit one immutable route and return acceptance or rejection."""
        ...

    def get_status(
        self,
        command_id: RouteCommandId,
    ) -> RouteExecutionStatus:
        """Return the latest immutable status for a submitted command."""
        ...

    async def wait_for_update(
        self,
        request: RouteExecutionUpdateRequest,
    ) -> RouteExecutionUpdateResult:
        """Wait for a newer status publication, terminal state, or timeout."""
        ...

    async def control(
        self,
        request: RouteControlRequest,
    ) -> RouteControlResult:
        """Request pause, resume, or graceful cancellation."""
        ...

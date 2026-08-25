"""Public facade and read-only stream of the Local Trajectory Engine."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    LocalTrajectoryPublication,
    LocalTrajectoryStreamErrorCode,
    LocalTrajectoryUpdateResult,
    WaitForLocalTrajectoryRequest,
)


class LocalTrajectoryStreamError(RuntimeError):
    """Structured stream contract or infrastructure failure."""

    def __init__(
        self,
        code: LocalTrajectoryStreamErrorCode,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@runtime_checkable
class LocalTrajectoryStream(Protocol):
    """Read-only output boundary consumed outside Navigation Harness."""

    def get_latest(self) -> LocalTrajectoryPublication:
        """Returns the newest immutable publication without waiting."""
        ...

    async def wait_for_update(
        self,
        request: WaitForLocalTrajectoryRequest,
    ) -> LocalTrajectoryUpdateResult:
        """Waits for a newer publication, stream reset, or normal timeout."""
        ...


@runtime_checkable
class LocalTrajectoryEngine(LocalTrajectoryStream, Protocol):
    """Route-bound engine that continuously publishes local trajectories."""

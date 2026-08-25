"""Public Localization Engine facade for the Longship navigation harness."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    BeliefUpdateResult,
    LocalizationEngineStatus,
    LocalizationErrorCode,
    LocationBelief,
    RelocalizationAcceptance,
    RelocalizationRequest,
    WaitForUpdateRequest,
)


class LocalizationEngineError(RuntimeError):
    """Structured contract or infrastructure failure."""

    def __init__(
        self,
        code: LocalizationErrorCode,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@runtime_checkable
class LocalizationEngine(Protocol):
    """System-level, continuously running localization state service."""

    def get_belief(self) -> LocationBelief:
        """Return the latest immutable belief without waiting."""
        ...

    async def wait_for_update(
        self,
        request: WaitForUpdateRequest,
    ) -> BeliefUpdateResult:
        """Wait for a newer belief, a stream reset, or timeout."""
        ...

    async def request_relocalization(
        self,
        request: RelocalizationRequest,
    ) -> RelocalizationAcceptance:
        """Ask the running engine to start or join a relocalization attempt."""
        ...

    def get_status(self) -> LocalizationEngineStatus:
        """Return operational health and active-map metadata."""
        ...

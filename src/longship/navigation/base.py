from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol


class NavigationAuthority:
    """Revocable in-process authority for one motion-command epoch.

    A conforming navigation provider checks ``ensure_active`` immediately
    before every side effect that can start or resume motion. Cross-process
    providers must map the same epoch onto a target-side expiring lease.
    """

    __slots__ = ("epoch", "_revoked")

    def __init__(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("navigation authority epoch must be non-negative")
        self.epoch = epoch
        self._revoked = asyncio.Event()

    @property
    def revoked(self) -> bool:
        return self._revoked.is_set()

    @property
    def revoked_event(self) -> asyncio.Event:
        return self._revoked

    def revoke(self) -> None:
        self._revoked.set()

    def ensure_active(self) -> None:
        if self.revoked:
            raise asyncio.CancelledError


@dataclass(frozen=True, slots=True)
class NavigationRequest:
    request_id: str
    authority_epoch: int
    map_id: str
    map_version: str
    route_id: str
    waypoint_id: str


@dataclass(frozen=True, slots=True)
class NavigationResult:
    arrived: bool
    request_id: str
    authority_epoch: int
    map_id: str
    map_version: str
    route_id: str
    waypoint_id: str
    evidence: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class NavigationStopRequest:
    request_id: str
    reason: str
    revoke_through_epoch: int


@dataclass(frozen=True, slots=True)
class StopResult:
    request_id: str
    revoked_through_epoch: int
    requested: bool
    verified_stopped: bool
    evidence: str
    detail: str = ""


class NavigationPort(Protocol):
    """Target-independent waypoint navigation seam.

    A future Nav2 or other navigation plugin resolves approved waypoint IDs to
    poses. Voice and Brain providers never receive permission to emit raw poses
    or target-specific velocity commands through this interface.
    """

    async def navigate_to(
        self, request: NavigationRequest, authority: NavigationAuthority
    ) -> NavigationResult:
        ...

    async def pause(self, authority: NavigationAuthority) -> None:
        ...

    async def resume(self, authority: NavigationAuthority) -> None:
        ...

    async def stop(self, request: NavigationStopRequest) -> StopResult:
        ...

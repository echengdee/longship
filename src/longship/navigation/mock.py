from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable

from .base import (
    NavigationAuthority,
    NavigationRequest,
    NavigationResult,
    NavigationStopRequest,
    StopResult,
)


class MockNavigation:
    """Deterministic mock used by the public voice-tour scenario and CI."""

    def __init__(
        self,
        *,
        travel_seconds: float = 0.05,
        failing_waypoints: Iterable[str] = (),
    ) -> None:
        if travel_seconds < 0:
            raise ValueError("travel_seconds must be non-negative")
        self.travel_seconds = travel_seconds
        self.failing_waypoints = frozenset(failing_waypoints)
        self.events: list[tuple[str, str, float]] = []
        self._pause_gate = asyncio.Event()
        self._pause_gate.set()
        self._stopped = asyncio.Event()

    async def navigate_to(
        self, request: NavigationRequest, authority: NavigationAuthority
    ) -> NavigationResult:
        if request.authority_epoch != authority.epoch:
            raise ValueError("navigation request authority epoch does not match")
        authority.ensure_active()
        waypoint_id = request.waypoint_id
        self._stopped.clear()
        self.events.append(("navigation.started", waypoint_id, time.monotonic()))
        remaining = self.travel_seconds
        active_since = time.monotonic()
        while remaining > 0:
            if authority.revoked or self._stopped.is_set():
                self.events.append(("navigation.cancelled", waypoint_id, time.monotonic()))
                raise asyncio.CancelledError
            if not self._pause_gate.is_set():
                await self._pause_gate.wait()
                active_since = time.monotonic()
                continue
            now = time.monotonic()
            remaining -= max(0.0, now - active_since)
            active_since = now
            await asyncio.sleep(min(0.01, max(0.0, remaining)))

        if waypoint_id in self.failing_waypoints:
            self.events.append(("navigation.blocked", waypoint_id, time.monotonic()))
            return NavigationResult(
                arrived=False,
                request_id=request.request_id,
                authority_epoch=request.authority_epoch,
                map_id=request.map_id,
                map_version=request.map_version,
                route_id=request.route_id,
                waypoint_id=waypoint_id,
                evidence="mock.navigation.failure",
                detail="configured mock failure",
            )
        self.events.append(("navigation.arrived", waypoint_id, time.monotonic()))
        return NavigationResult(
            arrived=True,
            request_id=request.request_id,
            authority_epoch=request.authority_epoch,
            map_id=request.map_id,
            map_version=request.map_version,
            route_id=request.route_id,
            waypoint_id=waypoint_id,
            evidence="mock.arrival.v0",
        )

    async def pause(self, authority: NavigationAuthority) -> None:
        authority.ensure_active()
        self._pause_gate.clear()
        self.events.append(("navigation.paused", "", time.monotonic()))

    async def resume(self, authority: NavigationAuthority) -> None:
        authority.ensure_active()
        self._pause_gate.set()
        self.events.append(("navigation.resumed", "", time.monotonic()))

    async def stop(self, request: NavigationStopRequest) -> StopResult:
        self._stopped.set()
        self._pause_gate.set()
        self.events.append(("navigation.stopped", request.reason, time.monotonic()))
        return StopResult(
            request_id=request.request_id,
            revoked_through_epoch=request.revoke_through_epoch,
            requested=True,
            verified_stopped=True,
            evidence="mock.zero-motion.v0",
        )

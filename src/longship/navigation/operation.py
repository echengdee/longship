"""Operation-oriented adapter for navigation providers with trajectory streams.

``NavigationPort`` is intentionally a terminal-result interface.  Providers
backed by Navigation Harness can additionally implement
``NavigationOperationStarter`` so callers that need live local trajectories
can obtain the read-only stream for the same navigation operation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .base import (
    NavigationAuthority,
    NavigationPort,
    NavigationRequest,
    NavigationResult,
    NavigationStopRequest,
    StopResult,
)
from .local_trajectory_engine import LocalTrajectoryStream


@runtime_checkable
class NavigationSession(Protocol):
    """One started navigation operation supplied by a Harness integration."""

    @property
    def trajectory_stream(self) -> LocalTrajectoryStream:
        """Returns the operation's read-only local-trajectory output."""
        ...

    async def wait_result(self) -> NavigationResult:
        """Waits for the terminal navigation result."""
        ...

    async def pause(self, authority: NavigationAuthority) -> None:
        """Pauses this operation without revoking its authority."""
        ...

    async def resume(self, authority: NavigationAuthority) -> None:
        """Resumes this operation when authority remains active."""
        ...

    async def stop(self, request: NavigationStopRequest) -> StopResult:
        """Requests a verified stop for this operation."""
        ...


@runtime_checkable
class NavigationSessionFactory(Protocol):
    """Creates deployment-specific Harness-backed navigation sessions."""

    async def start_session(
        self,
        request: NavigationRequest,
        authority: NavigationAuthority,
    ) -> NavigationSession:
        """Starts the mission and returns its terminal result and trajectory stream."""
        ...


@runtime_checkable
class NavigationSessionBuilder(Protocol):
    """Deployment-specific constructor for one Harness navigation session.

    A NoMaD/ROS 2 builder owns the concrete composition of its map resources,
    observation source, localization runtime, route plan, and trajectory
    service. It returns only the operation boundary needed by Longship.
    """

    async def build_session(
        self,
        request: NavigationRequest,
        authority: NavigationAuthority,
    ) -> NavigationSession:
        """Builds and starts one request-scoped Harness session."""
        ...


class NavigationHarnessFactory:
    """Concrete outer composition root for a Harness-backed navigation port.

    The integrating application constructs this class once with its selected
    session builder, then exposes the result of ``create_navigation_port()``
    to Longship Runtime. The factory deliberately does not import a policy,
    ROS, a map backend, or a controller.
    """

    def __init__(self, session_builder: NavigationSessionBuilder) -> None:
        self._session_builder = session_builder

    def create_navigation_port(self) -> StreamBackedNavigationPort:
        """Creates an independent outer navigation facade."""
        return StreamBackedNavigationPort(self)

    async def start_session(
        self,
        request: NavigationRequest,
        authority: NavigationAuthority,
    ) -> NavigationSession:
        """Creates one deployment-specific session for the navigation facade."""
        return await self._session_builder.build_session(request, authority)


@dataclass(frozen=True, slots=True)
class NavigationOperation:
    """Caller-facing handle for one active navigation request.

    ``trajectory_stream`` has no command surface.  It is safe to hand to an
    external visualization, safety, or trajectory-tracking adapter without
    exposing map, localization, planning, or policy internals.
    """

    request_id: str
    trajectory_stream: LocalTrajectoryStream
    _wait_result: Callable[[], Awaitable[NavigationResult]] = field(
        repr=False,
        compare=False,
    )

    async def wait_result(self) -> NavigationResult:
        """Waits for the terminal result and releases the active operation."""
        return await self._wait_result()


@runtime_checkable
class NavigationOperationStarter(Protocol):
    """Optional extension that exposes an operation's live trajectory stream."""

    async def start_navigation(
        self,
        request: NavigationRequest,
        authority: NavigationAuthority,
    ) -> NavigationOperation:
        """Starts navigation and returns immediately with its trajectory stream."""
        ...


@dataclass(frozen=True, slots=True)
class _ActiveSession:
    request: NavigationRequest
    authority: NavigationAuthority
    session: NavigationSession
    result_task: asyncio.Task[NavigationResult]


class StreamBackedNavigationPort(NavigationPort, NavigationOperationStarter):
    """Adapts Harness sessions to the existing Longship navigation contract.

    The adapter owns operation lookup and lifecycle only.  The injected factory
    is responsible for composing the concrete Map, Localization, Planning,
    Mission, and Local Trajectory Engine implementation for a deployment.
    """

    def __init__(self, factory: NavigationSessionFactory) -> None:
        self._factory = factory
        self._sessions: dict[str, _ActiveSession] = {}
        self._lock = asyncio.Lock()

    async def start_navigation(
        self,
        request: NavigationRequest,
        authority: NavigationAuthority,
    ) -> NavigationOperation:
        """Starts one operation and exposes its read-only trajectory stream."""
        self._validate_request(request, authority)
        authority.ensure_active()

        async with self._lock:
            if request.request_id in self._sessions:
                raise ValueError("navigation request is already active")
            if any(
                active.authority is authority
                for active in self._sessions.values()
            ):
                raise ValueError("navigation authority already owns an operation")

            session = await self._factory.start_session(request, authority)
            result_task = asyncio.create_task(
                self._await_session_result(request.request_id, session)
            )
            active = _ActiveSession(
                request=request,
                authority=authority,
                session=session,
                result_task=result_task,
            )
            self._sessions[request.request_id] = active

        return NavigationOperation(
            request_id=request.request_id,
            trajectory_stream=session.trajectory_stream,
            _wait_result=lambda: self._wait_for_result(active),
        )

    async def navigate_to(
        self,
        request: NavigationRequest,
        authority: NavigationAuthority,
    ) -> NavigationResult:
        """Compatibility convenience method for terminal-result callers."""
        operation = await self.start_navigation(request, authority)
        return await operation.wait_result()

    async def pause(self, authority: NavigationAuthority) -> None:
        """Forwards pause to the one operation owned by ``authority``."""
        authority.ensure_active()
        active = await self._session_for_authority(authority)
        await active.session.pause(authority)

    async def resume(self, authority: NavigationAuthority) -> None:
        """Forwards resume to the one operation owned by ``authority``."""
        authority.ensure_active()
        active = await self._session_for_authority(authority)
        await active.session.resume(authority)

    async def stop(self, request: NavigationStopRequest) -> StopResult:
        """Forwards a verified stop request to the matching operation."""
        async with self._lock:
            active = self._sessions.get(request.request_id)
        if active is None:
            raise ValueError("navigation request is not active")
        return await active.session.stop(request)

    async def _wait_for_result(self, active: _ActiveSession) -> NavigationResult:
        return await asyncio.shield(active.result_task)

    async def _await_session_result(
        self,
        request_id: str,
        session: NavigationSession,
    ) -> NavigationResult:
        try:
            return await session.wait_result()
        finally:
            async with self._lock:
                active = self._sessions.get(request_id)
                if active is not None and active.session is session:
                    del self._sessions[request_id]

    async def _session_for_authority(
        self,
        authority: NavigationAuthority,
    ) -> _ActiveSession:
        async with self._lock:
            for active in self._sessions.values():
                if active.authority is authority:
                    return active
        raise ValueError("navigation authority does not own an active operation")

    @staticmethod
    def _validate_request(
        request: NavigationRequest,
        authority: NavigationAuthority,
    ) -> None:
        if not request.request_id:
            raise ValueError("navigation request ID must be non-empty")
        if request.authority_epoch != authority.epoch:
            raise ValueError("navigation request authority epoch does not match")

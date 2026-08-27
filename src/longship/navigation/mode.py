"""Long-lived Navigation Mode lifecycle around request-scoped operations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
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
from .operation import (
    NavigationOperation,
    NavigationOperationStarter,
    NavigationSession,
    NavigationSessionFactory,
    StreamBackedNavigationPort,
)


class NavigationModeState(str, Enum):
    """Lifecycle state of the shared Navigation Mode resources."""

    CREATED = "created"
    ENTERING = "entering"
    RUNNING = "running"
    EXITING = "exiting"
    STOPPED = "stopped"
    FAULTED = "faulted"


@dataclass(frozen=True, slots=True)
class NavigationModeStatus:
    """Non-blocking lifecycle status for the outer application."""

    state: NavigationModeState
    has_trajectory_stream: bool
    detail_code: str | None = None
    last_error: str | None = None


class NavigationModeRuntimeError(RuntimeError):
    """Navigation Mode lifecycle or invocation failure."""


@runtime_checkable
class NavigationModeDriver(NavigationSessionFactory, Protocol):
    """Deployment-owned resources shared while Navigation Mode is active.

    ``enter`` starts the camera or replay source, model resources, map access,
    and localization at its configured default start. The stream is available
    after entering and must publish non-motion states until a route can produce
    an ``ACTIVE`` local trajectory.

    Every session returned by ``start_session`` must publish to this mode's
    trajectory stream. ``exit`` stops any active session, invalidates old
    trajectories, and releases all mode-owned resources.
    """

    @property
    def trajectory_stream(self) -> LocalTrajectoryStream:
        """Returns the shared stream available for this Navigation Mode."""
        ...

    async def enter(self) -> None:
        """Starts shared resources and establishes the configured start state."""
        ...

    async def exit(self) -> None:
        """Stops active navigation and releases the shared resources."""
        ...


@runtime_checkable
class NavigationModeDriverFactory(Protocol):
    """Creates one fresh deployment-specific driver per Navigation Mode."""

    def create_driver(self) -> NavigationModeDriver:
        """Returns an unentered driver with no active mode resources."""
        ...


class NavigationModeRuntime(
    NavigationPort,
    NavigationOperationStarter,
):
    """Outer facade for one entered Navigation Mode.

    ``enter`` and ``exit`` own shared resource lifetime. Navigation requests
    create only request-scoped operations within that mode; they do not reopen
    the camera, reload the model, or reset default-start localization.
    """

    def __init__(self, driver: NavigationModeDriver) -> None:
        self._driver = driver
        self._navigation = StreamBackedNavigationPort(driver)
        self._state = NavigationModeState.CREATED
        self._trajectory_stream: LocalTrajectoryStream | None = None
        self._detail_code: str | None = None
        self._last_error: str | None = None
        self._lifecycle_lock = asyncio.Lock()

    @property
    def trajectory_stream(self) -> LocalTrajectoryStream:
        """Returns the shared stream after the mode has been entered."""
        if self._trajectory_stream is None:
            raise NavigationModeRuntimeError(
                "navigation mode has no trajectory stream before enter"
            )
        return self._trajectory_stream

    def get_status(self) -> NavigationModeStatus:
        """Returns the current lifecycle state without waiting."""
        return NavigationModeStatus(
            state=self._state,
            has_trajectory_stream=self._trajectory_stream is not None,
            detail_code=self._detail_code,
            last_error=self._last_error,
        )

    async def enter(self) -> None:
        """Enters Navigation Mode and initializes its default start state."""
        async with self._lifecycle_lock:
            if self._state == NavigationModeState.RUNNING:
                return
            if self._state != NavigationModeState.CREATED:
                raise NavigationModeRuntimeError(
                    "navigation mode can only enter from CREATED"
                )
            self._state = NavigationModeState.ENTERING
            self._detail_code = "starting_shared_resources"
            try:
                await self._driver.enter()
                self._trajectory_stream = self._driver.trajectory_stream
            except BaseException as error:
                self._state = NavigationModeState.FAULTED
                self._detail_code = f"enter_failed:{type(error).__name__}"
                self._last_error = str(error)
                await self._attempt_exit_after_enter_failure()
                if isinstance(error, asyncio.CancelledError):
                    raise
                raise NavigationModeRuntimeError(
                    "navigation mode failed to enter"
                ) from error
            self._state = NavigationModeState.RUNNING
            self._detail_code = "running"

    async def exit(self) -> None:
        """Exits Navigation Mode and releases its shared resources."""
        async with self._lifecycle_lock:
            if self._state == NavigationModeState.STOPPED:
                return
            self._state = NavigationModeState.EXITING
            self._detail_code = "stopping_shared_resources"
            try:
                await self._driver.exit()
            except BaseException as error:
                self._state = NavigationModeState.FAULTED
                self._detail_code = f"exit_failed:{type(error).__name__}"
                self._last_error = str(error)
                if isinstance(error, asyncio.CancelledError):
                    raise
                raise NavigationModeRuntimeError(
                    "navigation mode failed to exit"
                ) from error
            self._state = NavigationModeState.STOPPED
            self._detail_code = "stopped"

    async def start_navigation(
        self,
        request: NavigationRequest,
        authority: NavigationAuthority,
    ) -> NavigationOperation:
        """Starts one target navigation within the entered mode."""
        async with self._lifecycle_lock:
            self._ensure_running()
            return await self._navigation.start_navigation(request, authority)

    async def navigate_to(
        self,
        request: NavigationRequest,
        authority: NavigationAuthority,
    ) -> NavigationResult:
        """Waits for a target navigation terminal result."""
        operation = await self.start_navigation(request, authority)
        return await operation.wait_result()

    async def pause(self, authority: NavigationAuthority) -> None:
        """Pauses the operation owned by the supplied authority."""
        self._ensure_running()
        await self._navigation.pause(authority)

    async def resume(self, authority: NavigationAuthority) -> None:
        """Resumes the operation owned by the supplied authority."""
        self._ensure_running()
        await self._navigation.resume(authority)

    async def stop(self, request: NavigationStopRequest) -> StopResult:
        """Stops the target navigation while keeping Navigation Mode entered."""
        self._ensure_running()
        return await self._navigation.stop(request)

    async def _attempt_exit_after_enter_failure(self) -> None:
        try:
            await self._driver.exit()
        except Exception as cleanup_error:
            self._last_error = (
                f"{self._last_error}; cleanup failed: {cleanup_error}"
            )

    def _ensure_running(self) -> None:
        if self._state != NavigationModeState.RUNNING:
            raise NavigationModeRuntimeError(
                "navigation mode is not running"
            )


class NavigationModeRuntimeFactory:
    """Creates independently enterable Navigation Mode runtime instances."""

    def __init__(self, driver_factory: NavigationModeDriverFactory) -> None:
        self._driver_factory = driver_factory

    def create_runtime(self) -> NavigationModeRuntime:
        """Creates a fresh Navigation Mode in the ``CREATED`` state."""
        return NavigationModeRuntime(self._driver_factory.create_driver())

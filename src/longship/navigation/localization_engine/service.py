"""System-owned continuous runner for a tick-driven Localization Engine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Protocol

from longship.navigation.common import TimePoint

from .models import LocationBelief


class LocalizationServiceState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAULTED = "faulted"


@dataclass(frozen=True, slots=True)
class LocalizationServiceConfig:
    """Scheduling and shutdown guards owned by the system supervisor."""

    tick_period_s: float = 0.25
    stop_timeout_s: float = 5.0

    def validate(self) -> None:
        values = (self.tick_period_s, self.stop_timeout_s)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("localization service durations must be finite")
        if self.tick_period_s <= 0.0:
            raise ValueError("tick_period_s must be positive")
        if self.stop_timeout_s <= 0.0:
            raise ValueError("stop_timeout_s must be positive")


@dataclass(frozen=True, slots=True)
class LocalizationServiceStatus:
    state: LocalizationServiceState
    tick_period_s: float
    ticks_completed: int
    skipped_tick_slots: int
    started_at: TimePoint | None
    last_tick_at: TimePoint | None
    detail_code: str | None = None
    last_error: str | None = None


class LocalizationTimeSource(Protocol):
    """Provides tick timestamps in the observation clock domain."""

    def now(self) -> TimePoint: ...


class TickableLocalizationEngine(Protocol):
    """Internal scheduling surface, deliberately absent from the facade."""

    async def tick(self, now: TimePoint) -> LocationBelief: ...


class MonotonicTimeSource:
    """Uses the host monotonic clock for observations and policy requests."""

    def __init__(self, clock_id: str = "monotonic") -> None:
        if not clock_id.strip():
            raise ValueError("clock_id must not be empty")
        self._clock_id = clock_id

    def now(self) -> TimePoint:
        return TimePoint(
            clock_id=self._clock_id,
            nanoseconds=time.monotonic_ns(),
        )


class ContinuousLocalizationService:
    """Runs one Localization Engine tick stream without overlapping ticks.

    This class belongs to Runtime Bootstrap/System Supervisor. It does not
    become part of the mission-facing ``LocalizationEngine`` facade, own a
    camera, decode images, or shut down a policy executor.
    """

    def __init__(
        self,
        *,
        engine: TickableLocalizationEngine,
        time_source: LocalizationTimeSource,
        config: LocalizationServiceConfig = LocalizationServiceConfig(),
    ) -> None:
        config.validate()
        self._engine = engine
        self._time_source = time_source
        self._config = config
        self._state = LocalizationServiceState.CREATED
        self._ticks_completed = 0
        self._skipped_tick_slots = 0
        self._started_at: TimePoint | None = None
        self._last_tick_at: TimePoint | None = None
        self._detail_code: str | None = None
        self._last_error: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop_requested = asyncio.Event()
        self._terminated = asyncio.Event()
        self._lifecycle_lock = asyncio.Lock()

    def get_status(self) -> LocalizationServiceStatus:
        return LocalizationServiceStatus(
            state=self._state,
            tick_period_s=self._config.tick_period_s,
            ticks_completed=self._ticks_completed,
            skipped_tick_slots=self._skipped_tick_slots,
            started_at=self._started_at,
            last_tick_at=self._last_tick_at,
            detail_code=self._detail_code,
            last_error=self._last_error,
        )

    async def start(self) -> None:
        """Starts exactly one tick task; a stopped instance is not reusable."""

        async with self._lifecycle_lock:
            if self._state != LocalizationServiceState.CREATED:
                raise RuntimeError(
                    "localization service can only start from CREATED"
                )
            started_at = self._time_source.now()
            self._validate_time(started_at, previous=None)
            self._started_at = started_at
            self._state = LocalizationServiceState.RUNNING
            self._detail_code = "running"
            self._task = asyncio.create_task(
                self._run(),
                name="continuous-localization-service",
            )

    async def stop(self) -> None:
        """Stops after the active tick, cancelling only after timeout."""

        async with self._lifecycle_lock:
            if self._state == LocalizationServiceState.CREATED:
                self._state = LocalizationServiceState.STOPPED
                self._detail_code = "stopped_before_start"
                self._terminated.set()
                return
            if self._state in (
                LocalizationServiceState.STOPPED,
                LocalizationServiceState.FAULTED,
            ):
                return
            self._state = LocalizationServiceState.STOPPING
            self._detail_code = "stop_requested"
            self._stop_requested.set()
            task = self._task

        if task is None:
            raise RuntimeError("running localization service has no task")
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self._config.stop_timeout_s,
            )
        except TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            self._state = LocalizationServiceState.STOPPED
            self._detail_code = "stop_timeout_cancelled"
            self._terminated.set()

    async def wait_stopped(
        self,
        timeout_s: float | None = None,
    ) -> LocalizationServiceStatus:
        """Waits for normal stop or fault without changing lifecycle state."""

        if timeout_s is not None:
            if not math.isfinite(timeout_s) or timeout_s < 0.0:
                raise ValueError("timeout_s must be finite and non-negative")
            await asyncio.wait_for(self._terminated.wait(), timeout=timeout_s)
        else:
            await self._terminated.wait()
        return self.get_status()

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        next_deadline = loop.time()
        try:
            while not self._stop_requested.is_set():
                tick_at = self._time_source.now()
                previous_tick = self._last_tick_at or self._started_at
                self._validate_time(tick_at, previous=previous_tick)
                self._last_tick_at = tick_at
                await self._engine.tick(tick_at)
                self._ticks_completed += 1

                if self._stop_requested.is_set():
                    break
                next_deadline += self._config.tick_period_s
                current_time = loop.time()
                if current_time > next_deadline:
                    skipped_slots = (
                        math.floor(
                            (current_time - next_deadline)
                            / self._config.tick_period_s
                        )
                        + 1
                    )
                    self._skipped_tick_slots += skipped_slots
                    next_deadline += (
                        skipped_slots * self._config.tick_period_s
                    )
                delay_s = max(0.0, next_deadline - loop.time())
                try:
                    await asyncio.wait_for(
                        self._stop_requested.wait(),
                        timeout=delay_s,
                    )
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._state = LocalizationServiceState.FAULTED
            self._detail_code = f"tick_failed:{type(error).__name__}"
            self._last_error = str(error)
        finally:
            if self._state in (
                LocalizationServiceState.RUNNING,
                LocalizationServiceState.STOPPING,
            ):
                self._state = LocalizationServiceState.STOPPED
                self._detail_code = "stopped"
            self._terminated.set()

    def _validate_time(
        self,
        value: TimePoint,
        *,
        previous: TimePoint | None,
    ) -> None:
        if not value.clock_id.strip():
            raise ValueError("localization clock_id must not be empty")
        if self._started_at is not None:
            if value.clock_id != self._started_at.clock_id:
                raise ValueError("localization time source changed clock domain")
        if previous is not None:
            if value.nanoseconds < previous.nanoseconds:
                raise ValueError("localization time source moved backward")

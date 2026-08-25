"""Localization-driven scheduling for a Local Trajectory Engine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
import math
from typing import Protocol

from longship.navigation.common import TimePoint
from longship.navigation.local_trajectory_engine.models import (
    LocalTrajectoryPublication,
)
from longship.navigation.localization_engine.interface import LocalizationEngine
from longship.navigation.localization_engine.models import (
    BeliefRevision,
    BeliefUpdateOutcome,
    WaitForUpdateRequest,
)


class LocalTrajectoryServiceState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAULTED = "faulted"


@dataclass(frozen=True, slots=True)
class LocalTrajectoryServiceConfig:
    belief_wait_timeout_s: float = 0.25
    stop_timeout_s: float = 5.0

    def validate(self) -> None:
        values = (self.belief_wait_timeout_s, self.stop_timeout_s)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError(
                "local trajectory service durations must be positive"
            )


@dataclass(frozen=True, slots=True)
class LocalTrajectoryServiceStatus:
    state: LocalTrajectoryServiceState
    ticks_completed: int
    coalesced_belief_updates: int
    detail_code: str | None = None
    last_error: str | None = None


class LocalTrajectoryTimeSource(Protocol):
    def now(self) -> TimePoint: ...


class _LocalTrajectoryTickEngine(Protocol):
    async def tick(self, now: TimePoint) -> LocalTrajectoryPublication: ...

    async def stop(self, now: TimePoint) -> LocalTrajectoryPublication: ...

    async def fault(
        self,
        now: TimePoint,
        detail_code: str,
    ) -> LocalTrajectoryPublication: ...


class LocalizationDrivenLocalTrajectoryService:
    """Ticks the engine after belief updates while prioritizing localization."""

    def __init__(
        self,
        *,
        engine: _LocalTrajectoryTickEngine,
        localization_engine: LocalizationEngine,
        time_source: LocalTrajectoryTimeSource,
        config: LocalTrajectoryServiceConfig = LocalTrajectoryServiceConfig(),
    ) -> None:
        config.validate()
        self._engine = engine
        self._localization_engine = localization_engine
        self._time_source = time_source
        self._config = config
        self._state = LocalTrajectoryServiceState.CREATED
        self._ticks_completed = 0
        self._coalesced_belief_updates = 0
        self._detail_code: str | None = None
        self._last_error: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop_requested = asyncio.Event()
        self._terminated = asyncio.Event()

    def get_status(self) -> LocalTrajectoryServiceStatus:
        return LocalTrajectoryServiceStatus(
            state=self._state,
            ticks_completed=self._ticks_completed,
            coalesced_belief_updates=self._coalesced_belief_updates,
            detail_code=self._detail_code,
            last_error=self._last_error,
        )

    async def start(self) -> None:
        if self._state != LocalTrajectoryServiceState.CREATED:
            raise RuntimeError("local trajectory service is not reusable")
        self._state = LocalTrajectoryServiceState.RUNNING
        self._detail_code = "running"
        self._task = asyncio.create_task(
            self._run(),
            name="localization-driven-local-trajectory-service",
        )

    async def stop(self) -> None:
        if self._state == LocalTrajectoryServiceState.CREATED:
            self._state = LocalTrajectoryServiceState.STOPPED
            self._detail_code = "stopped_before_start"
            await self._engine.stop(self._time_source.now())
            self._terminated.set()
            return
        if self._state in (
            LocalTrajectoryServiceState.STOPPED,
            LocalTrajectoryServiceState.FAULTED,
        ):
            return
        self._state = LocalTrajectoryServiceState.STOPPING
        self._stop_requested.set()
        if self._task is None:
            raise RuntimeError("running local trajectory service has no task")
        try:
            await asyncio.wait_for(
                asyncio.shield(self._task),
                timeout=self._config.stop_timeout_s,
            )
        except TimeoutError:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        await self._engine.stop(self._time_source.now())
        self._state = LocalTrajectoryServiceState.STOPPED
        self._detail_code = "stopped"
        self._terminated.set()

    async def wait_stopped(
        self,
        timeout_s: float | None = None,
    ) -> LocalTrajectoryServiceStatus:
        if timeout_s is None:
            await self._terminated.wait()
        else:
            await asyncio.wait_for(self._terminated.wait(), timeout=timeout_s)
        return self.get_status()

    async def _run(self) -> None:
        revision: BeliefRevision = (
            self._localization_engine.get_belief().revision
        )
        try:
            await self._engine.tick(self._time_source.now())
            self._ticks_completed += 1
            while not self._stop_requested.is_set():
                update = await self._localization_engine.wait_for_update(
                    WaitForUpdateRequest(
                        after_revision=revision,
                        timeout_s=self._config.belief_wait_timeout_s,
                    )
                )
                if update.outcome == BeliefUpdateOutcome.TIMED_OUT:
                    continue
                next_revision = update.belief.revision
                if (
                    update.outcome == BeliefUpdateOutcome.UPDATED
                    and next_revision.stream_id == revision.stream_id
                ):
                    self._coalesced_belief_updates += max(
                        0,
                        next_revision.sequence - revision.sequence - 1,
                    )
                revision = next_revision
                await self._engine.tick(self._time_source.now())
                self._ticks_completed += 1
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._state = LocalTrajectoryServiceState.FAULTED
            self._detail_code = f"tick_failed:{type(error).__name__}"
            self._last_error = str(error)
            await self._engine.fault(
                self._time_source.now(),
                self._detail_code,
            )
        finally:
            self._terminated.set()

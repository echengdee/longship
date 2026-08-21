"""Tests for the system-owned continuous Localization supervisor."""

from __future__ import annotations

import asyncio
import unittest

from longship.navigation.common import TimePoint
from longship.navigation.localization_engine.models import (
    BeliefRevision,
    BeliefStreamId,
    LocalizationStatus,
    LocationBelief,
)
from longship.navigation.localization_engine.service import (
    ContinuousLocalizationService,
    LocalizationServiceConfig,
    LocalizationServiceState,
)
from longship.navigation.map_engine.models import SnapshotId


class _StepTimeSource:
    def __init__(self) -> None:
        self._next_nanoseconds = 0

    def now(self) -> TimePoint:
        value = TimePoint(
            clock_id="test-clock",
            nanoseconds=self._next_nanoseconds,
        )
        self._next_nanoseconds += 1_000_000
        return value


class _ScriptedTimeSource:
    def __init__(self, nanoseconds: list[int]) -> None:
        self._nanoseconds = iter(nanoseconds)

    def now(self) -> TimePoint:
        return TimePoint(
            clock_id="test-clock",
            nanoseconds=next(self._nanoseconds),
        )


class _RecordingEngine:
    def __init__(
        self,
        *,
        signal_after: int = 1,
        delay_s: float = 0.0,
        fail_on_attempt: int | None = None,
    ) -> None:
        self.signal_after = signal_after
        self.delay_s = delay_s
        self.fail_on_attempt = fail_on_attempt
        self.tick_times: list[TimePoint] = []
        self.attempts = 0
        self.active_ticks = 0
        self.maximum_active_ticks = 0
        self.signaled = asyncio.Event()

    async def tick(self, now: TimePoint) -> LocationBelief:
        self.attempts += 1
        self.active_ticks += 1
        self.maximum_active_ticks = max(
            self.maximum_active_ticks,
            self.active_ticks,
        )
        try:
            if self.fail_on_attempt == self.attempts:
                raise RuntimeError("scripted tick failure")
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            self.tick_times.append(now)
            if len(self.tick_times) >= self.signal_after:
                self.signaled.set()
            return LocationBelief(
                snapshot_id=SnapshotId("test-snapshot"),
                revision=BeliefRevision(
                    stream_id=BeliefStreamId("test-stream"),
                    sequence=len(self.tick_times),
                ),
                estimate_time=now,
                published_at=now,
                status=LocalizationStatus.INITIALIZING,
                confidence=None,
            )
        finally:
            self.active_ticks -= 1


class _BlockedEngine(_RecordingEngine):
    def __init__(self) -> None:
        super().__init__()
        self.tick_started = asyncio.Event()
        self.release_tick = asyncio.Event()

    async def tick(self, now: TimePoint) -> LocationBelief:
        self.tick_started.set()
        await self.release_tick.wait()
        return await super().tick(now)


class ContinuousLocalizationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_periodically_without_overlapping_ticks(self) -> None:
        engine = _RecordingEngine(signal_after=3)
        service = ContinuousLocalizationService(
            engine=engine,
            time_source=_StepTimeSource(),
            config=LocalizationServiceConfig(
                tick_period_s=0.005,
                stop_timeout_s=0.5,
            ),
        )

        self.assertEqual(
            service.get_status().state,
            LocalizationServiceState.CREATED,
        )
        await service.start()
        await asyncio.wait_for(engine.signaled.wait(), timeout=0.5)
        await service.stop()

        status = service.get_status()
        self.assertEqual(status.state, LocalizationServiceState.STOPPED)
        self.assertGreaterEqual(status.ticks_completed, 3)
        self.assertEqual(engine.maximum_active_ticks, 1)
        self.assertEqual(
            engine.tick_times,
            sorted(engine.tick_times, key=lambda value: value.nanoseconds),
        )
        with self.assertRaisesRegex(RuntimeError, "only start from CREATED"):
            await service.start()

    async def test_skips_slots_instead_of_overlapping_slow_ticks(self) -> None:
        engine = _RecordingEngine(signal_after=2, delay_s=0.02)
        service = ContinuousLocalizationService(
            engine=engine,
            time_source=_StepTimeSource(),
            config=LocalizationServiceConfig(
                tick_period_s=0.005,
                stop_timeout_s=0.5,
            ),
        )

        await service.start()
        await asyncio.wait_for(engine.signaled.wait(), timeout=0.5)
        await service.stop()

        status = service.get_status()
        self.assertGreater(status.skipped_tick_slots, 0)
        self.assertEqual(engine.maximum_active_ticks, 1)

    async def test_faults_on_an_unhandled_tick_failure(self) -> None:
        engine = _RecordingEngine(fail_on_attempt=2)
        service = ContinuousLocalizationService(
            engine=engine,
            time_source=_StepTimeSource(),
            config=LocalizationServiceConfig(tick_period_s=0.005),
        )

        await service.start()
        status = await service.wait_stopped(timeout_s=0.5)

        self.assertEqual(status.state, LocalizationServiceState.FAULTED)
        self.assertEqual(status.ticks_completed, 1)
        self.assertEqual(status.detail_code, "tick_failed:RuntimeError")
        self.assertEqual(status.last_error, "scripted tick failure")
        await service.stop()

    async def test_faults_when_first_tick_precedes_service_start(self) -> None:
        service = ContinuousLocalizationService(
            engine=_RecordingEngine(),
            time_source=_ScriptedTimeSource([10, 9]),
            config=LocalizationServiceConfig(tick_period_s=0.005),
        )

        await service.start()
        status = await service.wait_stopped(timeout_s=0.5)

        self.assertEqual(status.state, LocalizationServiceState.FAULTED)
        self.assertEqual(status.detail_code, "tick_failed:ValueError")
        self.assertEqual(
            status.last_error,
            "localization time source moved backward",
        )

    async def test_stop_waits_for_the_active_tick(self) -> None:
        engine = _BlockedEngine()
        service = ContinuousLocalizationService(
            engine=engine,
            time_source=_StepTimeSource(),
            config=LocalizationServiceConfig(stop_timeout_s=0.5),
        )

        await service.start()
        await asyncio.wait_for(engine.tick_started.wait(), timeout=0.5)
        stop_task = asyncio.create_task(service.stop())
        await asyncio.sleep(0)
        self.assertEqual(
            service.get_status().state,
            LocalizationServiceState.STOPPING,
        )
        self.assertFalse(stop_task.done())

        engine.release_tick.set()
        await asyncio.wait_for(stop_task, timeout=0.5)

        self.assertEqual(
            service.get_status().state,
            LocalizationServiceState.STOPPED,
        )
        self.assertEqual(service.get_status().ticks_completed, 1)

    async def test_stop_before_start_is_terminal(self) -> None:
        service = ContinuousLocalizationService(
            engine=_RecordingEngine(),
            time_source=_StepTimeSource(),
        )

        await service.stop()

        self.assertEqual(
            service.get_status().state,
            LocalizationServiceState.STOPPED,
        )
        with self.assertRaisesRegex(RuntimeError, "only start from CREATED"):
            await service.start()


if __name__ == "__main__":
    unittest.main()

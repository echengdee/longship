"""Tests for localization-driven Local Trajectory Engine scheduling."""

from __future__ import annotations

import asyncio
import unittest

from longship.navigation.common import TimePoint
from longship.navigation.localization_engine.models import (
    BeliefRevision,
    BeliefStreamId,
    BeliefUpdateOutcome,
    BeliefUpdateResult,
    LocationBelief,
    LocalizationStatus,
    WaitForUpdateRequest,
)
from longship.navigation.map_engine.models import SnapshotId
from longship.navigation.runtime import (
    LocalizationDrivenLocalTrajectoryService,
    LocalTrajectoryServiceConfig,
    LocalTrajectoryServiceState,
)


class _LocalizationUpdates:
    def __init__(self) -> None:
        self._belief = _belief(0)
        self._updates: asyncio.Queue[LocationBelief] = asyncio.Queue()

    def get_belief(self) -> LocationBelief:
        return self._belief

    async def wait_for_update(
        self,
        request: WaitForUpdateRequest,
    ) -> BeliefUpdateResult:
        try:
            belief = await asyncio.wait_for(
                self._updates.get(),
                timeout=request.timeout_s,
            )
        except TimeoutError:
            return BeliefUpdateResult(
                outcome=BeliefUpdateOutcome.TIMED_OUT,
                belief=self._belief,
            )
        return BeliefUpdateResult(
            outcome=BeliefUpdateOutcome.UPDATED,
            belief=belief,
        )

    async def publish(self, sequence: int) -> None:
        self._belief = _belief(sequence)
        await self._updates.put(self._belief)


class _RecordingPublisher:
    def __init__(self) -> None:
        self.ticks: list[TimePoint] = []
        self.stopped = False
        self.faults: list[str] = []

    async def tick(self, now: TimePoint) -> None:
        self.ticks.append(now)

    async def stop(self, now: TimePoint) -> None:
        del now
        self.stopped = True

    async def fault(self, now: TimePoint, detail_code: str) -> None:
        del now
        self.faults.append(detail_code)


class _Clock:
    def __init__(self) -> None:
        self._sequence = 0

    def now(self) -> TimePoint:
        self._sequence += 1
        return TimePoint(
            clock_id="test",
            nanoseconds=self._sequence,
        )


class LocalizationDrivenLocalTrajectoryServiceTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_ticks_after_beliefs_and_reports_coalescing(self) -> None:
        localization = _LocalizationUpdates()
        publisher = _RecordingPublisher()
        service = LocalizationDrivenLocalTrajectoryService(
            engine=publisher,
            localization_engine=localization,
            time_source=_Clock(),
            config=LocalTrajectoryServiceConfig(
                belief_wait_timeout_s=0.01,
            ),
        )

        await service.start()
        await _wait_for_tick_count(publisher, 1)
        await localization.publish(3)
        await _wait_for_tick_count(publisher, 2)
        await service.stop()

        status = service.get_status()
        self.assertEqual(status.state, LocalTrajectoryServiceState.STOPPED)
        self.assertEqual(status.ticks_completed, 2)
        self.assertEqual(status.coalesced_belief_updates, 2)
        self.assertTrue(publisher.stopped)
        self.assertFalse(publisher.faults)


async def _wait_for_tick_count(
    publisher: _RecordingPublisher,
    expected: int,
) -> None:
    async with asyncio.timeout(0.5):
        while len(publisher.ticks) < expected:
            await asyncio.sleep(0)


def _belief(sequence: int) -> LocationBelief:
    now = TimePoint(clock_id="test", nanoseconds=sequence)
    return LocationBelief(
        snapshot_id=SnapshotId("snapshot"),
        revision=BeliefRevision(
            stream_id=BeliefStreamId("belief"),
            sequence=sequence,
        ),
        estimate_time=now,
        published_at=now,
        status=LocalizationStatus.TRACKING,
        confidence=None,
    )


if __name__ == "__main__":
    unittest.main()

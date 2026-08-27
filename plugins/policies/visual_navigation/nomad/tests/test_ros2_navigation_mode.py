"""Unit tests for the stable mode-level NoMaD trajectory stream."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
import unittest

from longship.navigation.common import TimePoint
from longship.navigation.local_trajectory_engine import (
    LocalTrajectoryPublication,
    LocalTrajectoryRevision,
    LocalTrajectoryState,
    LocalTrajectoryStreamId,
    LocalTrajectoryUpdateOutcome,
    LocalTrajectoryUpdateResult,
    WaitForLocalTrajectoryRequest,
)
from longship.navigation.localization_engine.service import MonotonicTimeSource
from longship.navigation.map_engine.models import SnapshotId
from longship.navigation.planning_engine.models import RouteId

from longship_adapter.ros2_navigation_mode import (
    NomadRos2NavigationModeConfig,
    _ModeTrajectoryStream,
)


class _SourceStream:
    def __init__(self, publication: LocalTrajectoryPublication) -> None:
        self._latest = publication
        self._condition = asyncio.Condition()

    def get_latest(self) -> LocalTrajectoryPublication:
        return self._latest

    async def publish(self, publication: LocalTrajectoryPublication) -> None:
        async with self._condition:
            self._latest = publication
            self._condition.notify_all()

    async def wait_for_update(
        self,
        request: WaitForLocalTrajectoryRequest,
    ) -> LocalTrajectoryUpdateResult:
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._latest.revision.sequence
                > request.after_revision.sequence
            )
        return LocalTrajectoryUpdateResult(
            outcome=LocalTrajectoryUpdateOutcome.UPDATED,
            publication=self._latest,
        )


def _publication(
    *,
    sequence: int,
    state: LocalTrajectoryState,
) -> LocalTrajectoryPublication:
    return LocalTrajectoryPublication(
        revision=LocalTrajectoryRevision(
            stream_id=LocalTrajectoryStreamId("route"),
            sequence=sequence,
        ),
        route_id=RouteId("route"),
        snapshot_id=SnapshotId("snapshot"),
        state=state,
        published_at=TimePoint(clock_id="test", nanoseconds=sequence),
    )


class ModeTrajectoryStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_forwards_route_updates_with_its_own_stable_revision(self) -> None:
        clock = MonotonicTimeSource(clock_id="test")
        stream = _ModeTrajectoryStream(
            stream_id=LocalTrajectoryStreamId("mode"),
            snapshot_id=SnapshotId("snapshot"),
            clock=clock,
        )
        source = _SourceStream(
            _publication(sequence=0, state=LocalTrajectoryState.INITIALIZING)
        )

        await stream.attach(source)
        await asyncio.sleep(0)
        revision = stream.get_latest().revision
        await source.publish(
            _publication(sequence=1, state=LocalTrajectoryState.ACTIVE)
        )

        update = await stream.wait_for_update(
            WaitForLocalTrajectoryRequest(
                after_revision=revision,
                timeout_s=1.0,
            )
        )

        self.assertEqual(update.outcome, LocalTrajectoryUpdateOutcome.UPDATED)
        self.assertEqual(update.publication.state, LocalTrajectoryState.ACTIVE)
        self.assertEqual(update.publication.revision.stream_id, "mode")
        self.assertGreater(update.publication.revision.sequence, revision.sequence)
        await stream.close()

    async def test_pause_replaces_active_output_with_holding(self) -> None:
        clock = MonotonicTimeSource(clock_id="test")
        stream = _ModeTrajectoryStream(
            stream_id=LocalTrajectoryStreamId("mode"),
            snapshot_id=SnapshotId("snapshot"),
            clock=clock,
        )
        source = _SourceStream(
            _publication(sequence=0, state=LocalTrajectoryState.ACTIVE)
        )
        await stream.attach(source)

        await stream.set_paused(True)

        self.assertEqual(stream.get_latest().state, LocalTrajectoryState.HOLDING)
        self.assertEqual(stream.get_latest().detail_code, "paused")
        await stream.close()


class NomadRos2NavigationModeConfigTests(unittest.TestCase):
    def test_requires_a_nonempty_ros_color_topic(self) -> None:
        config = NomadRos2NavigationModeConfig(
            topomap_root=Path("map"),
            color_topic="",
            checkpoint_path=Path("nomad.pth"),
        )

        with self.assertRaisesRegex(ValueError, "color_topic"):
            config.validate()


if __name__ == "__main__":
    unittest.main()

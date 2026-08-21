"""Tests for decoded-frame sampling into the NoMaD policy context."""

from __future__ import annotations

from collections import deque
import unittest

from longship.navigation.runtime import (
    LocalizationObservationProducer,
    LocalizationObservationProducerStatus,
)
from longship_adapter import (
    DecodedObservationFrame,
    NomadObservationFanout,
    NomadObservationProducer,
    NomadObservationProducerConfig,
    NomadObservationProducerState,
)


_IMAGE_PROFILE_ID = "nomad.direct-resize.v1"


class _FrameSource:
    def __init__(self, frames: list[DecodedObservationFrame]) -> None:
        self._frames = deque(frames)
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def read(self) -> DecodedObservationFrame | None:
        if not self._frames:
            return None
        return self._frames.popleft()

    async def stop(self) -> None:
        self.stopped = True


class _RecordingObservationSink:
    def __init__(self) -> None:
        self.clear_count = 0
        self.submissions: list[tuple[object, float, dict[str, str]]] = []

    def clear_observations(self) -> None:
        self.clear_count += 1

    def submit_observation(
        self,
        image: object,
        timestamp_s: float,
        **representation: str,
    ) -> None:
        self.submissions.append((image, timestamp_s, representation))


def _frame(
    sequence_id: int,
    timestamp_s: float,
    *,
    image_profile_id: str = _IMAGE_PROFILE_ID,
    source_timestamp_s: float | None = None,
) -> DecodedObservationFrame:
    return DecodedObservationFrame(
        image=f"frame-{sequence_id}",
        timestamp_s=timestamp_s,
        source_timestamp_s=(
            timestamp_s + 6.0
            if source_timestamp_s is None
            else source_timestamp_s
        ),
        sequence_id=sequence_id,
        image_profile_id=image_profile_id,
    )


class NomadObservationProducerTests(unittest.IsolatedAsyncioTestCase):
    async def test_fanout_keeps_distance_and_trajectory_contexts_aligned(
        self,
    ) -> None:
        distance_sink = _RecordingObservationSink()
        trajectory_sink = _RecordingObservationSink()
        fanout = NomadObservationFanout(
            (distance_sink, trajectory_sink)
        )

        fanout.submit_observation(
            "frame",
            1.25,
            layout="hwc",
            channel_order="rgb",
            value_range="byte",
        )
        fanout.clear_observations()

        self.assertEqual(
            distance_sink.submissions,
            trajectory_sink.submissions,
        )
        self.assertEqual(distance_sink.clear_count, 1)
        self.assertEqual(trajectory_sink.clear_count, 1)

    async def test_samples_fast_source_without_building_a_context_backlog(
        self,
    ) -> None:
        source = _FrameSource(
            [
                _frame(0, 0.0),
                _frame(1, 0.03),
                _frame(2, 0.12),
                _frame(3, 0.15),
                _frame(4, 0.24),
            ]
        )
        sink = _RecordingObservationSink()
        producer = NomadObservationProducer(
            source=source,
            policy=sink,
            config=NomadObservationProducerConfig(
                image_profile_id=_IMAGE_PROFILE_ID,
                sample_hz=9.0,
            ),
        )
        self.assertIsInstance(producer, LocalizationObservationProducer)

        await producer.start()
        status = await producer.wait_stopped(timeout_s=0.5)
        await producer.stop()

        self.assertEqual(
            status.state,
            NomadObservationProducerState.COMPLETED,
        )
        self.assertIsInstance(
            status,
            LocalizationObservationProducerStatus,
        )
        self.assertEqual(status.frames_received, 5)
        self.assertEqual(status.frames_submitted, 3)
        self.assertEqual(status.frames_dropped_by_sampler, 2)
        self.assertEqual(
            [timestamp for _, timestamp, _ in sink.submissions],
            [0.0, 0.12, 0.24],
        )
        self.assertEqual(sink.clear_count, 1)
        self.assertTrue(source.started)
        self.assertTrue(source.stopped)

    async def test_resets_context_after_a_source_gap(self) -> None:
        source = _FrameSource(
            [
                _frame(0, 0.0),
                _frame(1, 0.1),
                _frame(2, 1.0),
            ]
        )
        sink = _RecordingObservationSink()
        producer = NomadObservationProducer(
            source=source,
            policy=sink,
            config=NomadObservationProducerConfig(
                image_profile_id=_IMAGE_PROFILE_ID,
                sample_hz=10.0,
                maximum_frame_gap_s=0.5,
            ),
        )

        await producer.start()
        status = await producer.wait_stopped(timeout_s=0.5)

        self.assertEqual(status.context_resets, 1)
        self.assertEqual(status.frames_submitted, 3)
        self.assertEqual(sink.clear_count, 2)

    async def test_uses_a_fixed_sample_grid_for_a_thirty_hertz_source(
        self,
    ) -> None:
        source = _FrameSource(
            [_frame(index, index / 30.0) for index in range(30)]
        )
        sink = _RecordingObservationSink()
        producer = NomadObservationProducer(
            source=source,
            policy=sink,
            config=NomadObservationProducerConfig(
                image_profile_id=_IMAGE_PROFILE_ID,
                sample_hz=9.0,
            ),
        )

        await producer.start()
        status = await producer.wait_stopped(timeout_s=0.5)

        self.assertEqual(status.frames_received, 30)
        self.assertEqual(status.frames_submitted, 9)
        self.assertEqual(status.frames_dropped_by_sampler, 21)

    async def test_recorded_source_sampling_ignores_delivery_jitter(
        self,
    ) -> None:
        source = _FrameSource(
            [
                _frame(0, 0.0, source_timestamp_s=10.0),
                _frame(1, 0.03, source_timestamp_s=10.03),
                _frame(2, 0.60, source_timestamp_s=10.12),
                _frame(3, 0.61, source_timestamp_s=10.15),
                _frame(4, 0.62, source_timestamp_s=10.24),
            ]
        )
        sink = _RecordingObservationSink()
        producer = NomadObservationProducer(
            source=source,
            policy=sink,
            config=NomadObservationProducerConfig(
                image_profile_id=_IMAGE_PROFILE_ID,
                sample_hz=9.0,
                maximum_frame_gap_s=0.5,
            ),
        )

        await producer.start()
        status = await producer.wait_stopped(timeout_s=0.5)

        self.assertEqual(status.context_resets, 0)
        self.assertEqual(
            [image for image, _, _ in sink.submissions],
            ["frame-0", "frame-2", "frame-4"],
        )

    async def test_faults_on_an_incompatible_image_profile(self) -> None:
        source = _FrameSource(
            [_frame(0, 0.0, image_profile_id="other-profile")]
        )
        sink = _RecordingObservationSink()
        producer = NomadObservationProducer(
            source=source,
            policy=sink,
            config=NomadObservationProducerConfig(
                image_profile_id=_IMAGE_PROFILE_ID,
            ),
        )

        await producer.start()
        status = await producer.wait_stopped(timeout_s=0.5)
        await producer.stop()

        self.assertEqual(
            status.state,
            NomadObservationProducerState.FAULTED,
        )
        self.assertEqual(status.detail_code, "observation_failed:ValueError")
        self.assertIn("image profile", status.last_error or "")
        self.assertFalse(sink.submissions)

    async def test_faults_when_source_timestamps_move_backward(self) -> None:
        source = _FrameSource(
            [
                _frame(0, 1.0, source_timestamp_s=10.0),
                _frame(1, 2.0, source_timestamp_s=9.0),
            ]
        )
        sink = _RecordingObservationSink()
        producer = NomadObservationProducer(
            source=source,
            policy=sink,
            config=NomadObservationProducerConfig(
                image_profile_id=_IMAGE_PROFILE_ID,
            ),
        )

        await producer.start()
        status = await producer.wait_stopped(timeout_s=0.5)

        self.assertEqual(
            status.state,
            NomadObservationProducerState.FAULTED,
        )
        self.assertIn("source frame timestamps", status.last_error or "")


if __name__ == "__main__":
    unittest.main()

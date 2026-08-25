"""Tests for the live NoMaD diagnostic video frame cache."""

from __future__ import annotations

import torch

from longship.navigation.common import TimePoint
from tools.live_trajectory_video import ObservationFrameCache


class _Sink:
    def __init__(self) -> None:
        self.clear_calls = 0

    def submit_observation(
        self,
        image: object,
        timestamp_s: float,
        *,
        layout: str = "chw",
        channel_order: str = "rgb",
        value_range: str = "auto",
    ) -> None:
        del image, timestamp_s, layout, channel_order, value_range

    def clear_observations(self) -> None:
        self.clear_calls += 1


def test_keeps_an_inflight_frame_across_a_policy_context_reset() -> None:
    sink = _Sink()
    cache = ObservationFrameCache(sink)
    cache.submit_observation(
        torch.zeros((1, 2, 3), dtype=torch.uint8),
        1.0,
        layout="hwc",
        channel_order="rgb",
        value_range="byte",
    )

    cache.clear_observations()

    assert sink.clear_calls == 1
    assert cache.take(
        TimePoint(clock_id="monotonic", nanoseconds=1_000_000_000)
    ) is not None

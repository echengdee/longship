"""Tests for the optional ROS 2 image ingress adapter."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

import torch

from tools.ros2_image_source import (
    Ros2ImageFrameSource,
    Ros2ImageFrameSourceConfig,
)


_PROFILE_ID = "nomad.rgb.direct_resize_96x96.imagenet.v1"


@dataclass(frozen=True, slots=True)
class _Stamp:
    sec: int
    nanosec: int


@dataclass(frozen=True, slots=True)
class _Header:
    stamp: _Stamp


@dataclass(frozen=True, slots=True)
class _Image:
    height: int
    width: int
    step: int
    encoding: str
    data: bytes
    header: _Header


class _Backend:
    def __init__(self) -> None:
        self.callback = None
        self.started = False
        self.stopped = False

    def start(self, callback: object) -> None:
        self.callback = callback
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def emit(self, image: _Image) -> None:
        if self.callback is None:
            raise RuntimeError("backend has not started")
        self.callback(image)


def _image(
    *,
    encoding: str = "rgb8",
    timestamp_ns: int = 1_000_000_000,
    step: int = 6,
    data: bytes = bytes((1, 2, 3, 4, 5, 6)),
) -> _Image:
    return _Image(
        height=1,
        width=2,
        step=step,
        encoding=encoding,
        data=data,
        header=_Header(
            stamp=_Stamp(
                sec=timestamp_ns // 1_000_000_000,
                nanosec=timestamp_ns % 1_000_000_000,
            )
        ),
    )


class Ros2ImageFrameSourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_decodes_rgb_image_and_preserves_source_timestamp(self) -> None:
        backend = _Backend()
        source = Ros2ImageFrameSource(
            Ros2ImageFrameSourceConfig(
                topic_name="/camera/camera/color/image_raw",
                image_profile_id=_PROFILE_ID,
            ),
            backend=backend,
        )

        await source.start()
        backend.emit(_image(timestamp_ns=1_250_000_000))
        frame = await source.read()
        await source.stop()

        self.assertTrue(backend.started)
        self.assertTrue(backend.stopped)
        self.assertEqual(frame.image_profile_id, _PROFILE_ID)
        self.assertEqual(frame.layout, "hwc")
        self.assertEqual(frame.channel_order, "rgb")
        self.assertEqual(frame.value_range, "byte")
        self.assertEqual(frame.sequence_id, 0)
        self.assertEqual(frame.source_timestamp_s, 1.25)
        self.assertIsInstance(frame.image, torch.Tensor)
        self.assertEqual(frame.image.shape, (1, 2, 3))
        self.assertEqual(frame.image.tolist(), [[[1, 2, 3], [4, 5, 6]]])

    async def test_preserves_bgr_encoding_for_nomad_preprocessing(self) -> None:
        backend = _Backend()
        source = Ros2ImageFrameSource(
            Ros2ImageFrameSourceConfig(
                topic_name="/camera/camera/color/image_raw",
                image_profile_id=_PROFILE_ID,
            ),
            backend=backend,
        )

        await source.start()
        backend.emit(_image(encoding="bgr8"))
        frame = await source.read()
        await source.stop()

        self.assertEqual(frame.channel_order, "bgr")

    async def test_uses_the_latest_frame_when_callbacks_outpace_reads(self) -> None:
        backend = _Backend()
        source = Ros2ImageFrameSource(
            Ros2ImageFrameSourceConfig(
                topic_name="/camera/camera/color/image_raw",
                image_profile_id=_PROFILE_ID,
            ),
            backend=backend,
        )

        await source.start()
        backend.emit(_image(timestamp_ns=1_000_000_000))
        backend.emit(
            _image(
                timestamp_ns=1_033_000_000,
                data=bytes((7, 8, 9, 10, 11, 12)),
            )
        )
        frame = await source.read()
        await source.stop()

        self.assertEqual(frame.sequence_id, 0)
        self.assertEqual(frame.source_timestamp_s, 1.033)
        self.assertEqual(frame.image.tolist(), [[[7, 8, 9], [10, 11, 12]]])

    async def test_rejects_unsupported_encoding(self) -> None:
        backend = _Backend()
        source = Ros2ImageFrameSource(
            Ros2ImageFrameSourceConfig(
                topic_name="/camera/camera/color/image_raw",
                image_profile_id=_PROFILE_ID,
            ),
            backend=backend,
        )

        await source.start()
        backend.emit(_image(encoding="mono8"))
        with self.assertRaisesRegex(ValueError, "encoding"):
            await source.read()
        await source.stop()

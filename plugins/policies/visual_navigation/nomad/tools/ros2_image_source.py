"""ROS 2 image ingress for live NoMaD observation production.

This module deliberately lives in ``tools`` rather than the Longship engine
packages.  ROS 2 owns transport and device integration; the NoMaD adapter only
receives decoded RGB tensors through ``DecodedObservationSource``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import math
from threading import Lock, Thread
import time
from typing import Protocol

import torch

from longship_adapter import DecodedObservationFrame


@dataclass(frozen=True, slots=True)
class Ros2ImageFrameSourceConfig:
    """Configuration for one ROS 2 ``sensor_msgs/Image`` subscription."""

    topic_name: str
    image_profile_id: str
    node_name: str = "nomad_image_source"
    qos_depth: int = 1

    def validate(self) -> None:
        if not self.topic_name.strip():
            raise ValueError("ROS 2 image topic name must not be empty")
        if not self.image_profile_id.strip():
            raise ValueError("image profile id must not be empty")
        if not self.node_name.strip():
            raise ValueError("ROS 2 node name must not be empty")
        if self.qos_depth <= 0:
            raise ValueError("ROS 2 image QoS depth must be positive")


class Ros2ImageBackend(Protocol):
    """Starts and stops one image-message subscription."""

    def start(self, callback: Callable[[object], None]) -> None:
        """Starts delivery of image messages to ``callback``."""
        ...

    def stop(self) -> None:
        """Stops message delivery and releases ROS resources."""
        ...


class Ros2ImageFrameSource:
    """Turns a live ROS 2 RGB topic into decoded NoMaD observation frames.

    The ROS callback runs on a dedicated executor thread.  It forwards only
    the latest received message to the asyncio loop, so a temporarily busy
    NoMaD consumer does not accumulate an unbounded camera-frame backlog.

    Frame ``timestamp_s`` uses local ``time.monotonic()`` because it must share
    the policy clock domain.  The ROS header timestamp is preserved separately
    as ``source_timestamp_s`` for sampling and gap detection.
    """

    def __init__(
        self,
        config: Ros2ImageFrameSourceConfig,
        *,
        backend: Ros2ImageBackend | None = None,
    ) -> None:
        config.validate()
        self._config = config
        self._backend = backend or _RclpyImageBackend(config)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._frame_available: asyncio.Event | None = None
        self._latest_message: object | None = None
        self._next_sequence_id = 0
        self._last_source_timestamp_s: float | None = None
        self._running = False
        self._stopped = False
        self._callback_error: Exception | None = None

    async def start(self) -> None:
        if self._running:
            raise RuntimeError("ROS 2 image source is already running")
        self._loop = asyncio.get_running_loop()
        self._frame_available = asyncio.Event()
        self._running = True
        self._stopped = False
        self._callback_error = None
        try:
            await asyncio.to_thread(self._backend.start, self._on_message)
        except BaseException:
            self._running = False
            self._stopped = True
            raise

    async def read(self) -> DecodedObservationFrame | None:
        event = self._frame_available
        if not self._running or event is None:
            raise RuntimeError("ROS 2 image source is not running")
        while True:
            await event.wait()
            if self._callback_error is not None:
                raise RuntimeError("ROS 2 image callback failed") from (
                    self._callback_error
                )
            message = self._latest_message
            self._latest_message = None
            event.clear()
            if message is not None:
                return self._decode_message(message)
            if self._stopped:
                return None

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._running = False
        try:
            await asyncio.to_thread(self._backend.stop)
        finally:
            event = self._frame_available
            if event is not None:
                event.set()

    def _on_message(self, message: object) -> None:
        loop = self._loop
        if loop is None or self._stopped:
            return
        loop.call_soon_threadsafe(self._accept_message, message)

    def _accept_message(self, message: object) -> None:
        if self._stopped:
            return
        try:
            source_timestamp_s = _header_timestamp_s(message)
            if (
                source_timestamp_s is not None
                and self._last_source_timestamp_s is not None
                and source_timestamp_s <= self._last_source_timestamp_s
            ):
                return
            if source_timestamp_s is not None:
                self._last_source_timestamp_s = source_timestamp_s
            self._latest_message = message
            if self._frame_available is not None:
                self._frame_available.set()
        except Exception as error:
            self._callback_error = error
            if self._frame_available is not None:
                self._frame_available.set()

    def _decode_message(self, message: object) -> DecodedObservationFrame:
        try:
            height = int(getattr(message, "height"))
            width = int(getattr(message, "width"))
            step = int(getattr(message, "step"))
            encoding = str(getattr(message, "encoding")).lower()
            payload = getattr(message, "data")
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("ROS image message is malformed") from error
        if height <= 0 or width <= 0:
            raise ValueError("ROS image dimensions must be positive")
        channel_order = _channel_order(encoding)
        row_bytes = width * 3
        if step < row_bytes:
            raise ValueError("ROS image step is smaller than RGB row size")
        if len(payload) < height * step:
            raise ValueError("ROS image payload is shorter than declared size")

        image = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        image = image[: height * step].view(height, step)
        image = image[:, :row_bytes].view(height, width, 3)
        sequence_id = self._next_sequence_id
        self._next_sequence_id += 1
        return DecodedObservationFrame(
            image=image,
            timestamp_s=time.monotonic(),
            source_timestamp_s=_header_timestamp_s(message),
            sequence_id=sequence_id,
            image_profile_id=self._config.image_profile_id,
            layout="hwc",
            channel_order=channel_order,
            value_range="byte",
        )


class _RclpyImageBackend:
    """Lazy rclpy implementation that keeps ROS optional for unit tests."""

    def __init__(self, config: Ros2ImageFrameSourceConfig) -> None:
        self._config = config
        self._context: object | None = None
        self._executor: object | None = None
        self._node: object | None = None
        self._thread: Thread | None = None
        self._rclpy: object | None = None
        self._lock = Lock()

    def start(self, callback: Callable[[object], None]) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("ROS 2 image backend is already running")
            try:
                import rclpy
                from rclpy.context import Context
                from rclpy.executors import SingleThreadedExecutor
                from rclpy.qos import (
                    DurabilityPolicy,
                    HistoryPolicy,
                    QoSProfile,
                    ReliabilityPolicy,
                )
                from sensor_msgs.msg import Image
            except ImportError as error:
                raise RuntimeError(
                    "rclpy and sensor_msgs are required for live ROS 2 input"
                ) from error

            context = Context()
            rclpy.init(context=context)
            try:
                node = rclpy.create_node(
                    self._config.node_name,
                    context=context,
                )
                qos = QoSProfile(
                    depth=self._config.qos_depth,
                    history=HistoryPolicy.KEEP_LAST,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.VOLATILE,
                )
                node.create_subscription(
                    Image,
                    self._config.topic_name,
                    callback,
                    qos,
                )
                executor = SingleThreadedExecutor(context=context)
                executor.add_node(node)
            except BaseException:
                rclpy.shutdown(context=context)
                raise
            thread = Thread(
                target=executor.spin,
                name="nomad-ros2-image-source",
                daemon=True,
            )
            self._rclpy = rclpy
            self._context = context
            self._node = node
            self._executor = executor
            self._thread = thread
            thread.start()

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            executor = self._executor
            node = self._node
            context = self._context
            rclpy = self._rclpy
            self._thread = None
            self._executor = None
            self._node = None
            self._context = None
            self._rclpy = None
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy is not None and context is not None:
            rclpy.shutdown(context=context)
        if thread is not None:
            thread.join(timeout=2.0)


def _channel_order(encoding: str) -> str:
    if encoding == "rgb8":
        return "rgb"
    if encoding == "bgr8":
        return "bgr"
    raise ValueError(
        "ROS image encoding must be 'rgb8' or 'bgr8', "
        f"got {encoding!r}"
    )


def _header_timestamp_s(message: object) -> float | None:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    try:
        seconds = int(stamp.sec)
        nanoseconds = int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("ROS image header stamp is malformed") from error
    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        raise ValueError("ROS image header stamp is outside the valid range")
    timestamp_s = seconds + nanoseconds / 1_000_000_000.0
    if timestamp_s == 0.0:
        return None
    if not math.isfinite(timestamp_s):
        raise ValueError("ROS image header timestamp must be finite")
    return timestamp_s

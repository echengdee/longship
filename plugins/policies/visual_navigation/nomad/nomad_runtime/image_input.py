"""Tensor-only image ingress for NoMaD inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from threading import Lock

import torch


class ObservationContextError(RuntimeError):
    """Base class for observation-context availability failures."""


class ObservationContextNotReadyError(ObservationContextError):
    """Raised before the configured number of frames is available."""


class StaleObservationContextError(ObservationContextError):
    """Raised when the latest frame exceeds the caller's age limit."""


class ImageLayout(str, Enum):
    """Supported decoded image tensor layouts."""

    CHW = "chw"
    HWC = "hwc"


class ChannelOrder(str, Enum):
    """Supported color channel orders."""

    RGB = "rgb"
    BGR = "bgr"


class ImageValueRange(str, Enum):
    """How numeric image values should be interpreted."""

    AUTO = "auto"
    UNIT = "unit"
    BYTE = "byte"


@dataclass(frozen=True)
class ImageTensorSpec:
    """Describes the layout and numeric representation of an image tensor."""

    layout: ImageLayout = ImageLayout.CHW
    channel_order: ChannelOrder = ChannelOrder.RGB
    value_range: ImageValueRange = ImageValueRange.AUTO

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout", ImageLayout(self.layout))
        object.__setattr__(
            self,
            "channel_order",
            ChannelOrder(self.channel_order),
        )
        object.__setattr__(
            self,
            "value_range",
            ImageValueRange(self.value_range),
        )


@dataclass(frozen=True)
class ImageFrame:
    """One canonical RGB frame and its caller-provided timestamp."""

    image: torch.Tensor
    timestamp_s: float


@dataclass(frozen=True)
class ObservationContext:
    """A chronological NoMaD observation context."""

    images: torch.Tensor
    timestamps_s: tuple[float, ...]

    @property
    def latest_timestamp_s(self) -> float:
        return self.timestamps_s[-1]


def canonicalize_image(
    image: torch.Tensor,
    spec: ImageTensorSpec = ImageTensorSpec(),
) -> torch.Tensor:
    """Converts one decoded image to contiguous RGB ``float32`` CHW [0, 1]."""
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if image.ndim != 3:
        raise ValueError("image must be a three-dimensional CHW or HWC tensor")

    channel_axis = 0 if spec.layout == ImageLayout.CHW else 2
    if image.shape[channel_axis] != 3:
        raise ValueError("image must contain exactly three color channels")
    if any(dimension <= 0 for dimension in image.shape):
        raise ValueError("image dimensions must be positive")
    if image.dtype != torch.uint8 and not image.is_floating_point():
        raise TypeError("image dtype must be uint8 or floating point")

    chw_image = (
        image
        if spec.layout == ImageLayout.CHW
        else image.permute(2, 0, 1)
    )
    if spec.channel_order == ChannelOrder.BGR:
        chw_image = chw_image[[2, 1, 0], ...]

    value_range = spec.value_range
    if value_range == ImageValueRange.AUTO:
        value_range = (
            ImageValueRange.BYTE
            if image.dtype == torch.uint8
            else ImageValueRange.UNIT
        )

    float_image = chw_image.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(float_image).all().item()):
        raise ValueError("image values must be finite")

    minimum = float(float_image.amin().item())
    maximum = float(float_image.amax().item())
    upper_bound = 1.0 if value_range == ImageValueRange.UNIT else 255.0
    if minimum < 0.0 or maximum > upper_bound:
        raise ValueError(
            f"image values must be in [0, {upper_bound:g}] for "
            f"value_range={value_range.value!r}"
        )
    if value_range == ImageValueRange.BYTE:
        float_image = float_image.div(255.0)

    return float_image.contiguous()


class ObservationBuffer:
    """Thread-safe chronological frame history for one NoMaD camera stream."""

    def __init__(
        self,
        context_frames: int = 4,
        history_frames: int | None = None,
    ) -> None:
        if context_frames <= 0:
            raise ValueError("context_frames must be positive")
        if history_frames is None:
            history_frames = context_frames
        if history_frames < context_frames:
            raise ValueError(
                "history_frames must not be smaller than context_frames"
            )
        self._context_frames = context_frames
        self._frames: deque[ImageFrame] = deque(maxlen=history_frames)
        self._lock = Lock()

    @property
    def context_frames(self) -> int:
        return self._context_frames

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def ready(self) -> bool:
        return self.size >= self._context_frames

    @property
    def latest_timestamp_s(self) -> float | None:
        with self._lock:
            if not self._frames:
                return None
            return self._frames[-1].timestamp_s

    def clear(self) -> None:
        """Clears all buffered frames, for example after a camera restart."""
        with self._lock:
            self._frames.clear()

    def append(
        self,
        image: torch.Tensor,
        timestamp_s: int | float,
        spec: ImageTensorSpec = ImageTensorSpec(),
    ) -> ImageFrame:
        """Converts and appends a frame with a strictly increasing timestamp."""
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        canonical_image = canonicalize_image(image, spec).clone()

        with self._lock:
            if self._frames and timestamp <= self._frames[-1].timestamp_s:
                raise ValueError("frame timestamps must be strictly increasing")
            if (
                self._frames
                and canonical_image.shape != self._frames[-1].image.shape
            ):
                raise ValueError(
                    "image resolution changed; clear the observation buffer "
                    "before appending the new stream"
                )
            frame = ImageFrame(canonical_image, timestamp)
            self._frames.append(frame)
            return frame

    def snapshot(
        self,
        *,
        now_s: int | float | None = None,
        max_age_s: int | float | None = None,
    ) -> ObservationContext:
        """Returns the latest complete context at or before ``now_s``."""
        if (now_s is None) != (max_age_s is None):
            raise ValueError("now_s and max_age_s must be provided together")

        current_time = None
        maximum_age = None
        if now_s is not None and max_age_s is not None:
            current_time = float(now_s)
            maximum_age = float(max_age_s)
            if not math.isfinite(current_time):
                raise ValueError("now_s must be finite")
            if not math.isfinite(maximum_age) or maximum_age < 0.0:
                raise ValueError("max_age_s must be finite and non-negative")

        with self._lock:
            eligible_frames = (
                tuple(self._frames)
                if current_time is None
                else tuple(
                    frame
                    for frame in self._frames
                    if frame.timestamp_s <= current_time
                )
            )
            if len(eligible_frames) < self._context_frames:
                raise ObservationContextNotReadyError(
                    "observation context is not ready: "
                    f"received {len(eligible_frames)} eligible of "
                    f"{self._context_frames} frames"
                )
            frames = eligible_frames[-self._context_frames :]

        if current_time is not None and maximum_age is not None:
            age = current_time - frames[-1].timestamp_s
            if age > maximum_age:
                raise StaleObservationContextError(
                    f"latest image is stale: age={age:.6f}s, "
                    f"maximum={maximum_age:.6f}s"
                )

        return ObservationContext(
            images=torch.stack([frame.image for frame in frames]),
            timestamps_s=tuple(frame.timestamp_s for frame in frames),
        )

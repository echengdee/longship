"""Live diagnostic video output for accepted NoMaD trajectory proposals.

The video is intentionally a diagnostic artifact.  It draws trajectories in
their declared policy-native robot frame as an inset, rather than projecting
un-calibrated policy coordinates into the RGB camera perspective.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import TypeVar

from PIL import Image
import torch

from longship.navigation.common import TimePoint
from longship.navigation.local_trajectory_engine import (
    LocalTrajectoryPublication,
    LocalTrajectoryState,
)
from longship.navigation.ports.trajectory_policy import (
    PolicyNativeWaypoint,
    TrajectoryCandidate,
    TrajectoryCandidateId,
    TrajectoryCandidateSet,
)
from longship_adapter import NomadObservationSink
from tools.trajectory_overlay import (
    TrajectoryOverlayState,
    draw_trajectory_overlay,
)


@dataclass(frozen=True, slots=True)
class _CachedFrame:
    image: object
    timestamp_s: float
    source_timestamp_s: float | None
    layout: str
    channel_order: str
    value_range: str


class ObservationFrameCache(NomadObservationSink):
    """Caches only frames that the wrapped NoMaD sink accepted.

    The cache is a non-owning diagnostic tap: observation delivery still first
    succeeds at the wrapped policy fanout, so a rendering failure cannot alter
    the inference input sequence.
    """

    def __init__(
        self, sink: NomadObservationSink, *, maximum_frames: int = 32
    ) -> None:
        if maximum_frames <= 0:
            raise ValueError("maximum_frames must be positive")
        self._sink = sink
        self._maximum_frames = maximum_frames
        self._frames: OrderedDict[int, _CachedFrame] = OrderedDict()

    def submit_observation(
        self,
        image: object,
        timestamp_s: float,
        *,
        layout: str = "chw",
        channel_order: str = "rgb",
        value_range: str = "auto",
    ) -> None:
        self._sink.submit_observation(
            image,
            timestamp_s,
            layout=layout,
            channel_order=channel_order,
            value_range=value_range,
        )
        key = _timestamp_key(timestamp_s)
        self._frames[key] = _CachedFrame(
            image=image,
            timestamp_s=timestamp_s,
            source_timestamp_s=None,
            layout=layout,
            channel_order=channel_order,
            value_range=value_range,
        )
        self._frames.move_to_end(key)
        while len(self._frames) > self._maximum_frames:
            self._frames.popitem(last=False)

    def clear_observations(self) -> None:
        self._sink.clear_observations()
        # A source gap resets model context, but an already-started inference
        # may still legally publish its exact pre-gap observation. Keep the
        # bounded render cache so diagnostics cannot affect the Harness.

    def attach_source_timestamp(
        self, observation_time_s: float, source_timestamp_s: float | None
    ) -> None:
        """Adds source timing after the producer has accepted a source frame."""

        key = _timestamp_key(observation_time_s)
        frame = self._frames.get(key)
        if frame is None:
            return
        self._frames[key] = _CachedFrame(
            image=frame.image,
            timestamp_s=frame.timestamp_s,
            source_timestamp_s=source_timestamp_s,
            layout=frame.layout,
            channel_order=frame.channel_order,
            value_range=frame.value_range,
        )

    def take(self, observation_time: TimePoint) -> _CachedFrame | None:
        """Returns and removes the exact frame used for one policy request."""

        return self._frames.pop(_timestamp_key_from_timepoint(observation_time), None)


class LiveTrajectoryVideoWriter:
    """Writes one diagnostic MJPEG/AVI frame per accepted trajectory proposal."""

    def __init__(self, output_path: Path, *, frames_per_second: float) -> None:
        if not math.isfinite(frames_per_second) or frames_per_second <= 0.0:
            raise ValueError("frames_per_second must be finite and positive")
        self._output_path = output_path
        self._frames_per_second = frames_per_second
        self._writer: object | None = None
        self.frames_written = 0

    def write(
        self,
        publication: LocalTrajectoryPublication,
        frame: _CachedFrame,
        *,
        goal_image: Image.Image | None = None,
    ) -> None:
        rendered = draw_trajectory_overlay(
            as_rgb_image(
                frame.image,
                layout=frame.layout,
                channel_order=frame.channel_order,
                value_range=frame.value_range,
            ),
            _overlay_state(publication, frame, goal_image=goal_image),
        )
        self.write_rendered(rendered)

    def write_rendered(self, frame: Image.Image) -> None:
        """Writes a pre-rendered RGB diagnostic frame."""

        if self._writer is None:
            self._writer = self._open_writer(frame.size)
        self._write_rgb_frame(frame)
        self.frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()  # type: ignore[union-attr]
            self._writer = None

    def _open_writer(self, size: tuple[int, int]) -> object:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is required to encode overlay video") from error
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        if self._output_path.suffix.lower() != ".avi":
            raise ValueError(
                "live overlay output must use the supported '.avi' suffix: "
                f"{self._output_path}"
            )
        if self._output_path.exists():
            raise FileExistsError(
                "overlay video already exists; choose a new output path: "
                f"{self._output_path}"
            )
        writer = cv2.VideoWriter(
            str(self._output_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            self._frames_per_second,
            size,
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(
                "OpenCV could not open the requested MJPEG/AVI output: "
                f"{self._output_path}"
            )
        return writer

    def _write_rgb_frame(self, frame: Image.Image) -> None:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is required to encode overlay video") from error
        image = torch.frombuffer(
            bytearray(frame.tobytes()), dtype=torch.uint8
        ).view(frame.height, frame.width, 3)
        bgr = image[:, :, (2, 1, 0)].numpy()
        self._writer.write(bgr)  # type: ignore[union-attr]


def write_publication_frame(
    *,
    cache: ObservationFrameCache,
    writer: LiveTrajectoryVideoWriter,
    publication: LocalTrajectoryPublication,
    goal_image: Image.Image | None = None,
) -> bool:
    """Writes one frame for an active proposal, if its source frame remains."""

    if (
        publication.state != LocalTrajectoryState.ACTIVE
        or publication.observation_time is None
        or publication.trajectory is None
    ):
        return False
    frame = cache.take(publication.observation_time)
    if frame is None:
        return False
    writer.write(publication, frame, goal_image=goal_image)
    return True


def as_rgb_image(
    image: object,
    *,
    layout: str,
    channel_order: str,
    value_range: str,
) -> Image.Image:
    """Converts a byte tensor in an explicit source format to Pillow RGB."""

    if not isinstance(image, torch.Tensor):
        raise TypeError("live overlay requires a torch tensor observation")
    if image.dtype != torch.uint8 or value_range not in ("byte", "auto"):
        raise ValueError("live overlay requires byte uint8 observations")
    image = image.detach().cpu()
    if layout == "chw":
        image = image.permute(1, 2, 0)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("live overlay requires a three-channel image")
    if channel_order == "bgr":
        image = image[:, :, (2, 1, 0)]
    elif channel_order != "rgb":
        raise ValueError("live overlay requires RGB or BGR observations")
    height, width, _ = image.shape
    return Image.frombytes(
        "RGB",
        (int(width), int(height)),
        image.contiguous().numpy().tobytes(),
    )


def _overlay_state(
    publication: LocalTrajectoryPublication,
    frame: _CachedFrame,
    *,
    goal_image: Image.Image | None,
) -> TrajectoryOverlayState:
    trajectory = publication.trajectory
    if trajectory is None or publication.observation_time is None:
        raise ValueError("active publication must include trajectory and time")
    candidate = TrajectoryCandidate(
        candidate_id=TrajectoryCandidateId(trajectory.source_candidate_id),
        waypoints=tuple(
            PolicyNativeWaypoint(
                step_index=waypoint.step_index,
                x=waypoint.x,
                y=waypoint.y,
            )
            for waypoint in trajectory.waypoints
        ),
    )
    candidates = TrajectoryCandidateSet(
        snapshot_id=publication.snapshot_id,
        segment_id=_require(publication.segment_id, "segment_id"),
        source_node_id=_require(publication.source_node_id, "source_node_id"),
        target_node_id=_require(publication.target_node_id, "target_node_id"),
        target_anchor_id=_require(
            publication.target_anchor_id, "target_anchor_id"
        ),
        goal_resource_id=_require(
            publication.goal_resource_id, "goal_resource_id"
        ),
        observation_time=publication.observation_time,
        produced_at=publication.generated_at or publication.published_at,
        temporal_distance=trajectory.temporal_distance,
        coordinate_frame=trajectory.coordinate_frame,
        coordinate_units=trajectory.coordinate_units,
        sampling_seed=trajectory.sampling_seed,
        candidates=(candidate,),
        policy_id=trajectory.policy_id,
        image_profile_id=trajectory.image_profile_id,
        model_artifact_id=trajectory.model_artifact_id,
        model_artifact_digest=trajectory.model_artifact_digest,
    )
    timestamp_s = (
        frame.timestamp_s
        if frame.source_timestamp_s is None
        else frame.source_timestamp_s
    )
    return TrajectoryOverlayState(
        source_timestamp_s=timestamp_s,
        phase=publication.state.value,
        current_node=str(publication.source_node_id),
        target_node=str(publication.target_node_id),
        status_detail=(
            f"proposal={publication.revision.sequence}  "
            f"distance={trajectory.temporal_distance:.3f}  native units"
        ),
        candidate_set=candidates,
        goal_image=goal_image,
    )


_Value = TypeVar("_Value")


def _require(value: _Value | None, name: str) -> _Value:
    if value is None:
        raise ValueError(f"active publication missing {name}")
    return value


def _timestamp_key(timestamp_s: float) -> int:
    if not math.isfinite(timestamp_s):
        raise ValueError("observation timestamp must be finite")
    return round(timestamp_s * 1_000_000_000)


def _timestamp_key_from_timepoint(value: TimePoint) -> int:
    return value.nanoseconds

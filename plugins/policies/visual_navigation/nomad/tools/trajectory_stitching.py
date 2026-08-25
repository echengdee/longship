"""Diagnostic short-step stitching of raw NoMaD trajectory candidates.

This module deliberately produces neither odometry nor robot commands.  It
integrates a robust representative of successive policy predictions only so
that an offline replay can expose long-horizon drift and discontinuities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Protocol

from longship.navigation.ports.trajectory_policy import (
    TrajectoryCandidateSet,
)


@dataclass(frozen=True, slots=True)
class PlanarPoint:
    """One point in policy-native planar coordinates."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class StitchedPose:
    """One open-loop pose integrated into the diagnostic start frame."""

    x: float
    y: float
    heading_rad: float
    source_timestamp_s: float


@dataclass(frozen=True, slots=True)
class TrajectoryStitchUpdate:
    """The representative prediction and short step used for one update."""

    representative_path: tuple[PlanarPoint, ...]
    local_step: PlanarPoint
    actual_step_distance: float
    stitched_pose: StitchedPose
    selected_candidate_index: int | None = None
    selected_waypoint_index: int | None = None
    control_waypoint: PlanarPoint | None = None
    scaled_control_waypoint: PlanarPoint | None = None
    linear_velocity: float | None = None
    angular_velocity: float | None = None


class DiagnosticTrajectoryStitcher(Protocol):
    """Common rendering surface for diagnostic stitching strategies."""

    @property
    def method_id(self) -> str:
        ...

    @property
    def panel_title(self) -> str:
        ...

    @property
    def panel_detail(self) -> str:
        ...

    @property
    def metadata(self) -> dict[str, object]:
        ...

    @property
    def poses(self) -> tuple[StitchedPose, ...]:
        ...

    def append(
        self,
        candidate_set: TrajectoryCandidateSet,
        source_timestamp_s: float,
    ) -> TrajectoryStitchUpdate | None:
        ...


class ShortStepTrajectoryStitcher:
    """Composes one short, median-candidate step per policy inference."""

    def __init__(self, step_distance: float) -> None:
        if not math.isfinite(step_distance) or step_distance <= 0.0:
            raise ValueError("step_distance must be finite and positive")
        self._step_distance = step_distance
        self._poses = [
            StitchedPose(
                x=0.0,
                y=0.0,
                heading_rad=0.0,
                source_timestamp_s=0.0,
            )
        ]
        self._coordinate_frame: str | None = None
        self._coordinate_units: str | None = None

    @property
    def step_distance(self) -> float:
        return self._step_distance

    @property
    def method_id(self) -> str:
        return "coordinate_wise_median_short_step_se2_v1"

    @property
    def panel_title(self) -> str:
        return "Median short-step path (diagnostic)"

    @property
    def panel_detail(self) -> str:
        return f"{len(self._poses) - 1} steps x {self._step_distance:g} native"

    @property
    def metadata(self) -> dict[str, object]:
        return {"step_distance": self._step_distance}

    @property
    def poses(self) -> tuple[StitchedPose, ...]:
        return tuple(self._poses)

    def append(
        self,
        candidate_set: TrajectoryCandidateSet,
        source_timestamp_s: float,
    ) -> TrajectoryStitchUpdate | None:
        """Adds one fixed-distance step, or returns ``None`` for zero motion."""

        if not math.isfinite(source_timestamp_s):
            raise ValueError("source_timestamp_s must be finite")
        self._validate_coordinates(candidate_set)
        representative_path = median_candidate_path(candidate_set)
        local_step, actual_distance = _point_at_arc_distance(
            representative_path,
            self._step_distance,
        )
        if local_step is None:
            return None

        previous = self._poses[-1]
        cosine = math.cos(previous.heading_rad)
        sine = math.sin(previous.heading_rad)
        global_dx = cosine * local_step.x - sine * local_step.y
        global_dy = sine * local_step.x + cosine * local_step.y
        local_heading = math.atan2(local_step.y, local_step.x)
        pose = StitchedPose(
            x=previous.x + global_dx,
            y=previous.y + global_dy,
            heading_rad=_wrap_angle(previous.heading_rad + local_heading),
            source_timestamp_s=source_timestamp_s,
        )
        self._poses.append(pose)
        return TrajectoryStitchUpdate(
            representative_path=representative_path,
            local_step=local_step,
            actual_step_distance=actual_distance,
            stitched_pose=pose,
        )

    def _validate_coordinates(
        self,
        candidate_set: TrajectoryCandidateSet,
    ) -> None:
        if self._coordinate_frame is None:
            self._coordinate_frame = candidate_set.coordinate_frame
            self._coordinate_units = candidate_set.coordinate_units
            return
        if candidate_set.coordinate_frame != self._coordinate_frame:
            raise ValueError("candidate coordinate frame changed while stitching")
        if candidate_set.coordinate_units != self._coordinate_units:
            raise ValueError("candidate coordinate units changed while stitching")


class OfficialDemoTrajectoryStitcher:
    """Replays the released demo's sample/waypoint/controller decisions."""

    def __init__(
        self,
        *,
        sample_index: int = 0,
        waypoint_index: int = 2,
        frame_rate_hz: float = 4.0,
        max_linear_velocity: float = 0.2,
        max_angular_velocity: float = 0.4,
    ) -> None:
        if sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if waypoint_index < 0:
            raise ValueError("waypoint_index must be non-negative")
        values = (
            frame_rate_hz,
            max_linear_velocity,
            max_angular_velocity,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("controller rates and limits must be positive")
        self._sample_index = sample_index
        self._waypoint_index = waypoint_index
        self._frame_rate_hz = frame_rate_hz
        self._max_linear_velocity = max_linear_velocity
        self._max_angular_velocity = max_angular_velocity
        self._poses = [
            StitchedPose(
                x=0.0,
                y=0.0,
                heading_rad=0.0,
                source_timestamp_s=0.0,
            )
        ]
        self._coordinate_frame: str | None = None
        self._coordinate_units: str | None = None

    @property
    def method_id(self) -> str:
        return "official_demo_sample_waypoint_pd_unicycle_v1"

    @property
    def panel_title(self) -> str:
        return "Official demo controller mock"

    @property
    def panel_detail(self) -> str:
        return (
            f"{len(self._poses) - 1} steps; sample-{self._sample_index} / "
            f"waypoint-{self._waypoint_index}; meters"
        )

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "sample_index": self._sample_index,
            "waypoint_index": self._waypoint_index,
            "frame_rate_hz": self._frame_rate_hz,
            "max_linear_velocity": self._max_linear_velocity,
            "max_angular_velocity": self._max_angular_velocity,
            "waypoint_scale": (
                self._max_linear_velocity / self._frame_rate_hz
            ),
        }

    @property
    def poses(self) -> tuple[StitchedPose, ...]:
        return tuple(self._poses)

    def append(
        self,
        candidate_set: TrajectoryCandidateSet,
        source_timestamp_s: float,
    ) -> TrajectoryStitchUpdate:
        """Selects sample 0 / waypoint 2 and integrates the demo controller."""

        if not math.isfinite(source_timestamp_s):
            raise ValueError("source_timestamp_s must be finite")
        self._validate_coordinates(candidate_set)
        if self._sample_index >= len(candidate_set.candidates):
            raise ValueError("official demo sample index is unavailable")
        candidate = candidate_set.candidates[self._sample_index]
        if self._waypoint_index >= len(candidate.waypoints):
            raise ValueError("official demo waypoint index is unavailable")

        path = tuple(
            PlanarPoint(x=waypoint.x, y=waypoint.y)
            for waypoint in candidate.waypoints
        )
        waypoint = path[self._waypoint_index]
        waypoint_scale = self._max_linear_velocity / self._frame_rate_hz
        scaled_waypoint = PlanarPoint(
            x=waypoint.x * waypoint_scale,
            y=waypoint.y * waypoint_scale,
        )
        delta_time_s = 1.0 / self._frame_rate_hz
        linear_velocity, angular_velocity = self._controller(
            scaled_waypoint,
            delta_time_s,
        )
        local_step = _integrate_unicycle(
            linear_velocity,
            angular_velocity,
            delta_time_s,
        )
        previous = self._poses[-1]
        cosine = math.cos(previous.heading_rad)
        sine = math.sin(previous.heading_rad)
        pose = StitchedPose(
            x=previous.x + cosine * local_step.x - sine * local_step.y,
            y=previous.y + sine * local_step.x + cosine * local_step.y,
            heading_rad=_wrap_angle(
                previous.heading_rad + angular_velocity * delta_time_s
            ),
            source_timestamp_s=source_timestamp_s,
        )
        self._poses.append(pose)
        return TrajectoryStitchUpdate(
            representative_path=path,
            local_step=local_step,
            actual_step_distance=linear_velocity * delta_time_s,
            stitched_pose=pose,
            selected_candidate_index=self._sample_index,
            selected_waypoint_index=self._waypoint_index,
            control_waypoint=waypoint,
            scaled_control_waypoint=scaled_waypoint,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
        )

    def _controller(
        self,
        waypoint: PlanarPoint,
        delta_time_s: float,
    ) -> tuple[float, float]:
        if abs(waypoint.x) < 1.0e-8:
            linear_velocity = 0.0
            if abs(waypoint.y) < 1.0e-8:
                angular_velocity = 0.0
            else:
                angular_velocity = math.copysign(
                    math.pi / (2.0 * delta_time_s),
                    waypoint.y,
                )
        else:
            linear_velocity = waypoint.x / delta_time_s
            angular_velocity = math.atan(
                waypoint.y / waypoint.x
            ) / delta_time_s
        return (
            max(0.0, min(self._max_linear_velocity, linear_velocity)),
            max(
                -self._max_angular_velocity,
                min(self._max_angular_velocity, angular_velocity),
            ),
        )

    def _validate_coordinates(
        self,
        candidate_set: TrajectoryCandidateSet,
    ) -> None:
        if self._coordinate_frame is None:
            self._coordinate_frame = candidate_set.coordinate_frame
            self._coordinate_units = candidate_set.coordinate_units
            return
        if candidate_set.coordinate_frame != self._coordinate_frame:
            raise ValueError("candidate coordinate frame changed while stitching")
        if candidate_set.coordinate_units != self._coordinate_units:
            raise ValueError("candidate coordinate units changed while stitching")


def median_candidate_path(
    candidate_set: TrajectoryCandidateSet,
) -> tuple[PlanarPoint, ...]:
    """Returns the coordinate-wise median path across all candidates."""

    if not candidate_set.candidates:
        raise ValueError("candidate set must not be empty")
    waypoint_count = min(
        len(candidate.waypoints) for candidate in candidate_set.candidates
    )
    if waypoint_count <= 0:
        raise ValueError("trajectory candidates must contain waypoints")
    return tuple(
        PlanarPoint(
            x=float(
                median(
                    candidate.waypoints[index].x
                    for candidate in candidate_set.candidates
                )
            ),
            y=float(
                median(
                    candidate.waypoints[index].y
                    for candidate in candidate_set.candidates
                )
            ),
        )
        for index in range(waypoint_count)
    )


def _point_at_arc_distance(
    path: tuple[PlanarPoint, ...],
    requested_distance: float,
) -> tuple[PlanarPoint | None, float]:
    previous = PlanarPoint(0.0, 0.0)
    traversed = 0.0
    for point in path:
        segment_x = point.x - previous.x
        segment_y = point.y - previous.y
        segment_distance = math.hypot(segment_x, segment_y)
        if segment_distance <= 1.0e-9:
            previous = point
            continue
        remaining = requested_distance - traversed
        if segment_distance >= remaining:
            ratio = remaining / segment_distance
            return (
                PlanarPoint(
                    x=previous.x + ratio * segment_x,
                    y=previous.y + ratio * segment_y,
                ),
                requested_distance,
            )
        traversed += segment_distance
        previous = point
    if traversed <= 1.0e-9:
        return None, 0.0
    return previous, traversed


def _integrate_unicycle(
    linear_velocity: float,
    angular_velocity: float,
    delta_time_s: float,
) -> PlanarPoint:
    heading_delta = angular_velocity * delta_time_s
    if abs(angular_velocity) <= 1.0e-9:
        return PlanarPoint(x=linear_velocity * delta_time_s, y=0.0)
    radius = linear_velocity / angular_velocity
    return PlanarPoint(
        x=radius * math.sin(heading_delta),
        y=radius * (1.0 - math.cos(heading_delta)),
    )


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))

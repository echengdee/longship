from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from longship.contracts.skills.follow_person import finite_number


@dataclass(frozen=True, slots=True)
class RigidTransform:
    """Validated row-major transform from camera optical frame to robot base."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != 16 or not all(finite_number(v) for v in self.values):
            raise ValueError("rigid transform must contain 16 finite values")
        if any(abs(self.values[12 + index]) > 1e-6 for index in range(3)) or abs(
            self.values[15] - 1.0
        ) > 1e-6:
            raise ValueError("rigid transform has an invalid homogeneous row")
        rotation = tuple(self.values[row * 4 : row * 4 + 3] for row in range(3))
        for row in rotation:
            if abs(sum(item * item for item in row) - 1.0) > 1e-3:
                raise ValueError("rigid transform rotation rows must be unit length")
        for first, second in ((0, 1), (0, 2), (1, 2)):
            dot_product = sum(
                rotation[first][i] * rotation[second][i] for i in range(3)
            )
            if abs(dot_product) > 1e-3:
                raise ValueError("rigid transform rotation rows must be orthogonal")
        determinant = (
            rotation[0][0]
            * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
            - rotation[0][1]
            * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
            + rotation[0][2]
            * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
        )
        if abs(determinant - 1.0) > 1e-3:
            raise ValueError("rigid transform rotation must be right-handed")
        if any(abs(self.values[index]) > 5.0 for index in (3, 7, 11)):
            raise ValueError("camera translation exceeds the five metre sanity bound")

    def apply(
        self, optical_xyz_m: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        if len(optical_xyz_m) != 3 or not all(
            finite_number(item) for item in optical_xyz_m
        ):
            raise ValueError("camera point must contain three finite values")
        x, y, z = optical_xyz_m
        matrix = self.values
        return (
            matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3],
            matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7],
            matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11],
        )

    @classmethod
    def from_calibration(cls, value: object) -> tuple[str, "RigidTransform"]:
        expected = {
            "schema_version",
            "calibration_id",
            "confirmed",
            "camera_optical_to_base_row_major",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("camera calibration contains missing or unexpected fields")
        if value["schema_version"] != "longship.camera-extrinsic.v0":
            raise ValueError("unsupported camera calibration schema")
        calibration_id = value["calibration_id"]
        if not isinstance(calibration_id, str) or not calibration_id.strip():
            raise ValueError("calibration_id is required")
        if value["confirmed"] is not True:
            raise ValueError("camera calibration is not confirmed")
        raw = value["camera_optical_to_base_row_major"]
        if not isinstance(raw, list) or not all(finite_number(item) for item in raw):
            raise ValueError("camera transform must be an array")
        return calibration_id, cls(tuple(float(item) for item in raw))


@dataclass(frozen=True, slots=True)
class BoundingBox:
    left_px: int
    top_px: int
    width_px: int
    height_px: int
    confidence: float

    def __post_init__(self) -> None:
        if any(
            type(item) is not int or item < 0
            for item in (self.left_px, self.top_px, self.width_px, self.height_px)
        ):
            raise ValueError("bounding box pixel values must be non-negative integers")
        if self.width_px == 0 or self.height_px == 0:
            raise ValueError("bounding box must have non-zero area")
        if not finite_number(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("bounding box confidence must be between 0 and 1")

    @property
    def right_px(self) -> int:
        return self.left_px + self.width_px

    @property
    def bottom_px(self) -> int:
        return self.top_px + self.height_px

    def iou(self, other: "BoundingBox") -> float:
        left = max(self.left_px, other.left_px)
        top = max(self.top_px, other.top_px)
        right = min(self.right_px, other.right_px)
        bottom = min(self.bottom_px, other.bottom_px)
        intersection = max(0, right - left) * max(0, bottom - top)
        if intersection == 0:
            return 0.0
        first_area = self.width_px * self.height_px
        second_area = other.width_px * other.height_px
        return intersection / (first_area + second_area - intersection)


@dataclass(frozen=True, slots=True)
class TrackedBox:
    track_id: str
    box: BoundingBox


class ShortTrackAssigner:
    """Small IoU tracker for brief detector gaps; it is not re-identification."""

    def __init__(
        self,
        *,
        minimum_iou: float = 0.25,
        maximum_missed_frames: int = 4,
    ) -> None:
        if not 0.0 < minimum_iou < 1.0:
            raise ValueError("minimum_iou must be between zero and one")
        if type(maximum_missed_frames) is not int or maximum_missed_frames < 0:
            raise ValueError("maximum_missed_frames must be non-negative")
        self.minimum_iou = minimum_iou
        self.maximum_missed_frames = maximum_missed_frames
        self._next_id = 1
        self._tracks: dict[str, tuple[BoundingBox, int]] = {}

    def update(self, detections: tuple[BoundingBox, ...]) -> tuple[TrackedBox, ...]:
        candidates: list[tuple[float, str, int]] = []
        for track_id, (old_box, _) in self._tracks.items():
            for index, box in enumerate(detections):
                overlap = old_box.iou(box)
                if overlap >= self.minimum_iou:
                    candidates.append((overlap, track_id, index))
        candidates.sort(reverse=True)
        assigned_tracks: set[str] = set()
        assigned_detections: set[int] = set()
        output: dict[int, TrackedBox] = {}
        next_tracks: dict[str, tuple[BoundingBox, int]] = {}
        for _, track_id, index in candidates:
            if track_id in assigned_tracks or index in assigned_detections:
                continue
            assigned_tracks.add(track_id)
            assigned_detections.add(index)
            box = detections[index]
            output[index] = TrackedBox(track_id, box)
            next_tracks[track_id] = (box, 0)
        for track_id, (box, missed) in self._tracks.items():
            if track_id not in assigned_tracks and missed < self.maximum_missed_frames:
                next_tracks[track_id] = (box, missed + 1)
        for index, box in enumerate(detections):
            if index in assigned_detections:
                continue
            track_id = f"camera-track-{self._next_id}"
            self._next_id += 1
            output[index] = TrackedBox(track_id, box)
            next_tracks[track_id] = (box, 0)
        self._tracks = next_tracks
        return tuple(output[index] for index in sorted(output))

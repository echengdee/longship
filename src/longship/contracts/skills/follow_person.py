from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


def finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _number(value: object, field: str) -> float:
    if not finite_number(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


class FollowState(str, Enum):
    IDLE = "idle"
    ACQUIRING = "acquiring"
    FOLLOWING = "following"
    HOLDING = "holding_for_scene"
    LOST_APPROACH = "approaching_last_seen"
    BLOCKED = "blocked"
    PAUSED = "paused"
    STOPPED = "stopped"
    STOP_UNVERIFIED = "stop_unverified"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PersonTrack:
    track_id: str
    forward_m: float
    left_m: float
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, str) or not self.track_id.strip():
            raise ValueError("track_id must be a non-empty string")
        if len(self.track_id) > 128:
            raise ValueError("track_id exceeds 128 characters")
        if not finite_number(self.forward_m) or not finite_number(self.left_m):
            raise ValueError("track position must be finite")
        if not finite_number(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("track confidence must be between 0 and 1")

    @property
    def distance_m(self) -> float:
        return math.hypot(self.forward_m, self.left_m)

    @property
    def bearing_rad(self) -> float:
        return math.atan2(self.left_m, self.forward_m)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PersonTrack":
        expected = {"track_id", "forward_m", "left_m", "confidence"}
        if set(value) != expected:
            raise ValueError("person track contains missing or unexpected fields")
        return cls(
            track_id=value["track_id"],
            forward_m=_number(value["forward_m"], "forward_m"),
            left_m=_number(value["left_m"], "left_m"),
            confidence=_number(value["confidence"], "confidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "forward_m": self.forward_m,
            "left_m": self.left_m,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class ObstaclePoint:
    forward_m: float
    left_m: float
    radius_m: float = 0.05

    def __post_init__(self) -> None:
        if not all(
            finite_number(item)
            for item in (self.forward_m, self.left_m, self.radius_m)
        ):
            raise ValueError("obstacle geometry must be finite")
        if self.radius_m < 0.0 or self.radius_m > 5.0:
            raise ValueError("obstacle radius must be between 0 and 5 metres")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ObstaclePoint":
        expected = {"forward_m", "left_m", "radius_m"}
        if set(value) != expected:
            raise ValueError("obstacle contains missing or unexpected fields")
        return cls(
            forward_m=_number(value["forward_m"], "forward_m"),
            left_m=_number(value["left_m"], "left_m"),
            radius_m=_number(value["radius_m"], "radius_m"),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "forward_m": self.forward_m,
            "left_m": self.left_m,
            "radius_m": self.radius_m,
        }


@dataclass(frozen=True, slots=True)
class FollowScene:
    sequence: int
    captured_monotonic_ns: int
    received_monotonic_ns: int
    healthy: bool
    calibration_id: str
    calibration_valid: bool
    detector_ready: bool
    floor_valid: bool
    tracks: tuple[PersonTrack, ...]
    obstacles: tuple[ObstaclePoint, ...]
    raw_forward_clearance_m: float | None
    detail: str = ""
    schema_version: str = "longship.follow-scene.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "longship.follow-scene.v1":
            raise ValueError("unsupported follow scene schema version")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("scene sequence must be a non-negative integer")
        if (
            type(self.captured_monotonic_ns) is not int
            or type(self.received_monotonic_ns) is not int
            or self.captured_monotonic_ns < 0
            or self.received_monotonic_ns < 0
        ):
            raise ValueError("scene timestamps must be non-negative integers")
        for field in (
            self.healthy,
            self.calibration_valid,
            self.detector_ready,
            self.floor_valid,
        ):
            if type(field) is not bool:
                raise ValueError("scene health flags must be booleans")
        if not isinstance(self.calibration_id, str) or not self.calibration_id.strip():
            raise ValueError("scene calibration_id must be a non-empty string")
        if len(self.calibration_id) > 128:
            raise ValueError("scene calibration_id exceeds 128 characters")
        if self.raw_forward_clearance_m is not None and (
            not finite_number(self.raw_forward_clearance_m)
            or self.raw_forward_clearance_m < 0.0
        ):
            raise ValueError("raw forward clearance must be non-negative or null")
        if not isinstance(self.detail, str) or len(self.detail) > 1_000:
            raise ValueError("scene detail must be a string no longer than 1000 chars")
        if len(self.tracks) > 128 or len(self.obstacles) > 20_000:
            raise ValueError("follow scene exceeds bounded collection limits")
        if not all(isinstance(track, PersonTrack) for track in self.tracks):
            raise ValueError("scene tracks must be PersonTrack values")
        if not all(isinstance(item, ObstaclePoint) for item in self.obstacles):
            raise ValueError("scene obstacles must be ObstaclePoint values")

    def age_s(self, now_ns: int) -> float:
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a non-negative integer")
        return max(0.0, (now_ns - self.captured_monotonic_ns) / 1_000_000_000)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, received_monotonic_ns: int
    ) -> "FollowScene":
        expected = {
            "schema_version",
            "sequence",
            "captured_monotonic_ns",
            "healthy",
            "calibration_id",
            "calibration_valid",
            "detector_ready",
            "floor_valid",
            "tracks",
            "obstacles",
            "raw_forward_clearance_m",
            "detail",
        }
        if set(value) != expected:
            raise ValueError("follow scene contains missing or unexpected fields")
        raw_tracks = value["tracks"]
        raw_obstacles = value["obstacles"]
        if not isinstance(raw_tracks, list) or not isinstance(raw_obstacles, list):
            raise ValueError("tracks and obstacles must be arrays")
        if not all(isinstance(item, Mapping) for item in raw_tracks):
            raise ValueError("every track must be an object")
        if not all(isinstance(item, Mapping) for item in raw_obstacles):
            raise ValueError("every obstacle must be an object")
        clearance = value["raw_forward_clearance_m"]
        return cls(
            schema_version=value["schema_version"],
            sequence=value["sequence"],
            captured_monotonic_ns=value["captured_monotonic_ns"],
            received_monotonic_ns=received_monotonic_ns,
            healthy=value["healthy"],
            calibration_id=value["calibration_id"],
            calibration_valid=value["calibration_valid"],
            detector_ready=value["detector_ready"],
            floor_valid=value["floor_valid"],
            tracks=tuple(PersonTrack.from_mapping(item) for item in raw_tracks),
            obstacles=tuple(
                ObstaclePoint.from_mapping(item) for item in raw_obstacles
            ),
            raw_forward_clearance_m=(
                None if clearance is None else _number(clearance, "clearance")
            ),
            detail=value["detail"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "captured_monotonic_ns": self.captured_monotonic_ns,
            "healthy": self.healthy,
            "calibration_id": self.calibration_id,
            "calibration_valid": self.calibration_valid,
            "detector_ready": self.detector_ready,
            "floor_valid": self.floor_valid,
            "tracks": [track.to_dict() for track in self.tracks],
            "obstacles": [obstacle.to_dict() for obstacle in self.obstacles],
            "raw_forward_clearance_m": self.raw_forward_clearance_m,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class FollowCommand:
    session_id: str
    sequence: int
    issued_monotonic_ns: int
    expires_monotonic_ns: int
    forward_mps: float
    yaw_rate_radps: float
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("command session_id is required")
        if len(self.session_id) > 128:
            raise ValueError("command session_id exceeds 128 characters")
        if type(self.sequence) is not int or self.sequence <= 0:
            raise ValueError("command sequence must be positive")
        if (
            type(self.issued_monotonic_ns) is not int
            or type(self.expires_monotonic_ns) is not int
            or self.issued_monotonic_ns < 0
            or self.expires_monotonic_ns <= self.issued_monotonic_ns
        ):
            raise ValueError("command timestamps are invalid")
        if not finite_number(self.forward_mps) or not finite_number(
            self.yaw_rate_radps
        ):
            raise ValueError("command velocity must be finite")
        if self.forward_mps < 0.0:
            raise ValueError("FollowPerson command cannot request reverse motion")
        if self.expires_monotonic_ns - self.issued_monotonic_ns > 250_000_000:
            raise ValueError("FollowPerson command TTL exceeds 250 ms")
        if (
            not isinstance(self.reason, str)
            or not self.reason
            or len(self.reason) > 1_000
        ):
            raise ValueError("command reason is required")

    @property
    def is_zero(self) -> bool:
        return self.forward_mps == 0.0 and self.yaw_rate_radps == 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "sequence": self.sequence,
            "issued_monotonic_ns": self.issued_monotonic_ns,
            "expires_monotonic_ns": self.expires_monotonic_ns,
            "forward_mps": self.forward_mps,
            "yaw_rate_radps": self.yaw_rate_radps,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class MotionReceipt:
    accepted: bool
    detail: str
    verified_stopped: bool = False


@dataclass(frozen=True, slots=True)
class PlanDecision:
    forward_mps: float
    yaw_rate_radps: float
    path_robot_xy_m: tuple[tuple[float, float], ...]
    reached: bool
    blocked: bool
    detail: str


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    forward_mps: float
    yaw_rate_radps: float
    blocked: bool
    detail: str


@dataclass(frozen=True, slots=True)
class FollowSnapshot:
    session_id: str
    state: FollowState
    revision: int
    scene_sequence: int | None
    locked_track_id: str | None
    target_robot_xy_m: tuple[float, float] | None
    command: FollowCommand | None
    path_robot_xy_m: tuple[tuple[float, float], ...]
    detail: str
    stop_verified: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "longship.follow-runtime-snapshot.v1",
            "session_id": self.session_id,
            "state": self.state.value,
            "revision": self.revision,
            "scene_sequence": self.scene_sequence,
            "locked_track_id": self.locked_track_id,
            "target_robot_xy_m": (
                list(self.target_robot_xy_m)
                if self.target_robot_xy_m is not None
                else None
            ),
            "command": self.command.to_dict() if self.command else None,
            "path_robot_xy_m": [list(point) for point in self.path_robot_xy_m],
            "detail": self.detail,
            "stop_verified": self.stop_verified,
        }

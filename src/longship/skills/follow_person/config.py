from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

from longship.contracts.skills.follow_person import finite_number
from longship.safety.follow_obstacle import SafetySettings


class FollowConfigError(ValueError):
    pass


T = TypeVar("T")


def _strict_dataclass(
    cls: type[T], value: object, section: str
) -> T:
    if not isinstance(value, Mapping):
        raise FollowConfigError(f"{section} must be an object")
    expected = {field.name for field in fields(cls)}
    if set(value) != expected:
        raise FollowConfigError(
            f"{section} contains missing or unexpected fields: "
            f"expected {sorted(expected)}"
        )
    converted: dict[str, Any] = {}
    for name in expected:
        raw = value[name]
        if not finite_number(raw):
            raise FollowConfigError(f"{section}.{name} must be finite")
        converted[name] = float(raw)
    try:
        return cls(**converted)
    except ValueError as exc:
        raise FollowConfigError(f"invalid {section}: {exc}") from exc


def _positive(values: Mapping[str, float]) -> None:
    for name, value in values.items():
        if not finite_number(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    control_frequency_hz: float
    command_ttl_s: float
    scene_max_age_s: float
    scene_failure_grace_s: float
    acquire_timeout_s: float
    lost_target_timeout_s: float
    blocked_timeout_s: float

    def __post_init__(self) -> None:
        _positive({field.name: getattr(self, field.name) for field in fields(self)})
        if not 2.0 <= self.control_frequency_hz <= 100.0:
            raise ValueError("control_frequency_hz must be between 2 and 100")
        if not 0.05 <= self.command_ttl_s <= 0.25:
            raise ValueError("command_ttl_s must be between 0.05 and 0.25")
        if self.command_ttl_s < 1.5 / self.control_frequency_hz:
            raise ValueError("command TTL is too short for the configured control rate")


@dataclass(frozen=True, slots=True)
class ControlSettings:
    desired_distance_m: float
    distance_deadband_m: float
    minimum_distance_m: float
    maximum_forward_speed_mps: float
    maximum_yaw_rate_radps: float
    distance_gain: float
    heading_gain: float
    forward_disable_angle_deg: float
    minimum_track_confidence: float
    lost_target_standoff_m: float
    lost_target_goal_tolerance_m: float
    reacquire_gate_m: float
    maximum_linear_accel_mps2: float
    maximum_yaw_accel_radps2: float
    maximum_linear_jerk_mps3: float
    maximum_yaw_jerk_radps3: float

    def __post_init__(self) -> None:
        _positive(
            {
                field.name: getattr(self, field.name)
                for field in fields(self)
                if field.name != "minimum_track_confidence"
            }
        )
        if not 0.0 < self.minimum_track_confidence <= 1.0:
            raise ValueError("minimum_track_confidence must be in (0, 1]")
        if self.minimum_distance_m >= self.desired_distance_m:
            raise ValueError("minimum distance must be below desired distance")
        if self.distance_deadband_m >= self.desired_distance_m:
            raise ValueError("distance deadband is too large")
        if self.forward_disable_angle_deg >= 90.0:
            raise ValueError("forward disable angle must be below 90 degrees")

    @property
    def forward_disable_angle_rad(self) -> float:
        return math.radians(self.forward_disable_angle_deg)


@dataclass(frozen=True, slots=True)
class PlannerSettings:
    grid_resolution_m: float
    forward_extent_m: float
    side_extent_m: float
    robot_radius_m: float
    clearance_margin_m: float
    lookahead_distance_m: float
    target_exclusion_radius_m: float

    def __post_init__(self) -> None:
        _positive({field.name: getattr(self, field.name) for field in fields(self)})
        if not 0.05 <= self.grid_resolution_m <= 0.25:
            raise ValueError("grid resolution must be between 0.05 and 0.25 metres")
        if self.lookahead_distance_m > self.forward_extent_m:
            raise ValueError("lookahead exceeds planner extent")


@dataclass(frozen=True, slots=True)
class FollowProfile:
    profile_id: str
    runtime: RuntimeSettings
    control: ControlSettings
    safety: SafetySettings
    planner: PlannerSettings
    schema_version: str = "longship.follow-person-profile.v0"

    @classmethod
    def load(cls, path: str | Path) -> "FollowProfile":
        resolved = Path(path)
        if resolved.stat().st_size > 256_000:
            raise FollowConfigError("follow profile exceeds 256 KB")
        with resolved.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        return cls.from_mapping(value)

    @classmethod
    def from_mapping(cls, value: object) -> "FollowProfile":
        expected = {
            "schema_version",
            "profile_id",
            "runtime",
            "control",
            "safety",
            "planner",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise FollowConfigError(
                "follow profile contains missing or unexpected fields"
            )
        if value["schema_version"] != "longship.follow-person-profile.v0":
            raise FollowConfigError("unsupported follow profile schema version")
        profile_id = value["profile_id"]
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise FollowConfigError("profile_id must be a non-empty string")
        return cls(
            schema_version=value["schema_version"],
            profile_id=profile_id,
            runtime=_strict_dataclass(RuntimeSettings, value["runtime"], "runtime"),
            control=_strict_dataclass(
                ControlSettings,
                value["control"],
                "control",
            ),
            safety=_strict_dataclass(SafetySettings, value["safety"], "safety"),
            planner=_strict_dataclass(PlannerSettings, value["planner"], "planner"),
        )

    @property
    def control_period_s(self) -> float:
        return 1.0 / self.runtime.control_frequency_hz


@dataclass(frozen=True, slots=True)
class FollowQualification:
    qualification_id: str
    target_id: str
    profile_sha256: str
    calibration_id: str
    reviewer: str
    evidence_refs: tuple[str, ...]
    approved: bool
    expires_at_unix_s: int
    maximum_runtime_s: float
    schema_version: str = "longship.follow-qualification.v0"

    @classmethod
    def load(cls, path: str | Path) -> "FollowQualification":
        resolved = Path(path)
        if resolved.stat().st_size > 64_000:
            raise FollowConfigError("qualification record exceeds 64 KB")
        with resolved.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        expected = {
            "schema_version",
            "qualification_id",
            "target_id",
            "profile_sha256",
            "calibration_id",
            "reviewer",
            "evidence_refs",
            "approved",
            "expires_at_unix_s",
            "maximum_runtime_s",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise FollowConfigError(
                "qualification record contains missing or unexpected fields"
            )
        strings = (
            "qualification_id",
            "target_id",
            "calibration_id",
            "reviewer",
        )
        if not all(
            isinstance(value[name], str) and value[name].strip() for name in strings
        ):
            raise FollowConfigError("qualification identity fields are required")
        if any(len(value[name]) > 256 for name in strings):
            raise FollowConfigError("qualification identity field exceeds 256 chars")
        digest = value["profile_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise FollowConfigError("qualification profile_sha256 is invalid")
        raw_evidence = value["evidence_refs"]
        if (
            not isinstance(raw_evidence, list)
            or not raw_evidence
            or len(raw_evidence) > 64
            or not all(isinstance(item, str) and item.strip() for item in raw_evidence)
            or any(len(item) > 1_000 for item in raw_evidence)
        ):
            raise FollowConfigError("qualification evidence_refs must be non-empty")
        if len(set(raw_evidence)) != len(raw_evidence):
            raise FollowConfigError("qualification evidence_refs must be unique")
        if type(value["approved"]) is not bool:
            raise FollowConfigError("qualification approved must be boolean")
        if type(value["expires_at_unix_s"]) is not int:
            raise FollowConfigError("qualification expiry must be integer Unix seconds")
        maximum_runtime = value["maximum_runtime_s"]
        if not finite_number(maximum_runtime) or not 1.0 <= maximum_runtime <= 3_600.0:
            raise FollowConfigError("qualification maximum runtime is invalid")
        return cls(
            schema_version=value["schema_version"],
            qualification_id=value["qualification_id"],
            target_id=value["target_id"],
            profile_sha256=digest,
            calibration_id=value["calibration_id"],
            reviewer=value["reviewer"],
            evidence_refs=tuple(raw_evidence),
            approved=value["approved"],
            expires_at_unix_s=value["expires_at_unix_s"],
            maximum_runtime_s=float(maximum_runtime),
        )

    def __post_init__(self) -> None:
        if self.schema_version != "longship.follow-qualification.v0":
            raise FollowConfigError("unsupported follow qualification schema")

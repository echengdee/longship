from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from longship.contracts.skills.follow_person import (
    FollowScene,
    FollowState,
    ObstaclePoint,
    PersonTrack,
)
from longship.runtime.follow_person import FollowEventSink, FollowPersonRuntime
from longship.safety.follow_obstacle import ForwardObstacleGuard
from longship.skills.follow_person.config import FollowProfile
from longship.skills.follow_person.governor import MotionGovernor
from longship.skills.follow_person.planner import LocalFollowPlanner
from longship.targets.follow_person import RecordingMotion


class SimulationConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PersonKeyframe:
    time_s: float
    world_x_m: float
    world_y_m: float
    visible: bool
    track_id: str


@dataclass(frozen=True, slots=True)
class WorldObstacle:
    world_x_m: float
    world_y_m: float
    radius_m: float
    active_from_s: float
    active_until_s: float


@dataclass(frozen=True, slots=True)
class AcceptanceCriteria:
    minimum_following_steps: int
    maximum_unsafe_forward_commands: int
    maximum_final_distance_error_m: float


@dataclass(frozen=True, slots=True)
class FollowSimulationScenario:
    scenario_id: str
    duration_s: float
    step_s: float
    keyframes: tuple[PersonKeyframe, ...]
    obstacles: tuple[WorldObstacle, ...]
    acceptance: AcceptanceCriteria
    schema_version: str = "longship.follow-simulation.v0"

    @classmethod
    def load(cls, path: str | Path) -> "FollowSimulationScenario":
        resolved = Path(path)
        if resolved.stat().st_size > 256_000:
            raise SimulationConfigError("simulation scenario exceeds 256 KB")
        with resolved.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        return cls.from_mapping(value)

    @classmethod
    def from_mapping(cls, value: object) -> "FollowSimulationScenario":
        expected = {
            "schema_version",
            "scenario_id",
            "duration_s",
            "step_s",
            "person_keyframes",
            "obstacles",
            "acceptance",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise SimulationConfigError(
                "simulation scenario contains missing or unexpected fields"
            )
        if value["schema_version"] != "longship.follow-simulation.v0":
            raise SimulationConfigError("unsupported simulation schema version")
        scenario_id = value["scenario_id"]
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise SimulationConfigError("scenario_id must be a non-empty string")
        duration = _finite(value["duration_s"], "duration_s")
        step = _finite(value["step_s"], "step_s")
        if duration <= 0.0 or not 0.01 <= step <= 0.2:
            raise SimulationConfigError("duration or simulation step is out of range")
        raw_keyframes = value["person_keyframes"]
        raw_obstacles = value["obstacles"]
        if not isinstance(raw_keyframes, list) or len(raw_keyframes) < 2:
            raise SimulationConfigError("at least two person keyframes are required")
        if not isinstance(raw_obstacles, list):
            raise SimulationConfigError("obstacles must be an array")
        keyframes = tuple(_keyframe(item) for item in raw_keyframes)
        if tuple(item.time_s for item in keyframes) != tuple(
            sorted(item.time_s for item in keyframes)
        ):
            raise SimulationConfigError("person keyframes must be sorted by time")
        if keyframes[0].time_s != 0.0 or keyframes[-1].time_s < duration:
            raise SimulationConfigError(
                "person keyframes must cover the scenario duration"
            )
        obstacles = tuple(_obstacle(item) for item in raw_obstacles)
        acceptance = _acceptance(value["acceptance"])
        return cls(
            schema_version=value["schema_version"],
            scenario_id=scenario_id,
            duration_s=duration,
            step_s=step,
            keyframes=keyframes,
            obstacles=obstacles,
            acceptance=acceptance,
        )


@dataclass(frozen=True, slots=True)
class SimulationReport:
    passed: bool
    following_steps: int
    unsafe_forward_commands: int
    final_distance_error_m: float
    maximum_forward_speed_mps: float
    terminal_state: FollowState
    checks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "longship.follow-simulation-report.v0",
            "passed": self.passed,
            "following_steps": self.following_steps,
            "unsafe_forward_commands": self.unsafe_forward_commands,
            "final_distance_error_m": self.final_distance_error_m,
            "maximum_forward_speed_mps": self.maximum_forward_speed_mps,
            "terminal_state": self.terminal_state.value,
            "checks": list(self.checks),
        }


class FollowSimulationWorld:
    def __init__(
        self, scenario: FollowSimulationScenario, motion: RecordingMotion
    ) -> None:
        self.scenario = scenario
        self.motion = motion
        self.robot_x_m = 0.0
        self.robot_y_m = 0.0
        self.robot_yaw_rad = 0.0
        self.sequence = 0

    def scene(self, time_s: float, now_ns: int) -> FollowScene:
        self.sequence += 1
        person = self._person_at(time_s)
        person_robot = self._world_to_robot(person.world_x_m, person.world_y_m)
        active_obstacles = tuple(
            item
            for item in self.scenario.obstacles
            if item.active_from_s <= time_s <= item.active_until_s
        )
        obstacle_points = tuple(
            ObstaclePoint(
                *self._world_to_robot(item.world_x_m, item.world_y_m),
                item.radius_m,
            )
            for item in active_obstacles
        )
        tracks = (
            (
                PersonTrack(
                    track_id=person.track_id,
                    forward_m=person_robot[0],
                    left_m=person_robot[1],
                    confidence=0.92,
                ),
            )
            if person.visible and person_robot[0] > 0.0
            else ()
        )
        clearance_candidates = [
            obstacle.forward_m - obstacle.radius_m
            for obstacle in obstacle_points
            if obstacle.forward_m > 0.0
            and abs(obstacle.left_m) <= 0.45 + obstacle.radius_m
        ]
        if person_robot[0] > 0.0 and abs(person_robot[1]) <= 0.7:
            clearance_candidates.append(max(0.0, person_robot[0] - 0.25))
        raw_clearance = min(clearance_candidates, default=10.0)
        return FollowScene(
            sequence=self.sequence,
            captured_monotonic_ns=now_ns,
            received_monotonic_ns=now_ns,
            healthy=True,
            calibration_id="synthetic-camera-to-base-v0",
            calibration_valid=True,
            detector_ready=True,
            floor_valid=True,
            tracks=tracks,
            obstacles=obstacle_points,
            raw_forward_clearance_m=raw_clearance,
            detail="deterministic synthetic scene",
        )

    def advance(self, dt_s: float) -> None:
        command = self.motion.current_command
        if command is None:
            return
        forward = command.forward_mps * dt_s
        midpoint_yaw = self.robot_yaw_rad + command.yaw_rate_radps * dt_s / 2.0
        self.robot_x_m += math.cos(midpoint_yaw) * forward
        self.robot_y_m += math.sin(midpoint_yaw) * forward
        self.robot_yaw_rad = _wrap_angle(
            self.robot_yaw_rad + command.yaw_rate_radps * dt_s
        )

    def person_distance(self, time_s: float) -> float:
        person = self._person_at(time_s)
        return math.dist(
            (person.world_x_m, person.world_y_m),
            (self.robot_x_m, self.robot_y_m),
        )

    def _person_at(self, time_s: float) -> PersonKeyframe:
        keyframes = self.scenario.keyframes
        for first, second in zip(keyframes, keyframes[1:]):
            if first.time_s <= time_s <= second.time_s:
                span = second.time_s - first.time_s
                alpha = 0.0 if span == 0.0 else (time_s - first.time_s) / span
                return PersonKeyframe(
                    time_s=time_s,
                    world_x_m=first.world_x_m
                    + alpha * (second.world_x_m - first.world_x_m),
                    world_y_m=first.world_y_m
                    + alpha * (second.world_y_m - first.world_y_m),
                    visible=first.visible,
                    track_id=first.track_id,
                )
        return keyframes[-1]

    def _world_to_robot(self, world_x: float, world_y: float) -> tuple[float, float]:
        dx = world_x - self.robot_x_m
        dy = world_y - self.robot_y_m
        cosine = math.cos(self.robot_yaw_rad)
        sine = math.sin(self.robot_yaw_rad)
        return (cosine * dx + sine * dy, -sine * dx + cosine * dy)


def run_simulation(
    profile: FollowProfile,
    scenario: FollowSimulationScenario,
    *,
    event_sink: FollowEventSink | None = None,
    real_time: bool = False,
) -> SimulationReport:
    if abs(profile.control_period_s - scenario.step_s) > 1e-9:
        raise ValueError("scenario step must match the profile control period")
    motion = RecordingMotion()
    world = FollowSimulationWorld(scenario, motion)
    runtime = FollowPersonRuntime(
        profile,
        motion,
        planner=LocalFollowPlanner(profile.planner, profile.control),
        safety_guard=ForwardObstacleGuard(profile.safety),
        governor=MotionGovernor(profile.control),
        event_sink=event_sink,
        session_id=f"simulation:{scenario.scenario_id}",
    )
    now_ns = 1_000_000_000
    step_ns = int(scenario.step_s * 1_000_000_000)
    runtime.start(now_ns=now_ns - step_ns)
    following_steps = 0
    unsafe_commands = 0
    maximum_speed = 0.0
    step_count = round(scenario.duration_s / scenario.step_s)
    for index in range(step_count + 1):
        time_s = index * scenario.step_s
        scene = world.scene(time_s, now_ns)
        snapshot = runtime.tick(scene, now_ns=now_ns)
        command = snapshot.command
        if snapshot.state is FollowState.FOLLOWING:
            following_steps += 1
        if command is not None:
            maximum_speed = max(maximum_speed, abs(command.forward_mps))
            if (
                command.forward_mps > 0.0
                and scene.raw_forward_clearance_m is not None
                and scene.raw_forward_clearance_m
                <= profile.safety.nominal_stop_distance_m
            ):
                unsafe_commands += 1
        if snapshot.state is FollowState.FAILED:
            break
        world.advance(scenario.step_s)
        now_ns += step_ns
        if real_time:
            time.sleep(scenario.step_s)
    terminal_before_stop = runtime.state
    if runtime.is_active:
        runtime.stop("simulation complete", now_ns=now_ns)
    final_error = abs(
        world.person_distance(min(scenario.duration_s, index * scenario.step_s))
        - profile.control.desired_distance_m
    )
    criteria = scenario.acceptance
    checks = (
        f"following steps {following_steps} >= {criteria.minimum_following_steps}",
        f"unsafe forward commands {unsafe_commands} <= "
        f"{criteria.maximum_unsafe_forward_commands}",
        f"final distance error {final_error:.3f} m <= "
        f"{criteria.maximum_final_distance_error_m:.3f} m",
        f"runtime did not fail ({terminal_before_stop.value})",
    )
    passed = (
        following_steps >= criteria.minimum_following_steps
        and unsafe_commands <= criteria.maximum_unsafe_forward_commands
        and final_error <= criteria.maximum_final_distance_error_m
        and terminal_before_stop is not FollowState.FAILED
    )
    return SimulationReport(
        passed=passed,
        following_steps=following_steps,
        unsafe_forward_commands=unsafe_commands,
        final_distance_error_m=final_error,
        maximum_forward_speed_mps=maximum_speed,
        terminal_state=terminal_before_stop,
        checks=checks,
    )


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimulationConfigError(f"{field} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise SimulationConfigError(f"{field} must be a finite number")
    return converted


def _keyframe(value: object) -> PersonKeyframe:
    expected = {"time_s", "world_x_m", "world_y_m", "visible", "track_id"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SimulationConfigError("person keyframe has missing or unexpected fields")
    if type(value["visible"]) is not bool:
        raise SimulationConfigError("keyframe visible must be boolean")
    track_id = value["track_id"]
    if not isinstance(track_id, str) or not track_id:
        raise SimulationConfigError("keyframe track_id is required")
    return PersonKeyframe(
        time_s=_finite(value["time_s"], "time_s"),
        world_x_m=_finite(value["world_x_m"], "world_x_m"),
        world_y_m=_finite(value["world_y_m"], "world_y_m"),
        visible=value["visible"],
        track_id=track_id,
    )


def _obstacle(value: object) -> WorldObstacle:
    expected = {
        "world_x_m",
        "world_y_m",
        "radius_m",
        "active_from_s",
        "active_until_s",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SimulationConfigError("obstacle has missing or unexpected fields")
    obstacle = WorldObstacle(
        world_x_m=_finite(value["world_x_m"], "world_x_m"),
        world_y_m=_finite(value["world_y_m"], "world_y_m"),
        radius_m=_finite(value["radius_m"], "radius_m"),
        active_from_s=_finite(value["active_from_s"], "active_from_s"),
        active_until_s=_finite(value["active_until_s"], "active_until_s"),
    )
    if obstacle.radius_m <= 0.0 or obstacle.active_until_s < obstacle.active_from_s:
        raise SimulationConfigError("obstacle radius or active interval is invalid")
    return obstacle


def _acceptance(value: object) -> AcceptanceCriteria:
    expected = {
        "minimum_following_steps",
        "maximum_unsafe_forward_commands",
        "maximum_final_distance_error_m",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SimulationConfigError("acceptance has missing or unexpected fields")
    minimum_steps = value["minimum_following_steps"]
    maximum_unsafe = value["maximum_unsafe_forward_commands"]
    if type(minimum_steps) is not int or minimum_steps < 1:
        raise SimulationConfigError("minimum following steps must be positive")
    if type(maximum_unsafe) is not int or maximum_unsafe < 0:
        raise SimulationConfigError("maximum unsafe commands must be non-negative")
    maximum_error = _finite(
        value["maximum_final_distance_error_m"], "maximum_final_distance_error_m"
    )
    if maximum_error <= 0.0:
        raise SimulationConfigError("maximum final distance error must be positive")
    return AcceptanceCriteria(minimum_steps, maximum_unsafe, maximum_error)


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

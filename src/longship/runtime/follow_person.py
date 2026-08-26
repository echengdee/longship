from __future__ import annotations

import math
import time
import uuid
from typing import Any, Mapping, Protocol

from longship.contracts.skills.follow_person import (
    FollowCommand,
    FollowScene,
    FollowSnapshot,
    FollowState,
    MotionReceipt,
    ObstaclePoint,
    PersonTrack,
    PlanDecision,
    SafetyDecision,
)
from longship.skills.follow_person.config import FollowProfile


class FollowMotionPort(Protocol):
    def acquire(self, session_id: str, now_ns: int) -> MotionReceipt:
        ...

    def apply(self, command: FollowCommand) -> MotionReceipt:
        ...

    def protective_stop(self, reason: str) -> MotionReceipt:
        ...


class FollowEventSink(Protocol):
    def publish(self, event: Mapping[str, Any]) -> None:
        ...


class FollowPlannerPort(Protocol):
    def plan(
        self,
        target_robot_xy_m: tuple[float, float],
        obstacles: tuple[ObstaclePoint, ...],
        *,
        standoff_m: float,
        goal_tolerance_m: float,
    ) -> PlanDecision:
        ...


class FollowSafetyPort(Protocol):
    def apply(
        self,
        plan: PlanDecision,
        *,
        raw_clearance_m: float | None,
        current_forward_mps: float,
    ) -> SafetyDecision:
        ...


class MotionGovernorPort(Protocol):
    def apply(
        self, desired_forward_mps: float, desired_yaw_rate_radps: float, dt_s: float
    ) -> tuple[float, float]:
        ...

    def emergency_zero(self) -> tuple[float, float]:
        ...


class NullEventSink:
    def publish(self, event: Mapping[str, Any]) -> None:
        del event


class FollowPersonRuntime:
    """Own the FollowPerson session, freshness gates, and actuator authority.

    Person tracking and local planning remain provider concerns. This class
    admits their bounded outputs, owns state and command sequencing, applies an
    independent raw-clearance guard, and fences every command with an expiry.
    """

    _ACTIVE_STATES = frozenset(
        {
            FollowState.ACQUIRING,
            FollowState.FOLLOWING,
            FollowState.HOLDING,
            FollowState.LOST_APPROACH,
            FollowState.BLOCKED,
            FollowState.PAUSED,
        }
    )

    def __init__(
        self,
        profile: FollowProfile,
        motion: FollowMotionPort,
        *,
        planner: FollowPlannerPort,
        safety_guard: FollowSafetyPort,
        governor: MotionGovernorPort,
        event_sink: FollowEventSink | None = None,
        session_id: str | None = None,
        required_calibration_id: str | None = None,
    ) -> None:
        self.profile = profile
        self.motion = motion
        self.planner = planner
        self.safety_guard = safety_guard
        self.event_sink = event_sink or NullEventSink()
        self.session_id = session_id or f"follow-{uuid.uuid4().hex[:16]}"
        if required_calibration_id is not None and (
            not isinstance(required_calibration_id, str)
            or not required_calibration_id.strip()
        ):
            raise ValueError("required calibration identity must be non-empty")
        self.required_calibration_id = required_calibration_id
        self.state = FollowState.IDLE
        self.revision = 0
        self.locked_track_id: str | None = None
        self._predicted_target_xy: tuple[float, float] | None = None
        self._command_sequence = 0
        self._last_command: FollowCommand | None = None
        self._last_tick_ns: int | None = None
        self._phase_started_ns: int | None = None
        self._health_failure_started_ns: int | None = None
        self._lost_started_ns: int | None = None
        self._blocked_started_ns: int | None = None
        self._last_scene_sequence: int | None = None
        self._resume_reconfirmation = False
        self._governor = governor
        self.last_observability_error: str | None = None
        self._last_snapshot = self._snapshot(None, (), "not started")

    @property
    def snapshot(self) -> FollowSnapshot:
        return self._last_snapshot

    @property
    def is_active(self) -> bool:
        return self.state in self._ACTIVE_STATES

    def start(self, *, now_ns: int | None = None) -> FollowSnapshot:
        now = time.monotonic_ns() if now_ns is None else now_ns
        if self.state is not FollowState.IDLE:
            raise RuntimeError(f"cannot start from {self.state.value}")
        try:
            receipt = self.motion.acquire(self.session_id, now)
        except Exception as exc:
            return self._fail(
                now, f"motion authority raised {type(exc).__name__}"
            )
        if not receipt.accepted:
            return self._fail(now, f"motion authority rejected: {receipt.detail}")
        self._phase_started_ns = now
        self._last_tick_ns = now
        self._set_state(FollowState.ACQUIRING)
        return self._record(None, (), "waiting for a person target")

    def tick(self, scene: FollowScene, *, now_ns: int | None = None) -> FollowSnapshot:
        now = time.monotonic_ns() if now_ns is None else now_ns
        if self.state not in self._ACTIVE_STATES:
            raise RuntimeError(f"cannot tick while runtime is {self.state.value}")
        if self._last_tick_ns is None:
            raise RuntimeError("runtime has not been started")
        dt_s = (now - self._last_tick_ns) / 1_000_000_000
        if dt_s <= 0.0:
            return self._fail(now, "control clock did not advance", scene)
        if dt_s > self.profile.runtime.command_ttl_s:
            return self._fail(now, "control loop exceeded the command TTL", scene)
        self._advance_last_seen(dt_s)
        self._last_tick_ns = now

        scene_problem = self._scene_problem(scene, now)
        if scene_problem is not None:
            return self._handle_unhealthy_scene(scene, now, scene_problem)
        self._health_failure_started_ns = None
        if (
            self._last_scene_sequence is None
            or scene.sequence > self._last_scene_sequence
        ):
            self._last_scene_sequence = scene.sequence

        if self.state is FollowState.PAUSED:
            return self._command_zero(scene, now, "operator pause", ())

        target = self._resolve_target(scene)
        if self._resume_reconfirmation and target is None:
            assert self._phase_started_ns is not None
            if self._elapsed_s(self._phase_started_ns, now) > (
                self.profile.runtime.acquire_timeout_s
            ):
                return self._fail(
                    now,
                    "paused target was not freshly reconfirmed; "
                    "supervisor restart required",
                    scene,
                )
            self._set_state(FollowState.ACQUIRING)
            return self._command_zero(
                scene, now, "waiting to reconfirm the paused target", ()
            )
        if target is not None:
            self._resume_reconfirmation = False
            self.locked_track_id = target.track_id
            self._predicted_target_xy = (target.forward_m, target.left_m)
            self._lost_started_ns = None
            self._set_state(FollowState.FOLLOWING)
            planning_target = self._predicted_target_xy
            standoff_m = self.profile.control.desired_distance_m
            goal_tolerance_m = self.profile.control.distance_deadband_m
            lost_mode = False
        elif self.locked_track_id is None:
            assert self._phase_started_ns is not None
            if self._elapsed_s(self._phase_started_ns, now) > (
                self.profile.runtime.acquire_timeout_s
            ):
                return self._fail(now, "no eligible person was acquired", scene)
            self._set_state(FollowState.ACQUIRING)
            return self._command_zero(scene, now, "waiting for person", ())
        else:
            if self._lost_started_ns is None:
                self._lost_started_ns = now
            if self._predicted_target_xy is None:
                return self._fail(now, "locked target has no last-seen position", scene)
            if self._elapsed_s(self._lost_started_ns, now) > (
                self.profile.runtime.lost_target_timeout_s
            ):
                return self._fail(
                    now,
                    "last-seen recovery timed out; supervisor restart required",
                    scene,
                )
            self._set_state(FollowState.LOST_APPROACH)
            planning_target = self._predicted_target_xy
            standoff_m = self.profile.control.lost_target_standoff_m
            goal_tolerance_m = self.profile.control.lost_target_goal_tolerance_m
            lost_mode = True

        try:
            plan = self.planner.plan(
                planning_target,
                scene.obstacles,
                standoff_m=standoff_m,
                goal_tolerance_m=goal_tolerance_m,
            )
        except Exception as exc:
            return self._fail(now, f"planner raised {type(exc).__name__}", scene)
        if not isinstance(plan, PlanDecision):
            return self._fail(now, "planner returned an invalid decision", scene)
        if lost_mode and plan.reached:
            return self._fail(
                now,
                "reached the last-seen vicinity without reacquisition; "
                "supervisor restart required",
                scene,
            )

        if plan.blocked:
            return self._handle_blocked(scene, now, plan.detail, plan.path_robot_xy_m)

        try:
            guard = self._guard(scene, plan)
        except Exception as exc:
            return self._fail(now, f"safety guard raised {type(exc).__name__}", scene)
        if not isinstance(guard, SafetyDecision):
            return self._fail(now, "safety guard returned an invalid decision", scene)
        if guard.blocked:
            return self._handle_blocked(scene, now, guard.detail, plan.path_robot_xy_m)
        self._blocked_started_ns = None
        if target is None:
            self._set_state(FollowState.LOST_APPROACH)
        else:
            self._set_state(FollowState.FOLLOWING)
        try:
            governed = self._governor.apply(
                guard.forward_mps, guard.yaw_rate_radps, dt_s
            )
            forward, yaw = governed
        except Exception as exc:
            return self._fail(now, f"motion governor raised {type(exc).__name__}", scene)
        return self._send(
            scene,
            now,
            forward,
            yaw,
            guard.detail,
            plan.path_robot_xy_m,
        )

    def pause(self, *, now_ns: int | None = None) -> FollowSnapshot:
        now = time.monotonic_ns() if now_ns is None else now_ns
        if self.state not in self._ACTIVE_STATES - {FollowState.PAUSED}:
            raise RuntimeError(f"cannot pause from {self.state.value}")
        self._set_state(FollowState.PAUSED)
        return self._command_zero(None, now, "operator pause", ())

    def resume(self, *, now_ns: int | None = None) -> FollowSnapshot:
        now = time.monotonic_ns() if now_ns is None else now_ns
        if self.state is not FollowState.PAUSED:
            raise RuntimeError(f"cannot resume from {self.state.value}")
        self._phase_started_ns = now
        self._health_failure_started_ns = None
        self._blocked_started_ns = None
        self._resume_reconfirmation = True
        self._set_state(FollowState.ACQUIRING)
        return self._record(None, (), "resume accepted; target must be observed again")

    def stop(
        self, reason: str = "operator stop", *, now_ns: int | None = None
    ) -> FollowSnapshot:
        del now_ns
        if self.state in {FollowState.STOPPED, FollowState.STOP_UNVERIFIED}:
            return self._last_snapshot
        if self.state is FollowState.IDLE:
            raise RuntimeError("runtime has not been started")
        self._governor.emergency_zero()
        try:
            receipt = self.motion.protective_stop(reason)
        except Exception as exc:
            receipt = MotionReceipt(
                False, f"stop transport raised {type(exc).__name__}"
            )
        final_state = (
            FollowState.STOPPED
            if receipt.verified_stopped
            else FollowState.STOP_UNVERIFIED
        )
        self._set_state(final_state)
        detail = (
            f"{reason}; target stop verified"
            if receipt.verified_stopped
            else f"{reason}; target stop is unverified: {receipt.detail}"
        )
        self._last_snapshot = self._snapshot(
            None, (), detail, stop_verified=receipt.verified_stopped
        )
        self._publish(None)
        return self._last_snapshot

    def _resolve_target(self, scene: FollowScene) -> PersonTrack | None:
        eligible = tuple(
            track
            for track in scene.tracks
            if track.confidence >= self.profile.control.minimum_track_confidence
            and track.forward_m > 0.0
        )
        if not eligible:
            return None
        if self.locked_track_id is not None:
            for track in eligible:
                if track.track_id == self.locked_track_id:
                    if (
                        self._lost_started_ns is None
                        or self._within_reacquire_gate(track)
                    ):
                        return track
        if self._lost_started_ns is not None and self._predicted_target_xy is not None:
            nearby = tuple(
                track for track in eligible if self._within_reacquire_gate(track)
            )
            if nearby:
                return min(
                    nearby,
                    key=lambda item: math.dist(
                        (item.forward_m, item.left_m), self._predicted_target_xy
                    ),
                )
            return None
        if self.locked_track_id is not None:
            return None
        return min(
            eligible,
            key=lambda item: abs(item.bearing_rad) + 0.05 * item.distance_m,
        )

    def _within_reacquire_gate(self, track: PersonTrack) -> bool:
        assert self._predicted_target_xy is not None
        return math.dist(
            (track.forward_m, track.left_m), self._predicted_target_xy
        ) <= self.profile.control.reacquire_gate_m

    def _scene_problem(self, scene: FollowScene, now_ns: int) -> str | None:
        if (
            self._last_scene_sequence is not None
            and scene.sequence < self._last_scene_sequence
        ):
            return "scene sequence moved backwards"
        if scene.captured_monotonic_ns > now_ns + 50_000_000:
            return "scene timestamp is in the future"
        if scene.age_s(now_ns) > self.profile.runtime.scene_max_age_s:
            return "scene is stale"
        if not scene.healthy:
            return scene.detail or "scene provider is unhealthy"
        if not scene.calibration_valid:
            return "camera calibration is not validated"
        if (
            self.required_calibration_id is not None
            and scene.calibration_id != self.required_calibration_id
        ):
            return "camera calibration identity changed during the session"
        if not scene.detector_ready:
            return "person detector is unavailable"
        if not scene.floor_valid:
            return "ground-plane estimate is invalid"
        if scene.raw_forward_clearance_m is None:
            return "raw obstacle clearance is unavailable"
        return None

    def _handle_unhealthy_scene(
        self, scene: FollowScene, now_ns: int, problem: str
    ) -> FollowSnapshot:
        if self._health_failure_started_ns is None:
            self._health_failure_started_ns = now_ns
        self._set_state(FollowState.HOLDING)
        if self._elapsed_s(self._health_failure_started_ns, now_ns) > (
            self.profile.runtime.scene_failure_grace_s
        ):
            return self._fail(
                now_ns,
                f"scene remained unhealthy: {problem}; supervisor restart required",
                scene,
            )
        return self._command_zero(scene, now_ns, f"transient scene hold: {problem}", ())

    def _handle_blocked(
        self,
        scene: FollowScene,
        now_ns: int,
        detail: str,
        path: tuple[tuple[float, float], ...],
    ) -> FollowSnapshot:
        if self._blocked_started_ns is None:
            self._blocked_started_ns = now_ns
        self._set_state(FollowState.BLOCKED)
        if self._elapsed_s(self._blocked_started_ns, now_ns) > (
            self.profile.runtime.blocked_timeout_s
        ):
            return self._fail(
                now_ns,
                f"local route remained blocked: {detail}; supervisor restart required",
                scene,
            )
        return self._command_zero(scene, now_ns, detail, path)

    def _guard(self, scene: FollowScene, plan: PlanDecision) -> SafetyDecision:
        return self.safety_guard.apply(
            plan,
            raw_clearance_m=scene.raw_forward_clearance_m,
            current_forward_mps=(
                self._last_command.forward_mps if self._last_command else 0.0
            ),
        )

    def _command_zero(
        self,
        scene: FollowScene | None,
        now_ns: int,
        reason: str,
        path: tuple[tuple[float, float], ...],
    ) -> FollowSnapshot:
        self._governor.emergency_zero()
        return self._send(scene, now_ns, 0.0, 0.0, reason, path)

    def _send(
        self,
        scene: FollowScene | None,
        now_ns: int,
        forward_mps: float,
        yaw_rate_radps: float,
        reason: str,
        path: tuple[tuple[float, float], ...],
    ) -> FollowSnapshot:
        self._command_sequence += 1
        try:
            command = FollowCommand(
                session_id=self.session_id,
                sequence=self._command_sequence,
                issued_monotonic_ns=now_ns,
                expires_monotonic_ns=now_ns
                + int(self.profile.runtime.command_ttl_s * 1_000_000_000),
                forward_mps=forward_mps,
                yaw_rate_radps=yaw_rate_radps,
                reason=reason,
            )
        except (TypeError, ValueError) as exc:
            return self._fail(
                now_ns, f"command validation raised {type(exc).__name__}", scene
            )
        try:
            receipt = self.motion.apply(command)
        except Exception as exc:
            return self._fail(
                now_ns, f"motion command raised {type(exc).__name__}", scene
            )
        if not receipt.accepted:
            return self._fail(
                now_ns, f"motion command rejected: {receipt.detail}", scene
            )
        self._last_command = command
        return self._record(scene, path, reason, command=command)

    def _fail(
        self,
        now_ns: int,
        detail: str,
        scene: FollowScene | None = None,
    ) -> FollowSnapshot:
        del now_ns
        if self.state is FollowState.FAILED:
            return self._last_snapshot
        self._governor.emergency_zero()
        try:
            receipt = self.motion.protective_stop(detail)
        except Exception as exc:
            receipt = MotionReceipt(
                False, f"stop transport raised {type(exc).__name__}"
            )
        self._set_state(FollowState.FAILED)
        combined = detail
        if not receipt.verified_stopped:
            combined += f"; target stop is unverified: {receipt.detail}"
        self._last_snapshot = self._snapshot(
            scene,
            (),
            combined,
            stop_verified=receipt.verified_stopped,
        )
        self._publish(scene)
        return self._last_snapshot

    def _advance_last_seen(self, dt_s: float) -> None:
        if self._predicted_target_xy is None or self._last_command is None:
            return
        forward = self._predicted_target_xy[0] - self._last_command.forward_mps * dt_s
        left = self._predicted_target_xy[1]
        yaw = self._last_command.yaw_rate_radps * dt_s
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        self._predicted_target_xy = (
            cosine * forward + sine * left,
            -sine * forward + cosine * left,
        )

    def _record(
        self,
        scene: FollowScene | None,
        path: tuple[tuple[float, float], ...],
        detail: str,
        *,
        command: FollowCommand | None = None,
    ) -> FollowSnapshot:
        self.revision += 1
        self._last_snapshot = self._snapshot(scene, path, detail, command=command)
        self._publish(scene)
        return self._last_snapshot

    def _snapshot(
        self,
        scene: FollowScene | None,
        path: tuple[tuple[float, float], ...],
        detail: str,
        *,
        command: FollowCommand | None = None,
        stop_verified: bool | None = None,
    ) -> FollowSnapshot:
        return FollowSnapshot(
            session_id=self.session_id,
            state=self.state,
            revision=self.revision,
            scene_sequence=scene.sequence if scene else self._last_scene_sequence,
            locked_track_id=self.locked_track_id,
            target_robot_xy_m=self._predicted_target_xy,
            command=command,
            path_robot_xy_m=path,
            detail=detail,
            stop_verified=stop_verified,
        )

    def _publish(self, scene: FollowScene | None) -> None:
        event: dict[str, Any] = {
            "schema_version": "longship.follow-runtime-event.v1",
            "snapshot": self._last_snapshot.to_dict(),
            "scene": scene.to_dict() if scene else None,
        }
        try:
            self.event_sink.publish(event)
        except Exception as exc:
            self.last_observability_error = type(exc).__name__

    def _set_state(self, state: FollowState) -> None:
        if self.state is not state:
            self.state = state
            self.revision += 1

    @staticmethod
    def _elapsed_s(start_ns: int, now_ns: int) -> float:
        return max(0.0, (now_ns - start_ns) / 1_000_000_000)

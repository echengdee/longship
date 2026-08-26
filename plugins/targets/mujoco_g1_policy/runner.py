from __future__ import annotations

import argparse
import asyncio
import json
import math
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from longship.artifacts import (
    ArtifactStore,
    VerifiedArtifact,
    load_model_artifact_manifest,
    read_verified_artifact_bytes,
    sha256_directory,
)
from longship.brain.codex_follow import CodexFollowBrain
from longship.contracts.skills.follow_person import (
    FollowCommand,
    FollowScene,
    FollowState,
    MotionReceipt,
    ObstaclePoint,
    PersonTrack,
)
from longship.observability.follow_person import (
    CompositeEventSink,
    FollowDashboard,
    JsonlEventSink,
)
from longship.runtime.follow_person import NullEventSink
from longship.policies import (
    UNITREE_RL_GYM_G1_ACTION_SPACE,
    UNITREE_RL_GYM_G1_RESOURCE_SCOPE,
    PolicyActionFrame,
    PolicyCandidate,
    PolicyCandidateRejected,
    PolicyRequest,
    UnitreeRLGymG1Observation,
    UnitreeRLGymTorchScriptRunner,
    guard_candidate,
    unitree_rl_gym_g1_guard_profile,
)
from longship.simulation.follow_person import (
    FollowSimulationScenario,
    PersonKeyframe,
)
from longship.simulation.follow_stack import run_interactive_follow_stack
from longship.simulation.follow_system import run_system_with_world
from longship.skills.follow_person.config import FollowProfile


_CONFIG_FIELDS = {
    "policy_path",
    "xml_path",
    "simulation_duration",
    "simulation_dt",
    "control_decimation",
    "kps",
    "kds",
    "default_angles",
    "ang_vel_scale",
    "dof_pos_scale",
    "dof_vel_scale",
    "action_scale",
    "cmd_scale",
    "num_actions",
    "num_obs",
    "cmd_init",
}
_G1_ACTIONS = 12
_G1_OBSERVATIONS = 47
_GAIT_PERIOD_S = 0.8
_MAXIMUM_ABSOLUTE_RAW_ACTION = 10.0
_STOP_SETTLE_S = 1.5
_STOP_MEASUREMENT_S = 1.5
_PLUGIN_DIRECTORY = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _PLUGIN_DIRECTORY.parents[2]
_MODEL_MANIFEST_ID = (
    "longship.unitree-rl-gym.g1-12dof.official-276801e"
)


@dataclass(frozen=True, slots=True)
class ExternalPolicyConfig:
    simulation_dt_s: float
    control_decimation: int
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    default_position: tuple[float, ...]
    angular_velocity_scale: float
    joint_position_scale: float
    joint_velocity_scale: float
    action_scale: float
    command_scale: tuple[float, float, float]

    @property
    def policy_period_s(self) -> float:
        return self.simulation_dt_s * self.control_decimation

    @classmethod
    def load(cls, path: Path) -> "ExternalPolicyConfig":
        if path.stat().st_size > 64 * 1024:
            raise ValueError("external policy config exceeds 64 KiB")
        return cls._load_bytes(path.read_bytes())

    @classmethod
    def load_verified(cls, artifact: VerifiedArtifact) -> "ExternalPolicyConfig":
        return cls._load_bytes(
            read_verified_artifact_bytes(artifact, maximum_size_bytes=64 * 1024)
        )

    @classmethod
    def _load_bytes(cls, payload: bytes) -> "ExternalPolicyConfig":
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("external policy config is not UTF-8") from exc
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    "PyYAML is required to load a non-JSON external G1 config"
                ) from exc
            value = yaml.safe_load(text)
        if not isinstance(value, dict) or set(value) != _CONFIG_FIELDS:
            raise ValueError("external policy config has an unexpected shape")
        if value["num_actions"] != _G1_ACTIONS:
            raise ValueError("external policy must have exactly 12 actions")
        if value["num_obs"] != _G1_OBSERVATIONS:
            raise ValueError("external policy must have exactly 47 observations")
        simulation_dt = _finite(value["simulation_dt"], "simulation_dt")
        decimation = value["control_decimation"]
        if not 0.001 <= simulation_dt <= 0.005:
            raise ValueError("external simulation timestep is out of range")
        if type(decimation) is not int or not 1 <= decimation <= 50:
            raise ValueError("external control decimation is out of range")
        config = cls(
            simulation_dt_s=simulation_dt,
            control_decimation=decimation,
            stiffness=_vector(value["kps"], _G1_ACTIONS, "kps"),
            damping=_vector(value["kds"], _G1_ACTIONS, "kds"),
            default_position=_vector(
                value["default_angles"], _G1_ACTIONS, "default_angles"
            ),
            angular_velocity_scale=_finite(
                value["ang_vel_scale"], "ang_vel_scale"
            ),
            joint_position_scale=_finite(
                value["dof_pos_scale"], "dof_pos_scale"
            ),
            joint_velocity_scale=_finite(
                value["dof_vel_scale"], "dof_vel_scale"
            ),
            action_scale=_finite(value["action_scale"], "action_scale"),
            command_scale=_vector(value["cmd_scale"], 3, "cmd_scale"),
        )
        if not 0.005 <= config.policy_period_s <= 0.05:
            raise ValueError("external policy period is out of range")
        if any(item <= 0.0 for item in config.stiffness + config.damping):
            raise ValueError("external PD gains must be positive")
        if not 0.0 < config.action_scale <= 1.0:
            raise ValueError("external action scale is out of range")
        return config


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    manifest_id: str
    manifest_sha256: str
    scene_bundle_sha256: str
    policy_sha256: str
    config_sha256: str
    license_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "scene_bundle_sha256": self.scene_bundle_sha256,
            "policy_sha256": self.policy_sha256,
            "config_sha256": self.config_sha256,
            "license_sha256": self.license_sha256,
        }


@dataclass(frozen=True, slots=True)
class VerifiedExternalG1Artifacts:
    identity: ArtifactIdentity
    policy: VerifiedArtifact
    config: VerifiedArtifact
    license: VerifiedArtifact


class G1PolicyVelocityTarget:
    """Lease and freshness boundary for an in-process simulation policy."""

    def __init__(
        self,
        *,
        maximum_forward_mps: float,
        maximum_yaw_rate_radps: float,
    ) -> None:
        self.maximum_forward_mps = maximum_forward_mps
        self.maximum_yaw_rate_radps = maximum_yaw_rate_radps
        self.session_id: str | None = None
        self.lease_epoch = 0
        self.command: FollowCommand | None = None
        self.now_ns = 0
        self.stop_verified = False
        self._stop_handler: Callable[[], tuple[bool, str]] | None = None

    def bind_stop_handler(self, handler: Callable[[], tuple[bool, str]]) -> None:
        if self._stop_handler is not None:
            raise RuntimeError("G1 simulation stop handler is already bound")
        self._stop_handler = handler

    def set_time(self, now_ns: int) -> None:
        self.now_ns = now_ns

    def acquire(self, session_id: str, now_ns: int) -> MotionReceipt:
        if self.session_id is not None:
            return MotionReceipt(False, "G1 simulation authority is already held")
        self.session_id = session_id
        self.lease_epoch += 1
        self.now_ns = now_ns
        return MotionReceipt(True, "G1 policy simulation authority acquired")

    def policy_lease_is_current(self, request: PolicyRequest) -> bool:
        return bool(
            self.session_id is not None
            and request.lease_id == self.session_id
            and request.lease_epoch == self.lease_epoch
        )

    def apply(self, command: FollowCommand) -> MotionReceipt:
        if command.session_id != self.session_id:
            return MotionReceipt(False, "G1 simulation command owner mismatch")
        if abs(command.forward_mps) > self.maximum_forward_mps + 1e-9:
            return MotionReceipt(False, "G1 simulation forward command exceeds limit")
        if abs(command.yaw_rate_radps) > self.maximum_yaw_rate_radps + 1e-9:
            return MotionReceipt(False, "G1 simulation yaw command exceeds limit")
        if command.expires_monotonic_ns <= self.now_ns:
            return MotionReceipt(False, "G1 simulation command is expired")
        self.command = command
        return MotionReceipt(True, "G1 policy setpoint accepted")

    def desired_velocity(self) -> tuple[float, float, float]:
        command = self.command
        if command is None or command.expires_monotonic_ns <= self.now_ns:
            return (0.0, 0.0, 0.0)
        forward = max(
            -self.maximum_forward_mps,
            min(self.maximum_forward_mps, command.forward_mps),
        )
        yaw = max(
            -self.maximum_yaw_rate_radps,
            min(self.maximum_yaw_rate_radps, command.yaw_rate_radps),
        )
        return (forward, 0.0, yaw)

    def protective_stop(self, reason: str) -> MotionReceipt:
        del reason
        self.command = None
        if self._stop_handler is None:
            return MotionReceipt(False, "G1 simulation stop monitor is unavailable")
        verified, detail = self._stop_handler()
        self.stop_verified = verified
        return MotionReceipt(True, detail, verified_stopped=verified)


class ExternalG1PolicyWorld:
    """Official-asset G1 dynamics driven by an external TorchScript policy."""

    def __init__(
        self,
        scenario: FollowSimulationScenario,
        target: G1PolicyVelocityTarget,
        config: ExternalPolicyConfig,
        *,
        scene_path: Path,
        policy_artifact: VerifiedArtifact,
        show_viewer: bool,
        enable_hud_camera: bool,
        desired_follow_distance_m: float,
    ) -> None:
        try:
            import mujoco
            import numpy
        except ImportError as exc:
            raise RuntimeError(
                "external G1 simulation requires mujoco, numpy, and torch"
            ) from exc
        if tuple(int(item) for item in mujoco.__version__.split(".")[:2]) < (3, 11):
            raise RuntimeError("external G1 simulation requires MuJoCo 3.11+")
        self.mujoco = mujoco
        self.numpy = numpy
        self.scenario = scenario
        self.target = target
        self.config = config
        self.desired_follow_distance_m = desired_follow_distance_m
        self.enable_hud_camera = enable_hud_camera
        self.spec = mujoco.MjSpec.from_file(str(scene_path))
        self._add_scenario_bodies()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = config.simulation_dt_s
        if self.model.nq != 7 + _G1_ACTIONS:
            raise ValueError("external G1 model does not expose 12 policy joints")
        if self.model.nv != 6 + _G1_ACTIONS or self.model.nu != _G1_ACTIONS:
            raise ValueError("external G1 model dynamics do not match the policy")
        self.policy = UnitreeRLGymTorchScriptRunner(policy_artifact)
        policy_step_ms = round(config.policy_period_s * 1000.0)
        if not math.isclose(
            policy_step_ms / 1000.0,
            config.policy_period_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("external policy period must be an integer millisecond")
        self.policy_guard_profile = unitree_rl_gym_g1_guard_profile(
            policy_step_ms=policy_step_ms,
            maximum_absolute_action=_MAXIMUM_ABSOLUTE_RAW_ACTION,
        )
        self.model_binding_id = f"sha256:{policy_artifact.sha256}"
        self.stiffness = numpy.asarray(config.stiffness, dtype=numpy.float32)
        self.damping = numpy.asarray(config.damping, dtype=numpy.float32)
        self.default_position = numpy.asarray(
            config.default_position, dtype=numpy.float32
        )
        self.command_scale = numpy.asarray(
            config.command_scale, dtype=numpy.float32
        )
        self.action = numpy.zeros(_G1_ACTIONS, dtype=numpy.float32)
        self.target_position = self.default_position.copy()
        self.data.qpos[7:] = self.default_position
        self.sequence = 0
        self.physics_steps = 0
        self.policy_steps = 0
        self.maximum_absolute_policy_action = 0.0
        self.contact_steps = 0
        self.fallen = False
        self.current_tilt_rad = 0.0
        self.maximum_tilt_rad = 0.0
        self.stop_measurements: dict[str, float] = {}
        self._camera_renderer: Any | None = None
        self._camera_encoder: Any | None = None
        self._camera_frame: tuple[int, bytes] | None = None
        self._camera_frame_physics_step = -1
        self._last_time_s = 0.0
        self._person_mocap = self._mocap_index("walking_person")
        self._obstacle_mocap = tuple(
            self._mocap_index(f"obstacle_{index}")
            for index in range(len(scenario.obstacles))
        )
        self.viewer = self._launch_viewer() if show_viewer else None
        self.target.bind_stop_handler(self._settle_and_verify_stop)
        mujoco.mj_forward(self.model, self.data)
        if enable_hud_camera:
            self._start_hud_camera()

    def close(self) -> None:
        renderer = self._camera_renderer
        self._camera_renderer = None
        if renderer is not None:
            renderer.close()
        viewer = self.viewer
        self.viewer = None
        if viewer is not None:
            viewer.close()
            # launch_passive owns a native render thread. Give it time to
            # release model/data references before interpreter teardown.
            time.sleep(0.25)

    def scene(self, time_s: float, now_ns: int) -> FollowScene:
        self.target.set_time(now_ns)
        self._last_time_s = time_s
        person = self._update_fixtures(time_s)
        self.sequence += 1
        person_robot = self._world_to_robot(person.world_x_m, person.world_y_m)
        active_obstacles = tuple(
            item
            for item in self.scenario.obstacles
            if item.active_from_s <= time_s <= item.active_until_s
        )
        obstacles = tuple(
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
                    confidence=0.97,
                ),
            )
            if person.visible and person_robot[0] > 0.0
            else ()
        )
        clearances = [
            item.forward_m - item.radius_m
            for item in obstacles
            if item.forward_m > 0.0
            and abs(item.left_m) <= 0.45 + item.radius_m
        ]
        if person_robot[0] > 0.0 and abs(person_robot[1]) <= 0.7:
            clearances.append(max(0.0, person_robot[0] - 0.25))
        return FollowScene(
            sequence=self.sequence,
            captured_monotonic_ns=now_ns,
            received_monotonic_ns=now_ns,
            healthy=not self.fallen,
            calibration_id="external-g1-ground-truth-to-base-v0",
            calibration_valid=True,
            detector_ready=True,
            floor_valid=not self.fallen,
            tracks=tracks,
            obstacles=obstacles,
            raw_forward_clearance_m=min(clearances, default=10.0),
            detail=(
                "external G1 policy simulation"
                if not self.fallen
                else "G1 fall detector latched"
            ),
        )

    def advance(self, dt_s: float) -> None:
        step_count = round(dt_s / self.config.simulation_dt_s)
        if abs(step_count * self.config.simulation_dt_s - dt_s) > 1e-9:
            raise ValueError("Runtime period is not divisible by G1 physics timestep")
        self._advance_steps(step_count)
        if self._has_barrier_contact():
            self.contact_steps += 1
        self._update_fall_state()
        if self.viewer is not None:
            self.viewer.sync()

    def person_distance(self, time_s: float) -> float:
        person = self._person_at(time_s)
        robot_x, robot_y, _ = self.robot_pose()
        return math.hypot(person.world_x_m - robot_x, person.world_y_m - robot_y)

    def robot_pose(self) -> tuple[float, float, float]:
        quaternion = self.data.qpos[3:7]
        yaw = math.atan2(
            2.0 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
            1.0 - 2.0 * (quaternion[2] ** 2 + quaternion[3] ** 2),
        )
        return (float(self.data.qpos[0]), float(self.data.qpos[1]), yaw)

    def target_telemetry(self) -> dict[str, Any]:
        robot_x, robot_y, robot_yaw = self.robot_pose()
        person = self._person_at(self._last_time_s)
        delta_x = person.world_x_m - robot_x
        delta_y = person.world_y_m - robot_y
        distance = math.hypot(delta_x, delta_y)
        if distance > 1e-6:
            scale = max(0.0, distance - self.desired_follow_distance_m) / distance
            goal = (robot_x + delta_x * scale, robot_y + delta_y * scale)
        else:
            goal = (robot_x, robot_y)
        return {
            "schema_version": "longship.follow-target-telemetry.v0",
            "provider": "external-unitree-rl-gym-g1-12dof",
            "simulation_time_s": float(self.data.time),
            "robot_world_pose": [robot_x, robot_y, robot_yaw],
            "base_height_m": float(self.data.qpos[2]),
            "tilt_rad": self.current_tilt_rad,
            "fallen": self.fallen,
            "person_world_xy_m": [person.world_x_m, person.world_y_m],
            "follow_goal_world_xy_m": [goal[0], goal[1]],
            "desired_velocity": list(self.target.desired_velocity()),
            "barrier_contact_steps": self.contact_steps,
            "physics_steps": self.physics_steps,
            "policy_steps": self.policy_steps,
            "maximum_absolute_policy_action": self.maximum_absolute_policy_action,
            "policy_candidate_guard": True,
        }

    def render_camera_jpeg(self) -> tuple[int, bytes] | None:
        renderer = self._camera_renderer
        encoder = self._camera_encoder
        if renderer is None or encoder is None:
            return None
        minimum_steps = max(1, round(0.2 / self.config.simulation_dt_s))
        if (
            self._camera_frame is not None
            and self.physics_steps - self._camera_frame_physics_step < minimum_steps
        ):
            return self._camera_frame
        renderer.update_scene(self.data, camera="longship_follow_camera")
        rgb = renderer.render()
        accepted, encoded = encoder.imencode(".jpg", rgb[:, :, ::-1])
        if not accepted:
            raise RuntimeError("G1 HUD camera JPEG encoding failed")
        self._camera_frame = (self.sequence, encoded.tobytes())
        self._camera_frame_physics_step = self.physics_steps
        return self._camera_frame

    def _add_scenario_bodies(self) -> None:
        if self.enable_hud_camera:
            self.spec.body("pelvis").add_camera(
                name="longship_follow_camera",
                pos=[0.2, 0.0, 0.55],
                xyaxes=[0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
                fovy=70.0,
            )
        person = self.spec.worldbody.add_body(
            name="walking_person", mocap=True, pos=[2.2, 0.0, 0.9]
        )
        person.add_geom(
            name="walking_person_shape",
            type=self.mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[0.18, 0.55, 0.0],
            rgba=[0.2, 0.75, 0.3, 0.9],
            contype=0,
            conaffinity=0,
        )
        for index, obstacle in enumerate(self.scenario.obstacles):
            body = self.spec.worldbody.add_body(
                name=f"obstacle_{index}",
                mocap=True,
                pos=[100.0 + index, 100.0, 0.35],
            )
            body.add_geom(
                name=f"barrier_{index}",
                type=self.mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[obstacle.radius_m, 0.35, 0.0],
                rgba=[0.85, 0.2, 0.1, 1.0],
            )

    def _advance_steps(self, count: int) -> None:
        for _ in range(count):
            position = self.data.qpos[7:]
            velocity = self.data.qvel[6:]
            torque = self.stiffness * (self.target_position - position)
            torque -= self.damping * velocity
            if self.model.actuator_ctrllimited.all():
                torque = self.numpy.clip(
                    torque,
                    self.model.actuator_ctrlrange[:, 0],
                    self.model.actuator_ctrlrange[:, 1],
                )
            self.data.ctrl[:] = torque
            self.mujoco.mj_step(self.model, self.data)
            self.physics_steps += 1
            if self.physics_steps % self.config.control_decimation == 0:
                self._update_policy()

    def _update_policy(self) -> None:
        projected_gravity = self.numpy.zeros(3, dtype=self.numpy.float64)
        inverse_quaternion = self.data.qpos[3:7].copy()
        inverse_quaternion[1:] *= -1.0
        self.mujoco.mju_rotVecQuat(
            projected_gravity,
            self.numpy.asarray([0.0, 0.0, -1.0]),
            inverse_quaternion,
        )
        phase = (float(self.data.time) % _GAIT_PERIOD_S) / _GAIT_PERIOD_S
        command = self.numpy.asarray(
            self.target.desired_velocity(), dtype=self.numpy.float32
        )
        observation = UnitreeRLGymG1Observation(
            base_angular_velocity_scaled=tuple(
                self.data.qvel[3:6] * self.config.angular_velocity_scale
            ),
            projected_gravity=tuple(projected_gravity),
            velocity_command_scaled=tuple(command * self.command_scale),
            joint_position_relative_scaled=tuple(
                (self.data.qpos[7:] - self.default_position)
                * self.config.joint_position_scale
            ),
            joint_velocity_scaled=tuple(
                self.data.qvel[6:] * self.config.joint_velocity_scale
            ),
            last_action=tuple(self.action),
            gait_phase=(
                math.sin(2.0 * math.pi * phase),
                math.cos(2.0 * math.pi * phase),
            ),
        )
        lease_id = self.target.session_id
        if lease_id is None:
            raise PolicyCandidateRejected("G1 policy inference has no motion lease")
        now = float(self.data.time)
        call_id = f"g1-policy:{self.target.lease_epoch}:{self.policy_steps + 1}"
        request = PolicyRequest(
            call_id=call_id,
            model_binding_id=self.model_binding_id,
            lease_id=lease_id,
            lease_epoch=self.target.lease_epoch,
            observation_version=self.policy_steps + 1,
            deadline_monotonic=now + self.config.policy_period_s,
            max_action_horizon_ms=self.policy_guard_profile.max_action_horizon_ms,
            resource_scope=UNITREE_RL_GYM_G1_RESOURCE_SCOPE,
            payload={"observation": observation},
        )
        if not self.target.policy_lease_is_current(request):
            raise PolicyCandidateRejected("G1 policy lease is not current")
        values = self.policy.infer(observation.vector())
        if not self.target.policy_lease_is_current(request):
            raise PolicyCandidateRejected("G1 policy lease changed during inference")
        candidate = PolicyCandidate(
            call_id=call_id,
            model_binding_id=self.model_binding_id,
            lease_id=lease_id,
            lease_epoch=self.target.lease_epoch,
            observation_version=self.policy_steps + 1,
            generated_at_monotonic=now,
            expires_at_monotonic=now + self.config.policy_period_s,
            action_space_id=UNITREE_RL_GYM_G1_ACTION_SPACE,
            resource_scope=UNITREE_RL_GYM_G1_RESOURCE_SCOPE,
            frames=(PolicyActionFrame(offset_ms=0, values=values),),
        )
        guarded = guard_candidate(
            request,
            candidate,
            self.policy_guard_profile,
            now_monotonic=now,
        )
        action = self.numpy.asarray(
            guarded.frames[0].values, dtype=self.numpy.float32
        )
        self.action = action.astype(self.numpy.float32)
        self.maximum_absolute_policy_action = max(
            self.maximum_absolute_policy_action,
            float(self.numpy.max(self.numpy.abs(self.action))),
        )
        self.target_position = (
            self.default_position + self.config.action_scale * self.action
        )
        self.policy_steps += 1

    def _settle_and_verify_stop(self) -> tuple[bool, str]:
        command_x, command_y, _ = self.robot_pose()
        settle_steps = round(_STOP_SETTLE_S / self.config.simulation_dt_s)
        self._advance_steps(settle_steps)
        self._update_fall_state()
        initial_x, initial_y, initial_yaw = self.robot_pose()
        measurement_steps = round(
            _STOP_MEASUREMENT_S / self.config.simulation_dt_s
        )
        self._advance_steps(measurement_steps)
        self._update_fall_state()
        final_x, final_y, final_yaw = self.robot_pose()
        base_speed = float(self.numpy.linalg.norm(self.data.qvel[:3]))
        angular_speed = float(self.numpy.linalg.norm(self.data.qvel[3:6]))
        joint_speed = float(self.numpy.max(self.numpy.abs(self.data.qvel[6:])))
        displacement = math.hypot(final_x - initial_x, final_y - initial_y)
        braking_displacement = math.hypot(
            initial_x - command_x, initial_y - command_y
        )
        total_displacement = math.hypot(final_x - command_x, final_y - command_y)
        yaw_change = abs(
            math.atan2(
                math.sin(final_yaw - initial_yaw),
                math.cos(final_yaw - initial_yaw),
            )
        )
        self.stop_measurements = {
            "dwell_s": _STOP_SETTLE_S + _STOP_MEASUREMENT_S,
            "settle_s": _STOP_SETTLE_S,
            "measurement_s": _STOP_MEASUREMENT_S,
            "braking_displacement_m": braking_displacement,
            "planar_displacement_m": displacement,
            "total_planar_displacement_m": total_displacement,
            "yaw_change_rad": yaw_change,
            "final_base_speed_mps": base_speed,
            "final_angular_speed_radps": angular_speed,
            "maximum_joint_speed_radps": joint_speed,
        }
        verified = (
            not self.fallen
            and displacement <= 0.08
            and yaw_change <= 0.08
            and base_speed <= 0.10
        )
        detail = (
            "G1 simulation measured stationary base after zero-command dwell"
            if verified
            else "G1 simulation did not establish stationary base evidence"
        )
        return verified, detail

    def _update_fixtures(self, time_s: float) -> PersonKeyframe:
        person = self._person_at(time_s)
        self.data.mocap_pos[self._person_mocap] = (
            person.world_x_m,
            person.world_y_m,
            0.9,
        )
        for index, obstacle in enumerate(self.scenario.obstacles):
            active = obstacle.active_from_s <= time_s <= obstacle.active_until_s
            self.data.mocap_pos[self._obstacle_mocap[index]] = (
                obstacle.world_x_m if active else 100.0 + index,
                obstacle.world_y_m if active else 100.0,
                0.35,
            )
        self.mujoco.mj_forward(self.model, self.data)
        return person

    def _person_at(self, time_s: float) -> PersonKeyframe:
        frames = self.scenario.keyframes
        for first, second in zip(frames, frames[1:]):
            if first.time_s <= time_s <= second.time_s:
                width = second.time_s - first.time_s
                fraction = 0.0 if width == 0.0 else (time_s - first.time_s) / width
                return PersonKeyframe(
                    time_s=time_s,
                    world_x_m=first.world_x_m
                    + fraction * (second.world_x_m - first.world_x_m),
                    world_y_m=first.world_y_m
                    + fraction * (second.world_y_m - first.world_y_m),
                    visible=first.visible,
                    track_id=first.track_id,
                )
        return frames[-1]

    def _world_to_robot(self, world_x: float, world_y: float) -> tuple[float, float]:
        robot_x, robot_y, yaw = self.robot_pose()
        delta_x = world_x - robot_x
        delta_y = world_y - robot_y
        return (
            math.cos(yaw) * delta_x + math.sin(yaw) * delta_y,
            -math.sin(yaw) * delta_x + math.cos(yaw) * delta_y,
        )

    def _update_fall_state(self) -> None:
        gravity = self.numpy.zeros(3, dtype=self.numpy.float64)
        inverse_quaternion = self.data.qpos[3:7].copy()
        inverse_quaternion[1:] *= -1.0
        self.mujoco.mju_rotVecQuat(
            gravity,
            self.numpy.asarray([0.0, 0.0, -1.0]),
            inverse_quaternion,
        )
        tilt = math.acos(max(-1.0, min(1.0, -float(gravity[2]))))
        self.current_tilt_rad = tilt
        self.maximum_tilt_rad = max(self.maximum_tilt_rad, tilt)
        if float(self.data.qpos[2]) < 0.55 or tilt > 0.9:
            self.fallen = True

    def _has_barrier_contact(self) -> bool:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            names = (
                self._geom_name(int(contact.geom1)),
                self._geom_name(int(contact.geom2)),
            )
            if any(name.startswith("barrier_") for name in names):
                return True
        return False

    def _geom_name(self, geom_id: int) -> str:
        name = self.mujoco.mj_id2name(
            self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        return name or ""

    def _mocap_index(self, body_name: str) -> int:
        body_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id < 0:
            raise RuntimeError(f"MuJoCo body {body_name} is missing")
        mocap_id = int(self.model.body_mocapid[body_id])
        if mocap_id < 0:
            raise RuntimeError(f"MuJoCo body {body_name} is not mocap")
        return mocap_id

    def _launch_viewer(self) -> Any:
        try:
            import mujoco.viewer
        except ImportError as exc:
            raise RuntimeError("MuJoCo viewer support is unavailable") from exc
        return self.mujoco.viewer.launch_passive(self.model, self.data)

    def _start_hud_camera(self) -> None:
        try:
            import cv2

            self._camera_renderer = self.mujoco.Renderer(
                self.model, height=360, width=640
            )
        except Exception as exc:
            raise RuntimeError(
                "G1 HUD camera renderer failed; set MUJOCO_GL=egl for "
                "headless use or run under a working display"
            ) from exc
        self._camera_encoder = cv2


class G1HudEventSink:
    """Add cached target/camera telemetry without creating a command path."""

    def __init__(
        self,
        downstream: Any,
        dashboard: FollowDashboard,
        world: ExternalG1PolicyWorld,
        *,
        brain_provider: str,
        brain_model: str | None,
        brain_reasoning_effort: str | None,
    ) -> None:
        self.downstream = downstream
        self.dashboard = dashboard
        self.world = world
        self.brain_provider = brain_provider
        self.brain_model = brain_model
        self.brain_reasoning_effort = brain_reasoning_effort
        self.camera_error: str | None = None

    def publish_current_camera(self) -> bool:
        try:
            frame = self.world.render_camera_jpeg()
            if frame is None:
                return False
            self.dashboard.publish_camera_frame(
                frame[0], frame[1], source="g1-pelvis-sim-camera"
            )
        except Exception as exc:
            self.camera_error = type(exc).__name__
            return False
        self.camera_error = None
        return True

    def publish(self, event: Any) -> None:
        enriched = dict(event)
        if enriched.get("schema_version") == "longship.follow-runtime-event.v1":
            telemetry = self.world.target_telemetry()
            telemetry["brain_provider"] = self.brain_provider
            telemetry["brain_model"] = self.brain_model
            telemetry["brain_reasoning_effort"] = self.brain_reasoning_effort
            enriched["target_telemetry"] = telemetry
            if not self.publish_current_camera() and self.camera_error:
                enriched["hud_camera_error"] = self.camera_error
        self.downstream.publish(enriched)


@dataclass(frozen=True, slots=True)
class G1DynamicReport:
    mode: str
    pipeline: Any
    artifacts: ArtifactIdentity
    world: ExternalG1PolicyWorld
    target: G1PolicyVelocityTarget
    brain_provider: str = "deterministic"
    brain_model: str | None = None
    brain_reasoning_effort: str | None = None

    @property
    def passed(self) -> bool:
        if hasattr(self.pipeline, "passed"):
            pipeline_passed = bool(self.pipeline.passed)
        else:
            pipeline_passed = bool(
                self.pipeline.exit_reason == "operator_stop"
                and self.pipeline.terminal_state is FollowState.STOPPED
                and self.pipeline.brain_requests >= 1
                and self.pipeline.accepted_skill_calls == 1
                and self.pipeline.control_steps > 0
                and self.pipeline.target_command_steps > 0
            )
        return bool(
            pipeline_passed
            and not self.world.fallen
            and self.world.contact_steps == 0
            and self.target.stop_verified
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "longship.follow-g1-policy-mujoco-report.v0",
            "passed": self.passed,
            "mode": self.mode,
            "provider": "external-unitree-rl-gym-g1-12dof",
            "brain": {
                "provider": self.brain_provider,
                "model": self.brain_model,
                "reasoning_effort": self.brain_reasoning_effort,
            },
            "artifacts": self.artifacts.to_dict(),
            "pipeline": self.pipeline.to_dict(),
            "dynamics": {
                "physics_steps": self.world.physics_steps,
                "policy_steps": self.world.policy_steps,
                "maximum_absolute_policy_action": (
                    self.world.maximum_absolute_policy_action
                ),
                "policy_candidate_guard": {
                    "action_space_id": UNITREE_RL_GYM_G1_ACTION_SPACE,
                    "resource_scope": list(UNITREE_RL_GYM_G1_RESOURCE_SCOPE),
                    "maximum_absolute_raw_action": _MAXIMUM_ABSOLUTE_RAW_ACTION,
                },
                "fallen": self.world.fallen,
                "maximum_tilt_rad": self.world.maximum_tilt_rad,
                "barrier_contact_steps": self.world.contact_steps,
                "stop_verified": self.target.stop_verified,
                "stop_control_mode": "zero-velocity-policy-with-base-dwell",
                "stop_measurements": self.world.stop_measurements,
                "final_robot_pose": self.world.robot_pose(),
            },
            "limitations": [
                "12 actuated lower-body joints; this is not the 29-DOF policy",
                "ground-truth scripted perception; rendered RGB-D is not evaluated",
                "simulation evidence does not qualify physical deployment",
            ],
        }


def verify_artifacts(args: argparse.Namespace) -> VerifiedExternalG1Artifacts:
    bundle_root = args.scene_bundle_root.resolve()
    scene_path = args.scene.resolve()
    if not scene_path.is_relative_to(bundle_root):
        raise ValueError("external G1 scene is outside the verified asset bundle")
    manifest = load_model_artifact_manifest(args.artifact_manifest)
    if manifest.manifest_id != _MODEL_MANIFEST_ID:
        raise ValueError("external G1 model manifest identity mismatch")
    references = {
        "policy": manifest.artifact("motion.pt"),
        "config": manifest.artifact("g1.yaml"),
        "license": manifest.artifact("LICENSE"),
    }
    expected_hashes = {
        "policy": args.expected_policy_sha256,
        "config": args.expected_config_sha256,
        "license": args.expected_license_sha256,
    }
    for name, expected in expected_hashes.items():
        if references[name].sha256 != expected:
            raise ValueError(f"external G1 {name} lock disagrees with the manifest")
    store = ArtifactStore(args.artifact_cache)
    policy = store.verify(args.policy, references["policy"])
    config = store.verify(args.policy_config, references["config"])
    license_artifact = store.verify(args.license, references["license"])
    scene_digest, _, _ = sha256_directory(args.scene_bundle_root)
    identity = ArtifactIdentity(
        manifest_id=manifest.manifest_id,
        manifest_sha256=manifest.source_sha256,
        scene_bundle_sha256=scene_digest,
        policy_sha256=policy.sha256,
        config_sha256=config.sha256,
        license_sha256=license_artifact.sha256,
    )
    expected = ArtifactIdentity(
        manifest_id=manifest.manifest_id,
        manifest_sha256=manifest.source_sha256,
        scene_bundle_sha256=args.expected_scene_bundle_sha256,
        policy_sha256=args.expected_policy_sha256,
        config_sha256=args.expected_config_sha256,
        license_sha256=args.expected_license_sha256,
    )
    if identity != expected:
        raise ValueError("external G1 artifact identity mismatch")
    try:
        license_text = read_verified_artifact_bytes(
            license_artifact, maximum_size_bytes=64 * 1024
        ).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("external G1 license record is not UTF-8") from exc
    if "BSD 3-Clause License" not in license_text:
        raise ValueError(
            "external G1 license record is not the reviewed BSD-3-Clause text"
        )
    return VerifiedExternalG1Artifacts(
        identity=identity,
        policy=policy,
        config=config,
        license=license_artifact,
    )


async def _run_pipeline(
    args: argparse.Namespace,
    profile: FollowProfile,
    scenario: FollowSimulationScenario,
    target: G1PolicyVelocityTarget,
    world: ExternalG1PolicyWorld,
    sink: Any,
) -> Any:
    async def invoke(brain: Any = None) -> Any:
        if args.mode == "system":
            return await run_system_with_world(
                profile,
                scenario,
                target,
                world,
                instruction=args.instruction,
                brain=brain,
                event_sink=sink,
                real_time=args.real_time or args.viewer or bool(args.hud_port),
            )
        return await run_interactive_follow_stack(
            profile,
            scenario,
            target,
            world,
            brain=brain,
            event_sink=sink,
            keep_running=(
                None
                if world.viewer is None
                else lambda: bool(world.viewer.is_running())
            ),
        )

    if args.brain == "deterministic":
        return await invoke()
    with tempfile.TemporaryDirectory(prefix="longship-g1-codex-") as workspace:
        async with CodexFollowBrain(
            workspace,
            model=args.codex_model or "gpt-5.6-terra",
            reasoning_effort=args.codex_reasoning_effort,
            timeout_s=args.codex_timeout_s,
        ) as brain:
            return await invoke(brain)


def run(
    args: argparse.Namespace,
    event_sink: Any | None = None,
    dashboard: FollowDashboard | None = None,
) -> G1DynamicReport:
    verified_artifacts = verify_artifacts(args)
    profile = FollowProfile.load(args.profile)
    scenario = FollowSimulationScenario.load(args.scenario)
    config = ExternalPolicyConfig.load_verified(verified_artifacts.config)
    target = G1PolicyVelocityTarget(
        maximum_forward_mps=profile.control.maximum_forward_speed_mps,
        maximum_yaw_rate_radps=profile.control.maximum_yaw_rate_radps,
    )
    world = ExternalG1PolicyWorld(
        scenario,
        target,
        config,
        scene_path=args.scene,
        policy_artifact=verified_artifacts.policy,
        show_viewer=args.viewer,
        enable_hud_camera=dashboard is not None,
        desired_follow_distance_m=profile.control.desired_distance_m,
    )
    try:
        sink = event_sink if event_sink is not None else NullEventSink()
        if dashboard is not None:
            hud_sink = G1HudEventSink(
                sink,
                dashboard,
                world,
                brain_provider=args.brain,
                brain_model=(
                    (args.codex_model or "gpt-5.6-terra")
                    if args.brain == "codex"
                    else None
                ),
                brain_reasoning_effort=(
                    args.codex_reasoning_effort
                    if args.brain == "codex"
                    else None
                ),
            )
            sink = hud_sink
            if not hud_sink.publish_current_camera():
                print(
                    "HUD camera has no initial frame; control remains available"
                )
        pipeline = asyncio.run(
            _run_pipeline(
                args,
                profile,
                scenario,
                target,
                world,
                sink,
            )
        )
        return G1DynamicReport(
            args.mode,
            pipeline,
            verified_artifacts.identity,
            world,
            target,
            brain_provider=args.brain,
            brain_model=(
                (args.codex_model or "gpt-5.6-terra")
                if args.brain == "codex"
                else None
            ),
            brain_reasoning_effort=(
                args.codex_reasoning_effort if args.brain == "codex" else None
            ),
        )
    finally:
        if args.keep_viewer and world.viewer is not None:
            print("G1 run complete; close the viewer window to exit")
            while world.viewer.is_running():
                world.viewer.sync()
                time.sleep(0.05)
        if args.keep_hud and dashboard is not None:
            try:
                input("Press Enter to close the read-only mission HUD. ")
            except (EOFError, KeyboardInterrupt):
                pass
        world.close()


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _vector(value: object, size: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{field} must contain {size} values")
    return tuple(_finite(item, field) for item in value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Longship FollowPerson on external Unitree G1 dynamics"
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--scene-bundle-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=_PLUGIN_DIRECTORY / "model-artifacts.experimental.json",
    )
    parser.add_argument(
        "--artifact-cache",
        type=Path,
        default=_REPOSITORY_ROOT / ".longship/artifacts",
    )
    parser.add_argument("--expected-scene-bundle-sha256", required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-license-sha256", required=True)
    parser.add_argument("--mode", choices=("system", "stack"), default="system")
    parser.add_argument("--instruction", default="Jackie，跟着我走")
    parser.add_argument(
        "--brain", choices=("deterministic", "codex"), default="deterministic"
    )
    parser.add_argument("--codex-model", default=None)
    parser.add_argument(
        "--codex-reasoning-effort",
        choices=(
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ),
        default="none",
    )
    parser.add_argument("--codex-timeout-s", type=float, default=60.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--keep-viewer", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--hud-host", default="127.0.0.1")
    parser.add_argument("--hud-port", type=int, default=0)
    parser.add_argument("--keep-hud", action="store_true")
    parser.add_argument("--events", type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.keep_viewer and not args.viewer:
        parser.error("--keep-viewer requires --viewer")
    if args.keep_hud and not args.hud_port:
        parser.error("--keep-hud requires --hud-port")
    if args.codex_model is not None and args.brain != "codex":
        parser.error("--codex-model requires --brain codex")
    if not 5.0 <= args.codex_timeout_s <= 120.0:
        parser.error("--codex-timeout-s must be between 5 and 120 seconds")
    try:
        with ExitStack() as resources:
            sinks: list[Any] = []
            if args.events:
                sinks.append(resources.enter_context(JsonlEventSink(args.events)))
            dashboard = None
            if args.hud_port:
                dashboard = resources.enter_context(
                    FollowDashboard(host=args.hud_host, port=args.hud_port)
                )
                sinks.append(dashboard)
                print(f"Read-only mission HUD: http://{args.hud_host}:{dashboard.port}")
            sink = CompositeEventSink(sinks) if sinks else NullEventSink()
            report = run(args, sink, dashboard)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(3, f"BLOCKED: {exc}\n")
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    print(payload)
    if args.report:
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(payload + "\n")
    raise SystemExit(0 if report.passed else 2)


if __name__ == "__main__":
    main()

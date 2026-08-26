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
from typing import Any

from longship.brain.codex_follow import CodexFollowBrain
from longship.contracts.skills.follow_person import (
    FollowCommand,
    FollowScene,
    FollowState,
    MotionReceipt,
    ObstaclePoint,
    PersonTrack,
)
from longship.observability.follow_person import JsonlEventSink
from longship.runtime.follow_person import FollowPersonRuntime, NullEventSink
from longship.safety.follow_obstacle import ForwardObstacleGuard
from longship.simulation.follow_person import (
    FollowSimulationScenario,
    PersonKeyframe,
)
from longship.simulation.follow_stack import run_interactive_follow_stack
from longship.simulation.follow_system import run_system_with_world
from longship.skills.follow_person.config import FollowProfile
from longship.skills.follow_person.governor import MotionGovernor
from longship.skills.follow_person.planner import LocalFollowPlanner


@dataclass(frozen=True, slots=True)
class MujocoAcceptanceReport:
    passed: bool
    following_steps: int
    unsafe_forward_commands: int
    robot_obstacle_contact_steps: int
    final_distance_error_m: float
    maximum_forward_speed_mps: float
    terminal_state: FollowState
    final_robot_pose: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "longship.follow-mujoco-report.v0",
            "passed": self.passed,
            "following_steps": self.following_steps,
            "unsafe_forward_commands": self.unsafe_forward_commands,
            "robot_obstacle_contact_steps": self.robot_obstacle_contact_steps,
            "final_distance_error_m": self.final_distance_error_m,
            "maximum_forward_speed_mps": self.maximum_forward_speed_mps,
            "terminal_state": self.terminal_state.value,
            "final_robot_pose": {
                "world_x_m": self.final_robot_pose[0],
                "world_y_m": self.final_robot_pose[1],
                "yaw_rad": self.final_robot_pose[2],
            },
        }


class MujocoVelocityTarget:
    """Process-local simulator target implementing the Runtime motion port."""

    def __init__(self) -> None:
        self.session_id: str | None = None
        self.current_command: FollowCommand | None = None

    def acquire(self, session_id: str, now_ns: int) -> MotionReceipt:
        del now_ns
        if self.session_id is not None:
            return MotionReceipt(False, "MuJoCo target authority is already held")
        self.session_id = session_id
        return MotionReceipt(True, "MuJoCo planar velocity target acquired")

    def apply(self, command: FollowCommand) -> MotionReceipt:
        if command.session_id != self.session_id:
            return MotionReceipt(False, "MuJoCo command lease owner mismatch")
        self.current_command = command
        return MotionReceipt(True, "MuJoCo setpoint accepted")

    def protective_stop(self, reason: str) -> MotionReceipt:
        del reason
        self.current_command = None
        return MotionReceipt(True, "MuJoCo velocity is zero", verified_stopped=True)


class MujocoFollowWorld:
    """MuJoCo-backed target and ground-truth perception fixture."""

    def __init__(
        self,
        mujoco: Any,
        scenario: FollowSimulationScenario,
        target: MujocoVelocityTarget,
        *,
        viewer: bool,
    ) -> None:
        self._mj = mujoco
        self.scenario = scenario
        self.target = target
        self.model = mujoco.MjModel.from_xml_string(
            _model_xml(tuple(item.radius_m for item in scenario.obstacles))
        )
        self.data = mujoco.MjData(self.model)
        self.sequence = 0
        self.contact_steps = 0
        self._viewer = self._launch_viewer() if viewer else None
        self._robot_qpos = tuple(
            self._joint_address(name, position=True)
            for name in ("robot_x", "robot_y", "robot_yaw")
        )
        self._robot_qvel = tuple(
            self._joint_address(name, position=False)
            for name in ("robot_x", "robot_y", "robot_yaw")
        )
        self._person_mocap = self._mocap_index("walking_person")
        self._obstacle_mocap = tuple(
            self._mocap_index(f"obstacle_{index}")
            for index in range(len(scenario.obstacles))
        )
        mujoco.mj_forward(self.model, self.data)

    @property
    def viewer(self) -> Any | None:
        return self._viewer

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()

    def update_fixture(self, time_s: float) -> PersonKeyframe:
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
        self._mj.mj_forward(self.model, self.data)
        return person

    def scene(self, time_s: float, now_ns: int) -> FollowScene:
        person = self.update_fixture(time_s)
        self.sequence += 1
        target_xy = self._world_to_robot(person.world_x_m, person.world_y_m)
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
                    forward_m=target_xy[0],
                    left_m=target_xy[1],
                    confidence=0.96,
                ),
            )
            if person.visible and target_xy[0] > 0.0
            else ()
        )
        corridor_clearances = [
            item.forward_m - item.radius_m
            for item in obstacles
            if item.forward_m > 0.0
            and abs(item.left_m) <= 0.45 + item.radius_m
        ]
        if target_xy[0] > 0.0 and abs(target_xy[1]) <= 0.7:
            corridor_clearances.append(max(0.0, target_xy[0] - 0.25))
        return FollowScene(
            sequence=self.sequence,
            captured_monotonic_ns=now_ns,
            received_monotonic_ns=now_ns,
            healthy=True,
            calibration_id="mujoco-ground-truth-to-base-v0",
            calibration_valid=True,
            detector_ready=True,
            floor_valid=True,
            tracks=tracks,
            obstacles=obstacles,
            raw_forward_clearance_m=min(corridor_clearances, default=10.0),
            detail="MuJoCo ground-truth evaluation scene",
        )

    def advance(self, dt_s: float) -> None:
        substeps = max(1, round(dt_s / float(self.model.opt.timestep)))
        for _ in range(substeps):
            command = self.target.current_command
            desired_forward = command.forward_mps if command is not None else 0.0
            desired_yaw_rate = command.yaw_rate_radps if command is not None else 0.0
            yaw = float(self.data.qpos[self._robot_qpos[2]])
            desired_vx = math.cos(yaw) * desired_forward
            desired_vy = math.sin(yaw) * desired_forward
            actual_vx = float(self.data.qvel[self._robot_qvel[0]])
            actual_vy = float(self.data.qvel[self._robot_qvel[1]])
            actual_yaw_rate = float(self.data.qvel[self._robot_qvel[2]])
            self.data.ctrl[0] = _clamp(180.0 * (desired_vx - actual_vx), 160.0)
            self.data.ctrl[1] = _clamp(180.0 * (desired_vy - actual_vy), 160.0)
            self.data.ctrl[2] = _clamp(
                45.0 * (desired_yaw_rate - actual_yaw_rate), 60.0
            )
            self._mj.mj_step(self.model, self.data)
        if self._has_robot_obstacle_contact():
            self.contact_steps += 1
        if self._viewer is not None:
            self._viewer.sync()

    def robot_pose(self) -> tuple[float, float, float]:
        return tuple(float(self.data.qpos[index]) for index in self._robot_qpos)

    def person_distance(self, time_s: float) -> float:
        person = self._person_at(time_s)
        robot_x, robot_y, _ = self.robot_pose()
        return math.hypot(person.world_x_m - robot_x, person.world_y_m - robot_y)

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

    def _joint_address(self, name: str, *, position: bool) -> int:
        joint_id = self._mj.mj_name2id(
            self.model, self._mj.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise RuntimeError(f"MuJoCo model is missing joint {name}")
        table = self.model.jnt_qposadr if position else self.model.jnt_dofadr
        return int(table[joint_id])

    def _mocap_index(self, body_name: str) -> int:
        body_id = self._mj.mj_name2id(
            self.model, self._mj.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id < 0:
            raise RuntimeError(f"MuJoCo model is missing body {body_name}")
        mocap_id = int(self.model.body_mocapid[body_id])
        if mocap_id < 0:
            raise RuntimeError(f"MuJoCo body {body_name} is not a mocap body")
        return mocap_id

    def _has_robot_obstacle_contact(self) -> bool:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first = self._geom_name(int(contact.geom1))
            second = self._geom_name(int(contact.geom2))
            if (
                first.startswith("robot_") and second.startswith("barrier_")
            ) or (
                second.startswith("robot_") and first.startswith("barrier_")
            ):
                return True
        return False

    def _geom_name(self, geom_id: int) -> str:
        value = self._mj.mj_id2name(
            self.model, self._mj.mjtObj.mjOBJ_GEOM, geom_id
        )
        return value or ""

    def _launch_viewer(self) -> Any:
        try:
            import mujoco.viewer
        except ImportError as exc:
            raise RuntimeError(
                "this MuJoCo installation has no viewer support"
            ) from exc
        return mujoco.viewer.launch_passive(self.model, self.data)


def run_mujoco(
    profile: FollowProfile,
    scenario: FollowSimulationScenario,
    *,
    show_viewer: bool,
    real_time: bool,
    keep_viewer: bool,
    event_sink: Any = None,
) -> MujocoAcceptanceReport:
    if abs(profile.control_period_s - scenario.step_s) > 1e-9:
        raise ValueError("scenario step must match the profile control period")
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is not installed; install the optional 'mujoco' dependency"
        ) from exc

    target = MujocoVelocityTarget()
    world = MujocoFollowWorld(
        mujoco,
        scenario,
        target,
        viewer=show_viewer,
    )
    runtime = FollowPersonRuntime(
        profile,
        target,
        planner=LocalFollowPlanner(profile.planner, profile.control),
        safety_guard=ForwardObstacleGuard(profile.safety),
        governor=MotionGovernor(profile.control),
        event_sink=event_sink or NullEventSink(),
        session_id=f"mujoco:{scenario.scenario_id}",
    )
    now_ns = 1_000_000_000
    step_ns = int(scenario.step_s * 1_000_000_000)
    following_steps = 0
    unsafe_commands = 0
    maximum_speed = 0.0
    final_time_s = 0.0
    try:
        runtime.start(now_ns=now_ns - step_ns)
        step_count = round(scenario.duration_s / scenario.step_s)
        for index in range(step_count + 1):
            final_time_s = index * scenario.step_s
            scene = world.scene(final_time_s, now_ns)
            snapshot = runtime.tick(scene, now_ns=now_ns)
            if snapshot.state is FollowState.FOLLOWING:
                following_steps += 1
            command = snapshot.command
            if command is not None:
                maximum_speed = max(maximum_speed, command.forward_mps)
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
            if real_time or show_viewer:
                time.sleep(scenario.step_s)
            if world.viewer is not None and not world.viewer.is_running():
                break
        terminal = runtime.state
        if runtime.is_active:
            runtime.stop("MuJoCo simulation complete", now_ns=now_ns)
        final_error = abs(
            world.person_distance(min(scenario.duration_s, final_time_s))
            - profile.control.desired_distance_m
        )
        criteria = scenario.acceptance
        passed = (
            following_steps >= criteria.minimum_following_steps
            and unsafe_commands <= criteria.maximum_unsafe_forward_commands
            and world.contact_steps == 0
            and final_error <= criteria.maximum_final_distance_error_m
            and terminal is not FollowState.FAILED
        )
        report = MujocoAcceptanceReport(
            passed=passed,
            following_steps=following_steps,
            unsafe_forward_commands=unsafe_commands,
            robot_obstacle_contact_steps=world.contact_steps,
            final_distance_error_m=final_error,
            maximum_forward_speed_mps=maximum_speed,
            terminal_state=terminal,
            final_robot_pose=world.robot_pose(),
        )
        if keep_viewer and world.viewer is not None:
            print("MuJoCo run complete; close the viewer window to exit")
            while world.viewer.is_running():
                world.viewer.sync()
                time.sleep(0.05)
        return report
    finally:
        if runtime.is_active:
            runtime.stop("MuJoCo runner shutdown", now_ns=now_ns)
        world.close()


def run_mujoco_system(
    profile: FollowProfile,
    scenario: FollowSimulationScenario,
    *,
    instruction: str,
    show_viewer: bool,
    real_time: bool,
    keep_viewer: bool,
    event_sink: Any = None,
) -> Any:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is not installed; install the optional 'mujoco' dependency"
        ) from exc
    target = MujocoVelocityTarget()
    world = MujocoFollowWorld(
        mujoco,
        scenario,
        target,
        viewer=show_viewer,
    )
    try:
        report = asyncio.run(
            run_system_with_world(
                profile,
                scenario,
                target,
                world,
                instruction=instruction,
                event_sink=event_sink,
                real_time=real_time or show_viewer,
            )
        )
        if keep_viewer and world.viewer is not None:
            print("System run complete; close the MuJoCo viewer window to exit")
            while world.viewer.is_running():
                world.viewer.sync()
                time.sleep(0.05)
        return report
    finally:
        world.close()


def run_mujoco_stack(
    profile: FollowProfile,
    scenario: FollowSimulationScenario,
    *,
    show_viewer: bool,
    keep_viewer: bool,
    brain_provider: str = "deterministic",
    codex_model: str | None = None,
    event_sink: Any = None,
) -> Any:
    """Run the Longship-native interactive stack against this target plugin."""

    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is not installed; install the optional 'mujoco' dependency"
        ) from exc
    target = MujocoVelocityTarget()
    world = MujocoFollowWorld(mujoco, scenario, target, viewer=show_viewer)
    try:
        report = asyncio.run(
            _run_mujoco_stack_session(
                profile,
                scenario,
                target,
                world,
                brain_provider=brain_provider,
                codex_model=codex_model,
                event_sink=event_sink,
            )
        )
        if keep_viewer and world.viewer is not None:
            print("Interactive stack stopped; close the MuJoCo viewer window to exit")
            while world.viewer.is_running():
                world.viewer.sync()
                time.sleep(0.05)
        return report
    finally:
        world.close()


async def _run_mujoco_stack_session(
    profile: FollowProfile,
    scenario: FollowSimulationScenario,
    target: MujocoVelocityTarget,
    world: MujocoFollowWorld,
    *,
    brain_provider: str,
    codex_model: str | None,
    event_sink: Any,
) -> Any:
    keep_running = (
        None
        if world.viewer is None
        else lambda: bool(world.viewer.is_running())
    )
    if brain_provider == "deterministic":
        return await run_interactive_follow_stack(
            profile,
            scenario,
            target,
            world,
            event_sink=event_sink,
            keep_running=keep_running,
        )
    if brain_provider != "codex":
        raise ValueError("unsupported Brain provider")
    with tempfile.TemporaryDirectory(prefix="longship-follow-codex-") as workspace:
        async with CodexFollowBrain(workspace, model=codex_model) as brain:
            return await run_interactive_follow_stack(
                profile,
                scenario,
                target,
                world,
                brain=brain,
                event_sink=event_sink,
                keep_running=keep_running,
            )


def _model_xml(obstacle_radii_m: tuple[float, ...]) -> str:
    if len(obstacle_radii_m) > 256:
        raise ValueError("MuJoCo scenario obstacle count is out of range")
    if any(
        not math.isfinite(radius) or radius <= 0.0 or radius > 5.0
        for radius in obstacle_radii_m
    ):
        raise ValueError("MuJoCo obstacle radius is out of range")
    obstacle_bodies = "\n".join(
        f"""
    <body name="obstacle_{index}" mocap="true" pos="{100 + index} 100 0.35">
      <geom name="barrier_{index}" type="cylinder" size="{radius:.9g} 0.35"
            rgba="0.85 0.25 0.15 1" friction="1.2 0.02 0.002"/>
    </body>"""
        for index, radius in enumerate(obstacle_radii_m)
    )
    return f"""<mujoco model="longship_follow_person_proxy">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.7 0.7 0.7"/>
    <global azimuth="135" elevation="-28"/>
  </visual>
  <default>
    <geom solref="0.01 1" solimp="0.9 0.95 0.001" condim="3"/>
  </default>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="12 12 0.1"
          rgba="0.82 0.84 0.86 1" friction="1.0 0.02 0.002"/>
    <body name="robot" pos="0 0 0">
      <joint name="robot_x" type="slide" axis="1 0 0" damping="10"/>
      <joint name="robot_y" type="slide" axis="0 1 0" damping="10"/>
      <joint name="robot_yaw" type="hinge" axis="0 0 1" damping="8"/>
      <geom name="robot_base" type="cylinder" pos="0 0 0.27" size="0.27 0.24"
            mass="34" rgba="0.12 0.35 0.72 1" friction="1.1 0.02 0.002"/>
      <geom name="robot_torso" type="capsule" fromto="0 0 0.46 0 0 1.28"
            size="0.16" mass="18" rgba="0.16 0.42 0.82 1"
            contype="0" conaffinity="0"/>
      <geom name="robot_head" type="sphere" pos="0 0 1.53" size="0.16"
            mass="4" rgba="0.75 0.82 0.9 1" contype="0" conaffinity="0"/>
      <site name="robot_forward" pos="0.38 0 0.55" size="0.055"
            rgba="0.1 0.9 0.3 1"/>
    </body>
    <body name="walking_person" mocap="true" pos="2.2 0 0.9">
      <geom name="person_body" type="capsule" fromto="0 0 -0.65 0 0 0.45"
            size="0.18" rgba="0.2 0.72 0.3 0.9" contype="0" conaffinity="0"/>
      <geom name="person_head" type="sphere" pos="0 0 0.7" size="0.16"
            rgba="0.95 0.72 0.48 1" contype="0" conaffinity="0"/>
    </body>
{obstacle_bodies}
  </worldbody>
  <actuator>
    <motor name="drive_x" joint="robot_x" ctrllimited="true" ctrlrange="-160 160"/>
    <motor name="drive_y" joint="robot_y" ctrllimited="true" ctrlrange="-160 160"/>
    <motor name="turn" joint="robot_yaw" ctrllimited="true" ctrlrange="-60 60"/>
  </actuator>
</mujoco>"""


def _clamp(value: float, magnitude: float) -> float:
    return max(-magnitude, min(magnitude, value))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FollowPerson against a visible MuJoCo physics proxy"
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--system",
        action="store_true",
        help="include mock voice input, Brain proposal, and Skill admission",
    )
    parser.add_argument(
        "--stack",
        action="store_true",
        help="open Longship's persistent interactive terminal composition",
    )
    parser.add_argument("--instruction", default="Jackie，跟着我走")
    parser.add_argument(
        "--brain", choices=("deterministic", "codex"), default="deterministic"
    )
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--keep-viewer", action="store_true")
    parser.add_argument("--events", type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.keep_viewer and not args.viewer:
        raise SystemExit("BLOCKED: --keep-viewer requires --viewer")
    if args.system and args.stack:
        raise SystemExit("BLOCKED: choose only one of --system and --stack")
    if args.codex_model is not None and (not args.stack or args.brain != "codex"):
        raise SystemExit("BLOCKED: --codex-model requires --stack --brain codex")
    if args.brain == "codex" and not args.stack:
        raise SystemExit("BLOCKED: --brain codex requires --stack")
    profile = FollowProfile.load(args.profile)
    scenario = FollowSimulationScenario.load(args.scenario)
    with ExitStack() as stack:
        sink = (
            stack.enter_context(JsonlEventSink(args.events))
            if args.events
            else NullEventSink()
        )
        try:
            if args.stack:
                report = run_mujoco_stack(
                    profile,
                    scenario,
                    show_viewer=args.viewer,
                    keep_viewer=args.keep_viewer,
                    brain_provider=args.brain,
                    codex_model=args.codex_model,
                    event_sink=sink,
                )
            elif args.system:
                report = run_mujoco_system(
                    profile,
                    scenario,
                    instruction=args.instruction,
                    show_viewer=args.viewer,
                    real_time=args.real_time,
                    keep_viewer=args.keep_viewer,
                    event_sink=sink,
                )
            else:
                report = run_mujoco(
                    profile,
                    scenario,
                    show_viewer=args.viewer,
                    real_time=args.real_time,
                    keep_viewer=args.keep_viewer,
                    event_sink=sink,
                )
        except (OSError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"BLOCKED: {exc}") from exc
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    print(payload)
    if args.report:
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(payload + "\n")
    passed = (
        report.passed
        if hasattr(report, "passed")
        else report.exit_reason != "runtime_failed"
    )
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()

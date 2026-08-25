#!/usr/bin/env python3
"""The single Longship MuJoCo/DDS simulator used by every policy backend."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from longship.rl.sim2sim.dds import (
    DEPTH_TOPIC,
    G1_29DOF_JOINTS,
    SIM_CONTROL_TOPIC,
    DdsContract,
)
DEPTH_HEIGHT = 270
DEPTH_WIDTH = 480
VIEWER_FREQUENCY_HZ = 60.0

HIKING_FOOT_CAPSULES = {
    "left": (
        (0.1, -0.026, -0.025, 0.05, -0.027, -0.025),
        (-0.044, -0.018, -0.025, 0.123, -0.018, -0.025),
        (-0.052, -0.01, -0.025, 0.13, -0.01, -0.025),
        (-0.054, 0.0, -0.025, 0.132, 0.0, -0.025),
        (-0.052, 0.01, -0.025, 0.13, 0.01, -0.025),
        (-0.044, 0.018, -0.025, 0.123, 0.018, -0.025),
        (0.1, 0.026, -0.025, 0.05, 0.026, -0.025),
    ),
    "right": (
        (0.1, -0.026, -0.025, 0.05, -0.026, -0.025),
        (-0.044, -0.018, -0.025, 0.123, -0.018, -0.025),
        (-0.052, -0.01, -0.025, 0.13, -0.01, -0.025),
        (-0.054, 0.0, -0.025, 0.132, 0.0, -0.025),
        (-0.052, 0.01, -0.025, 0.13, 0.01, -0.025),
        (-0.044, 0.018, -0.025, 0.123, 0.018, -0.025),
        (0.1, 0.026, -0.025, 0.05, 0.026, -0.025),
    ),
}
def _install_foot_collision(spec: Any, preset: str) -> None:
    if preset == "native":
        return
    if preset != "hiking_training_v1":
        raise ValueError(f"unsupported foot collision preset {preset!r}")
    import mujoco

    for side, segments in HIKING_FOOT_CAPSULES.items():
        foot = spec.body(f"{side}_ankle_roll_link")
        if foot is None:
            raise RuntimeError(f"Hiking scene has no {side}_ankle_roll_link")
        for geom in list(foot.geoms):
            if int(geom.contype) != 0 or int(geom.conaffinity) != 0:
                spec.delete(geom)
        for index, fromto in enumerate(segments, 1):
            foot.add_geom(
                name=f"{side}_foot{index}_collision",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=fromto,
                size=(0.01, 0.0, 0.0),
                contype=1,
                conaffinity=1,
                condim=3,
                priority=1,
                friction=(0.6, 0.005, 0.0001),
                group=3,
            )


def _actuator_ids(model: Any, joint_names: tuple[str, ...]) -> np.ndarray:
    """Map by transmission target so model-specific actuator names are harmless."""
    import mujoco

    result = []
    for name in joint_names:
        joint_id = model.joint(name).id
        matches = np.flatnonzero(
            (model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT))
            & (model.actuator_trnid[:, 0] == joint_id)
        )
        if len(matches) != 1:
            raise RuntimeError(f"joint {name!r} has {len(matches)} actuator transmissions")
        result.append(int(matches[0]))
    return np.asarray(result, dtype=np.int64)


def _foot_body_ids(model: Any) -> tuple[int, int]:
    import mujoco

    contact_ids = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot_contact_point")
        for side in ("left", "right")
    )
    if min(contact_ids) >= 0:
        return contact_ids
    return tuple(model.body(f"{side}_ankle_roll_link").id for side in ("left", "right"))


def _apply_hiking_spotter(
    model: Any,
    data: Any,
    pelvis_id: int,
    torso_id: int,
    catch_height: float,
    horizontal_anchor: np.ndarray,
) -> None:
    """Source-aligned Hiking cold-start/stand assistance."""
    import mujoco

    data.xfrc_applied[pelvis_id] = 0.0
    data.xfrc_applied[torso_id] = 0.0
    height = float(data.qpos[2])
    vertical_velocity = float(data.qvel[2])
    if height < catch_height:
        data.xfrc_applied[pelvis_id, 2] = max(
            2000.0 * (catch_height - height) - 200.0 * vertical_velocity,
            0.0,
        )
    offset = data.qpos[:2] - horizontal_anchor
    data.xfrc_applied[pelvis_id, 0] = -120.0 * offset[0] - 30.0 * data.qvel[0]
    data.xfrc_applied[pelvis_id, 1] = -120.0 * offset[1] - 30.0 * data.qvel[1]

    pelvis_up = data.xmat[pelvis_id].reshape(3, 3)[:, 2]
    pelvis_tilt = math.degrees(math.acos(float(np.clip(pelvis_up[2], -1.0, 1.0))))
    if pelvis_tilt > 12.0:
        data.xfrc_applied[pelvis_id, 3:] = (
            150.0 * np.cross(pelvis_up, np.asarray((0.0, 0.0, 1.0)))
            - 10.0 * data.qvel[3:6]
        )
    torso_velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, torso_id, torso_velocity, 0
    )
    torso_up = data.xmat[torso_id].reshape(3, 3)[:, 2]
    data.xfrc_applied[torso_id, 3:] = (
        40.0 * np.cross(torso_up, np.asarray((0.0, 0.0, 1.0)))
        - 3.0 * torso_velocity[:3]
    )
GROUNDED_GANTRY_SUPPORT_FRACTION = 0.90


@dataclass(slots=True)
class InteractiveControls:
    """Thread-safe viewer controls and virtual-gantry physics state."""

    enabled: bool = True
    length: float = 1.0
    stiffness: float = 200.0
    damping: float = 100.0
    height: float = 3.0
    anchor: np.ndarray = field(default_factory=lambda: np.asarray((0.0, 0.0, 3.0)))
    camera_tracking: bool = False
    _reset_requested: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def key_callback(self, keycode: int) -> None:
        with self._lock:
            if keycode == 55:  # GLFW KEY_7
                self.length -= 0.1
                print(f"viewer: gantry length={self.length:.2f} m", flush=True)
            elif keycode == 56:  # GLFW KEY_8
                self.length += 0.1
                print(f"viewer: gantry length={self.length:.2f} m", flush=True)
            elif keycode == 57:  # GLFW KEY_9
                self.enabled = not self.enabled
                state = "enabled" if self.enabled else "disabled"
                print(f"viewer: gantry {state}", flush=True)
            elif keycode == 259:  # GLFW KEY_BACKSPACE
                self._reset_requested = True
                print("viewer: reset requested", flush=True)
            elif keycode in (89, 121):  # GLFW KEY_Y / ASCII y
                self.camera_tracking = not self.camera_tracking
                state = "enabled" if self.camera_tracking else "disabled"
                print(f"viewer: robot camera tracking {state}", flush=True)

    def consume_reset(self) -> bool:
        with self._lock:
            requested = self._reset_requested
            self._reset_requested = False
            return requested

    def disable_gantry(self) -> None:
        with self._lock:
            self.enabled = False

    def reset_anchor(self, position: np.ndarray) -> None:
        with self._lock:
            self.anchor = np.asarray((position[0], position[1], self.height), dtype=np.float64)

    def follow_horizontally(self, position: np.ndarray) -> None:
        """Move the virtual trolley above the robot without changing rope height."""
        with self._lock:
            self.anchor[:2] = np.asarray(position[:2], dtype=np.float64)

    def set_grounded_length(
        self,
        position: np.ndarray,
        total_mass: float,
        gravity: float,
        support_fraction: float = GROUNDED_GANTRY_SUPPORT_FRACTION,
    ) -> None:
        """Set gantry preload while keeping both feet at the grounded pose."""
        if total_mass <= 0 or gravity <= 0:
            raise ValueError("gantry mass and gravity must be positive")
        if not 0.0 <= support_fraction <= 1.0:
            raise ValueError("gantry support_fraction must be in [0, 1]")
        with self._lock:
            distance = float(np.linalg.norm(self.anchor - position))
            preload_extension = support_fraction * total_mass * gravity / self.stiffness
            self.length = max(0.0, distance - preload_extension)

    def tracking_enabled(self) -> bool:
        with self._lock:
            return self.camera_tracking

    def visualization_state(self) -> tuple[bool, np.ndarray]:
        with self._lock:
            return self.enabled, self.anchor.copy()

    def force(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        with self._lock:
            if not self.enabled:
                return np.zeros(3)
            delta = self.anchor - position
            distance = float(np.linalg.norm(delta))
            if distance < 1e-9:
                return np.zeros(3)
            direction = delta / distance
            radial_velocity = float(np.dot(velocity, direction))
            return (self.stiffness * (distance - self.length) - self.damping * radial_velocity) * direction


@dataclass(frozen=True, slots=True)
class LowCommandSnapshot:
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    kp: np.ndarray
    kd: np.ndarray

    @classmethod
    def from_message(cls, message: Any) -> "LowCommandSnapshot":
        motors = message.motor_cmd
        return cls(
            q=np.asarray([motor.q for motor in motors[:29]], dtype=np.float64),
            dq=np.asarray([motor.dq for motor in motors[:29]], dtype=np.float64),
            tau=np.asarray([motor.tau for motor in motors[:29]], dtype=np.float64),
            kp=np.asarray([motor.kp for motor in motors[:29]], dtype=np.float64),
            kd=np.asarray([motor.kd for motor in motors[:29]], dtype=np.float64),
        )

    def has_control_authority(self) -> bool:
        values = (self.q, self.dq, self.tau, self.kp, self.kd)
        return all(np.all(np.isfinite(value)) for value in values) and bool(
            np.any(self.kp > 0.0) or np.any(self.kd > 0.0) or np.any(self.tau != 0.0)
        )


def _draw_virtual_gantry(scene: Any, controls: InteractiveControls, position: np.ndarray) -> None:
    """Populate the viewer user scene with a visible rope and anchor."""
    import mujoco

    scene.ngeom = 0
    enabled, anchor = controls.visualization_state()
    if not enabled:
        return

    rope = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        rope,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray((0.1, 0.35, 1.0, 1.0), dtype=np.float32),
    )
    mujoco.mjv_connector(
        rope,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        0.012,
        np.asarray(position, dtype=np.float64),
        anchor,
    )
    scene.ngeom += 1
    anchor_marker = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        anchor_marker,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray((0.05, 0.05, 0.05), dtype=np.float64),
        anchor,
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray((1.0, 0.1, 0.1, 1.0), dtype=np.float32),
    )
    scene.ngeom += 1


def _update_tracking_camera(camera: Any, controls: InteractiveControls, position: np.ndarray) -> None:
    """Follow the robot without overwriting viewer mouse orbit or zoom."""
    if controls.tracking_enabled():
        camera.lookat[:] = position


def _reset_robot(
    model: Any,
    data: Any,
    qpos_addresses: np.ndarray | None = None,
    reset_q: np.ndarray | None = None,
) -> None:
    import mujoco

    mujoco.mj_resetData(model, data)
    if reset_q is not None:
        if qpos_addresses is None:
            raise ValueError("qpos addresses are required with a reset pose")
        data.qpos[qpos_addresses] = reset_q
    data.qpos[2] = 0.76
    mujoco.mj_forward(model, data)
    # Calibrate root height from the model's actual left/right foot contact
    # points instead of relying on a robot-specific magic base height.
    foot_ids = _foot_body_ids(model)
    contact_geoms = np.flatnonzero(
        np.isin(model.geom_bodyid, foot_ids)
        & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    )
    if len(contact_geoms):
        bottoms = []
        for geom_id in contact_geoms:
            radius = float(model.geom_size[geom_id, 0])
            half_length = float(model.geom_size[geom_id, 1])
            axis_z = abs(float(data.geom_xmat[geom_id].reshape(3, 3)[2, 2]))
            bottoms.append(float(data.geom_xpos[geom_id, 2]) - radius - half_length * axis_z)
        foot_z = min(bottoms)
    else:
        foot_z = min(float(data.xpos[body_id, 2]) for body_id in foot_ids)
    data.qpos[2] -= foot_z
    mujoco.mj_forward(model, data)


def _initialize_dds(contract: DdsContract) -> tuple[Any, Any]:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber

    ChannelFactoryInitialize(contract.domain_id, contract.interface)
    return ChannelPublisher, ChannelSubscriber


def encode_depth(depth: np.ndarray, sim_time: float) -> Any:
    """Encode a metric depth image in the shared DDS camera representation."""
    from unitree_sdk2py.idl.builtin_interfaces.msg.dds_ import Time_
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_

    millimetres = np.clip(depth * 1000.0, 0, 65535).astype("<u2")
    seconds = int(sim_time)
    return PointCloud2_(
        header=Header_(
            stamp=Time_(sec=seconds, nanosec=int((sim_time - seconds) * 1_000_000_000)),
            frame_id="longship_depth_optical",
        ),
        height=DEPTH_HEIGHT,
        width=DEPTH_WIDTH,
        fields=[],
        is_bigendian=False,
        point_step=2,
        row_step=DEPTH_WIDTH * 2,
        # CycloneDDS accepts the uint8 sequence as bytes directly.  Expanding
        # every 480x270x2 frame into ~259k Python integers made Hiking run at
        # roughly one sixth real time and broke wall-clock policy handoffs.
        data=millimetres.tobytes(),
        is_dense=True,
    )


def decode_depth(message: Any) -> np.ndarray:
    if int(message.height) != DEPTH_HEIGHT or int(message.width) != DEPTH_WIDTH:
        raise ValueError(f"unexpected depth shape {message.height}x{message.width}")
    raw = bytes(message.data)
    expected = DEPTH_HEIGHT * DEPTH_WIDTH * 2
    if len(raw) != expected:
        raise ValueError(f"depth payload has {len(raw)} bytes; expected {expected}")
    return np.frombuffer(raw, dtype="<u2").reshape(DEPTH_HEIGHT, DEPTH_WIDTH).astype(np.float32) * 0.001


def run(args: argparse.Namespace, contract: DdsContract) -> None:
    if args.viewer:
        # The interactive launcher must create an actual GLFW window even if a
        # parent shell previously selected EGL for headless jobs.
        os.environ["MUJOCO_GL"] = "glfw"
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import mujoco.viewer
    from unitree_sdk2py.idl.default import (
        unitree_hg_msg_dds__IMUState_,
        unitree_hg_msg_dds__LowState_,
    )
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_

    ChannelPublisher, ChannelSubscriber = _initialize_dds(contract)
    scene = args.scene or (
        args.root
        / "third_party/holosoma/src/holosoma/holosoma/data/robots/g1/scenes/scene_g1_29dof_wbt_plane.xml"
    )
    spec = mujoco.MjSpec.from_file(str(scene))
    _install_foot_collision(spec, args.foot_collision)
    torso = spec.body("torso_link")
    if torso is None:
        raise RuntimeError("G1 MuJoCo model has no torso_link")
    if args.depth:
        torso.add_camera(
            name="longship_depth",
            pos=(0.0488, 0.01, 0.4378),
            quat=(-0.65795401, -0.25558131, 0.25121801, 0.66231731),
            fovy=58.29,
        )
    model = spec.compile()
    model.opt.timestep = 1.0 / contract.state_frequency_hz
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, DEPTH_WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, DEPTH_HEIGHT)
    data = mujoco.MjData(model)
    qpos = np.asarray([model.joint(name).qposadr[0] for name in G1_29DOF_JOINTS])
    dofs = np.asarray([model.joint(name).dofadr[0] for name in G1_29DOF_JOINTS])
    actuators = _actuator_ids(model, G1_29DOF_JOINTS)
    reset_q = None if args.reset_q is None else np.asarray(args.reset_q, dtype=np.float64)
    _reset_robot(model, data, qpos, reset_q)
    torso_id = model.body("torso_link").id
    imu_body_id = model.body("pelvis").id
    initial_root_height = float(data.qpos[2])
    horizontal_anchor = data.qpos[:2].copy()
    controls = InteractiveControls(enabled=args.gantry, length=args.gantry_length)
    controls.reset_anchor(data.xpos[torso_id])
    print(
        f"virtual gantry: {'enabled' if controls.enabled else 'disabled'} "
        f"length={controls.length:.3f} m",
        flush=True,
    )
    lock = threading.Lock()
    command: LowCommandSnapshot | None = None

    def on_command(message: Any) -> None:
        nonlocal command
        snapshot = LowCommandSnapshot.from_message(message)
        with lock:
            command = snapshot

    state_publisher = ChannelPublisher(contract.lowstate_topic, LowState_)
    state_publisher.Init()
    secondary_imu_publisher = ChannelPublisher(contract.secondary_imu_topic, IMUState_)
    secondary_imu_publisher.Init()
    command_subscriber = ChannelSubscriber(contract.lowcmd_topic, LowCmd_)
    command_subscriber.Init(on_command, 1)

    def on_sim_control(message: Any) -> None:
        if str(message.data) == "release_gantry":
            controls.disable_gantry()
            print("virtual gantry: released by policy handoff", flush=True)

    sim_control_subscriber = ChannelSubscriber(SIM_CONTROL_TOPIC, String_)
    sim_control_subscriber.Init(on_sim_control, 1)
    depth_publisher = renderer = camera_id = None
    if args.depth:
        depth_publisher = ChannelPublisher(DEPTH_TOPIC, PointCloud2_)
        depth_publisher.Init()
        renderer = mujoco.Renderer(model, height=DEPTH_HEIGHT, width=DEPTH_WIDTH)
        renderer.enable_depth_rendering()
        camera_id = model.camera("longship_depth").id

    viewer = (
        mujoco.viewer.launch_passive(model, data, key_callback=controls.key_callback)
        if args.viewer
        else None
    )
    state = unitree_hg_msg_dds__LowState_()
    secondary_imu = unitree_hg_msg_dds__IMUState_()
    foot_ids = _foot_body_ids(model)
    min_torso_z = float(data.xpos[torso_id, 2])
    initial_torso_xy = data.xpos[torso_id, :2].copy()
    max_torso_tilt_deg = 0.0
    torso_tilt_deg = 0.0
    next_depth_wall = time.monotonic()
    next_viewer_sync = 0.0
    next_tick = time.monotonic()
    deadline = None if args.duration == 0 else time.monotonic() + args.duration
    received = 0
    published = 0
    gantry_release_logged = False
    try:
        while (deadline is None or time.monotonic() < deadline) and (
            viewer is None or viewer.is_running()
        ) and (args.sim_duration == 0 or data.time < args.sim_duration):
            if controls.consume_reset():
                _reset_robot(model, data, qpos, reset_q)
                controls.reset_anchor(data.xpos[torso_id])
                with lock:
                    command = None
                next_depth_wall = time.monotonic()
                next_viewer_sync = 0.0
                next_tick = time.monotonic()
            with lock:
                current = command
            authorized = current is not None and current.has_control_authority()
            if authorized:
                assert current is not None
                received += 1
                if (
                    args.gantry_release_after is not None
                    and not gantry_release_logged
                    and data.time >= args.gantry_release_after
                ):
                    controls.disable_gantry()
                    gantry_release_logged = True
                    print(
                        f"virtual gantry: automatically disabled at sim_time={data.time:.3f}s",
                        flush=True,
                    )
                data.ctrl[actuators] = (
                    current.tau
                    + current.kp * (current.q - data.qpos[qpos])
                    + current.kd * (current.dq - data.qvel[dofs])
                )
                torso_velocity_world = np.zeros(6)
                mujoco.mj_objectVelocity(
                    model, data, mujoco.mjtObj.mjOBJ_BODY, torso_id, torso_velocity_world, 0
                )
                data.xfrc_applied[torso_id, :] = 0.0
                if args.gantry_mode == "rope":
                    controls.follow_horizontally(data.xpos[torso_id])
                    data.xfrc_applied[torso_id, :3] = controls.force(
                        data.xpos[torso_id], torso_velocity_world[3:]
                    )
                elif controls.enabled:
                    _apply_hiking_spotter(
                        model,
                        data,
                        imu_body_id,
                        torso_id,
                        initial_root_height - 0.06,
                        horizontal_anchor,
                    )
                else:
                    data.xfrc_applied[imu_body_id, :] = 0.0
                mujoco.mj_step(model, data)
            else:
                # The simulator owns no policy pose or PD gains.  Keep the
                # reset state frozen until an adapter supplies a complete,
                # finite LowCmd with control authority.
                data.ctrl[actuators] = 0.0
            min_torso_z = min(min_torso_z, float(data.xpos[torso_id, 2]))
            torso_rotation = data.xmat[torso_id].reshape(3, 3)
            torso_tilt_deg = math.degrees(
                math.acos(float(np.clip(torso_rotation[2, 2], -1.0, 1.0)))
            )
            max_torso_tilt_deg = max(max_torso_tilt_deg, torso_tilt_deg)
            imu_velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, imu_body_id, imu_velocity, 1
            )
            state.tick = int(data.time * 1000)
            state.mode_machine = 5
            # Match HoloSoma's bridge contract: Unitree LowState IMU comes from
            # the free robot root, not the articulated torso above the waist.
            state.imu_state.quaternion[:] = data.xquat[imu_body_id]
            state.imu_state.gyroscope[:] = imu_velocity[:3]
            # Match SONIC's own MuJoCo SDK2 bridge: the floating-base linear
            # acceleration is the first three free-joint accelerations.
            state.imu_state.accelerometer[:] = data.qacc[:3]
            torso_imu_velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, torso_id, torso_imu_velocity, 1
            )
            secondary_imu.quaternion[:] = data.xquat[torso_id]
            secondary_imu.gyroscope[:] = torso_imu_velocity[:3]
            for index in range(29):
                motor = state.motor_state[index]
                motor.mode = 1
                motor.q = float(data.qpos[qpos[index]])
                motor.dq = float(data.qvel[dofs[index]])
                motor.tau_est = float(data.ctrl[actuators[index]])
            state_publisher.Write(state)
            secondary_imu_publisher.Write(secondary_imu)
            published += 1
            now_wall = time.monotonic()
            if renderer is not None and now_wall >= next_depth_wall:
                renderer.update_scene(data, camera=camera_id)
                depth_publisher.Write(encode_depth(np.asarray(renderer.render()), data.time))
                next_depth_wall += 1.0 / 30.0
                if next_depth_wall < now_wall:
                    next_depth_wall = now_wall
            wall_now = time.monotonic()
            if viewer is not None and wall_now >= next_viewer_sync:
                with viewer.lock():
                    _draw_virtual_gantry(viewer.user_scn, controls, data.xpos[torso_id])
                    _update_tracking_camera(viewer.cam, controls, data.xpos[torso_id])
                viewer.sync()
                next_viewer_sync = wall_now + 1.0 / VIEWER_FREQUENCY_HZ
            next_tick += 1.0 / contract.state_frequency_hz
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    finally:
        if renderer is not None:
            renderer.close()
        if viewer is not None:
            viewer.close()
    print(
        f"LONGSHIP MUJOCO SIM DONE: lowstate={published} "
        f"command_cycles={received} sim_time={data.time:.3f}s depth={args.depth} "
        f"torso_z={data.xpos[torso_id, 2]:.3f}m min_torso_z={min_torso_z:.3f}m "
        f"final_tilt={torso_tilt_deg:.1f}deg max_tilt={max_torso_tilt_deg:.1f}deg "
        f"displacement_xy=({data.xpos[torso_id, 0] - initial_torso_xy[0]:.3f},"
        f"{data.xpos[torso_id, 1] - initial_torso_xy[1]:.3f})m "
        f"feet_z=({data.xpos[foot_ids[0], 2]:.3f},{data.xpos[foot_ids[1], 2]:.3f})m"
    )
    if args.require_command and received == 0:
        raise RuntimeError("simulator received no rt/lowcmd samples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scene", type=Path)
    parser.add_argument(
        "--foot-collision",
        choices=("native", "hiking_training_v1"),
        default="native",
    )
    parser.add_argument(
        "--gantry-mode",
        choices=("rope", "hiking_spotter_v1"),
        default="rope",
    )
    parser.add_argument("--reset-q", type=float, nargs=29)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--state-frequency-hz", type=int, default=500)
    parser.add_argument("--control-frequency-hz", type=int, default=50)
    parser.add_argument("--command-frequency-hz", type=int, default=200)
    parser.add_argument("--duration", type=float, default=0.0, help="0 runs until interrupted")
    parser.add_argument(
        "--sim-duration",
        type=float,
        default=0.0,
        help="0 disables; otherwise stop at this MuJoCo simulation time",
    )
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo window")
    parser.add_argument("--gantry", action="store_true", help="start with the virtual gantry enabled")
    parser.add_argument(
        "--gantry-release-after",
        type=float,
        help="headless validation: disable the gantry after this many simulated seconds",
    )
    parser.add_argument(
        "--gantry-length",
        type=float,
        default=1.0,
        help="virtual gantry rest length in metres (HoloSoma guide default: 1.0)",
    )
    parser.add_argument("--require-command", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.root = args.root.resolve()
    if (
        args.duration < 0
        or args.sim_duration < 0
        or args.gantry_length < 0
        or (args.gantry_release_after is not None and args.gantry_release_after < 0)
    ):
        raise ValueError(
            "duration, sim duration, gantry length, and gantry release time must be non-negative"
        )
    contract = DdsContract(
        domain_id=args.domain_id,
        interface=args.interface,
        depth_topic=DEPTH_TOPIC if args.depth else None,
        state_frequency_hz=args.state_frequency_hz,
        control_frequency_hz=args.control_frequency_hz,
        command_frequency_hz=args.command_frequency_hz,
    )
    contract.validate()
    run(args, contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

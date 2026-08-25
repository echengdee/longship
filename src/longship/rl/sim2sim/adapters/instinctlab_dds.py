#!/usr/bin/env python3
"""Hiking-in-the-Wild policy adapter for the Longship DDS simulator."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from longship.rl.sim2sim.dds import (
    DEPTH_TOPIC,
    LOWCMD_TOPIC,
    LOWSTATE_TOPIC,
    SECONDARY_IMU_TOPIC,
    SIM_CONTROL_TOPIC,
    DdsContract,
)
from longship.rl.sim2sim.control import ControlMode, PolicyControl
from longship.rl.sim2sim.hiking_pipeline import (
    HikingModeCommand,
    HikingOnnxPolicy,
    dds_to_policy,
    policy_to_dds,
    projected_gravity,
)
from longship.rl.sim2sim.profile import bundled_profile_path, load_control_profile
from longship.rl.sim2sim.simulator import decode_depth
from longship.rl.sim2sim.teleop import TeleopSubscriber


def _initialize_dds(contract: DdsContract) -> tuple[Any, Any, Any]:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber

    # Unitree SDK2 raises when initialization fails and returns ``None`` on
    # success.  Do not interpret its return value as a boolean status.
    ChannelFactoryInitialize(contract.domain_id, contract.interface)
    return ChannelPublisher, ChannelSubscriber, contract


def run_controller(args: argparse.Namespace, contract: DdsContract) -> None:
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_

    ChannelPublisher, ChannelSubscriber, _ = _initialize_dds(contract)
    profile = load_control_profile(args.profile, "instinctlab")
    stand_checkpoint = profile.resolve_artifact(args.root, profile.initialization)
    parkour_checkpoint = profile.resolve_artifact(args.root, profile.policy)
    if stand_checkpoint is None or parkour_checkpoint is None:
        raise ValueError("InstinctLab profile must identify stand and parkour checkpoints")
    parkour = HikingOnnxPolicy(parkour_checkpoint, args.provider)
    # The source-aligned Hiking Sim2Sim warm-up uses the rendered depth stream
    # for the released stand checkpoint before handing off to parkour.
    stand = HikingOnnxPolicy(stand_checkpoint, args.provider)
    policies = {"stand": stand, "parkour": parkour}
    mode_command = HikingModeCommand()
    teleop = TeleopSubscriber(args.teleop_endpoint, "instinctlab") if args.teleop_endpoint else None
    control = PolicyControl(
        stand.default_q.copy(),
        init_duration=(
            profile.initialization_duration_s if args.init_duration is None else args.init_duration
        ),
        linear_step=float(profile.policy_options.get("linear_speed_mps", 0.4)),
        angular_step=float(profile.policy_options.get("yaw_rate_radps", 0.5)),
    )
    if teleop is None:
        control.mode = ControlMode.ENABLED
        control.lin_x = args.command_x
    condition = threading.Condition()
    latest: dict[str, Any] = {}

    def on_state(message: Any) -> None:
        with condition:
            latest["state"] = message
            condition.notify_all()

    def on_depth(message: Any) -> None:
        try:
            depth = decode_depth(message)
        except Exception as exc:
            with condition:
                latest["depth_error"] = exc
                condition.notify_all()
            return
        with condition:
            latest["depth"] = depth
            condition.notify_all()

    def on_torso_imu(message: Any) -> None:
        with condition:
            latest["torso_imu"] = message
            condition.notify_all()

    publisher = ChannelPublisher(LOWCMD_TOPIC, LowCmd_)
    publisher.Init()
    sim_control_publisher = ChannelPublisher(SIM_CONTROL_TOPIC, String_)
    sim_control_publisher.Init()
    state_subscriber = ChannelSubscriber(LOWSTATE_TOPIC, LowState_)
    state_subscriber.Init(on_state, 2)
    torso_imu_subscriber = ChannelSubscriber(SECONDARY_IMU_TOPIC, IMUState_)
    torso_imu_subscriber.Init(on_torso_imu, 2)
    depth_subscriber = ChannelSubscriber(DEPTH_TOPIC, PointCloud2_)
    depth_subscriber.Init(on_depth, 2)
    command = unitree_hg_msg_dds__LowCmd_()
    started = time.monotonic()
    last_policy_sim_time: float | None = None
    target = stand.default_q.copy()
    inference_count = 0
    publish_count = 0
    auto_parkour_started = False
    while args.duration == 0 or time.monotonic() - started < args.duration:
        now = time.monotonic()
        with condition:
            if not {"state", "torso_imu", "depth"}.issubset(latest):
                condition.wait(timeout=0.1)
                continue
            state = latest["state"]
            torso_imu = latest["torso_imu"]
            depth = latest["depth"].copy()
        if args.sim_duration > 0 and float(state.tick) * 0.001 >= args.sim_duration:
            break
        dds_q = np.asarray([state.motor_state[i].q for i in range(29)])
        dds_dq = np.asarray([state.motor_state[i].dq for i in range(29)])
        q = dds_to_policy(dds_q)
        dq = dds_to_policy(dds_dq)
        if teleop is not None:
            for event in teleop.poll():
                was_enabled = control.policy_enabled
                previous_mode = mode_command.mode
                if event.key in "12np":
                    result = mode_command.handle(event.key)
                    if mode_command.mode != previous_mode:
                        # The released agents read last_action from the most
                        # recently sent joint target across agent handoffs.
                        policies[mode_command.mode].last_action = policies[
                            previous_mode
                        ].last_action.copy()
                elif event.key in "wqe" and mode_command.mode == "stand":
                    result = "motion ignored while Hiking agent=stand; select parkour with 2"
                else:
                    result = control.handle(event.key, q, lateral=False, backward=False)
                if (
                    bool(profile.policy_options.get("release_spotter_on_parkour", False))
                    and mode_command.mode == "parkour"
                    and control.policy_enabled
                    and (previous_mode != "parkour" or not was_enabled)
                ):
                    sim_control_publisher.Write(String_("release_gantry"))
                    result += "; Hiking spotter release requested"
                print(f"teleop {event.key!r}: {result}")
        if (
            args.auto_parkour_at_sim is not None
            and not auto_parkour_started
            and control.policy_enabled
            and float(state.tick) * 0.001 >= args.auto_parkour_at_sim
        ):
            previous = policies[mode_command.mode]
            mode_command.handle("2")
            parkour.last_action = previous.last_action.copy()
            control.lin_x = args.command_x
            sim_control_publisher.Write(String_("release_gantry"))
            auto_parkour_started = True
            print(
                f"Hiking regression handoff: sim_time={float(state.tick) * 0.001:.3f}s "
                f"agent=parkour command_x={control.lin_x:.2f}",
                flush=True,
            )
        legacy_standing = teleop is None and now - started < args.stand_duration
        selected_mode = "stand" if legacy_standing else mode_command.mode
        active = policies[selected_mode] if control.mode is ControlMode.ENABLED else stand
        sim_time = float(state.tick) * 0.001
        if (
            last_policy_sim_time is None
            or sim_time - last_policy_sim_time >= 1.0 / contract.control_frequency_hz - 1.0e-6
        ):
            hold_target = control.target(q, now)
            if control.mode is ControlMode.ENABLED:
                # Hiking was trained and deployed against the torso-mounted
                # secondary IMU, not LowState's pelvis IMU.
                quaternion = np.asarray(torso_imu.quaternion, dtype=np.float64)
                gyro = np.asarray(torso_imu.gyroscope, dtype=np.float64)
                terms = (
                    gyro * 0.25,
                    projected_gravity(quaternion),
                    np.asarray(
                        (0.0, 0.0, 0.0) if legacy_standing else (control.lin_x, 0.0, control.yaw)
                    ),
                    q - active.default_q,
                    dq * 0.05,
                    active.last_action.astype(np.float64),
                )
                target = active.infer(terms, depth)
                inference_count += 1
            elif hold_target is not None:
                target = hold_target
            # Policies are trained at 50 Hz of physical time.  Scheduling
            # from wall time over-runs the actor whenever high-fidelity
            # contacts make MuJoCo slower than real time.
            last_policy_sim_time = sim_time
        if control.mode is ControlMode.IDLE:
            time.sleep(1.0 / contract.command_frequency_hz)
            continue
        dds_target = policy_to_dds(target)
        gain_scale = (
            float(profile.policy_options.get("initialization_gain_scale", 1.0))
            if control.mode is not ControlMode.ENABLED
            else 1.0
        )
        dds_kp = policy_to_dds(gain_scale * active.kp, signed=False)
        dds_kd = policy_to_dds(gain_scale * active.kd, signed=False)
        for index in range(29):
            motor = command.motor_cmd[index]
            motor.mode = 1
            motor.q = float(dds_target[index])
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = float(dds_kp[index])
            motor.kd = float(dds_kd[index])
        publisher.Write(command)
        publish_count += 1
        time.sleep(1.0 / contract.command_frequency_hz)
    print(f"HIKING DDS POLICY DONE: inference={inference_count} lowcmd={publish_count}")
    if teleop is None and inference_count == 0:
        raise RuntimeError("controller received no synchronized LowState/depth observations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=bundled_profile_path("instinctlab"))
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 runs until interrupted")
    parser.add_argument(
        "--sim-duration",
        type=float,
        default=0.0,
        help="0 disables; otherwise stop at the LowState simulation timestamp",
    )
    parser.add_argument("--stand-duration", type=float, default=2.0)
    parser.add_argument("--command-x", type=float, default=0.3)
    parser.add_argument("--auto-parkour-at-sim", type=float)
    parser.add_argument("--teleop-endpoint")
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--init-duration", type=float, default=None, help="override profile value")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.root = args.root.resolve()
    profile = load_control_profile(args.profile, "instinctlab")
    contract = replace(profile.dds, domain_id=args.domain_id, interface=args.interface)
    contract.validate()
    if (
        args.duration < 0
        or args.sim_duration < 0
        or (args.auto_parkour_at_sim is not None and args.auto_parkour_at_sim < 0)
        or (args.init_duration is not None and args.init_duration <= 0)
    ):
        raise ValueError(
            "durations/auto parkour time must be non-negative and --init-duration must be positive"
        )
    run_controller(args, contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Perceptive Humanoid Parkour student-policy adapter for Longship DDS."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from longship.rl.sim2sim.control import ControlMode, PolicyControl
from longship.rl.sim2sim.dds import (
    DEPTH_TOPIC,
    LOWCMD_TOPIC,
    LOWSTATE_TOPIC,
    SECONDARY_IMU_TOPIC,
    DdsContract,
)
from longship.rl.sim2sim.php_pipeline import PhpOnnxPolicy, command_vector
from longship.rl.sim2sim.profile import bundled_profile_path, load_control_profile
from longship.rl.sim2sim.simulator import decode_depth
from longship.rl.sim2sim.teleop import TeleopSubscriber


def _initialize_dds(contract: DdsContract) -> tuple[Any, Any]:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )

    ChannelFactoryInitialize(contract.domain_id, contract.interface)
    return ChannelPublisher, ChannelSubscriber


def run_controller(args: argparse.Namespace, contract: DdsContract) -> None:
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_

    ChannelPublisher, ChannelSubscriber = _initialize_dds(contract)
    profile = load_control_profile(args.profile, "php")
    policy_path = profile.resolve_artifact(args.root, profile.policy)
    depth_path = args.root / str(profile.policy_options["depth_backbone"])
    if policy_path is None:
        raise ValueError("PHP profile must identify the released student policy")
    policy = PhpOnnxPolicy(policy_path, depth_path, args.provider)
    control = PolicyControl(
        policy.policy_to_dds_vector(policy.default_q),
        init_duration=profile.initialization_duration_s,
    )
    selected_command = args.command
    high_speed = not args.low_speed
    teleop = TeleopSubscriber(args.teleop_endpoint, "php") if args.teleop_endpoint else None
    if teleop is None:
        control.mode = ControlMode.ENABLED
    condition = threading.Condition()
    latest: dict[str, Any] = {}

    def store(name: str, value: Any) -> None:
        with condition:
            latest[name] = value
            condition.notify_all()

    def on_depth(message: Any) -> None:
        try:
            store("depth", decode_depth(message))
        except Exception as exc:
            store("depth_error", exc)

    publisher = ChannelPublisher(LOWCMD_TOPIC, LowCmd_)
    publisher.Init()
    state_subscriber = ChannelSubscriber(LOWSTATE_TOPIC, LowState_)
    state_subscriber.Init(lambda message: store("state", message), 2)
    torso_subscriber = ChannelSubscriber(SECONDARY_IMU_TOPIC, IMUState_)
    torso_subscriber.Init(lambda message: store("torso_imu", message), 2)
    depth_subscriber = ChannelSubscriber(DEPTH_TOPIC, PointCloud2_)
    depth_subscriber.Init(on_depth, 2)
    command = unitree_hg_msg_dds__LowCmd_()
    target_dds = policy.policy_to_dds_vector(policy.default_q)
    kp_dds = policy.policy_to_dds_vector(policy.kp)
    kd_dds = policy.policy_to_dds_vector(policy.kd)
    started = time.monotonic()
    last_policy_sim_time: float | None = None
    inference_count = 0
    publish_count = 0
    command_keys = {
        "w": "forward", "q": "left_forward", "e": "right_forward",
        "a": "left", "d": "right", "s": "stop",
    }
    while args.duration == 0 or time.monotonic() - started < args.duration:
        now = time.monotonic()
        with condition:
            if not {"state", "torso_imu", "depth"}.issubset(latest):
                condition.wait(timeout=0.1)
                continue
            state = latest["state"]
            torso_imu = latest["torso_imu"]
            depth = latest["depth"].copy()
        sim_time = float(state.tick) * 0.001
        if args.sim_duration > 0 and sim_time >= args.sim_duration:
            break
        q_dds = np.asarray([state.motor_state[index].q for index in range(29)])
        dq_dds = np.asarray([state.motor_state[index].dq for index in range(29)])
        if teleop is not None:
            for event in teleop.poll():
                if event.key in command_keys:
                    selected_command = command_keys[event.key]
                    result = f"PHP command={selected_command}"
                elif event.key == "y":
                    high_speed = not high_speed
                    result = f"PHP speed={'high' if high_speed else 'low'}"
                else:
                    result = control.handle(event.key, q_dds, lateral=False, backward=False)
                    if control.mode is ControlMode.ENABLED:
                        policy.reset()
                print(f"teleop {event.key!r}: {result}", flush=True)
        if (
            last_policy_sim_time is None
            or sim_time - last_policy_sim_time >= 1.0 / contract.control_frequency_hz - 1.0e-6
        ):
            hold_target = control.target(q_dds, now)
            if control.mode is ControlMode.ENABLED:
                target_policy = policy.infer(
                    torso_quaternion_wxyz=torso_imu.quaternion,
                    base_angular_velocity=state.imu_state.gyroscope,
                    joint_position=q_dds,
                    joint_velocity=dq_dds,
                    command=command_vector(selected_command, high_speed=high_speed),
                    depth=depth,
                )
                target_dds = policy.policy_to_dds_vector(target_policy)
                inference_count += 1
            elif hold_target is not None:
                target_dds = hold_target
            last_policy_sim_time = sim_time
        if control.mode is ControlMode.IDLE:
            time.sleep(1.0 / contract.command_frequency_hz)
            continue
        for index in range(29):
            motor = command.motor_cmd[index]
            motor.mode = 1
            motor.q = float(target_dds[index])
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = float(kp_dds[index])
            motor.kd = float(kd_dds[index])
        publisher.Write(command)
        publish_count += 1
        time.sleep(1.0 / contract.command_frequency_hz)
    print(
        f"PHP DDS POLICY DONE: inference={inference_count} lowcmd={publish_count} "
        f"command={selected_command} speed={'high' if high_speed else 'low'}"
    )
    if teleop is None and inference_count == 0:
        raise RuntimeError("PHP controller received no synchronized LowState/depth observations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=bundled_profile_path("php"))
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 runs until interrupted")
    parser.add_argument("--sim-duration", type=float, default=0.0)
    parser.add_argument(
        "--command",
        choices=("stop", "forward", "left_forward", "left", "right_forward", "right"),
        default="stop",
    )
    parser.add_argument("--low-speed", action="store_true")
    parser.add_argument("--teleop-endpoint")
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.root = args.root.resolve()
    profile = load_control_profile(args.profile, "php")
    contract = replace(profile.dds, domain_id=args.domain_id, interface=args.interface)
    contract.validate()
    if args.duration < 0 or args.sim_duration < 0:
        raise ValueError("duration and sim duration must be non-negative")
    run_controller(args, contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

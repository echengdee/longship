#!/usr/bin/env python3
"""Run SONIC planner, encoder, and policy through Longship's Python ONNX runtime."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from longship.rl.sim2sim.dds import DdsContract, SIM_CONTROL_TOPIC
from longship.rl.sim2sim.profile import bundled_profile_path, load_control_profile
from longship.rl.sim2sim.sonic_pipeline import DEFAULT_Q, KD, KP, SonicOnnxPipeline, SonicRobotState
from longship.rl.sim2sim.teleop import TeleopSubscriber


class SonicStateReceiver:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.lowstate: Any | None = None
        self.torso_imu: Any | None = None

    def on_lowstate(self, message: Any) -> None:
        with self.condition:
            self.lowstate = message
            self.condition.notify_all()

    def on_torso_imu(self, message: Any) -> None:
        with self.condition:
            self.torso_imu = message
            self.condition.notify_all()

    def snapshot(self) -> SonicRobotState | None:
        with self.condition:
            if self.lowstate is None or self.torso_imu is None:
                return None
            state, torso = self.lowstate, self.torso_imu
            motors = state.motor_state
            return SonicRobotState(
                tick=int(state.tick),
                q=np.asarray([motor.q for motor in motors[:29]], dtype=np.float64),
                dq=np.asarray([motor.dq for motor in motors[:29]], dtype=np.float64),
                quaternion=np.asarray(state.imu_state.quaternion, dtype=np.float64).copy(),
                gyroscope=np.asarray(state.imu_state.gyroscope, dtype=np.float64).copy(),
                torso_quaternion=np.asarray(torso.quaternion, dtype=np.float64).copy(),
                torso_gyroscope=np.asarray(torso.gyroscope, dtype=np.float64).copy(),
            )


def run(args: argparse.Namespace, contract: DdsContract) -> None:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_

    teleop = TeleopSubscriber(args.teleop_endpoint, "sonic")
    print("SONIC ONNX: loading planner, encoder, and decoder", flush=True)
    pipeline = SonicOnnxPipeline(
        args.decoder,
        args.encoder,
        args.planner,
        provider=args.provider,
    )
    print(f"SONIC ONNX READY: providers={pipeline.providers}", flush=True)
    ChannelFactoryInitialize(contract.domain_id, contract.interface)
    receiver = SonicStateReceiver()
    publisher = ChannelPublisher(contract.lowcmd_topic, LowCmd_)
    publisher.Init()
    sim_control_publisher = ChannelPublisher(SIM_CONTROL_TOPIC, String_)
    sim_control_publisher.Init()
    lowstate_subscriber = ChannelSubscriber(contract.lowstate_topic, LowState_)
    lowstate_subscriber.Init(receiver.on_lowstate, 2)
    torso_subscriber = ChannelSubscriber(contract.secondary_imu_topic, IMUState_)
    torso_subscriber.Init(receiver.on_torso_imu, 2)
    command = unitree_hg_msg_dds__LowCmd_()

    mode = "idle"
    init_started = 0.0
    init_from = DEFAULT_Q.copy()
    queued_enable = False
    target = DEFAULT_Q.copy()
    target_lock = threading.Lock()
    started = time.monotonic()
    inference_count = {"value": 0}
    enabled_started: float | None = None
    publish_count = 0
    policy_stop = threading.Event()
    policy_thread: threading.Thread | None = None
    policy_error: list[BaseException] = []
    auto_walk_sent = False
    policy_input_file = (
        args.policy_input_logfile.open("w", encoding="utf-8", buffering=1)
        if args.policy_input_logfile is not None
        else None
    )

    def policy_loop() -> None:
        nonlocal target
        last_policy_sim_time: float | None = None
        while not policy_stop.is_set():
            state = receiver.snapshot()
            sim_time = None if state is None else float(state.tick) * 0.001
            due = (
                sim_time is not None
                and (
                    last_policy_sim_time is None
                    or sim_time - last_policy_sim_time
                    >= 1.0 / contract.control_frequency_hz - 1.0e-6
                )
            )
            if state is not None and due:
                try:
                    next_target = pipeline.infer(state)
                except BaseException as exc:
                    policy_error.append(exc)
                    policy_stop.set()
                    return
                with target_lock:
                    target = next_target
                if policy_input_file is not None and pipeline.last_policy_observation is not None:
                    policy_input_file.write(
                        ",".join(str(float(value)) for value in pipeline.last_policy_observation)
                        + "\n"
                    )
                inference_count["value"] += 1
                last_policy_sim_time = sim_time
            else:
                with receiver.condition:
                    receiver.condition.wait(timeout=0.01)

    def enable(state: SonicRobotState) -> None:
        nonlocal mode, target, policy_thread, enabled_started
        print("SONIC ONNX: seeding measured tracking reference", flush=True)
        pipeline.initialize_planner(state)
        with target_lock:
            target = DEFAULT_Q.copy()
        mode = "enabled"
        enabled_started = time.monotonic()
        policy_thread = threading.Thread(target=policy_loop, daemon=True, name="sonic-onnx-policy")
        policy_thread.start()
        print("SONIC ONNX: tracking policy enabled; planner starts on a motion key", flush=True)

    try:
        while (
            (args.duration == 0 or time.monotonic() - started < args.duration)
            and (
                args.sim_duration == 0
                or receiver.snapshot() is None
                or float(receiver.snapshot().tick) * 0.001 < args.sim_duration
            )
        ):
            state = receiver.snapshot()
            if state is None:
                with receiver.condition:
                    receiver.condition.wait(timeout=0.05)
                continue
            now = time.monotonic()
            if args.auto_enable and mode == "idle":
                mode = "initializing"
                init_started = now
                init_from = state.q.copy()
                queued_enable = True
                print(
                    f"SONIC regression: initializing over {args.init_duration:g} s",
                    flush=True,
                )
            if policy_error:
                raise RuntimeError("SONIC ONNX policy loop failed") from policy_error[0]
            for event in teleop.poll():
                key = event.key
                if key == "i":
                    if mode == "idle":
                        mode = "initializing"
                        init_started = now
                        init_from = state.q.copy()
                        print(
                            f"teleop 'i': SONIC ONNX initializing over "
                            f"{args.init_duration:g} s",
                            flush=True,
                        )
                    else:
                        print(f"teleop 'i': ignored while mode={mode}", flush=True)
                elif key == "]":
                    if mode == "initializing":
                        queued_enable = True
                        print("teleop ']': queued until initialization completes", flush=True)
                    elif mode == "ready":
                        enable(state)
                    elif mode == "idle":
                        print("teleop ']': ignored; press i first", flush=True)
                elif mode == "enabled":
                    print(f"teleop {key!r}: {pipeline.handle(key, state)}", flush=True)
                else:
                    print(f"teleop {key!r}: ignored while mode={mode}", flush=True)

            if mode == "initializing":
                ratio = min(1.0, (now - init_started) / args.init_duration)
                target = (1.0 - ratio) * init_from + ratio * DEFAULT_Q
                if ratio >= 1.0:
                    mode = "ready"
                    print("SONIC ONNX: initialization complete", flush=True)
                    if queued_enable:
                        queued_enable = False
                        enable(state)

            sim_time = float(state.tick) * 0.001
            if (
                mode == "enabled"
                and args.auto_walk_at_sim is not None
                and not auto_walk_sent
                and sim_time >= args.auto_walk_at_sim
            ):
                auto_walk_sent = True
                mode_message = pipeline.handle(args.auto_mode_key, state)
                message = pipeline.handle("w", state)
                sim_control_publisher.Write(String_("release_gantry"))
                print(
                    f"SONIC regression handoff: sim_time={sim_time:.3f}s "
                    f"{mode_message}; {message}",
                    flush=True,
                )

            if mode != "idle":
                with target_lock:
                    command_target = target.copy()
                for index in range(29):
                    motor = command.motor_cmd[index]
                    motor.mode = 1
                    motor.q = float(command_target[index])
                    motor.dq = 0.0
                    motor.tau = 0.0
                    motor.kp = float(KP[index])
                    motor.kd = float(KD[index])
                publisher.Write(command)
                publish_count += 1
            time.sleep(1.0 / contract.command_frequency_hz)
    finally:
        policy_stop.set()
        if policy_thread is not None:
            policy_thread.join(timeout=2.0)
        pipeline.close()
        if policy_input_file is not None:
            policy_input_file.close()
    elapsed = max(time.monotonic() - started, 1.0e-9)
    policy_elapsed = max(
        time.monotonic() - (enabled_started if enabled_started is not None else started),
        1.0e-9,
    )
    print(
        f"SONIC ONNX DDS DONE: inference={inference_count['value']} "
        f"({inference_count['value'] / policy_elapsed:.1f} Hz) lowcmd={publish_count} "
        f"({publish_count / elapsed:.1f} Hz)",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=bundled_profile_path("sonic"))
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--teleop-endpoint", required=True)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--init-duration", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--sim-duration", type=float, default=0.0)
    parser.add_argument("--auto-enable", action="store_true")
    parser.add_argument("--auto-walk-at-sim", type=float)
    parser.add_argument("--auto-mode-key", choices=tuple("12345678"), default="1")
    parser.add_argument("--policy-input-logfile", type=Path)
    args = parser.parse_args()
    if (
        args.init_duration <= 0
        or args.duration < 0
        or args.sim_duration < 0
        or (args.auto_walk_at_sim is not None and args.auto_walk_at_sim < 0)
    ):
        raise ValueError("init duration must be positive and duration must be non-negative")
    profile = load_control_profile(args.profile, "sonic")
    contract = replace(profile.dds, domain_id=args.domain_id, interface=args.interface)
    contract.validate()
    run(args, contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

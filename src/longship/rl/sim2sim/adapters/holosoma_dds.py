#!/usr/bin/env python3
"""Run the released HoloSoma locomotion policy on Longship's DDS contract."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import replace
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from longship.rl.sim2sim.dds import DdsContract
from longship.rl.sim2sim.control import ControlMode, PolicyControl
from longship.rl.sim2sim.profile import bundled_profile_path, load_control_profile
from longship.rl.sim2sim.teleop import TeleopSubscriber


@dataclass(frozen=True, slots=True)
class LowStateSnapshot:
    q: np.ndarray
    dq: np.ndarray
    gyroscope: np.ndarray
    quaternion: np.ndarray
    tick: int

    @classmethod
    def from_message(cls, message: Any) -> "LowStateSnapshot":
        motors = message.motor_state
        return cls(
            q=np.asarray([motor.q for motor in motors[:29]], dtype=np.float64),
            dq=np.asarray([motor.dq for motor in motors[:29]], dtype=np.float64),
            gyroscope=np.asarray(message.imu_state.gyroscope, dtype=np.float64).copy(),
            quaternion=np.asarray(message.imu_state.quaternion, dtype=np.float64).copy(),
            tick=int(message.tick),
        )


def _projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion_wxyz / np.linalg.norm(quaternion_wxyz)
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )
    return rotation.T @ np.asarray((0.0, 0.0, -1.0))


def _advance_phase(
    phase: np.ndarray,
    is_standing: bool,
    velocity: tuple[float, float, float],
) -> tuple[np.ndarray, bool]:
    """Match HoloSoma's locomotion phase update, including its zero-command stance."""
    phase = (phase + 2.0 * math.pi / 50.0 + math.pi) % (2.0 * math.pi) - math.pi
    if np.linalg.norm(velocity) < 0.01:
        phase[:] = math.pi
        return phase, True
    if is_standing:
        # This actor's smallest stand-to-walk action jump is [pi, 0].  The
        # opposite phase makes the other leg lunge at the first motion tick.
        phase[:] = (math.pi, 0.0)
    return phase, False


class ReleasedPolicy:
    def __init__(self, model_path: Path, default_q: np.ndarray, action_scale: float) -> None:
        import onnx
        import onnxruntime as ort

        model = onnx.load(str(model_path))
        metadata = {item.key: json.loads(item.value) for item in model.metadata_props}
        self.kp = np.asarray(metadata["kp"], dtype=np.float64)
        self.kd = np.asarray(metadata["kd"], dtype=np.float64)
        if self.kp.shape != (29,) or self.kd.shape != (29,):
            raise ValueError("HoloSoma ONNX metadata kp/kd must contain 29 values")
        if np.any(self.kp <= 0.0) or np.any(self.kd <= 0.0):
            raise ValueError("HoloSoma ONNX metadata kp/kd must be positive")
        self.default_q = default_q.copy()
        self.action_scale = action_scale
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.last_action = np.zeros(29, dtype=np.float32)
        self.phase = np.asarray((0.0, math.pi), dtype=np.float64)
        self.is_standing = False

    def infer(self, state: Any, velocity: tuple[float, float, float]) -> np.ndarray:
        self.phase, self.is_standing = _advance_phase(self.phase, self.is_standing, velocity)
        q = state.q
        dq = state.dq
        # HoloSoma sorts observation term names before concatenation.
        observation = np.concatenate(
            (
                self.last_action,
                state.gyroscope * 0.25,
                np.asarray((velocity[2],)),
                np.asarray(velocity[:2]),
                np.cos(self.phase),
                q - self.default_q,
                dq * 0.05,
                _projected_gravity(state.quaternion),
                np.sin(self.phase),
            )
        )[None].astype(np.float32)
        action = self.session.run(None, {"actor_obs": observation})[0].reshape(29)
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("HoloSoma policy produced NaN or infinity")
        self.last_action = np.clip(action, -100.0, 100.0)
        return self.default_q + self.action_scale * self.last_action


def run(args: argparse.Namespace, contract: DdsContract) -> None:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

    ChannelFactoryInitialize(contract.domain_id, contract.interface)
    profile = load_control_profile(args.profile, "holosoma")
    initialization = profile.initialization
    if initialization.default_q is None or initialization.kp is None or initialization.kd is None:
        raise ValueError("HoloSoma profile must provide initialization default_q/kp/kd")
    policy = ReleasedPolicy(
        args.model,
        initialization.default_q,
        float(profile.policy_options.get("action_scale", 0.25)),
    )
    teleop = TeleopSubscriber(args.teleop_endpoint, "holosoma") if args.teleop_endpoint else None
    control = PolicyControl(
        initialization.default_q.copy(),
        init_duration=(
            profile.initialization_duration_s if args.init_duration is None else args.init_duration
        ),
        linear_step=float(profile.policy_options.get("linear_speed_mps", 0.4)),
        angular_step=float(profile.policy_options.get("yaw_rate_radps", 0.5)),
        require_walk_enable=teleop is not None
        and bool(profile.policy_options.get("require_walk_enable", True)),
        smooth_velocity=teleop is not None
        and bool(profile.policy_options.get("smooth_velocity", True)),
    )
    if teleop is None:
        control.mode = ControlMode.ENABLED
        control.lin_x = args.command_x
    condition = threading.Condition()
    latest: dict[str, Any] = {}

    def on_state(message: Any) -> None:
        snapshot = LowStateSnapshot.from_message(message)
        with condition:
            latest["state"] = snapshot
            condition.notify_all()

    publisher = ChannelPublisher(contract.lowcmd_topic, LowCmd_)
    publisher.Init()
    subscriber = ChannelSubscriber(contract.lowstate_topic, LowState_)
    subscriber.Init(on_state, 2)
    command = unitree_hg_msg_dds__LowCmd_()
    started = time.monotonic()
    next_policy = started
    target = initialization.default_q.copy()
    inference_count = publish_count = 0
    while args.duration == 0 or time.monotonic() - started < args.duration:
        now = time.monotonic()
        with condition:
            if "state" not in latest:
                condition.wait(timeout=0.1)
                continue
            state = latest["state"]
        current_q = state.q
        if teleop is not None:
            for event in teleop.poll():
                print(f"teleop {event.key!r}: {control.handle(event.key, current_q, lateral=True, backward=True)}")
        if now >= next_policy:
            hold_target = control.target(current_q, now)
            velocity = control.update_velocity(now)
            if control.policy_enabled:
                target = policy.infer(state, velocity)
                inference_count += 1
            elif hold_target is not None:
                target = hold_target
            next_policy += 1.0 / contract.control_frequency_hz
        if control.mode is ControlMode.IDLE:
            time.sleep(1.0 / contract.command_frequency_hz)
            continue
        for index in range(29):
            motor = command.motor_cmd[index]
            motor.mode = 1
            motor.q = float(target[index])
            motor.dq = 0.0
            motor.tau = 0.0
            if control.policy_enabled:
                motor.kp = float(policy.kp[index])
                motor.kd = float(policy.kd[index])
            else:
                motor.kp = float(initialization.kp[index])
                motor.kd = float(initialization.kd[index])
        publisher.Write(command)
        publish_count += 1
        time.sleep(1.0 / contract.command_frequency_hz)
    print(f"HOLOSOMA DDS POLICY DONE: inference={inference_count} lowcmd={publish_count}")
    if teleop is None and inference_count == 0:
        raise RuntimeError("controller received no rt/lowstate samples")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=bundled_profile_path("holosoma"))
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 runs until interrupted")
    parser.add_argument("--command-x", type=float, default=0.3)
    parser.add_argument("--teleop-endpoint")
    parser.add_argument("--init-duration", type=float, default=None, help="override profile value")
    args = parser.parse_args()
    if args.duration < 0 or (args.init_duration is not None and args.init_duration <= 0):
        raise ValueError("--duration must be non-negative and --init-duration must be positive")
    profile = load_control_profile(args.profile, "holosoma")
    contract = replace(profile.dds, domain_id=args.domain_id, interface=args.interface)
    contract.validate()
    run(args, contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Hiking-in-the-Wild model pipeline behind Longship's common ONNX runtime."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from longship.rl.runtime import OnnxEngine
from longship.rl.sim2sim.dds import G1_29DOF_JOINTS


POLICY_JOINTS = (
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "waist_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "waist_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "waist_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_hip_pitch_joint",
    "right_hip_pitch_joint", "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint", "left_knee_joint",
    "right_knee_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
)
POLICY_TO_DDS = np.asarray([G1_29DOF_JOINTS.index(name) for name in POLICY_JOINTS])
POLICY_SIGNS = np.asarray(
    (1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    dtype=np.float64,
)
HISTORY_LENGTH = 8
DEPTH_HISTORY_LENGTH = 37
DEPTH_HISTORY_INDICES = np.asarray((-36, -31, -26, -21, -16, -11, -6, -1))
HIKING_MODES = ("stand", "parkour")


def resolve_regex_values(config: object, default: float = 0.0) -> np.ndarray:
    values = np.full(29, default, dtype=np.float64)
    if isinstance(config, (int, float)):
        values.fill(float(config))
        return values
    if not isinstance(config, dict):
        raise TypeError(f"expected scalar or regex mapping, got {type(config).__name__}")
    for expression, value in config.items():
        for index, name in enumerate(POLICY_JOINTS):
            if re.search(expression, name):
                values[index] = float(value)
    return values


def resolve_pd(actuators: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    kp = np.zeros(29, dtype=np.float64)
    kd = np.zeros(29, dtype=np.float64)
    for actuator in actuators.values():
        selected = np.asarray(
            [any(re.search(expr, name) for expr in actuator["joint_names_expr"]) for name in POLICY_JOINTS]
        )
        local_kp = resolve_regex_values(actuator["stiffness"])
        local_kd = resolve_regex_values(actuator["damping"])
        kp[selected] = local_kp[selected]
        kd[selected] = local_kd[selected]
    if np.any(kp <= 0.0) or np.any(kd <= 0.0):
        raise ValueError("env.yaml did not resolve positive gains for every G1 joint")
    return kp, kd


def policy_to_dds(values: np.ndarray, *, signed: bool = True) -> np.ndarray:
    result = np.empty(29, dtype=np.float64)
    result[POLICY_TO_DDS] = np.asarray(values) * (POLICY_SIGNS if signed else 1.0)
    return result


def dds_to_policy(values: np.ndarray) -> np.ndarray:
    return np.asarray(values)[POLICY_TO_DDS] * POLICY_SIGNS


def projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion_wxyz))
    if norm <= 1.0e-8:
        return np.asarray((0.0, 0.0, -1.0))
    w, x, y, z = quaternion_wxyz / norm
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )
    return rotation.T @ np.asarray((0.0, 0.0, -1.0))


@dataclass(slots=True)
class HikingModeCommand:
    """Expose only the two agents registered by upstream g1_parkour.py."""

    index: int = 0

    @property
    def mode(self) -> str:
        return HIKING_MODES[self.index]

    def handle(self, key: str) -> str:
        if key in "np":
            self.index = (self.index + (1 if key == "n" else -1)) % len(HIKING_MODES)
        elif key in "12":
            self.index = int(key) - 1
        else:
            return f"Hiking mode key {key!r} unsupported"
        return f"Hiking agent={self.mode} ({self.index + 1})"


class HikingOnnxPolicy:
    """One released Hiking agent, with model-owned observations and control values."""

    def __init__(
        self,
        checkpoint: Path,
        provider: str = "auto",
        *,
        perceive_depth: bool = True,
    ) -> None:
        import cv2  # noqa: F401 -- validate preprocessing dependency before loading models
        import yaml

        with (checkpoint / "params/env.yaml").open(encoding="utf-8") as stream:
            config = yaml.unsafe_load(stream)
        self.depth_encoder = OnnxEngine(
            checkpoint / "exported/0-depth_encoder.onnx", provider=provider, intra_op_threads=4
        )
        self.actor = OnnxEngine(
            checkpoint / "exported/actor.onnx", provider=provider, intra_op_threads=4
        )
        self.default_q = resolve_regex_values(config["scene"]["robot"]["init_state"]["joint_pos"])
        self.action_scale = resolve_regex_values(config["actions"]["joint_pos"]["scale"], 1.0)
        self.kp, self.kd = resolve_pd(config["scene"]["robot"]["actuators"])
        self.perceive_depth = perceive_depth
        self.histories = [deque(maxlen=HISTORY_LENGTH) for _ in range(6)]
        self.depth_history: deque[np.ndarray] = deque(maxlen=DEPTH_HISTORY_LENGTH)
        self.last_action = np.zeros(29, dtype=np.float32)
        self.last_depth_input = np.zeros((18, 32), dtype=np.float32)

    @property
    def providers(self) -> tuple[str, ...]:
        return self.actor.providers

    def reset_history(self, last_action: np.ndarray | None = None) -> None:
        """Start a fresh observation window at an agent handoff.

        The next inference seeds every history slot from the current robot
        state.  ``last_action`` remains continuous with the command most
        recently sent by the outgoing agent.
        """
        for history in self.histories:
            history.clear()
        self.depth_history.clear()
        if last_action is not None:
            action = np.asarray(last_action, dtype=np.float32)
            if action.shape != self.last_action.shape:
                raise ValueError(
                    f"last_action shape {action.shape} does not match {self.last_action.shape}"
                )
            self.last_action = action.copy()

    @staticmethod
    def preprocess_depth(depth: np.ndarray) -> np.ndarray:
        import cv2

        resized = cv2.resize(depth, (64, 36), interpolation=cv2.INTER_NEAREST)
        cropped = resized[18:, 16:-16]
        invalid = (cropped < 0.2).astype(np.uint8)
        cropped = cv2.inpaint(cropped, invalid, 3, cv2.INPAINT_NS)
        cropped = cv2.GaussianBlur(cropped, (3, 3), 1.0, 1.0)
        return np.clip(cropped, 0.0, 2.5).astype(np.float32) / 2.5

    def infer(self, terms: tuple[np.ndarray, ...], raw_depth: np.ndarray) -> np.ndarray:
        # Upstream ParkourStandAgent deliberately replaces the complete depth
        # observation with zeros.  Feeding live depth into the stand actor is
        # not equivalent even though it shares the same encoder topology.
        depth = (
            self.preprocess_depth(raw_depth)
            if self.perceive_depth
            else np.zeros((18, 32), dtype=np.float32)
        )
        self.last_depth_input = depth.copy()
        if not self.depth_history:
            for history, term in zip(self.histories, terms, strict=True):
                history.extend(term.copy() for _ in range(HISTORY_LENGTH))
            self.depth_history.extend(depth.copy() for _ in range(DEPTH_HISTORY_LENGTH))
        else:
            for history, term in zip(self.histories, terms, strict=True):
                history.append(term.copy())
            self.depth_history.append(depth.copy())
        proprio = np.concatenate([np.concatenate(tuple(history)) for history in self.histories])
        depth_stack = np.stack(tuple(self.depth_history))[DEPTH_HISTORY_INDICES][None].astype(np.float32)
        depth_input = self.depth_encoder.input_names[0]
        latent = self.depth_encoder.infer({depth_input: depth_stack})[0]
        actor_input = np.concatenate((proprio[None].astype(np.float32), latent), axis=1)
        action = self.actor.infer({self.actor.input_names[0]: actor_input})[0].reshape(29).astype(np.float64)
        self.last_action = action.astype(np.float32)
        return self.default_q + self.action_scale * action

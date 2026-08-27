"""Perceptive Humanoid Parkour released student-policy inference pipeline."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from longship.rl.runtime import OnnxEngine
from longship.rl.sim2sim.dds import G1_29DOF_JOINTS


DEPTH_SOURCE_SIZE = (106, 60)
DEPTH_INPUT_SIZE = (87, 58)
DEPTH_LATENCY_STEPS = 7
COMMAND_DIM = 15


def _metadata_vector(metadata: Mapping[str, str], key: str, length: int) -> np.ndarray:
    raw = metadata.get(key)
    if raw is None:
        raise ValueError(f"PHP policy metadata is missing {key!r}")
    values = np.fromstring(raw, dtype=np.float64, sep=",")
    if values.size == 1:
        values = np.full(length, values.item(), dtype=np.float64)
    if values.shape != (length,) or not np.all(np.isfinite(values)):
        raise ValueError(f"PHP policy metadata {key!r} must contain {length} finite values")
    return values


def projected_gravity(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-8:
        return np.asarray((0.0, 0.0, -1.0))
    w, x, y, z = quaternion / norm
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )
    return rotation.T @ np.asarray((0.0, 0.0, -1.0))


def command_vector(command: str = "stop", *, high_speed: bool = True) -> np.ndarray:
    """Build the released policy's one-hot 15-D joystick placeholder."""
    low_speed = {"stop": 0, "forward": 1, "left_forward": 2, "left": 3,
                 "right_forward": 4, "right": 5}
    if command not in low_speed:
        raise ValueError(f"unsupported PHP command {command!r}")
    index = low_speed[command]
    if high_speed and index:
        index += 5
    result = np.zeros(COMMAND_DIM, dtype=np.float32)
    result[index] = 1.0
    return result


def _resize_bilinear(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """NumPy implementation of policyController.js's half-pixel resize."""
    source = np.asarray(image, dtype=np.float32)
    in_height, in_width = source.shape
    source_x = (np.arange(width, dtype=np.float64) + 0.5) * in_width / width - 0.5
    source_y = (np.arange(height, dtype=np.float64) + 0.5) * in_height / height - 0.5
    x0 = np.clip(np.floor(source_x).astype(np.int64), 0, in_width - 1)
    y0 = np.clip(np.floor(source_y).astype(np.int64), 0, in_height - 1)
    x1 = np.minimum(x0 + 1, in_width - 1)
    y1 = np.minimum(y0 + 1, in_height - 1)
    wx = (source_x - x0)[None]
    wy = (source_y - y0)[:, None]
    top = source[y0[:, None], x0[None]] * (1.0 - wx) + source[y0[:, None], x1[None]] * wx
    bottom = source[y1[:, None], x0[None]] * (1.0 - wx) + source[y1[:, None], x1[None]] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


class PhpOnnxPolicy:
    """Python port of the policyController.js path published by PHP."""

    def __init__(self, policy: Path, depth_backbone: Path, provider: str = "auto") -> None:
        self.policy = OnnxEngine(policy, provider=provider, intra_op_threads=4)
        self.depth_backbone = OnnxEngine(depth_backbone, provider=provider, intra_op_threads=4)
        metadata = self.policy.session.get_modelmeta().custom_metadata_map
        names = tuple(
            value.strip()
            for value in metadata.get("joint_names", "").split(",")
            if value.strip()
        )
        if len(names) != 29 or set(names) != set(G1_29DOF_JOINTS):
            raise ValueError("PHP policy joint_names do not match the Unitree G1 29-DoF contract")
        observations = tuple(
            value.strip()
            for value in metadata.get("observation_names", "").split(",")
            if value.strip()
        )
        expected = (
            "robot_anchor_projected_gravity", "base_ang_vel", "joint_pos",
            "joint_vel", "actions", "placeholder",
        )
        if observations != expected:
            raise ValueError(f"unsupported PHP observation contract: {observations}")
        self.joint_names = names
        self.policy_to_dds = np.asarray([G1_29DOF_JOINTS.index(name) for name in names])
        self.default_q = _metadata_vector(metadata, "default_joint_pos", 29)
        self.action_scale = _metadata_vector(metadata, "action_scale", 29)
        self.kp = _metadata_vector(metadata, "joint_stiffness", 29)
        self.kd = _metadata_vector(metadata, "joint_damping", 29)
        self.last_action = np.zeros(29, dtype=np.float32)
        self.depth_latents: deque[np.ndarray] = deque()
        if self.policy.input_names != ("obs", "time_step"):
            raise ValueError(f"unsupported PHP policy inputs: {self.policy.input_names}")
        if self.depth_backbone.input_names != ("depth_image",):
            raise ValueError(f"unsupported PHP depth inputs: {self.depth_backbone.input_names}")

    @property
    def providers(self) -> tuple[str, ...]:
        return self.policy.providers

    def dds_to_policy(self, values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (29,):
            raise ValueError("G1 joint vector must contain 29 values")
        return array[self.policy_to_dds]

    def policy_to_dds_vector(self, values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (29,):
            raise ValueError("PHP policy joint vector must contain 29 values")
        result = np.empty(29, dtype=np.float64)
        result[self.policy_to_dds] = array
        return result

    @staticmethod
    def preprocess_depth(depth: np.ndarray) -> np.ndarray:
        image = np.asarray(depth, dtype=np.float32)
        if image.ndim != 2:
            raise ValueError(f"PHP depth image must be 2-D, got {image.shape}")
        # Reconstruct the browser path: 106x60 render, crop 2 top/4 each side,
        # bilinear resize to 87x58, metric normalization, then OpenGL row flip.
        source = _resize_bilinear(image, *DEPTH_SOURCE_SIZE)
        cropped = source[2:, 4:-4]
        resized = _resize_bilinear(cropped, *DEPTH_INPUT_SIZE)
        normalized = (resized - 0.3) / (3.0 - 0.3) - 0.5
        return np.flipud(normalized).copy().astype(np.float32)

    def reset(self) -> None:
        self.last_action.fill(0.0)
        self.depth_latents.clear()

    def infer(
        self,
        *,
        torso_quaternion_wxyz: Sequence[float],
        base_angular_velocity: Sequence[float],
        joint_position: Sequence[float],
        joint_velocity: Sequence[float],
        command: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        q = self.dds_to_policy(joint_position)
        dq = self.dds_to_policy(joint_velocity)
        joystick = np.asarray(command, dtype=np.float32)
        if joystick.shape != (COMMAND_DIM,):
            raise ValueError(f"PHP command must have shape ({COMMAND_DIM},)")
        proprio = np.concatenate(
            (
                projected_gravity(torso_quaternion_wxyz),
                np.asarray(base_angular_velocity, dtype=np.float64),
                q - self.default_q,
                dq,
                self.last_action,
                joystick,
            )
        ).astype(np.float32)
        if proprio.shape != (108,):
            raise RuntimeError(
                f"PHP proprioception contract produced {proprio.shape}, expected (108,)"
            )
        depth_image = self.preprocess_depth(depth)[None]
        latent = self.depth_backbone.infer(
            {"depth_image": depth_image}, ("depth_latent",)
        )[0].reshape(32)
        self.depth_latents.append(latent.astype(np.float32))
        delayed = self.depth_latents[0]
        if len(self.depth_latents) > DEPTH_LATENCY_STEPS:
            delayed = self.depth_latents.popleft()
        observation = np.concatenate((proprio, delayed))[None].astype(np.float32)
        actions = self.policy.infer(
            {"obs": observation, "time_step": np.zeros((1, 1), dtype=np.float32)},
            ("actions",),
        )[0].reshape(29)
        self.last_action = actions.astype(np.float32)
        return self.default_q + self.action_scale * actions

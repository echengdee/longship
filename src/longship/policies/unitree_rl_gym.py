from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Sequence

from longship.artifacts import VerifiedArtifact, read_verified_artifact_bytes

from .base import PolicyError, PolicyGuardProfile


UNITREE_RL_GYM_G1_ACTION_SPACE = (
    "unitree.g1.12dof.rl-gym.raw-position-residual.v0"
)
UNITREE_RL_GYM_G1_OBSERVATION_DIM = 47
UNITREE_RL_GYM_G1_ACTION_DIM = 12
UNITREE_RL_GYM_G1_RESOURCE_SCOPE = ("g1_lower_body_motion",)
UNITREE_RL_GYM_G1_POLICY_ARTIFACT_ID = "motion.pt"
UNITREE_RL_GYM_G1_POLICY_SHA256 = (
    "cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d"
)
UNITREE_RL_GYM_G1_POLICY_SIZE_BYTES = 145745


def _finite_tuple(
    values: Sequence[float], *, length: int, field: str
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != length:
        raise ValueError(f"{field} must contain exactly {length} values")
    if any(isinstance(value, bool) for value in values):
        raise ValueError(f"{field} must not contain booleans")
    normalized = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{field} must contain only finite values")
    return normalized


@dataclass(frozen=True, slots=True)
class UnitreeRLGymG1Observation:
    """Scaled fields in the inspected external 12-DoF actor input order."""

    base_angular_velocity_scaled: tuple[float, ...]
    projected_gravity: tuple[float, ...]
    velocity_command_scaled: tuple[float, ...]
    joint_position_relative_scaled: tuple[float, ...]
    joint_velocity_scaled: tuple[float, ...]
    last_action: tuple[float, ...]
    gait_phase: tuple[float, ...]

    def __post_init__(self) -> None:
        dimensions = {
            "base_angular_velocity_scaled": 3,
            "projected_gravity": 3,
            "velocity_command_scaled": 3,
            "joint_position_relative_scaled": UNITREE_RL_GYM_G1_ACTION_DIM,
            "joint_velocity_scaled": UNITREE_RL_GYM_G1_ACTION_DIM,
            "last_action": UNITREE_RL_GYM_G1_ACTION_DIM,
            "gait_phase": 2,
        }
        for field, length in dimensions.items():
            object.__setattr__(
                self,
                field,
                _finite_tuple(getattr(self, field), length=length, field=field),
            )
        if not math.isclose(
            math.hypot(*self.gait_phase), 1.0, rel_tol=0.0, abs_tol=1e-5
        ):
            raise ValueError("gait_phase must lie on the unit circle")

    def vector(self) -> tuple[float, ...]:
        vector = (
            self.base_angular_velocity_scaled
            + self.projected_gravity
            + self.velocity_command_scaled
            + self.joint_position_relative_scaled
            + self.joint_velocity_scaled
            + self.last_action
            + self.gait_phase
        )
        if len(vector) != UNITREE_RL_GYM_G1_OBSERVATION_DIM:
            raise AssertionError("unexpected Unitree RL Gym observation dimension")
        return vector


class UnitreeRLGymTorchScriptRunner:
    """Load the pinned external policy from a revalidated byte snapshot."""

    def __init__(self, verified_model: VerifiedArtifact) -> None:
        if not isinstance(verified_model, VerifiedArtifact):
            raise TypeError("verified_model must come from ArtifactStore verification")
        if (
            verified_model.artifact_id != UNITREE_RL_GYM_G1_POLICY_ARTIFACT_ID
            or verified_model.sha256 != UNITREE_RL_GYM_G1_POLICY_SHA256
            or verified_model.size_bytes != UNITREE_RL_GYM_G1_POLICY_SIZE_BYTES
        ):
            raise PolicyError("verified artifact is not the pinned RL Gym policy")
        try:
            model_bytes = read_verified_artifact_bytes(
                verified_model,
                maximum_size_bytes=2 * 1024 * 1024,
            )
        except Exception as exc:
            raise PolicyError("verified RL Gym policy identity changed") from exc
        try:
            import torch
        except ImportError as exc:
            raise PolicyError(
                "install the 'mujoco-g1' optional dependencies to load TorchScript"
            ) from exc
        self._torch = torch
        try:
            self._policy = torch.jit.load(io.BytesIO(model_bytes), map_location="cpu")
            self._policy.eval()
        except Exception as exc:
            raise PolicyError("RL Gym TorchScript policy could not be loaded") from exc

        output = self.infer((0.0,) * UNITREE_RL_GYM_G1_OBSERVATION_DIM)
        if len(output) != UNITREE_RL_GYM_G1_ACTION_DIM:
            raise PolicyError("RL Gym policy tensor contract is incompatible")

    def infer(self, observation: Sequence[float]) -> tuple[float, ...]:
        vector = _finite_tuple(
            observation,
            length=UNITREE_RL_GYM_G1_OBSERVATION_DIM,
            field="observation",
        )
        tensor = self._torch.tensor([vector], dtype=self._torch.float32)
        try:
            with self._torch.no_grad():
                output = self._policy(tensor)
            flattened = output.detach().cpu().numpy().reshape(-1)
        except Exception as exc:
            raise PolicyError("RL Gym policy inference failed") from exc
        return _finite_tuple(
            flattened,
            length=UNITREE_RL_GYM_G1_ACTION_DIM,
            field="policy output",
        )


def unitree_rl_gym_g1_guard_profile(
    *,
    policy_step_ms: int,
    maximum_absolute_action: float,
) -> PolicyGuardProfile:
    if (
        isinstance(maximum_absolute_action, bool)
        or not isinstance(maximum_absolute_action, (int, float))
        or not math.isfinite(float(maximum_absolute_action))
        or maximum_absolute_action <= 0
    ):
        raise ValueError("maximum_absolute_action must be positive and finite")
    bound = float(maximum_absolute_action)
    return PolicyGuardProfile(
        action_space_id=UNITREE_RL_GYM_G1_ACTION_SPACE,
        action_dimension=UNITREE_RL_GYM_G1_ACTION_DIM,
        permitted_resource_scope=UNITREE_RL_GYM_G1_RESOURCE_SCOPE,
        max_action_horizon_ms=policy_step_ms,
        minimum_action_value=-bound,
        maximum_action_value=bound,
    )

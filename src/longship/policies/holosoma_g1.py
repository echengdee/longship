from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from .base import (
    PolicyActionFrame,
    PolicyCandidate,
    PolicyError,
    PolicyGuardProfile,
    PolicyRequest,
)
from .g1 import G1_29DOF_JOINT_NAMES
from .worker import SingleFlightInferenceWorker


HOLOSOMA_G1_ACTION_SPACE = "holosoma.g1.29dof.loco.raw-position-residual.v1"
HOLOSOMA_G1_OBSERVATION_DIM = 100
HOLOSOMA_G1_ACTION_DIM = 29
HOLOSOMA_G1_POLICY_STEP_MS = 20
HOLOSOMA_G1_ACTION_SCALE = 0.25
HOLOSOMA_G1_BASE_ANGULAR_VELOCITY_SCALE = 0.25
HOLOSOMA_G1_JOINT_VELOCITY_SCALE = 0.05
HOLOSOMA_G1_RESOURCE_SCOPE = ("whole_body_motion",)


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
class HolosomaG1LocomotionObservation:
    """Official G1 locomotion actor fields in the exported alphabetical order."""

    joint_names: tuple[str, ...]
    last_action: tuple[float, ...]
    base_angular_velocity: tuple[float, ...]
    command_angular_velocity: tuple[float, ...]
    command_linear_velocity: tuple[float, ...]
    cosine_phase: tuple[float, ...]
    joint_position_relative: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    projected_gravity: tuple[float, ...]
    sine_phase: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.joint_names != G1_29DOF_JOINT_NAMES:
            raise ValueError("joint_names must match the pinned G1 29-DoF order")
        dimensions = {
            "last_action": HOLOSOMA_G1_ACTION_DIM,
            "base_angular_velocity": 3,
            "command_angular_velocity": 1,
            "command_linear_velocity": 2,
            "cosine_phase": 2,
            "joint_position_relative": HOLOSOMA_G1_ACTION_DIM,
            "joint_velocity": HOLOSOMA_G1_ACTION_DIM,
            "projected_gravity": 3,
            "sine_phase": 2,
        }
        for field, length in dimensions.items():
            object.__setattr__(
                self,
                field,
                _finite_tuple(getattr(self, field), length=length, field=field),
            )
        for index in range(2):
            phase_norm = math.hypot(
                self.sine_phase[index], self.cosine_phase[index]
            )
            if not math.isclose(phase_norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
                raise ValueError(
                    "each Holosoma gait phase must lie on the unit circle"
                )

    def vector(self) -> tuple[float, ...]:
        vector = (
            self.last_action
            + tuple(
                value * HOLOSOMA_G1_BASE_ANGULAR_VELOCITY_SCALE
                for value in self.base_angular_velocity
            )
            + self.command_angular_velocity
            + self.command_linear_velocity
            + self.cosine_phase
            + self.joint_position_relative
            + tuple(
                value * HOLOSOMA_G1_JOINT_VELOCITY_SCALE
                for value in self.joint_velocity
            )
            + self.projected_gravity
            + self.sine_phase
        )
        if len(vector) != HOLOSOMA_G1_OBSERVATION_DIM:
            raise AssertionError("unexpected Holosoma G1 observation dimension")
        return vector


class HolosomaVectorPolicyRunner(Protocol):
    def infer(self, observation: tuple[float, ...]) -> Sequence[float]:
        ...


class HolosomaG1LocomotionBackend:
    """Side-effect-free seam for an externally verified Holosoma runner."""

    def __init__(
        self,
        runner: HolosomaVectorPolicyRunner,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._worker = SingleFlightInferenceWorker(
            runner.infer, thread_name="longship-holosoma-policy"
        )

    def close(self, *, wait: bool = False) -> None:
        self._worker.close(wait=wait)

    async def infer(self, request: PolicyRequest) -> PolicyCandidate:
        observation = request.payload.get("observation")
        if not isinstance(observation, HolosomaG1LocomotionObservation):
            raise PolicyError(
                "Holosoma request payload requires a validated 'observation'"
            )
        if not set(HOLOSOMA_G1_RESOURCE_SCOPE).issubset(request.resource_scope):
            raise PolicyError("Holosoma policy requires the whole_body_motion lease")

        output = await self._worker.run(observation.vector())
        try:
            values = _finite_tuple(
                output,
                length=HOLOSOMA_G1_ACTION_DIM,
                field="policy output",
            )
        except (TypeError, ValueError) as exc:
            raise PolicyError("Holosoma policy produced an invalid action") from exc
        generated = self._clock()
        return PolicyCandidate(
            call_id=request.call_id,
            model_binding_id=request.model_binding_id,
            lease_id=request.lease_id,
            lease_epoch=request.lease_epoch,
            observation_version=request.observation_version,
            generated_at_monotonic=generated,
            expires_at_monotonic=generated + HOLOSOMA_G1_POLICY_STEP_MS / 1000.0,
            action_space_id=HOLOSOMA_G1_ACTION_SPACE,
            resource_scope=HOLOSOMA_G1_RESOURCE_SCOPE,
            frames=(PolicyActionFrame(offset_ms=0, values=values),),
        )


def holosoma_g1_guard_profile(
    *, minimum_action_value: float, maximum_action_value: float
) -> PolicyGuardProfile:
    """Build a guard using bounds supplied by target qualification."""

    return PolicyGuardProfile(
        action_space_id=HOLOSOMA_G1_ACTION_SPACE,
        action_dimension=HOLOSOMA_G1_ACTION_DIM,
        permitted_resource_scope=HOLOSOMA_G1_RESOURCE_SCOPE,
        max_action_horizon_ms=HOLOSOMA_G1_POLICY_STEP_MS,
        minimum_action_value=minimum_action_value,
        maximum_action_value=maximum_action_value,
    )

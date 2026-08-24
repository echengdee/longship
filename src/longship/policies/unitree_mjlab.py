from __future__ import annotations

import hashlib
import math
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

from longship.artifacts import VerifiedArtifact

from .base import (
    PolicyActionFrame,
    PolicyCandidate,
    PolicyError,
    PolicyGuardProfile,
    PolicyRequest,
)
from .g1 import G1_29DOF_JOINT_NAMES
from .worker import SingleFlightInferenceWorker


UNITREE_G1_29DOF_ACTION_SPACE = (
    "unitree.g1.29dof.mjlab.velocity-v0.raw-joint-position-action"
)
UNITREE_G1_29DOF_OBSERVATION_DIM = 98
UNITREE_G1_29DOF_ACTION_DIM = 29
UNITREE_G1_POLICY_STEP_MS = 20
UNITREE_G1_GAIT_PERIOD_S = 0.6
UNITREE_G1_RESOURCE_SCOPE = ("whole_body_motion",)
UNITREE_G1_POLICY_ARTIFACT_ID = "policy.onnx"
UNITREE_G1_POLICY_SHA256 = (
    "2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28"
)
UNITREE_G1_POLICY_SIZE_BYTES = 878421


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
class UnitreeG1VelocityCommand:
    linear_x_m_s: float
    linear_y_m_s: float
    angular_z_rad_s: float

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool)
            for value in (
                self.linear_x_m_s,
                self.linear_y_m_s,
                self.angular_z_rad_s,
            )
        ):
            raise ValueError("velocity command must not contain booleans")
        values = (
            float(self.linear_x_m_s),
            float(self.linear_y_m_s),
            float(self.angular_z_rad_s),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("velocity command must contain only finite values")
        if not -0.5 <= values[0] <= 1.0:
            raise ValueError("linear_x_m_s is outside the official v0 profile")
        if not -0.5 <= values[1] <= 0.5:
            raise ValueError("linear_y_m_s is outside the official v0 profile")
        if not -1.0 <= values[2] <= 1.0:
            raise ValueError("angular_z_rad_s is outside the official v0 profile")

    def as_tuple(self) -> tuple[float, float, float]:
        return (
            float(self.linear_x_m_s),
            float(self.linear_y_m_s),
            float(self.angular_z_rad_s),
        )


@dataclass(frozen=True, slots=True)
class UnitreeG1VelocityObservation:
    """Raw fields in the official G1 velocity-v0 observation order.

    The ONNX graph contains its own observation normalizer. Callers must not
    apply a second hidden normalization pass.
    """

    joint_names: tuple[str, ...]
    base_angular_velocity: tuple[float, ...]
    projected_gravity: tuple[float, ...]
    command: UnitreeG1VelocityCommand
    gait_phase: tuple[float, ...]
    joint_position_relative: tuple[float, ...]
    joint_velocity_relative: tuple[float, ...]
    last_action: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.joint_names != G1_29DOF_JOINT_NAMES:
            raise ValueError("joint_names must match the pinned G1 29-DoF order")
        object.__setattr__(
            self,
            "base_angular_velocity",
            _finite_tuple(
                self.base_angular_velocity,
                length=3,
                field="base_angular_velocity",
            ),
        )
        object.__setattr__(
            self,
            "projected_gravity",
            _finite_tuple(
                self.projected_gravity, length=3, field="projected_gravity"
            ),
        )
        if not isinstance(self.command, UnitreeG1VelocityCommand):
            raise TypeError("command must be UnitreeG1VelocityCommand")
        object.__setattr__(
            self,
            "gait_phase",
            _finite_tuple(self.gait_phase, length=2, field="gait_phase"),
        )
        phase_norm = math.hypot(*self.gait_phase)
        if phase_norm > 1e-9 and not math.isclose(
            phase_norm, 1.0, rel_tol=0.0, abs_tol=1e-5
        ):
            raise ValueError("gait_phase must be zero or lie on the unit circle")
        for field in (
            "joint_position_relative",
            "joint_velocity_relative",
            "last_action",
        ):
            object.__setattr__(
                self,
                field,
                _finite_tuple(
                    getattr(self, field),
                    length=UNITREE_G1_29DOF_ACTION_DIM,
                    field=field,
                ),
            )

    def vector(self) -> tuple[float, ...]:
        vector = (
            self.base_angular_velocity
            + self.projected_gravity
            + self.command.as_tuple()
            + self.gait_phase
            + self.joint_position_relative
            + self.joint_velocity_relative
            + self.last_action
        )
        if len(vector) != UNITREE_G1_29DOF_OBSERVATION_DIM:
            raise AssertionError("unexpected Unitree G1 observation dimension")
        return vector


class VectorPolicyRunner(Protocol):
    def infer(self, observation: tuple[float, ...]) -> Sequence[float]:
        ...


class OnnxRuntimeVectorRunner:
    """Optional ONNX Runtime loader for a previously verified policy file.

    This class performs inference only. It neither downloads artifacts nor
    translates its output into target commands.
    """

    def __init__(self, verified_model: VerifiedArtifact) -> None:
        if not isinstance(verified_model, VerifiedArtifact):
            raise TypeError("verified_model must come from ArtifactStore verification")
        model_path = Path(verified_model.path)
        if (
            verified_model.artifact_id != UNITREE_G1_POLICY_ARTIFACT_ID
            or verified_model.sha256 != UNITREE_G1_POLICY_SHA256
            or verified_model.size_bytes != UNITREE_G1_POLICY_SIZE_BYTES
        ):
            raise PolicyError("verified artifact is not the pinned Unitree policy")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(model_path, flags)
        except OSError as exc:
            raise PolicyError(
                "verified Unitree policy cannot be opened safely"
            ) from exc
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_dev != verified_model.device
                or before.st_ino != verified_model.inode
                or before.st_size != UNITREE_G1_POLICY_SIZE_BYTES
            ):
                raise PolicyError("verified Unitree policy file identity changed")
            digest = hashlib.sha256()
            chunks: list[bytes] = []
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                    chunks.append(chunk)
            after = os.fstat(descriptor)
            if (
                after.st_dev != before.st_dev
                or after.st_ino != before.st_ino
                or after.st_size != before.st_size
                or after.st_mtime_ns != before.st_mtime_ns
            ):
                raise PolicyError("Unitree policy changed while being loaded")
            if digest.hexdigest() != UNITREE_G1_POLICY_SHA256:
                raise PolicyError("Unitree policy digest changed after verification")
            model_bytes = b"".join(chunks)
        finally:
            os.close(descriptor)
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise PolicyError(
                "install the 'unitree-mjlab' optional dependencies to load ONNX"
            ) from exc

        self._np = np
        self._session = ort.InferenceSession(
            model_bytes, providers=["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise PolicyError("Unitree policy must expose exactly one input and output")
        if (
            inputs[0].name != "obs"
            or inputs[0].type != "tensor(float)"
            or list(inputs[0].shape) != [1, UNITREE_G1_29DOF_OBSERVATION_DIM]
        ):
            raise PolicyError("Unitree policy input contract is not obs float32[1,98]")
        if (
            outputs[0].name != "actions"
            or outputs[0].type != "tensor(float)"
            or list(outputs[0].shape) != [1, UNITREE_G1_29DOF_ACTION_DIM]
        ):
            raise PolicyError(
                "Unitree policy output contract is not actions float32[1,29]"
            )
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

    def infer(self, observation: tuple[float, ...]) -> tuple[float, ...]:
        vector = _finite_tuple(
            observation,
            length=UNITREE_G1_29DOF_OBSERVATION_DIM,
            field="observation",
        )
        input_array = self._np.asarray([vector], dtype=self._np.float32)
        output = self._session.run(
            [self._output_name], {self._input_name: input_array}
        )[0]
        flattened = tuple(float(value) for value in output.reshape(-1))
        return _finite_tuple(
            flattened,
            length=UNITREE_G1_29DOF_ACTION_DIM,
            field="policy output",
        )


class UnitreeG1VelocityBackend:
    """Simulation-only backend for the official G1 velocity-v0 policy seam."""

    def __init__(
        self,
        runner: VectorPolicyRunner,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._worker = SingleFlightInferenceWorker(
            runner.infer, thread_name="longship-unitree-policy"
        )

    def close(self, *, wait: bool = False) -> None:
        self._worker.close(wait=wait)

    async def infer(self, request: PolicyRequest) -> PolicyCandidate:
        observation = request.payload.get("observation")
        if not isinstance(observation, UnitreeG1VelocityObservation):
            raise PolicyError(
                "Unitree request payload requires a validated 'observation'"
            )
        if not set(UNITREE_G1_RESOURCE_SCOPE).issubset(request.resource_scope):
            raise PolicyError("Unitree policy requires the whole_body_motion lease")

        output = await self._worker.run(observation.vector())
        try:
            values = _finite_tuple(
                output,
                length=UNITREE_G1_29DOF_ACTION_DIM,
                field="policy output",
            )
        except (TypeError, ValueError) as exc:
            raise PolicyError("Unitree policy produced an invalid action") from exc
        generated = self._clock()
        return PolicyCandidate(
            call_id=request.call_id,
            model_binding_id=request.model_binding_id,
            lease_id=request.lease_id,
            lease_epoch=request.lease_epoch,
            observation_version=request.observation_version,
            generated_at_monotonic=generated,
            expires_at_monotonic=generated + UNITREE_G1_POLICY_STEP_MS / 1000.0,
            action_space_id=UNITREE_G1_29DOF_ACTION_SPACE,
            resource_scope=UNITREE_G1_RESOURCE_SCOPE,
            frames=(PolicyActionFrame(offset_ms=0, values=values),),
        )


def unitree_g1_velocity_guard_profile(
    *, minimum_action_value: float, maximum_action_value: float
) -> PolicyGuardProfile:
    """Build a target-qualified profile with explicit raw-action bounds.

    The upstream output layer is linear and its deployment file declares no
    clip. Longship therefore refuses to invent implicit bounds: a simulator or
    target qualification profile must provide them explicitly.
    """

    return PolicyGuardProfile(
        action_space_id=UNITREE_G1_29DOF_ACTION_SPACE,
        action_dimension=UNITREE_G1_29DOF_ACTION_DIM,
        permitted_resource_scope=UNITREE_G1_RESOURCE_SCOPE,
        max_action_horizon_ms=UNITREE_G1_POLICY_STEP_MS,
        minimum_action_value=minimum_action_value,
        maximum_action_value=maximum_action_value,
    )


def unitree_g1_gait_phase(
    elapsed_s: float, command: UnitreeG1VelocityCommand
) -> tuple[float, float]:
    """Build the official 0.6 s sin/cos phase, zeroed near standstill."""

    if not isinstance(command, UnitreeG1VelocityCommand):
        raise TypeError("command must be UnitreeG1VelocityCommand")
    if isinstance(elapsed_s, bool) or not isinstance(elapsed_s, (int, float)):
        raise TypeError("elapsed_s must be a finite number")
    elapsed = float(elapsed_s)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed_s must be finite and non-negative")
    if math.sqrt(sum(value * value for value in command.as_tuple())) < 0.1:
        return (0.0, 0.0)
    cycle = (elapsed % UNITREE_G1_GAIT_PERIOD_S) / UNITREE_G1_GAIT_PERIOD_S
    angle = 2.0 * math.pi * cycle
    return (math.sin(angle), math.cos(angle))

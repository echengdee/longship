from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from longship.rl.sim2sim.dds import DdsContract, G1_29DOF_JOINTS


PROFILE_DIR = Path(__file__).with_name("profiles")
PARAMETER_SOURCES = frozenset(
    ("profile", "onnx_metadata", "checkpoint", "native_controller", "python_pipeline")
)


def bundled_profile_path(backend: str) -> Path:
    return PROFILE_DIR / f"{backend}.yaml"


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _vector(values: object, name: str, *, positive: bool = False) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(G1_29DOF_JOINTS),):
        raise ValueError(f"{name} must contain {len(G1_29DOF_JOINTS)} values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} must contain only positive values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ParameterSet:
    source: str
    default_q: np.ndarray | None = None
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None
    artifact: str | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], name: str) -> "ParameterSet":
        values = dict(values)
        source = str(values.pop("source", "profile"))
        if source not in PARAMETER_SOURCES:
            raise ValueError(f"{name}.source must be one of {sorted(PARAMETER_SOURCES)}")
        artifact = values.pop("artifact", None)
        default_q = values.pop("default_q", None)
        kp = values.pop("kp", None)
        kd = values.pop("kd", None)
        if values:
            raise ValueError(f"unknown {name} fields: {sorted(values)}")
        if source == "profile" and any(item is None for item in (default_q, kp, kd)):
            raise ValueError(f"{name} with source=profile requires default_q, kp and kd")
        if source != "profile" and not artifact:
            raise ValueError(f"{name} with source={source} requires artifact")
        return cls(
            source=source,
            default_q=None if default_q is None else _vector(default_q, f"{name}.default_q"),
            kp=None if kp is None else _vector(kp, f"{name}.kp", positive=True),
            kd=None if kd is None else _vector(kd, f"{name}.kd", positive=True),
            artifact=None if artifact is None else str(artifact),
        )


@dataclass(frozen=True, slots=True)
class SimulatorSettings:
    gantry_enabled: bool
    gantry_length_m: float
    depth_enabled: bool
    scene: str | None
    foot_collision: str
    gantry_mode: str
    reset_q: tuple[float, ...] | None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SimulatorSettings":
        values = dict(values)
        result = cls(
            gantry_enabled=bool(values.pop("gantry_enabled", True)),
            gantry_length_m=float(values.pop("gantry_length_m", 1.0)),
            depth_enabled=bool(values.pop("depth_enabled", False)),
            scene=None if (scene := values.pop("scene", None)) is None else str(scene),
            foot_collision=str(values.pop("foot_collision", "native")),
            gantry_mode=str(values.pop("gantry_mode", "rope")),
            reset_q=(
                None
                if (reset_q := values.pop("reset_q", None)) is None
                else tuple(float(value) for value in reset_q)
            ),
        )
        if values:
            raise ValueError(f"unknown simulator fields: {sorted(values)}")
        if result.gantry_length_m < 0.0:
            raise ValueError("simulator.gantry_length_m must be non-negative")
        if result.foot_collision not in ("native", "hiking_training_v1"):
            raise ValueError("simulator.foot_collision must be native or hiking_training_v1")
        if result.gantry_mode not in ("rope", "hiking_spotter_v1"):
            raise ValueError("simulator.gantry_mode must be rope or hiking_spotter_v1")
        if result.reset_q is not None and len(result.reset_q) != len(G1_29DOF_JOINTS):
            raise ValueError(f"simulator.reset_q must contain {len(G1_29DOF_JOINTS)} values")
        return result


@dataclass(frozen=True, slots=True)
class ControlProfile:
    backend: str
    robot: str
    dds: DdsContract
    simulator: SimulatorSettings
    initialization_duration_s: float
    initialization: ParameterSet
    policy: ParameterSet
    policy_options: Mapping[str, Any]
    path: Path

    def resolve_artifact(self, root: Path, parameters: ParameterSet) -> Path | None:
        if parameters.artifact is None:
            return None
        path = Path(parameters.artifact)
        return path if path.is_absolute() else root / path


def load_control_profile(path: str | Path, expected_backend: str | None = None) -> ControlProfile:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    values = _mapping(raw, "control profile")
    if values.pop("schema_version", None) != "longship.sim2sim-control.v1":
        raise ValueError("unsupported Sim2Sim control profile schema_version")
    backend = str(values.pop("backend"))
    if expected_backend is not None and backend != expected_backend:
        raise ValueError(f"profile backend {backend!r} does not match {expected_backend!r}")
    robot = str(values.pop("robot"))
    if robot != "unitree_g1_29dof":
        raise ValueError(f"unsupported Sim2Sim robot {robot!r}")
    dds_values = _mapping(values.pop("dds", {}), "dds")
    simulator = SimulatorSettings.from_mapping(_mapping(values.pop("simulator", {}), "simulator"))
    initialization_values = _mapping(values.pop("initialization"), "initialization")
    duration = float(initialization_values.pop("duration_s"))
    if duration <= 0.0:
        raise ValueError("initialization.duration_s must be positive")
    initialization = ParameterSet.from_mapping(initialization_values, "initialization")
    policy_values = _mapping(values.pop("policy"), "policy")
    policy_parameters = ParameterSet.from_mapping(
        _mapping(policy_values.pop("parameters"), "policy.parameters"), "policy.parameters"
    )
    if values:
        raise ValueError(f"unknown control profile fields: {sorted(values)}")
    dds = DdsContract.from_mapping(dds_values)
    if simulator.depth_enabled and dds.depth_topic is None:
        raise ValueError("depth-enabled profile must define dds.depth_topic")
    return ControlProfile(
        backend=backend,
        robot=robot,
        dds=dds,
        simulator=simulator,
        initialization_duration_s=duration,
        initialization=initialization,
        policy=policy_parameters,
        policy_options=policy_values,
        path=path,
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from longship.rl.sim2sim.profile import load_control_profile


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


@dataclass(frozen=True, slots=True)
class DdsSettings:
    interface: str
    domain_id: int


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    provider: str
    input_timeout_s: float
    startup_timeout_s: float
    teleop_endpoint: str


@dataclass(frozen=True, slots=True)
class VisualizationSettings:
    enabled: bool
    bind_host: str
    port: int
    frame_endpoint: str
    fps: float


@dataclass(frozen=True, slots=True)
class SensorProfile:
    name: str
    type: str
    required: bool
    config: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DeploymentProfile:
    name: str
    backend: str
    target: str
    control_profile: Path
    dds: DdsSettings
    runtime: RuntimeSettings
    sensors: tuple[SensorProfile, ...]
    visualization: VisualizationSettings | None

    @classmethod
    def load(cls, path: Path, root: Path) -> "DeploymentProfile":
        values = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "profile")
        if values.pop("schema_version", None) != "longship.rl.deploy.v1":
            raise ValueError("unsupported deployment profile schema")
        dds_values = _mapping(values.pop("dds"), "dds")
        runtime_values = _mapping(values.pop("runtime", {}), "runtime")
        visualization_value = values.pop("visualization", None)
        sensor_values = values.pop("sensors", [])
        if not isinstance(sensor_values, list):
            raise ValueError("sensors must be a list")
        control_path = Path(str(values.pop("control_profile")))
        profile = cls(
            name=str(values.pop("name")),
            backend=str(values.pop("backend")),
            target=str(values.pop("target")),
            control_profile=control_path if control_path.is_absolute() else root / control_path,
            dds=DdsSettings(
                interface=str(dds_values.pop("interface", "")),
                domain_id=int(dds_values.pop("domain_id", 0)),
            ),
            runtime=RuntimeSettings(
                provider=str(runtime_values.pop("provider", "cpu")),
                input_timeout_s=float(runtime_values.pop("input_timeout_s", 0.25)),
                startup_timeout_s=float(runtime_values.pop("startup_timeout_s", 15.0)),
                teleop_endpoint=str(
                    runtime_values.pop("teleop_endpoint", "tcp://127.0.0.1:5560")
                ),
            ),
            sensors=tuple(_load_sensor(item, index) for index, item in enumerate(sensor_values)),
            visualization=(
                None
                if visualization_value is None
                else _load_visualization(visualization_value)
            ),
        )
        if values or dds_values or runtime_values:
            unknown = sorted((*values, *dds_values, *runtime_values))
            raise ValueError(f"unknown deployment profile fields: {unknown}")
        profile.validate()
        return profile

    def validate(self) -> None:
        if not self.name or not self.backend or not self.target:
            raise ValueError("name, backend, and target are required")
        if not self.control_profile.is_file():
            raise ValueError(f"control profile does not exist: {self.control_profile}")
        load_control_profile(self.control_profile, self.backend)
        if not self.dds.interface or self.dds.interface in ("lo", "auto"):
            raise ValueError("a concrete non-loopback robot interface is required")
        if not 0 <= self.dds.domain_id <= 232:
            raise ValueError("domain_id must be 0..232")
        if self.runtime.provider not in ("auto", "cpu", "cuda"):
            raise ValueError("runtime.provider must be auto, cpu, or cuda")
        if self.runtime.input_timeout_s <= 0 or self.runtime.startup_timeout_s <= 0:
            raise ValueError("deployment timeouts must be positive")
        if self.visualization is not None:
            visual = self.visualization
            if not visual.bind_host.strip() or not 1 <= visual.port <= 65535:
                raise ValueError("visualization bind_host or port is invalid")
            if not visual.frame_endpoint.startswith("tcp://") or visual.fps <= 0:
                raise ValueError("visualization frame_endpoint or fps is invalid")
        names = [sensor.name for sensor in self.sensors]
        if len(names) != len(set(names)):
            raise ValueError("sensor names must be unique")


def _load_sensor(value: object, index: int) -> SensorProfile:
    item = _mapping(value, f"sensors[{index}]")
    name = str(item.pop("name", ""))
    sensor_type = str(item.pop("type", ""))
    required = bool(item.pop("required", True))
    config = _mapping(item.pop("config", {}), f"sensors[{index}].config")
    if item:
        raise ValueError(f"unknown sensor fields: {sorted(item)}")
    if not name or not sensor_type:
        raise ValueError("each sensor requires name and type")
    return SensorProfile(name=name, type=sensor_type, required=required, config=config)


def _load_visualization(value: object) -> VisualizationSettings:
    item = _mapping(value, "visualization")
    result = VisualizationSettings(
        enabled=bool(item.pop("enabled", True)),
        bind_host=str(item.pop("bind_host", "127.0.0.1")),
        port=int(item.pop("port", 8080)),
        frame_endpoint=str(item.pop("frame_endpoint", "tcp://127.0.0.1:5570")),
        fps=float(item.pop("fps", 10.0)),
    )
    if item:
        raise ValueError(f"unknown visualization fields: {sorted(item)}")
    return result


def bundled_deployment_profile(name: str) -> Path:
    return Path(__file__).with_name("profiles") / f"{name}.yaml"

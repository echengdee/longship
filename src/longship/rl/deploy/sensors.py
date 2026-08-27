from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from longship.rl.deploy.profile import DeploymentProfile, SensorProfile
from longship.rl.runtime.process import ProcessSpec


@dataclass(frozen=True, slots=True)
class RealSensor:
    process: ProcessSpec
    ready_marker: str
    required_modules: tuple[str, ...]


def _integer(config: Mapping[str, Any], name: str, default: int) -> int:
    value = int(config.get(name, default))
    if not 1 <= value <= 4096:
        raise ValueError(f"sensor {name} must be between 1 and 4096")
    return value


def _realsense(
    root: Path, python: str, profile: DeploymentProfile, sensor: SensorProfile
) -> RealSensor:
    config = sensor.config
    serial = str(config.get("serial", "")).strip()
    if not serial:
        raise ValueError(f"sensor {sensor.name!r} requires a serial")
    fps = int(config.get("fps", 30))
    warmup = int(config.get("warmup_frames", 15))
    if not 1 <= fps <= 90 or warmup < 0:
        raise ValueError("RealSense fps or warmup_frames is invalid")
    known = {
        "serial", "expected_model", "raw_width", "raw_height", "output_width",
        "output_height", "fps", "warmup_frames",
    }
    unknown = set(config) - known
    if unknown:
        raise ValueError(f"unknown RealSense config fields: {sorted(unknown)}")
    argv = (
        python, "-m", "longship.rl.deploy.realsense_depth_dds",
        "--interface", profile.dds.interface,
        "--domain-id", str(profile.dds.domain_id),
        "--serial", serial,
        "--expected-model", str(config.get("expected_model", "D435I")),
        "--raw-width", str(_integer(config, "raw_width", 848)),
        "--raw-height", str(_integer(config, "raw_height", 480)),
        "--output-width", str(_integer(config, "output_width", 480)),
        "--output-height", str(_integer(config, "output_height", 270)),
        "--fps", str(fps),
        "--warmup-frames", str(warmup),
    )
    visual = profile.visualization
    if visual is not None and visual.enabled:
        argv += (
            "--debug-frame-endpoint", visual.frame_endpoint,
            "--debug-frame-fps", str(visual.fps),
        )
    return RealSensor(
        ProcessSpec(sensor.name, root, argv, environment=(("PYTHONPATH", str(root / "src")),)),
        "REALSENSE DDS READY",
        ("pyrealsense2",),
    )


_BUILDERS: Mapping[
    str, Callable[[Path, str, DeploymentProfile, SensorProfile], RealSensor]
] = {"realsense_depth_dds": _realsense}


def build_sensors(root: Path, python: str, profile: DeploymentProfile) -> tuple[RealSensor, ...]:
    sensors: list[RealSensor] = []
    for sensor in profile.sensors:
        try:
            builder = _BUILDERS[sensor.type]
        except KeyError as exc:
            raise ValueError(f"unknown physical sensor type {sensor.type!r}") from exc
        sensors.append(builder(root, python, profile, sensor))
    return tuple(sensors)

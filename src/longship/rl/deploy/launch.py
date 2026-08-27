from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from longship.rl.deploy.backends import RealBackend, build_real_backend
from longship.rl.deploy.profile import DeploymentProfile
from longship.rl.deploy.sensors import RealSensor, build_sensors
from longship.rl.deploy.targets.unitree_g1 import release_motion as unitree_g1_release
from longship.rl.runtime.process import ProcessSpec


@dataclass(frozen=True, slots=True)
class DeploymentLaunch:
    profile: DeploymentProfile
    backend: RealBackend
    sensors: tuple[RealSensor, ...]
    release_motion: ProcessSpec
    monitor: ProcessSpec | None


_TARGETS: Mapping[str, Callable[[Path, str, DeploymentProfile], ProcessSpec]] = {
    "unitree_g1_29dof": unitree_g1_release,
}


def build_deployment_launch(
    root: Path, python: str, profile: DeploymentProfile
) -> DeploymentLaunch:
    try:
        target_builder = _TARGETS[profile.target]
    except KeyError as exc:
        raise ValueError(f"unknown physical target {profile.target!r}") from exc
    monitor = None
    visual = profile.visualization
    if visual is not None and visual.enabled:
        monitor = ProcessSpec(
            "web_monitor",
            root,
            (
                python, "-m", "longship.rl.deploy.web_monitor",
                "--bind-host", visual.bind_host,
                "--port", str(visual.port),
                "--frame-endpoint", visual.frame_endpoint,
            ),
            environment=(("PYTHONPATH", str(root / "src")),),
        )
    return DeploymentLaunch(
        profile=profile,
        backend=build_real_backend(root, python, profile),
        sensors=build_sensors(root, python, profile),
        release_motion=target_builder(root, python, profile),
        monitor=monitor,
    )

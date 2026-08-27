from __future__ import annotations

from pathlib import Path

from longship.rl.deploy.profile import DeploymentProfile
from longship.rl.runtime.process import ProcessSpec


def release_motion(root: Path, python: str, profile: DeploymentProfile) -> ProcessSpec:
    return ProcessSpec(
        "release_motion",
        root,
        (
            python, "-m", "longship.rl.deploy.unitree_motion",
            "--interface", profile.dds.interface,
            "--domain-id", str(profile.dds.domain_id),
            "--release",
        ),
        environment=(("PYTHONPATH", str(root / "src")),),
    )

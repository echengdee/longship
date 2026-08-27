from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from longship.rl.deploy.profile import DeploymentProfile
from longship.rl.runtime.process import ProcessSpec


@dataclass(frozen=True, slots=True)
class RealBackend:
    controller: ProcessSpec
    teleop: ProcessSpec
    ready_marker: str
    required_modules: tuple[str, ...]
    competing_process_patterns: tuple[str, ...]
    operator_hint: str


def _pythonpath(root: Path) -> str:
    return ":".join(
        (
            str(root / "src"),
            str(root / "third_party/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python"),
        )
    )


def _instinctlab(root: Path, python: str, profile: DeploymentProfile) -> RealBackend:
    common = (
        "--interface", profile.dds.interface,
        "--domain-id", str(profile.dds.domain_id),
    )
    environment = (("PYTHONPATH", _pythonpath(root)),)
    debug = ()
    visual = profile.visualization
    if visual is not None and visual.enabled:
        debug = (
            "--debug-frame-endpoint", visual.frame_endpoint,
            "--debug-frame-fps", str(visual.fps),
        )
    return RealBackend(
        controller=ProcessSpec(
            "controller",
            root,
            (
                python,
                str(root / "src/longship/rl/sim2sim/adapters/instinctlab_dds.py"),
                "--root", str(root),
                "--profile", str(profile.control_profile),
                *common,
                "--provider", profile.runtime.provider,
                "--teleop-endpoint", profile.runtime.teleop_endpoint,
                "--clock", "wall",
                "--real-robot",
                "--input-timeout", str(profile.runtime.input_timeout_s),
                *debug,
            ),
            environment=environment,
        ),
        teleop=ProcessSpec(
            "keyboard",
            root,
            (
                python, "-m", "longship.rl.sim2sim.teleop", profile.backend,
                "--endpoint", profile.runtime.teleop_endpoint,
            ),
            environment=(("PYTHONPATH", str(root / "src")),),
        ),
        ready_marker="POLICY DDS READY",
        required_modules=("cyclonedds", "cv2", "onnxruntime", "zmq"),
        competing_process_patterns=(
            "unitree_split_controller", "g1_deploy_onnx_ref", "run_policy.py",
            "instinctlab_dds.py",
        ),
        operator_hint="Use: i -> wait -> ] -> 2 -> w; 1 returns to stand.",
    )


_BUILDERS: Mapping[str, Callable[[Path, str, DeploymentProfile], RealBackend]] = {
    "instinctlab": _instinctlab,
}


def build_real_backend(root: Path, python: str, profile: DeploymentProfile) -> RealBackend:
    try:
        builder = _BUILDERS[profile.backend]
    except KeyError as exc:
        raise ValueError(f"backend {profile.backend!r} has no physical deployment adapter") from exc
    return builder(root, python, profile)

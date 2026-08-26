from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Mapping

from longship.rl.sim2sim.dds import DdsContract
from longship.rl.sim2sim.profile import ControlProfile, bundled_profile_path, load_control_profile
from longship.rl.sim2sim.teleop import DEFAULT_ENDPOINT


@dataclass(frozen=True, slots=True)
class ProcessSpec:
    name: str
    cwd: Path
    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...] = ()
    stdin: str | None = None

    def shell_command(self) -> str:
        prefix = tuple(f"{key}={value}" for key, value in self.environment)
        command = shlex.join(("env",) + prefix + self.argv) if prefix else shlex.join(self.argv)
        return f"cd {shlex.quote(str(self.cwd))} && {command}"


@dataclass(frozen=True, slots=True)
class BackendLaunch:
    backend: str
    contract: DdsContract
    simulator: ProcessSpec
    controller: ProcessSpec
    teleop: ProcessSpec
    notes: tuple[str, ...] = ()


def _simulator(root: Path, python: str, profile: ControlProfile) -> ProcessSpec:
    contract = profile.dds
    argv = (
        python,
        "-m",
        "longship.rl.sim2sim.simulator",
        "--root",
        str(root),
        "--interface",
        contract.interface,
        "--domain-id",
        str(contract.domain_id),
        "--state-frequency-hz",
        str(contract.state_frequency_hz),
        "--control-frequency-hz",
        str(contract.control_frequency_hz),
        "--command-frequency-hz",
        str(contract.command_frequency_hz),
        "--viewer",
    )
    if profile.simulator.gantry_enabled:
        argv += ("--gantry",)
    argv += ("--gantry-length", str(profile.simulator.gantry_length_m))
    if profile.simulator.scene is not None:
        scene = Path(profile.simulator.scene)
        argv += ("--scene", str(scene if scene.is_absolute() else root / scene))
    argv += ("--foot-collision", profile.simulator.foot_collision)
    argv += ("--gantry-mode", profile.simulator.gantry_mode)
    if profile.simulator.reset_q is not None:
        argv += ("--reset-q",) + tuple(str(value) for value in profile.simulator.reset_q)
    if contract.depth_topic:
        argv += ("--depth",)
    pythonpath = (
        f"{root / 'src'}:"
        f"{root / 'third_party/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python'}"
    )
    return ProcessSpec("simulator", root, argv, environment=(("PYTHONPATH", pythonpath),))


def _teleop(root: Path, python: str, backend: str) -> ProcessSpec:
    return ProcessSpec(
        "keyboard",
        root,
        (
            python,
            "-m",
            "longship.rl.sim2sim.teleop",
            backend,
            "--endpoint",
            DEFAULT_ENDPOINT,
        ),
        environment=(("PYTHONPATH", str(root / "src")),),
    )


def _holosoma(root: Path, python: str) -> BackendLaunch:
    source = root / "third_party/holosoma"
    model = source / "src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx"
    adapter = root / "src/longship/rl/sim2sim/adapters/holosoma_dds.py"
    profile_path = bundled_profile_path("holosoma")
    profile = load_control_profile(profile_path, "holosoma")
    contract = profile.dds
    pythonpath = (
        f"{root / 'src'}:"
        f"{root / 'third_party/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python'}"
    )
    return BackendLaunch(
        backend="holosoma",
        contract=contract,
        simulator=_simulator(root, python, profile),
        controller=ProcessSpec(
            "controller",
            root,
            (
                python,
                str(adapter),
                "--model",
                str(model),
                "--profile",
                str(profile_path),
                "--interface",
                contract.interface,
                "--domain-id",
                str(contract.domain_id),
                "--teleop-endpoint",
                DEFAULT_ENDPOINT,
            ),
            environment=(("PYTHONPATH", pythonpath),),
        ),
        teleop=_teleop(root, python, "holosoma"),
        notes=("HoloSoma inference runs through the Longship Unitree SDK2 policy adapter.",),
    )


def _sonic(root: Path, python: str) -> BackendLaunch:
    profile_path = bundled_profile_path("sonic")
    profile = load_control_profile(profile_path, "sonic")
    contract = profile.dds
    pythonpath = (
        f"{root / 'src'}:"
        f"{root / 'third_party/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python'}"
    )
    def option_path(name: str) -> str:
        return str(root / str(profile.policy_options[name]))
    return BackendLaunch(
        backend="sonic",
        contract=contract,
        simulator=_simulator(root, python, profile),
        controller=ProcessSpec(
            "controller",
            root,
            (
                python,
                str(root / "src/longship/rl/sim2sim/adapters/sonic_onnx.py"),
                "--root",
                str(root),
                "--profile",
                str(profile_path),
                "--decoder",
                option_path("decoder"),
                "--encoder",
                option_path("encoder"),
                "--planner",
                option_path("planner"),
                "--provider",
                str(profile.policy_options.get("provider", "auto")),
                "--interface",
                contract.interface,
                "--domain-id",
                str(contract.domain_id),
                "--init-duration",
                str(profile.initialization_duration_s),
                "--teleop-endpoint",
                DEFAULT_ENDPOINT,
            ),
            environment=(("PYTHONPATH", pythonpath),),
        ),
        teleop=_teleop(root, python, "sonic"),
        notes=("SONIC planner, encoder and decoder execute in Python ONNX Runtime.",),
    )


def _instinctlab(root: Path, python: str) -> BackendLaunch:
    source = root / "third_party/InstinctLab"
    adapter = root / "src/longship/rl/sim2sim/adapters/instinctlab_dds.py"
    profile_path = bundled_profile_path("instinctlab")
    profile = load_control_profile(profile_path, "instinctlab")
    contract = profile.dds
    common = (
        python,
        str(adapter),
        "--root",
        str(root),
        "--profile",
        str(profile_path),
        "--interface",
        contract.interface,
        "--domain-id",
        str(contract.domain_id),
    )
    pythonpath = (
        f"{root / 'src'}:"
        f"{root / 'third_party/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python'}"
    )
    return BackendLaunch(
        backend="instinctlab",
        contract=contract,
        simulator=_simulator(root, python, profile),
        controller=ProcessSpec(
            "controller",
            source,
            common
            + (
                "--provider",
                str(profile.policy_options.get("provider", "auto")),
                "--teleop-endpoint",
                DEFAULT_ENDPOINT,
            ),
            environment=(("PYTHONPATH", pythonpath),),
        ),
        teleop=_teleop(root, python, "instinctlab"),
        notes=(
            "Hiking stand/depth-parkour agents execute in the shared Python ONNX Runtime.",
            "Depth observations use DDS topic rt/camera/depth in addition to Unitree low-level topics.",
        ),
    )


_BUILDERS: Mapping[str, object] = {
    "holosoma": _holosoma,
    "sonic": _sonic,
    "instinctlab": _instinctlab,
}


def backend_launch(root: str | Path, backend: str, python: str = "python") -> BackendLaunch:
    try:
        builder = _BUILDERS[backend]
    except KeyError as exc:
        raise ValueError(f"unknown Sim2Sim backend {backend!r}") from exc
    return builder(Path(root).resolve(), python)  # type: ignore[operator]

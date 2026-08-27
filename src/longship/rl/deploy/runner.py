from __future__ import annotations

from datetime import datetime
import importlib.util
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any

from longship.rl.deploy.launch import DeploymentLaunch
from longship.rl.sim2sim.preflight import preflight_backend


REQUIRED_GATES = (
    "REAL_ROBOT_ENABLED",
    "GANTRY_CONFIRMED",
    "ESTOP_CONFIRMED",
    "REMOTE_CONFIRMED",
    "ROBOT_MODE_CONFIRMED",
    "FALL_ZONE_CLEAR_CONFIRMED",
)


def preflight(root: Path, launch: DeploymentLaunch) -> tuple[str, ...]:
    profile = launch.profile
    interfaces = {name for _, name in socket.if_nameindex()}
    if profile.dds.interface not in interfaces:
        raise RuntimeError(f"robot interface not found: {profile.dds.interface}")
    checks = [f"robot interface exists: {profile.dds.interface}"]
    backend_result = preflight_backend(root, profile.backend)
    if not backend_result.ready:
        raise RuntimeError(
            f"{profile.backend} artifacts are not ready:\n  - "
            + "\n  - ".join(backend_result.blockers)
        )
    checks.extend(backend_result.checks)
    for module in launch.backend.required_modules:
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(f"Python dependency is unavailable: {module}")
    for sensor_profile, sensor in zip(profile.sensors, launch.sensors, strict=True):
        missing = [
            module
            for module in sensor.required_modules
            if importlib.util.find_spec(module) is None
        ]
        if missing and sensor_profile.required:
            raise RuntimeError(
                f"required sensor {sensor_profile.name!r} is unavailable; missing: "
                + ", ".join(missing)
            )
        if not missing:
            checks.append(f"sensor adapter is ready: {sensor_profile.name}")
    checks.append("controller dependencies are importable")
    return tuple(checks)


def _environment(spec_environment: tuple[tuple[str, str], ...]) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment.update(spec_environment)
    return environment


def _stop(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def _wait_for_marker(
    process: subprocess.Popen[Any], log_path: Path, marker: str, timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
            raise RuntimeError(f"{process.args[0]} exited during startup:\n{tail}")
        if log_path.exists() and marker in log_path.read_text(
            encoding="utf-8", errors="replace"
        ):
            return
        time.sleep(0.1)
    raise RuntimeError(f"startup timeout waiting for {marker!r}; inspect {log_path}")


def competing_controller_processes(patterns: tuple[str, ...]) -> str:
    if not patterns:
        return ""
    result = subprocess.run(
        ("pgrep", "-af", "|".join(patterns)),
        check=False,
        capture_output=True,
        text=True,
    )
    current_pid = str(os.getpid())
    lines = [
        line
        for line in result.stdout.splitlines()
        if line.split(maxsplit=1)[0] != current_pid and "pgrep -af" not in line
    ]
    return "\n".join(lines)


def _require_operator_gates() -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError("physical deployment requires an interactive terminal")
    for gate in REQUIRED_GATES:
        if os.environ.get(gate) != "1":
            raise RuntimeError(f"{gate}=1 is required for physical deployment")
    if os.environ.get("REAL_ROBOT_CONFIRM") != "I_UNDERSTAND_THE_RISK":
        raise RuntimeError("REAL_ROBOT_CONFIRM=I_UNDERSTAND_THE_RISK is required")


def run(root: Path, launch: DeploymentLaunch) -> int:
    _require_operator_gates()
    competing = competing_controller_processes(
        launch.backend.competing_process_patterns
    )
    if competing:
        raise RuntimeError(f"a competing LowCmd controller is running:\n{competing}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = root / "outputs/deploy" / launch.profile.name / stamp
    output.mkdir(parents=True, exist_ok=False)
    processes: list[subprocess.Popen[Any]] = []
    print(f"LOG_DIR={output}")
    try:
        if launch.monitor is not None:
            monitor_log = output / "web_monitor.log"
            stream = monitor_log.open("w", encoding="utf-8", buffering=1)
            monitor = subprocess.Popen(
                launch.monitor.argv,
                cwd=launch.monitor.cwd,
                env=_environment(launch.monitor.environment),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            stream.close()
            processes.append(monitor)
            _wait_for_marker(
                monitor,
                monitor_log,
                "WEB MONITOR READY",
                launch.profile.runtime.startup_timeout_s,
            )
            assert launch.profile.visualization is not None
            visual = launch.profile.visualization
            print(f"VISION_URL=http://{visual.bind_host}:{visual.port}")
        for sensor_profile, sensor in zip(
            launch.profile.sensors, launch.sensors, strict=True
        ):
            missing = any(
                importlib.util.find_spec(module) is None
                for module in sensor.required_modules
            )
            if missing and not sensor_profile.required:
                print(f"SKIP optional sensor: {sensor_profile.name}")
                continue
            log_path = output / f"{sensor.process.name}.log"
            stream = log_path.open("w", encoding="utf-8", buffering=1)
            process = subprocess.Popen(
                sensor.process.argv,
                cwd=sensor.process.cwd,
                env=_environment(sensor.process.environment),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            stream.close()
            processes.append(process)
            _wait_for_marker(
                process,
                log_path,
                sensor.ready_marker,
                launch.profile.runtime.startup_timeout_s,
            )
        controller = launch.backend.controller
        controller_log = output / "controller.log"
        stream = controller_log.open("w", encoding="utf-8", buffering=1)
        process = subprocess.Popen(
            controller.argv,
            cwd=controller.cwd,
            env=_environment(controller.environment),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        stream.close()
        processes.append(process)
        _wait_for_marker(
            process,
            controller_log,
            launch.backend.ready_marker,
            launch.profile.runtime.startup_timeout_s,
        )
        release = launch.release_motion
        subprocess.run(
            release.argv,
            cwd=release.cwd,
            env=_environment(release.environment),
            check=True,
        )
        print(f"Physical deployment {launch.profile.name!r} is ready.")
        print(launch.backend.operator_hint)
        teleop = launch.backend.teleop
        return subprocess.run(
            teleop.argv,
            cwd=teleop.cwd,
            env=_environment(teleop.environment),
            check=False,
        ).returncode
    finally:
        for process in reversed(processes):
            _stop(process)

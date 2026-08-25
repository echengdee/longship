#!/usr/bin/env python3
"""Launch a complete simulator/controller/ZMQ-keyboard stack."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import IO

from longship.rl.sim2sim.launch import ProcessSpec, backend_launch
from longship.rl.sim2sim.preflight import preflight_backend


def _environment(spec: ProcessSpec) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(spec.environment)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    return environment


_STATUS_MARKERS = (
    "teleop",
    "viewer:",
    "Init Done",
    "Loading policy model",
    "Loading encoder model",
    "Initialize Engine",
    "SONIC ONNX READY",
    "tracking policy enabled",
    "policy enabled",
    "Planner enabled",
    "ERROR",
    "Error",
)


def _pump_output(process: subprocess.Popen[str], log: IO[str], label: str) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        log.write(line)
        log.flush()
        if any(marker in line for marker in _STATUS_MARKERS):
            print(f"[{label}] {line}", end="", flush=True)


def _start_background(
    spec: ProcessSpec, log: IO[str], *, echo_status: bool = False
) -> subprocess.Popen[str]:
    print(f"starting {spec.name}: {' '.join(spec.argv)}", flush=True)
    process = subprocess.Popen(
        spec.argv,
        cwd=spec.cwd,
        env=_environment(spec),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if echo_status else log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    if echo_status:
        threading.Thread(target=_pump_output, args=(process, log, spec.name), daemon=True).start()
    return process


def _stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run(root: Path, backend: str, python: str) -> int:
    result = preflight_backend(root, backend)
    if not result.ready:
        print(f"{backend} preflight is BLOCKED:", file=sys.stderr)
        for blocker in result.blockers:
            print(f"  - {blocker}", file=sys.stderr)
        return 2

    launch = backend_launch(root, backend, python)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = root / "outputs/sim2sim" / backend / timestamp
    output.mkdir(parents=True, exist_ok=False)
    simulator_log_path = output / "simulator.log"
    controller_log_path = output / "controller.log"
    print(f"logs: {output}")

    processes: list[subprocess.Popen[str]] = []
    with simulator_log_path.open("w", encoding="utf-8", buffering=1) as simulator_log, \
         controller_log_path.open("w", encoding="utf-8", buffering=1) as controller_log:
        try:
            processes.append(_start_background(launch.simulator, simulator_log, echo_status=True))
            time.sleep(1.0)
            if processes[-1].poll() is not None:
                print(f"simulator exited early; inspect {simulator_log_path}", file=sys.stderr)
                return 3

            processes.append(_start_background(launch.controller, controller_log, echo_status=True))
            # Policy adapters allocate ONNX sessions before opening their ZMQ
            # subscriber.  Do not expose the keyboard until that subscriber is
            # ready, otherwise the first `i` event could be lost (PUB/SUB has no
            # replay for late joiners).
            time.sleep(3.0)
            if processes[-1].poll() is not None:
                print(f"controller exited early; inspect {controller_log_path}", file=sys.stderr)
                return 4

            if backend == "holosoma":
                print(
                    "\nKeyboard ready: i = initialize, ] = enable/queue policy, "
                    "= = stand/walk, Ctrl-C = stop all\n"
                )
            else:
                print("\nKeyboard ready: i = initialize, ] = enable/queue policy, Ctrl-C = stop all\n")
            keyboard = subprocess.run(
                launch.teleop.argv,
                cwd=launch.teleop.cwd,
                env=_environment(launch.teleop),
                check=False,
            )
            return keyboard.returncode
        except KeyboardInterrupt:
            return 130
        finally:
            for process in reversed(processes):
                _stop(process)
            print(f"Sim2Sim stopped. Logs kept in {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("holosoma", "sonic", "instinctlab"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    return run(args.root.resolve(), args.backend, args.python)


if __name__ == "__main__":
    raise SystemExit(main())

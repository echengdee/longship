from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from longship.brain.codex_follow import CodexFollowBrain
from longship.targets.unitree_sdk2 import VelocityLimits, connect_unitree_g1

from longship.contracts.skills.follow_person import FollowState
from longship.observability.follow_person import (
    BufferedEventSink,
    CompositeEventSink,
    FollowDashboard,
    JsonlEventSink,
)
from longship.perception.follow_http import HttpFollowSceneSource
from longship.runtime.follow_person import FollowPersonRuntime, NullEventSink
from longship.safety.follow_obstacle import ForwardObstacleGuard
from longship.simulation.follow_person import (
    FollowSimulationScenario,
    FollowSimulationWorld,
    run_simulation,
)
from longship.simulation.follow_stack import run_interactive_follow_stack
from longship.simulation.follow_system import run_system_simulation
from longship.skills.follow_person.config import FollowProfile, FollowQualification
from longship.skills.follow_person.governor import MotionGovernor
from longship.skills.follow_person.planner import LocalFollowPlanner
from longship.targets.follow_person import RecordingMotion, UnitreeFollowMotion

_HARDWARE_TOKEN = "SUPERVISED-GANTRY-ONLY"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Longship's provider-neutral FollowPerson V0 slice"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    simulate = commands.add_parser(
        "simulate", help="run a deterministic closed-loop world"
    )
    simulate.add_argument("--profile", type=Path, required=True)
    simulate.add_argument("--scenario", type=Path, required=True)
    simulate.add_argument("--events", type=Path)
    simulate.add_argument("--real-time", action="store_true")
    simulate.add_argument("--dashboard-host", default="127.0.0.1")
    simulate.add_argument("--dashboard-port", type=int, default=0)
    simulate.add_argument(
        "--keep-dashboard",
        action="store_true",
        help="wait for Enter after simulation so the final state remains visible",
    )

    system_simulate = commands.add_parser(
        "system-simulate",
        help="run input, Brain, Skill, Runtime, Safety, and target end to end",
    )
    system_simulate.add_argument("--profile", type=Path, required=True)
    system_simulate.add_argument("--scenario", type=Path, required=True)
    system_simulate.add_argument("--instruction", default="Jackie，跟着我走")
    system_simulate.add_argument("--events", type=Path)
    system_simulate.add_argument("--real-time", action="store_true")

    stack = commands.add_parser(
        "stack",
        help="run a persistent terminal-to-Brain-to-Skill simulation stack",
    )
    stack.add_argument("--profile", type=Path, required=True)
    stack.add_argument("--scenario", type=Path, required=True)
    stack.add_argument("--events", type=Path)
    stack.add_argument("--dashboard-host", default="127.0.0.1")
    stack.add_argument("--dashboard-port", type=int, default=0)
    stack.add_argument(
        "--brain",
        choices=("deterministic", "codex"),
        default="deterministic",
        help="high-level semantic provider; fixed controls and STOP bypass it",
    )
    stack.add_argument("--codex-model", default=None)
    stack.add_argument(
        "--codex-reasoning-effort",
        choices=(
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ),
        default="none",
        help="Codex app-server effort; GPT-5.6 uses none to disable reasoning",
    )
    stack.add_argument("--codex-timeout-s", type=float, default=60.0)

    probe = commands.add_parser(
        "probe", help="validate a perception service without actuator access"
    )
    probe.add_argument("--perception-url", required=True)
    probe.add_argument("--timeout-s", type=float, default=0.1)

    heartbeat = commands.add_parser(
        "heartbeat", help="run the manual operator deadman file writer"
    )
    heartbeat.add_argument("--file", type=Path, required=True)

    deploy = commands.add_parser(
        "deploy", help="run a gated, supervised Unitree G1 session"
    )
    deploy.add_argument("--profile", type=Path, required=True)
    deploy.add_argument("--qualification", type=Path, required=True)
    deploy.add_argument("--perception-url", required=True)
    deploy.add_argument("--interface", required=True)
    deploy.add_argument("--target-id", required=True)
    deploy.add_argument("--boot-id", required=True)
    deploy.add_argument("--heartbeat-file", type=Path, required=True)
    deploy.add_argument("--heartbeat-timeout-s", type=float, default=1.0)
    deploy.add_argument("--maximum-runtime-s", type=float, default=120.0)
    deploy.add_argument("--events", type=Path, required=True)
    deploy.add_argument("--dashboard-host", default="127.0.0.1")
    deploy.add_argument("--dashboard-port", type=int, default=8093)
    deploy.add_argument("--hardware-enable-token", default="")
    deploy.add_argument("--doctor-passed", action="store_true")
    deploy.add_argument("--physical-estop-verified", action="store_true")
    deploy.add_argument("--camera-calibration-verified", action="store_true")
    return parser


def _simulate(args: argparse.Namespace) -> int:
    profile = FollowProfile.load(args.profile)
    scenario = FollowSimulationScenario.load(args.scenario)
    with ExitStack() as stack:
        sinks: list[Any] = []
        if args.events:
            sinks.append(stack.enter_context(JsonlEventSink(args.events)))
        dashboard = None
        if args.dashboard_port:
            dashboard = stack.enter_context(
                FollowDashboard(host=args.dashboard_host, port=args.dashboard_port)
            )
            sinks.append(dashboard)
            print(
                f"Read-only dashboard: http://{args.dashboard_host}:{dashboard.port}"
            )
        sink = CompositeEventSink(sinks) if sinks else NullEventSink()
        report = run_simulation(
            profile,
            scenario,
            event_sink=sink,
            real_time=args.real_time,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        if dashboard is not None and args.keep_dashboard:
            try:
                input("Press Enter to close the read-only dashboard. ")
            except (EOFError, KeyboardInterrupt):
                pass
    return 0 if report.passed else 2


def _probe(args: argparse.Namespace) -> int:
    source = HttpFollowSceneSource(args.perception_url, timeout_s=args.timeout_s)
    scene = source.require_ready()
    print(json.dumps(scene.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _system_simulate(args: argparse.Namespace) -> int:
    profile = FollowProfile.load(args.profile)
    scenario = FollowSimulationScenario.load(args.scenario)
    with ExitStack() as stack:
        sink = (
            stack.enter_context(JsonlEventSink(args.events))
            if args.events
            else NullEventSink()
        )
        report = run_system_simulation(
            profile,
            scenario,
            instruction=args.instruction,
            event_sink=sink,
            real_time=args.real_time,
        )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.passed else 2


def _stack(args: argparse.Namespace) -> int:
    profile = FollowProfile.load(args.profile)
    scenario = FollowSimulationScenario.load(args.scenario)
    motion = RecordingMotion()
    world = FollowSimulationWorld(scenario, motion)
    with ExitStack() as resources:
        sinks: list[Any] = []
        if args.events:
            sinks.append(resources.enter_context(JsonlEventSink(args.events)))
        if args.dashboard_port:
            dashboard = resources.enter_context(
                FollowDashboard(host=args.dashboard_host, port=args.dashboard_port)
            )
            sinks.append(dashboard)
            print(
                f"Read-only dashboard: http://{args.dashboard_host}:{dashboard.port}"
            )
        sink = CompositeEventSink(sinks) if sinks else NullEventSink()
        report = asyncio.run(
            _run_stack_with_brain(args, profile, scenario, motion, world, sink)
        )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 2 if report.exit_reason == "runtime_failed" else 0


async def _run_stack_with_brain(
    args: argparse.Namespace,
    profile: FollowProfile,
    scenario: FollowSimulationScenario,
    motion: RecordingMotion,
    world: FollowSimulationWorld,
    sink: Any,
) -> Any:
    if args.brain == "deterministic":
        return await run_interactive_follow_stack(
            profile, scenario, motion, world, event_sink=sink
        )
    with tempfile.TemporaryDirectory(prefix="longship-follow-codex-") as workspace:
        async with CodexFollowBrain(
            workspace,
            model=args.codex_model or "gpt-5.6-terra",
            reasoning_effort=args.codex_reasoning_effort,
            timeout_s=args.codex_timeout_s,
        ) as brain:
            return await run_interactive_follow_stack(
                profile,
                scenario,
                motion,
                world,
                brain=brain,
                event_sink=sink,
            )


def _heartbeat(args: argparse.Namespace) -> int:
    path: Path = args.file
    path.parent.mkdir(parents=True, exist_ok=True)
    sequence = 0
    print("Manual deadman active. Press Enter at least once per timeout window.")
    print("Closing this process or stopping key presses makes deployment fail closed.")
    try:
        while True:
            input()
            sequence += 1
            temporary = path.with_name(path.name + ".next")
            value = {
                "schema_version": "longship.operator-heartbeat.v0",
                "sequence": sequence,
                "updated_unix_ns": time.time_ns(),
                "writer_pid": os.getpid(),
            }
            temporary.write_text(
                json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8"
            )
            temporary.replace(path)
    except (EOFError, KeyboardInterrupt):
        print("Heartbeat stopped; any attached deployment will stop on timeout.")
        return 0


def _deploy(args: argparse.Namespace) -> int:
    _require_hardware_gates(args)
    if not 0.2 <= args.heartbeat_timeout_s <= 5.0:
        raise ValueError("heartbeat timeout must be between 0.2 and 5 seconds")
    if not 1.0 <= args.maximum_runtime_s <= 3_600.0:
        raise ValueError("maximum runtime must be between 1 and 3600 seconds")
    _require_fresh_heartbeat(args.heartbeat_file, args.heartbeat_timeout_s)
    profile = FollowProfile.load(args.profile)
    qualification = FollowQualification.load(args.qualification)
    profile_digest = hashlib.sha256(args.profile.read_bytes()).hexdigest()
    if not qualification.approved:
        raise RuntimeError("FollowPerson qualification record is not approved")
    if qualification.expires_at_unix_s <= int(time.time()):
        raise RuntimeError("FollowPerson qualification record has expired")
    if qualification.target_id != args.target_id:
        raise RuntimeError("qualification target does not match --target-id")
    if qualification.profile_sha256 != profile_digest:
        raise RuntimeError("qualification is bound to a different profile digest")
    if args.maximum_runtime_s > qualification.maximum_runtime_s:
        raise RuntimeError("requested runtime exceeds the qualification bound")
    source = HttpFollowSceneSource(args.perception_url)
    initial_scene = source.require_ready()
    if initial_scene.calibration_id != qualification.calibration_id:
        raise RuntimeError("qualification is bound to a different camera calibration")

    exit_code = 0
    with ExitStack() as stack:
        journal = stack.enter_context(JsonlEventSink(args.events))
        dashboard = stack.enter_context(
            FollowDashboard(
                host=args.dashboard_host,
                port=args.dashboard_port,
                camera_preview_url=args.perception_url.rstrip("/")
                + "/preview.jpg",
            )
        )
        sink = stack.enter_context(
            BufferedEventSink(CompositeEventSink((journal, dashboard)))
        )
        target = connect_unitree_g1(
            args.interface,
            target_id=args.target_id,
            boot_id=args.boot_id,
            hardware_enabled=True,
            limits=VelocityLimits(
                max_abs_vx_mps=profile.control.maximum_forward_speed_mps,
                max_abs_vy_mps=0.0,
                max_abs_yaw_rate_radps=profile.control.maximum_yaw_rate_radps,
            ),
            max_command_ttl_s=profile.runtime.command_ttl_s,
            max_lease_ttl_s=2.0,
        )
        motion = UnitreeFollowMotion(target, lease_ttl_s=1.0)
        stack.callback(motion.close)
        runtime = FollowPersonRuntime(
            profile,
            motion,
            planner=LocalFollowPlanner(profile.planner, profile.control),
            safety_guard=ForwardObstacleGuard(profile.safety),
            governor=MotionGovernor(profile.control),
            event_sink=sink,
            required_calibration_id=qualification.calibration_id,
        )
        start_ns = time.monotonic_ns()
        runtime.start(now_ns=start_ns)
        print(f"Read-only dashboard: http://{args.dashboard_host}:{dashboard.port}")
        if runtime.state is FollowState.FAILED:
            print("FollowPerson failed during motion-authority acquisition.")
            exit_code = 5
        else:
            print("FollowPerson armed. Ctrl-C requests the reviewed stop path.")
        try:
            while runtime.is_active:
                cycle_started_ns = time.monotonic_ns()
                try:
                    _require_fresh_heartbeat(
                        args.heartbeat_file, args.heartbeat_timeout_s
                    )
                except RuntimeError as exc:
                    runtime.stop(str(exc))
                    exit_code = 4
                    break
                if (cycle_started_ns - start_ns) / 1_000_000_000 >= (
                    args.maximum_runtime_s
                ):
                    runtime.stop("maximum supervised runtime reached")
                    break
                runtime.tick(source.read(), now_ns=time.monotonic_ns())
                if runtime.state is FollowState.FAILED:
                    exit_code = 5
                    break
                elapsed_s = (
                    time.monotonic_ns() - cycle_started_ns
                ) / 1_000_000_000
                time.sleep(max(0.0, profile.control_period_s - elapsed_s))
        except KeyboardInterrupt:
            runtime.stop("operator keyboard interrupt")
        finally:
            if runtime.is_active:
                runtime.stop("deployment process exiting")
        print(json.dumps(runtime.snapshot.to_dict(), ensure_ascii=False, indent=2))
        if runtime.snapshot.stop_verified is False:
            print(
                "STOP UNVERIFIED: retain the physical E-stop and use the "
                "separately qualified motion monitor before approaching the robot."
            )
            exit_code = exit_code or 6
        if sink.dropped_events or sink.last_error:
            print(
                "OBSERVABILITY DEGRADED: "
                f"dropped_events={sink.dropped_events}, "
                f"last_error={sink.last_error or 'none'}"
            )
    return exit_code


def _require_hardware_gates(args: argparse.Namespace) -> None:
    if args.hardware_enable_token != _HARDWARE_TOKEN:
        raise RuntimeError(
            f"hardware disabled; pass --hardware-enable-token {_HARDWARE_TOKEN} "
            "only for a supervised gantry session"
        )
    missing = [
        label
        for label, ready in (
            ("robot doctor", args.doctor_passed),
            ("physical E-stop", args.physical_estop_verified),
            ("camera calibration", args.camera_calibration_verified),
        )
        if not ready
    ]
    if missing:
        raise RuntimeError("hardware gates not confirmed: " + ", ".join(missing))


def _require_fresh_heartbeat(path: Path, maximum_age_s: float) -> None:
    try:
        if path.stat().st_size > 4_096:
            raise RuntimeError("operator heartbeat file is oversized")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("operator heartbeat is unavailable") from exc
    expected = {"schema_version", "sequence", "updated_unix_ns", "writer_pid"}
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeError("operator heartbeat has an invalid shape")
    if value["schema_version"] != "longship.operator-heartbeat.v0":
        raise RuntimeError("operator heartbeat schema is unsupported")
    if type(value["sequence"]) is not int or value["sequence"] <= 0:
        raise RuntimeError("operator heartbeat sequence is invalid")
    if type(value["updated_unix_ns"]) is not int:
        raise RuntimeError("operator heartbeat timestamp is invalid")
    if type(value["writer_pid"]) is not int or value["writer_pid"] <= 0:
        raise RuntimeError("operator heartbeat writer identity is invalid")
    age_s = (time.time_ns() - value["updated_unix_ns"]) / 1_000_000_000
    if age_s < -0.2 or age_s > maximum_age_s:
        raise RuntimeError(f"operator heartbeat is stale ({age_s:.2f} s)")


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if (
        args.command == "stack"
        and args.codex_model is not None
        and args.brain != "codex"
    ):
        parser.error("--codex-model requires --brain codex")
    if args.command == "stack" and not 5.0 <= args.codex_timeout_s <= 120.0:
        parser.error("--codex-timeout-s must be between 5 and 120 seconds")
    try:
        if args.command == "simulate":
            code = _simulate(args)
        elif args.command == "system-simulate":
            code = _system_simulate(args)
        elif args.command == "stack":
            code = _stack(args)
        elif args.command == "probe":
            code = _probe(args)
        elif args.command == "heartbeat":
            code = _heartbeat(args)
        else:
            code = _deploy(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(3, f"BLOCKED: {exc}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()

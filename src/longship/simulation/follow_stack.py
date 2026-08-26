from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from longship.brain.follow_person import DeterministicFollowBrain, FollowBrainPort
from longship.contracts.skills.follow_person import FollowState
from longship.runtime.follow_mission import FollowMissionRuntime
from longship.runtime.follow_person import FollowEventSink, FollowPersonRuntime
from longship.safety.follow_obstacle import ForwardObstacleGuard
from longship.simulation.follow_person import FollowSimulationScenario
from longship.simulation.follow_system import (
    FollowSystemMotion,
    FollowSystemWorld,
    SystemTraceSink,
)
from longship.skills.follow_person.config import FollowProfile
from longship.skills.follow_person.governor import MotionGovernor
from longship.skills.follow_person.planner import LocalFollowPlanner
from longship.tour.interaction import CommandKind


LineReader = Callable[[str], Awaitable[str]]
LineWriter = Callable[[str], None]
KeepRunning = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class FollowStackReport:
    terminal_state: FollowState
    exit_reason: str
    brain_requests: int
    accepted_skill_calls: int
    control_steps: int
    target_command_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "longship.follow-interactive-stack-report.v0",
            "terminal_state": self.terminal_state.value,
            "exit_reason": self.exit_reason,
            "brain_requests": self.brain_requests,
            "accepted_skill_calls": self.accepted_skill_calls,
            "control_steps": self.control_steps,
            "target_command_steps": self.target_command_steps,
        }


async def console_line_reader(prompt: str) -> str:
    """Read one terminal line without blocking the Runtime event loop."""

    print(prompt, end="", flush=True)
    loop = asyncio.get_running_loop()
    result: asyncio.Future[str] = loop.create_future()

    def readable() -> None:
        try:
            line = sys.stdin.readline()
        except Exception as exc:
            if not result.done():
                result.set_exception(exc)
            return
        if result.done():
            return
        if line == "":
            result.set_exception(EOFError())
        else:
            result.set_result(line.rstrip("\r\n"))

    try:
        loop.add_reader(sys.stdin.fileno(), readable)
    except (AttributeError, NotImplementedError, OSError):
        line = await asyncio.to_thread(sys.stdin.readline)
        if line == "":
            raise EOFError
        return line.rstrip("\r\n")
    try:
        return await result
    finally:
        loop.remove_reader(sys.stdin.fileno())


async def run_interactive_follow_stack(
    profile: FollowProfile,
    scenario: FollowSimulationScenario,
    motion: FollowSystemMotion,
    world: FollowSystemWorld,
    *,
    brain: FollowBrainPort | None = None,
    event_sink: FollowEventSink | None = None,
    read_line: LineReader = console_line_reader,
    write_line: LineWriter = print,
    keep_running: KeepRunning | None = None,
) -> FollowStackReport:
    """Compose Longship's terminal, Brain, Skill, Runtime, Safety, and target.

    The terminal is only a text-input provider. It cannot emit velocity. STOP
    remains a reserved Runtime command and runs in a separate task so it can
    overtake a slow Brain request.
    """

    if abs(profile.control_period_s - scenario.step_s) > 1e-9:
        raise ValueError("scenario step must match the profile control period")

    trace = SystemTraceSink(event_sink)
    step_ns = int(scenario.step_s * 1_000_000_000)
    clock_ns = [1_000_000_000]
    skill = FollowPersonRuntime(
        profile,
        motion,
        planner=LocalFollowPlanner(profile.planner, profile.control),
        safety_guard=ForwardObstacleGuard(profile.safety),
        governor=MotionGovernor(profile.control),
        event_sink=trace,
        session_id=f"interactive-simulation:{scenario.scenario_id}",
    )
    mission = FollowMissionRuntime(
        skill,
        brain=brain or DeterministicFollowBrain(),
        event_sink=trace,
        clock_ns=lambda: clock_ns[0],
    )
    done = asyncio.Event()
    exit_reason = "operator_exit"
    scenario_time_s = 0.0
    control_steps = 0
    target_command_steps = 0
    command_tasks: set[asyncio.Task[str]] = set()
    stop_task: asyncio.Task[str] | None = None

    write_line(
        "Commands: 跟着我走/follow me, pause/暂停, resume/继续, "
        "status/状态, stop/停止, exit/退出"
    )

    def command_completed(task: asyncio.Task[str]) -> None:
        nonlocal stop_task
        command_tasks.discard(task)
        if task is stop_task:
            stop_task = None
        if task.cancelled():
            return
        try:
            write_line(task.result())
        except Exception as exc:
            write_line(f"Command failed closed: {type(exc).__name__}")

    async def control_loop() -> None:
        nonlocal control_steps, exit_reason, scenario_time_s, target_command_steps
        while not done.is_set():
            await asyncio.sleep(scenario.step_s)
            if keep_running is not None and not keep_running():
                if skill.is_active:
                    mission.stop("simulation target closed")
                exit_reason = "target_closed"
                done.set()
                return
            if not skill.is_active:
                if skill.state in {
                    FollowState.FAILED,
                    FollowState.STOPPED,
                    FollowState.STOP_UNVERIFIED,
                }:
                    exit_reason = (
                        "runtime_failed"
                        if skill.state is FollowState.FAILED
                        else "operator_stop"
                    )
                    done.set()
                continue

            clock_ns[0] += step_ns
            scene = world.scene(scenario_time_s, clock_ns[0])
            snapshot = mission.tick(scene, now_ns=clock_ns[0])
            control_steps += 1
            if snapshot.command is not None:
                target_command_steps += 1
            if snapshot.state is FollowState.FAILED:
                exit_reason = "runtime_failed"
                done.set()
                return
            world.advance(scenario.step_s)
            scenario_time_s = min(
                scenario.duration_s, scenario_time_s + scenario.step_s
            )

    control_task = asyncio.create_task(
        control_loop(), name="follow-interactive-control-loop"
    )
    try:
        while not done.is_set():
            line_task = asyncio.create_task(read_line("> "))
            done_task = asyncio.create_task(done.wait())
            completed, _ = await asyncio.wait(
                {line_task, done_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if done_task in completed:
                line_task.cancel()
                break
            done_task.cancel()
            try:
                text = line_task.result()
            except (EOFError, KeyboardInterrupt):
                if skill.is_active:
                    mission.stop("terminal input closed")
                exit_reason = "terminal_closed"
                done.set()
                break

            normalized = " ".join(text.casefold().strip().split())
            if normalized in {"exit", "quit", "退出"}:
                if skill.is_active:
                    write_line(mission.stop("operator exited interactive stack"))
                exit_reason = "operator_exit"
                done.set()
                break
            routed = mission.router.route(text)
            if (
                routed.kind is CommandKind.STOP
                and stop_task is not None
                and not stop_task.done()
            ):
                write_line("Protective stop is already in progress.")
                continue
            if len(command_tasks) >= 8 and routed.kind is not CommandKind.STOP:
                write_line(
                    "Too many pending requests; deterministic STOP remains available."
                )
                continue
            task = asyncio.create_task(mission.handle_text(text))
            if routed.kind is CommandKind.STOP:
                stop_task = task
            command_tasks.add(task)
            task.add_done_callback(command_completed)
            await asyncio.sleep(0)
            if routed.kind is CommandKind.STOP:
                await asyncio.gather(task, return_exceptions=True)
                exit_reason = "operator_stop"
                done.set()
                break
    finally:
        done.set()
        if skill.is_active:
            mission.stop("interactive stack shutdown")
        for task in command_tasks:
            task.cancel()
        if command_tasks:
            await asyncio.wait(command_tasks, timeout=1.0)
        control_task.cancel()
        await asyncio.gather(control_task, return_exceptions=True)

    return FollowStackReport(
        terminal_state=skill.state,
        exit_reason=exit_reason,
        brain_requests=mission.brain_request_count,
        accepted_skill_calls=mission.accepted_skill_calls,
        control_steps=control_steps,
        target_command_steps=target_command_steps,
    )

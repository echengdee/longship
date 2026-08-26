from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from longship.audio import (
    MockVoiceInput,
    VoiceInputEvent,
    VoiceInputEventType,
    WakeDictationController,
)
from longship.brain.follow_person import DeterministicFollowBrain, FollowBrainPort
from longship.contracts.skills.follow_person import FollowScene, FollowState
from longship.runtime.follow_mission import FollowMissionRuntime
from longship.runtime.follow_person import FollowEventSink, FollowPersonRuntime, NullEventSink
from longship.safety.follow_obstacle import ForwardObstacleGuard
from longship.simulation.follow_person import (
    FollowSimulationScenario,
    FollowSimulationWorld,
)
from longship.skills.follow_person.config import FollowProfile
from longship.skills.follow_person.governor import MotionGovernor
from longship.skills.follow_person.planner import LocalFollowPlanner
from longship.targets.follow_person import RecordingMotion


class FollowSystemWorld(Protocol):
    def scene(self, time_s: float, now_ns: int) -> FollowScene:
        ...

    def advance(self, dt_s: float) -> None:
        ...

    def person_distance(self, time_s: float) -> float:
        ...


class FollowSystemMotion(Protocol):
    def acquire(self, session_id: str, now_ns: int) -> Any:
        ...

    def apply(self, command: Any) -> Any:
        ...

    def protective_stop(self, reason: str) -> Any:
        ...


class SystemTraceSink:
    def __init__(self, downstream: FollowEventSink | None = None) -> None:
        self.downstream = downstream or NullEventSink()
        self.events: list[Mapping[str, Any]] = []

    def publish(self, event: Mapping[str, Any]) -> None:
        self.events.append(dict(event))
        self.downstream.publish(event)


@dataclass(frozen=True, slots=True)
class FollowSystemReport:
    passed: bool
    instruction: str
    brain_requests: int
    accepted_skill_calls: int
    following_steps: int
    target_command_steps: int
    unsafe_forward_commands: int
    physical_contact_steps: int | None
    final_distance_error_m: float
    state_before_operator_stop: FollowState
    terminal_state: FollowState
    stop_verified: bool | None
    observed_stages: tuple[str, ...]
    checks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "longship.follow-system-simulation-report.v0",
            "passed": self.passed,
            "instruction": self.instruction,
            "brain_requests": self.brain_requests,
            "accepted_skill_calls": self.accepted_skill_calls,
            "following_steps": self.following_steps,
            "target_command_steps": self.target_command_steps,
            "unsafe_forward_commands": self.unsafe_forward_commands,
            "physical_contact_steps": self.physical_contact_steps,
            "final_distance_error_m": self.final_distance_error_m,
            "state_before_operator_stop": self.state_before_operator_stop.value,
            "terminal_state": self.terminal_state.value,
            "stop_verified": self.stop_verified,
            "observed_stages": list(self.observed_stages),
            "checks": list(self.checks),
        }


async def run_system_with_world(
    profile: FollowProfile,
    scenario: FollowSimulationScenario,
    motion: FollowSystemMotion,
    world: FollowSystemWorld,
    *,
    instruction: str = "Jackie，跟着我走",
    brain: FollowBrainPort | None = None,
    event_sink: FollowEventSink | None = None,
    real_time: bool = False,
) -> FollowSystemReport:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("system simulation instruction is required")
    if len(instruction) > 500:
        raise ValueError("system simulation instruction is too long")
    if abs(profile.control_period_s - scenario.step_s) > 1e-9:
        raise ValueError("scenario step must match the profile control period")

    trace = SystemTraceSink(event_sink)
    clock = [1_000_000_000 - int(scenario.step_s * 1_000_000_000)]
    skill = FollowPersonRuntime(
        profile,
        motion,
        planner=LocalFollowPlanner(profile.planner, profile.control),
        safety_guard=ForwardObstacleGuard(profile.safety),
        governor=MotionGovernor(profile.control),
        event_sink=trace,
        session_id=f"system-simulation:{scenario.scenario_id}",
    )
    mission = FollowMissionRuntime(
        skill,
        brain=brain or DeterministicFollowBrain(),
        event_sink=trace,
        clock_ns=lambda: clock[0],
    )
    voice = WakeDictationController(MockVoiceInput(), mission)
    input_time_s = clock[0] / 1_000_000_000
    await voice.handle_event(
        VoiceInputEvent(VoiceInputEventType.WAKE, "follow-start", input_time_s)
    )
    await voice.handle_event(
        VoiceInputEvent(
            VoiceInputEventType.FINAL,
            "follow-start",
            input_time_s + 0.001,
            instruction,
            1.0,
        )
    )
    input_completed = await voice.wait_idle(timeout_s=2.0)

    following_steps = 0
    target_command_steps = 0
    unsafe_commands = 0
    step_ns = int(scenario.step_s * 1_000_000_000)
    final_time_s = 0.0
    if input_completed and skill.is_active:
        step_count = round(scenario.duration_s / scenario.step_s)
        for index in range(step_count + 1):
            final_time_s = index * scenario.step_s
            clock[0] += step_ns
            scene = world.scene(final_time_s, clock[0])
            snapshot = mission.tick(scene, now_ns=clock[0])
            if snapshot.state is FollowState.FOLLOWING:
                following_steps += 1
            command = snapshot.command
            if command is not None:
                target_command_steps += 1
                if (
                    command.forward_mps > 0.0
                    and scene.raw_forward_clearance_m is not None
                    and scene.raw_forward_clearance_m
                    <= profile.safety.nominal_stop_distance_m
                ):
                    unsafe_commands += 1
            if snapshot.state is FollowState.FAILED:
                break
            world.advance(scenario.step_s)
            if real_time:
                await asyncio.sleep(scenario.step_s)

    before_stop = skill.state
    if skill.is_active:
        stop_time_s = clock[0] / 1_000_000_000 + 0.001
        await voice.handle_event(
            VoiceInputEvent(
                VoiceInputEventType.PARTIAL,
                "ambient-stop",
                stop_time_s,
                "停止",
                1.0,
            )
        )
        await voice.wait_idle(timeout_s=2.0)
    await voice.aclose()

    final_error = abs(
        world.person_distance(min(scenario.duration_s, final_time_s))
        - profile.control.desired_distance_m
    )
    observed_stages = _observed_stages(trace.events)
    criteria = scenario.acceptance
    raw_contact_steps = getattr(world, "contact_steps", None)
    physical_contact_steps = (
        raw_contact_steps if type(raw_contact_steps) is int else None
    )
    stage_requirements = (
        "input",
        "brain",
        "skill",
        "runtime",
        "safety",
        "target",
    )
    missing_stages = tuple(
        stage for stage in stage_requirements if stage not in observed_stages
    )
    checks = (
        f"input completed: {input_completed}",
        f"brain requests {mission.brain_request_count} >= 1",
        f"accepted Skill calls {mission.accepted_skill_calls} == 1",
        f"following steps {following_steps} >= {criteria.minimum_following_steps}",
        f"target command steps {target_command_steps} > 0",
        f"unsafe forward commands {unsafe_commands} <= "
        f"{criteria.maximum_unsafe_forward_commands}",
        f"physical contact steps: {physical_contact_steps}",
        f"final distance error {final_error:.3f} m <= "
        f"{criteria.maximum_final_distance_error_m:.3f} m",
        f"missing pipeline stages: {list(missing_stages)}",
        f"operator stop verified: {skill.snapshot.stop_verified}",
    )
    passed = (
        input_completed
        and not voice.task_errors
        and mission.brain_request_count >= 1
        and mission.accepted_skill_calls == 1
        and following_steps >= criteria.minimum_following_steps
        and target_command_steps > 0
        and unsafe_commands <= criteria.maximum_unsafe_forward_commands
        and (physical_contact_steps is None or physical_contact_steps == 0)
        and final_error <= criteria.maximum_final_distance_error_m
        and before_stop is not FollowState.FAILED
        and skill.state is FollowState.STOPPED
        and skill.snapshot.stop_verified is True
        and not missing_stages
    )
    return FollowSystemReport(
        passed=passed,
        instruction=instruction,
        brain_requests=mission.brain_request_count,
        accepted_skill_calls=mission.accepted_skill_calls,
        following_steps=following_steps,
        target_command_steps=target_command_steps,
        unsafe_forward_commands=unsafe_commands,
        physical_contact_steps=physical_contact_steps,
        final_distance_error_m=final_error,
        state_before_operator_stop=before_stop,
        terminal_state=skill.state,
        stop_verified=skill.snapshot.stop_verified,
        observed_stages=observed_stages,
        checks=checks,
    )


def run_system_simulation(
    profile: FollowProfile,
    scenario: FollowSimulationScenario,
    *,
    instruction: str = "Jackie，跟着我走",
    brain: FollowBrainPort | None = None,
    event_sink: FollowEventSink | None = None,
    real_time: bool = False,
) -> FollowSystemReport:
    motion = RecordingMotion()
    world = FollowSimulationWorld(scenario, motion)
    return asyncio.run(
        run_system_with_world(
            profile,
            scenario,
            motion,
            world,
            instruction=instruction,
            brain=brain,
            event_sink=event_sink,
            real_time=real_time,
        )
    )


def _observed_stages(events: list[Mapping[str, Any]]) -> tuple[str, ...]:
    stages: set[str] = set()
    for event in events:
        if event.get("schema_version") == "longship.follow-system-event.v0":
            event_type = event.get("event_type")
            if isinstance(event_type, str):
                if event_type.startswith("input."):
                    stages.add("input")
                elif event_type.startswith("brain."):
                    stages.add("brain")
                elif event_type.startswith("skill."):
                    stages.add("skill")
                elif event_type.startswith("safety."):
                    stages.add("safety")
        if event.get("schema_version") == "longship.follow-runtime-event.v1":
            stages.add("runtime")
            snapshot = event.get("snapshot")
            if isinstance(snapshot, Mapping) and snapshot.get("command") is not None:
                stages.add("target")
    return tuple(sorted(stages))

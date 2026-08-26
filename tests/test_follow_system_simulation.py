from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from longship.brain.codex_follow import parse_follow_brain_response
from longship.brain.codex_local import CodexProviderError
from longship.brain.follow_person import (
    FOLLOW_PERSON_SKILL_ID,
    DeterministicFollowBrain,
    FollowBrainAction,
    FollowBrainContext,
    FollowBrainDecision,
    FollowTaskDraft,
    FollowTaskOperation,
    FollowTaskStep,
)
from longship.contracts.skills.follow_person import FollowState
from longship.contracts.runtime.task_graph import MissionTaskGraphState
from longship.runtime.follow_mission import FollowMissionRuntime
from longship.runtime.follow_person import FollowPersonRuntime
from longship.safety.follow_obstacle import ForwardObstacleGuard
from longship.simulation.follow_person import (
    FollowSimulationScenario,
    FollowSimulationWorld,
)
from longship.simulation.follow_system import run_system_simulation
from longship.skills.follow_person.config import FollowProfile
from longship.skills.follow_person.governor import MotionGovernor
from longship.skills.follow_person.planner import LocalFollowPlanner
from longship.targets.follow_person import RecordingMotion


ROOT = Path(__file__).resolve().parents[1]


class DeterministicFollowBrainTests(unittest.IsolatedAsyncioTestCase):
    async def test_follow_instruction_proposes_only_semantic_skill(self) -> None:
        brain = DeterministicFollowBrain()
        context = FollowBrainContext(
            request_id="request-1",
            runtime_revision=4,
            active_skill_call_id=None,
            skill_state=FollowState.IDLE,
        )

        decision = await brain.decide("请跟着我走", context)

        self.assertIs(decision.action, FollowBrainAction.CALL_SKILL)
        self.assertEqual(decision.skill_id, FOLLOW_PERSON_SKILL_ID)
        self.assertEqual(decision.based_on_runtime_revision, 4)

    async def test_negated_follow_instruction_does_not_start_skill(self) -> None:
        decision = await DeterministicFollowBrain().decide(
            "不要跟着我",
            FollowBrainContext(
                request_id="request-2",
                runtime_revision=0,
                active_skill_call_id=None,
                skill_state=FollowState.IDLE,
            ),
        )

        self.assertIs(decision.action, FollowBrainAction.RESPOND)
        self.assertIsNone(decision.skill_id)


class CodexFollowBrainContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = FollowBrainContext(
            request_id="request-codex-1",
            runtime_revision=3,
            active_skill_call_id=None,
            skill_state=FollowState.IDLE,
        )

    def test_semantic_skill_proposal_is_bound_to_runtime_context(self) -> None:
        decision = parse_follow_brain_response(
            '{"action":"call_skill","skill_id":"navigation.follow_person",'
            '"summary":"timed follow request","steps":['
            '{"operation":"follow","duration_s":3},'
            '{"operation":"pause","duration_s":1},'
            '{"operation":"resume","duration_s":null}]}',
            self.context,
        )

        self.assertEqual(decision.request_id, "request-codex-1")
        self.assertEqual(decision.based_on_runtime_revision, 3)
        self.assertEqual(decision.skill_id, FOLLOW_PERSON_SKILL_ID)
        assert decision.task_draft is not None
        self.assertEqual(
            tuple(step.operation for step in decision.task_draft.steps),
            (
                FollowTaskOperation.FOLLOW,
                FollowTaskOperation.PAUSE,
                FollowTaskOperation.RESUME,
            ),
        )
        self.assertEqual(
            tuple(step.duration_s for step in decision.task_draft.steps),
            (3, 1, None),
        )

    def test_motion_or_unknown_skill_output_fails_closed(self) -> None:
        with self.assertRaises(CodexProviderError):
            parse_follow_brain_response(
                '{"action":"call_skill","skill_id":"publish_velocity",'
                '"summary":"move","steps":['
                '{"operation":"follow","duration_s":null}]}',
                self.context,
            )

    def test_missing_task_steps_fail_closed(self) -> None:
        with self.assertRaises(CodexProviderError):
            parse_follow_brain_response(
                '{"action":"call_skill",'
                '"skill_id":"navigation.follow_person",'
                '"summary":"follow request"}',
                self.context,
            )


class _SlowFollowBrain:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def decide(
        self, text: str, context: FollowBrainContext
    ) -> FollowBrainDecision:
        del text
        self.started.set()
        await self.release.wait()
        return FollowBrainDecision(
            request_id=context.request_id,
            based_on_runtime_revision=context.runtime_revision,
            action=FollowBrainAction.CALL_SKILL,
            skill_id=FOLLOW_PERSON_SKILL_ID,
            summary="late proposal",
            confidence=1.0,
            task_draft=FollowTaskDraft(
                (FollowTaskStep(FollowTaskOperation.FOLLOW, None),)
            ),
        )


class _TimedFollowBrain:
    async def decide(
        self, text: str, context: FollowBrainContext
    ) -> FollowBrainDecision:
        del text
        return FollowBrainDecision(
            request_id=context.request_id,
            based_on_runtime_revision=context.runtime_revision,
            action=FollowBrainAction.CALL_SKILL,
            skill_id=FOLLOW_PERSON_SKILL_ID,
            summary="follow, pause, then resume",
            confidence=1.0,
            task_draft=FollowTaskDraft(
                (
                    FollowTaskStep(FollowTaskOperation.FOLLOW, 0.1),
                    FollowTaskStep(FollowTaskOperation.PAUSE, 0.1),
                    FollowTaskStep(FollowTaskOperation.RESUME, None),
                )
            ),
        )


class FollowMissionConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_invalidates_an_inflight_brain_proposal_while_idle(self) -> None:
        profile = FollowProfile.load(
            ROOT / "scenarios/follow_person/profile.v0.json"
        )
        motion = RecordingMotion()
        skill = FollowPersonRuntime(
            profile,
            motion,
            planner=LocalFollowPlanner(profile.planner, profile.control),
            safety_guard=ForwardObstacleGuard(profile.safety),
            governor=MotionGovernor(profile.control),
        )
        brain = _SlowFollowBrain()
        mission = FollowMissionRuntime(skill, brain=brain)

        pending = asyncio.create_task(mission.handle_text("follow me"))
        await brain.started.wait()
        stop_result = await mission.handle_text("stop")
        brain.release.set()
        late_result = await pending

        self.assertEqual(stop_result, "FollowPerson is not active.")
        self.assertIn("rejected", late_result)
        self.assertEqual(mission.accepted_skill_calls, 0)
        self.assertEqual(skill.state, FollowState.IDLE)

    async def test_runtime_advances_a_timed_brain_draft_during_control_ticks(
        self,
    ) -> None:
        profile = FollowProfile.load(
            ROOT / "scenarios/follow_person/profile.v0.json"
        )
        scenario = FollowSimulationScenario.load(
            ROOT / "scenarios/follow_person/closed_loop.v0.json"
        )
        motion = RecordingMotion()
        world = FollowSimulationWorld(scenario, motion)
        clock = [1_000_000_000]
        skill = FollowPersonRuntime(
            profile,
            motion,
            planner=LocalFollowPlanner(profile.planner, profile.control),
            safety_guard=ForwardObstacleGuard(profile.safety),
            governor=MotionGovernor(profile.control),
        )
        mission = FollowMissionRuntime(
            skill,
            brain=_TimedFollowBrain(),
            clock_ns=lambda: clock[0],
        )

        result = await mission.handle_text(
            "跟我走一小段，暂停一下，然后继续走"
        )

        self.assertIn("follow 0.1s -> pause 0.1s -> resume", result)
        assert mission.task_graph_snapshot is not None
        self.assertEqual(
            mission.task_graph_snapshot.current_operation_id,
            "navigation.follow_person.start",
        )

        step_ns = int(scenario.step_s * 1_000_000_000)
        for index in range(2):
            clock[0] += step_ns
            scene = world.scene(index * scenario.step_s, clock[0])
            mission.tick(scene, now_ns=clock[0])
            world.advance(scenario.step_s)

        self.assertEqual(skill.state, FollowState.PAUSED)
        self.assertIn("navigation.follow_person.pause", mission.status_message())

        for index in range(2, 4):
            clock[0] += step_ns
            scene = world.scene(index * scenario.step_s, clock[0])
            mission.tick(scene, now_ns=clock[0])
            world.advance(scenario.step_s)

        assert mission.task_graph_snapshot is not None
        self.assertEqual(
            mission.task_graph_snapshot.current_operation_id,
            "navigation.follow_person.resume",
        )
        self.assertNotEqual(skill.state, FollowState.PAUSED)

        stop_result = mission.stop("test operator stop")
        self.assertIn("stopped", stop_result)
        assert mission.task_graph_snapshot is not None
        self.assertEqual(
            mission.task_graph_snapshot.state,
            MissionTaskGraphState.CANCELLED,
        )


class FollowSystemSimulationTests(unittest.TestCase):
    def test_instruction_to_brain_to_skill_to_target_passes(self) -> None:
        profile = FollowProfile.load(
            ROOT / "scenarios/follow_person/profile.v0.json"
        )
        scenario = FollowSimulationScenario.load(
            ROOT / "scenarios/follow_person/closed_loop.v0.json"
        )

        report = run_system_simulation(profile, scenario)

        self.assertTrue(report.passed, report.checks)
        self.assertEqual(report.brain_requests, 1)
        self.assertEqual(report.accepted_skill_calls, 1)
        self.assertEqual(report.terminal_state, FollowState.STOPPED)
        self.assertEqual(
            set(report.observed_stages),
            {"input", "brain", "skill", "runtime", "safety", "target"},
        )

    def test_unrecognized_instruction_cannot_start_motion(self) -> None:
        profile = FollowProfile.load(
            ROOT / "scenarios/follow_person/profile.v0.json"
        )
        scenario = FollowSimulationScenario.load(
            ROOT / "scenarios/follow_person/closed_loop.v0.json"
        )

        report = run_system_simulation(
            profile,
            scenario,
            instruction="Jackie，今天天气怎么样",
        )

        self.assertFalse(report.passed)
        self.assertEqual(report.accepted_skill_calls, 0)
        self.assertEqual(report.target_command_steps, 0)


if __name__ == "__main__":
    unittest.main()

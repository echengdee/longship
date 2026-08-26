from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from longship.contracts.skills.follow_person import FollowState
from longship.simulation.follow_person import (
    FollowSimulationScenario,
    FollowSimulationWorld,
)
from longship.simulation.follow_stack import run_interactive_follow_stack
from longship.skills.follow_person.config import FollowProfile
from longship.targets.follow_person import RecordingMotion


ROOT = Path(__file__).resolve().parents[1]


class _ScriptedInput:
    def __init__(self, steps: tuple[tuple[float, str], ...]) -> None:
        self._steps = iter(steps)

    async def __call__(self, prompt: str) -> str:
        del prompt
        delay_s, text = next(self._steps)
        await asyncio.sleep(delay_s)
        return text


class FollowInteractiveStackTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_instruction_reaches_target_and_stop(self) -> None:
        profile = FollowProfile.load(
            ROOT / "scenarios/follow_person/profile.v0.json"
        )
        scenario = FollowSimulationScenario.load(
            ROOT / "scenarios/follow_person/closed_loop.v0.json"
        )
        motion = RecordingMotion()
        world = FollowSimulationWorld(scenario, motion)
        output: list[str] = []

        report = await run_interactive_follow_stack(
            profile,
            scenario,
            motion,
            world,
            read_line=_ScriptedInput(
                (
                    (0.0, "Jackie，跟着我走"),
                    (0.12, "状态"),
                    (0.08, "停止"),
                )
            ),
            write_line=output.append,
        )

        self.assertEqual(report.brain_requests, 1)
        self.assertEqual(report.accepted_skill_calls, 1)
        self.assertGreater(report.control_steps, 0)
        self.assertGreater(report.target_command_steps, 0)
        self.assertEqual(report.terminal_state, FollowState.STOPPED)
        self.assertEqual(report.exit_reason, "operator_stop")
        self.assertTrue(
            any("active_call=follow-skill-call-1" in item for item in output)
        )

    async def test_terminal_exit_stops_an_active_skill(self) -> None:
        profile = FollowProfile.load(
            ROOT / "scenarios/follow_person/profile.v0.json"
        )
        scenario = FollowSimulationScenario.load(
            ROOT / "scenarios/follow_person/closed_loop.v0.json"
        )
        motion = RecordingMotion()
        world = FollowSimulationWorld(scenario, motion)

        report = await run_interactive_follow_stack(
            profile,
            scenario,
            motion,
            world,
            read_line=_ScriptedInput(
                ((0.0, "follow me"), (0.08, "exit"))
            ),
            write_line=lambda _: None,
        )

        self.assertEqual(report.exit_reason, "operator_exit")
        self.assertEqual(report.terminal_state, FollowState.STOPPED)
        self.assertTrue(motion.stop_reasons)


if __name__ == "__main__":
    unittest.main()

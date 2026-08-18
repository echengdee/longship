from __future__ import annotations

import asyncio
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from longship.brain.base import TourBrainAction
from longship.brain.codex_local import (
    CodexLocalBrain,
    CodexProviderError,
    parse_brain_proposal,
)
from longship.tour.models import TourSnapshot, TourState


class CodexProviderTests(unittest.TestCase):
    def test_valid_structured_decision(self) -> None:
        decision = parse_brain_proposal('{"action":"respond","message":"Hello"}')
        self.assertIs(decision.action, TourBrainAction.RESPOND)
        self.assertEqual(decision.message, "Hello")

    def test_extra_field_fails_closed(self) -> None:
        with self.assertRaises(CodexProviderError):
            parse_brain_proposal(
                '{"action":"respond","message":"ok","velocity":{"vx":1}}'
            )

    def test_unapproved_action_fails_closed(self) -> None:
        with self.assertRaises(CodexProviderError):
            parse_brain_proposal('{"action":"run_shell","message":"ignored"}')

    def test_non_json_fails_closed(self) -> None:
        with self.assertRaises(CodexProviderError):
            parse_brain_proposal("start moving now")

    def test_terminal_control_text_fails_closed(self) -> None:
        with self.assertRaises(CodexProviderError):
            parse_brain_proposal(
                '{"action":"respond","message":"\\u001b[31mnot speech"}'
            )


class FakeTurnResult:
    def __init__(self, response: str) -> None:
        self.final_response = response


class FakeTurnHandle:
    def __init__(
        self, *, slow: bool = False, interrupt_stalls: bool = False
    ) -> None:
        self.slow = slow
        self.interrupt_stalls = interrupt_stalls
        self.interrupted = False
        self.run_cancelled = False
        self.closed = asyncio.Event()

    async def run(self):
        try:
            if self.slow:
                await self.closed.wait()
        except asyncio.CancelledError:
            self.run_cancelled = True
            raise
        return FakeTurnResult('{"action":"respond","message":"Hello"}')

    async def interrupt(self) -> None:
        self.interrupted = True
        if self.interrupt_stalls:
            await self.closed.wait()


class FakeThread:
    def __init__(self, handle: FakeTurnHandle) -> None:
        self.handle = handle
        self.turn_kwargs = None

    async def turn(self, prompt, **kwargs):
        self.turn_kwargs = kwargs
        return self.handle


class FakeAsyncCodex:
    instance = None
    instances = []
    total_thread_start_calls = 0
    first_thread_slow = False
    first_interrupt_stalls = False
    block_first_exit = False
    first_exit_started = None
    release_first_exit = None

    def __init__(self) -> None:
        type(self).instance = self
        type(self).instances.append(self)
        self.thread_start_calls = 0
        self.threads: list[FakeThread] = []
        self.exited = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        first_instance = self is type(self).instances[0]
        if self.block_first_exit and first_instance:
            type(self).first_exit_started.set()
            await type(self).release_first_exit.wait()
        self.exited = True
        for thread in self.threads:
            thread.handle.closed.set()
        return None

    async def thread_start(self, **kwargs):
        self.thread_start_calls += 1
        type(self).total_thread_start_calls += 1
        first_global_thread = type(self).total_thread_start_calls == 1
        slow = self.first_thread_slow and first_global_thread
        interrupt_stalls = self.first_interrupt_stalls and first_global_thread
        thread = FakeThread(
            FakeTurnHandle(slow=slow, interrupt_stalls=interrupt_stalls)
        )
        self.threads.append(thread)
        return thread


def snapshot() -> TourSnapshot:
    return TourSnapshot("tour", TourState.IDLE, 0, None, None, None, 1)


class CodexProviderLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        FakeAsyncCodex.instances = []
        FakeAsyncCodex.total_thread_start_calls = 0
        FakeAsyncCodex.first_thread_slow = False
        FakeAsyncCodex.first_interrupt_stalls = False
        FakeAsyncCodex.block_first_exit = False
        FakeAsyncCodex.first_exit_started = None
        FakeAsyncCodex.release_first_exit = None

    def fake_module(self):
        return SimpleNamespace(
            ApprovalMode=SimpleNamespace(deny_all="deny_all"),
            AsyncCodex=FakeAsyncCodex,
            Sandbox=SimpleNamespace(read_only="read_only"),
        )

    async def test_sdk_lifecycle_uses_non_actuating_options_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, patch.dict(
            sys.modules, {"openai_codex": self.fake_module()}
        ):
            async with CodexLocalBrain(workspace) as brain:
                proposal = await brain.decide("hello", snapshot())

        self.assertIs(proposal.action, TourBrainAction.RESPOND)
        sdk = FakeAsyncCodex.instance
        self.assertEqual(sdk.thread_start_calls, 1)
        self.assertEqual(sdk.threads[0].turn_kwargs["approval_mode"], "deny_all")
        self.assertEqual(sdk.threads[0].turn_kwargs["sandbox"], "read_only")
        self.assertIn("output_schema", sdk.threads[0].turn_kwargs)

    async def test_stuck_run_closes_app_server_without_cancelling_waiter(self) -> None:
        FakeAsyncCodex.first_thread_slow = True
        with tempfile.TemporaryDirectory() as workspace, patch.dict(
            sys.modules, {"openai_codex": self.fake_module()}
        ):
            async with CodexLocalBrain(workspace, timeout_s=0.01) as brain:
                with self.assertRaises(CodexProviderError):
                    await brain.decide("slow", snapshot())
                proposal = await brain.decide("retry", snapshot())

        self.assertIs(proposal.action, TourBrainAction.RESPOND)
        self.assertEqual(FakeAsyncCodex.total_thread_start_calls, 2)
        self.assertEqual(len(FakeAsyncCodex.instances), 2)
        first = FakeAsyncCodex.instances[0]
        self.assertTrue(first.exited)
        self.assertFalse(first.threads[0].handle.run_cancelled)

    async def test_exit_blocks_recovery_from_reopening_app_server(self) -> None:
        FakeAsyncCodex.first_thread_slow = True
        FakeAsyncCodex.block_first_exit = True
        FakeAsyncCodex.first_exit_started = asyncio.Event()
        FakeAsyncCodex.release_first_exit = asyncio.Event()

        with tempfile.TemporaryDirectory() as workspace, patch.dict(
            sys.modules, {"openai_codex": self.fake_module()}
        ):
            brain = CodexLocalBrain(workspace, timeout_s=0.01)
            await brain.__aenter__()
            decision = asyncio.create_task(brain.decide("slow", snapshot()))
            await asyncio.wait_for(FakeAsyncCodex.first_exit_started.wait(), 1.0)
            exit_task = asyncio.create_task(brain.__aexit__(None, None, None))
            await asyncio.sleep(0)
            self.assertTrue(brain._closing)
            FakeAsyncCodex.release_first_exit.set()
            results = await asyncio.gather(
                decision, exit_task, return_exceptions=True
            )

        self.assertIsInstance(results[0], CodexProviderError)
        self.assertIsNone(results[1])
        self.assertEqual(len(FakeAsyncCodex.instances), 1)
        self.assertTrue(FakeAsyncCodex.instances[0].exited)
        self.assertIsNone(brain._codex)
        self.assertFalse(brain._entered)


if __name__ == "__main__":
    unittest.main()

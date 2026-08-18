from __future__ import annotations

import asyncio
import unittest

from longship.audio import (
    MockVoiceInput,
    VoiceInputEvent,
    VoiceInputEventType,
    WakeDictationController,
    WakeDictationState,
)


def event(
    event_type: VoiceInputEventType,
    session_id: str,
    text: str | None = None,
    *,
    timestamp: float = 1.0,
    confidence: float | None = None,
) -> VoiceInputEvent:
    return VoiceInputEvent(event_type, session_id, timestamp, text, confidence)


class RecordingRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.brain_calls: list[str] = []
        self.stop_seen = asyncio.Event()

    async def handle_text(self, text: str, *, partial: bool = False) -> str:
        self.calls.append((text, partial))
        normalized = text.casefold()
        if partial and ("stop" in normalized or "停下" in normalized):
            self.stop_seen.set()
            return "stopped"
        if partial:
            return "ignored"
        self.brain_calls.append(text)
        return "brain"


class SlowBrainRuntime(RecordingRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.brain_started = asyncio.Event()
        self.release_brain = asyncio.Event()
        self.brain_cancelled = asyncio.Event()

    async def handle_text(self, text: str, *, partial: bool = False) -> str:
        self.calls.append((text, partial))
        if partial:
            if "stop" in text.casefold() or "停下" in text:
                self.stop_seen.set()
                return "stopped"
            return "ignored"
        self.brain_calls.append(text)
        self.brain_started.set()
        try:
            await self.release_brain.wait()
        except asyncio.CancelledError:
            self.brain_cancelled.set()
            raise
        return "brain"


class FixedControlRuntime(SlowBrainRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.pause_seen = asyncio.Event()

    async def handle_text(self, text: str, *, partial: bool = False) -> str:
        if not partial and text.casefold() == "pause":
            self.calls.append((text, partial))
            self.pause_seen.set()
            return "paused"
        return await super().handle_text(text, partial=partial)


class CloseTrackingVoiceInput(MockVoiceInput):
    def __init__(self, events: list[VoiceInputEvent]) -> None:
        super().__init__(events)
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        await super().aclose()


class BlockingRuntime:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started = 0
        self.cancelled = 0

    async def handle_text(self, text: str, *, partial: bool = False) -> str:
        self.started += 1
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return "done"


class SelectiveBlockingRuntime:
    def __init__(self) -> None:
        self.release_slow = asyncio.Event()
        self.slow_cancelled = asyncio.Event()

    async def handle_text(self, text: str, *, partial: bool = False) -> str:
        if text != "slow":
            return "done"
        try:
            await self.release_slow.wait()
        except asyncio.CancelledError:
            self.slow_cancelled.set()
            raise
        return "done"


class VoiceInputEventTests(unittest.TestCase):
    def test_event_validation(self) -> None:
        valid = event(VoiceInputEventType.FINAL, "session-1", "hello", confidence=0.9)
        self.assertEqual(valid.text, "hello")
        with self.assertRaises(ValueError):
            event(VoiceInputEventType.PARTIAL, "session-1")
        with self.assertRaises(ValueError):
            event(VoiceInputEventType.WAKE, "", timestamp=1.0)
        with self.assertRaises(ValueError):
            event(VoiceInputEventType.WAKE, "session-1", confidence=1.1)
        with self.assertRaises(TypeError):
            VoiceInputEvent("wake", "session-1", 1.0)  # type: ignore[arg-type]

    def test_pending_limits_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            WakeDictationController(
                MockVoiceInput(), RecordingRuntime(), max_pending_normal_calls=0
            )


class WakeDictationControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_mock_replays_events_in_order(self) -> None:
        events = [
            event(VoiceInputEventType.WAKE, "s1", timestamp=1.0),
            event(VoiceInputEventType.FINAL, "s1", "Jackie, hello", timestamp=2.0),
        ]
        source = MockVoiceInput(events)
        observed = [item async for item in source]
        self.assertEqual(observed, events)
        self.assertEqual(source.remaining, 0)

    async def test_wake_accepts_final_and_strips_only_leading_phrase(self) -> None:
        runtime = RecordingRuntime()
        controller = WakeDictationController(MockVoiceInput(), runtime)

        await controller.handle_event(event(VoiceInputEventType.WAKE, "s1"))
        await controller.handle_event(
            event(VoiceInputEventType.FINAL, "s1", "  Hey Jackie，带我去展厅")
        )
        await controller.wait_idle()

        self.assertEqual(runtime.brain_calls, ["带我去展厅"])
        self.assertEqual(controller.state, WakeDictationState.ARMED)
        self.assertIsNone(controller.active_session_id)
        await controller.aclose()

    async def test_custom_wake_phrase_and_callback_injection(self) -> None:
        calls: list[tuple[str, bool]] = []

        async def handle_text(text: str, *, partial: bool = False) -> str:
            calls.append((text, partial))
            return "ok"

        controller = WakeDictationController(
            MockVoiceInput(), handle_text, wake_phrases=("Hello Jackie",)
        )
        await controller.handle_event(event(VoiceInputEventType.WAKE, "s1"))
        await controller.handle_event(
            event(VoiceInputEventType.FINAL, "s1", "Hello Jackie: status")
        )
        await controller.wait_idle()
        self.assertEqual(calls, [("status", False)])
        await controller.aclose()

    async def test_does_not_strip_nonleading_or_ascii_prefix(self) -> None:
        runtime = RecordingRuntime()
        controller = WakeDictationController(MockVoiceInput(), runtime)
        self.assertEqual(
            controller.strip_leading_wake_phrase("please ask Jackie to start"),
            "please ask Jackie to start",
        )
        self.assertEqual(
            controller.strip_leading_wake_phrase("Jackiechan should stay"),
            "Jackiechan should stay",
        )
        self.assertEqual(
            controller.strip_leading_wake_phrase("你好杰基带我参观"),
            "带我参观",
        )
        await controller.aclose()

    async def test_partial_never_enters_brain(self) -> None:
        runtime = RecordingRuntime()
        controller = WakeDictationController(MockVoiceInput(), runtime)
        await controller.handle_event(
            event(VoiceInputEventType.PARTIAL, "ambient", "tell me about this")
        )
        await controller.wait_idle()
        self.assertEqual(runtime.calls, [("tell me about this", True)])
        self.assertEqual(runtime.brain_calls, [])
        await controller.aclose()

    async def test_unawakened_final_uses_safety_only_path(self) -> None:
        runtime = RecordingRuntime()
        controller = WakeDictationController(MockVoiceInput(), runtime)
        await controller.handle_event(
            event(VoiceInputEventType.FINAL, "ambient", "What is this?")
        )
        await controller.wait_idle()
        self.assertEqual(runtime.calls, [("What is this?", True)])
        self.assertEqual(runtime.brain_calls, [])
        await controller.aclose()

    async def test_stop_works_without_wake_for_partial_and_final(self) -> None:
        runtime = RecordingRuntime()
        controller = WakeDictationController(MockVoiceInput(), runtime)
        await controller.handle_event(
            event(VoiceInputEventType.PARTIAL, "ambient", "stop now")
        )
        await controller.handle_event(
            event(VoiceInputEventType.FINAL, "ambient", "停下")
        )
        await controller.wait_idle()
        self.assertEqual(runtime.calls, [("stop now", True), ("停下", True)])
        self.assertEqual(runtime.brain_calls, [])
        self.assertTrue(runtime.stop_seen.is_set())
        await controller.aclose()

    async def test_repeated_wake_fences_stale_session_events(self) -> None:
        runtime = RecordingRuntime()
        controller = WakeDictationController(MockVoiceInput(), runtime)
        await controller.handle_event(
            event(VoiceInputEventType.WAKE, "old", timestamp=1.0)
        )
        await controller.handle_event(
            event(VoiceInputEventType.WAKE, "new", timestamp=2.0)
        )
        await controller.handle_event(
            event(VoiceInputEventType.WAKE, "old", timestamp=1.0)
        )
        await controller.handle_event(
            event(VoiceInputEventType.TIMEOUT, "old", timestamp=3.0)
        )
        self.assertEqual(controller.state, WakeDictationState.LISTENING)
        self.assertEqual(controller.active_session_id, "new")

        await controller.handle_event(
            event(VoiceInputEventType.FINAL, "old", "old request", timestamp=3.0)
        )
        await controller.handle_event(
            event(
                VoiceInputEventType.FINAL,
                "new",
                "Jackie, new request",
                timestamp=4.0,
            )
        )
        await controller.wait_idle()
        self.assertEqual(runtime.brain_calls, ["new request"])
        self.assertIn(("old request", True), runtime.calls)
        await controller.aclose()

    async def test_delayed_wake_cannot_reopen_after_newer_final(self) -> None:
        runtime = RecordingRuntime()
        controller = WakeDictationController(MockVoiceInput(), runtime)
        await controller.handle_event(
            event(VoiceInputEventType.WAKE, "current", timestamp=1.0)
        )
        await controller.handle_event(
            event(
                VoiceInputEventType.FINAL,
                "current",
                "Jackie, current request",
                timestamp=10.0,
            )
        )
        await controller.handle_event(
            event(VoiceInputEventType.WAKE, "delayed", timestamp=2.0)
        )
        self.assertEqual(controller.state, WakeDictationState.ARMED)
        await controller.handle_event(
            event(
                VoiceInputEventType.FINAL,
                "delayed",
                "delayed request",
                timestamp=3.0,
            )
        )
        await controller.wait_idle()

        self.assertEqual(runtime.brain_calls, ["current request"])
        self.assertIn(("delayed request", True), runtime.calls)
        await controller.aclose()

    async def test_timeout_and_error_return_matching_session_to_armed(self) -> None:
        runtime = RecordingRuntime()
        controller = WakeDictationController(MockVoiceInput(), runtime)
        for index, terminal in enumerate(
            (VoiceInputEventType.TIMEOUT, VoiceInputEventType.ERROR)
        ):
            wake_time = float(index * 2 + 1)
            await controller.handle_event(
                event(VoiceInputEventType.WAKE, f"s{index}", timestamp=wake_time)
            )
            await controller.handle_event(
                event(
                    terminal,
                    f"s{index}",
                    "diagnostic",
                    timestamp=wake_time + 1.0,
                )
            )
            self.assertEqual(controller.state, WakeDictationState.ARMED)
            self.assertIsNone(controller.active_session_id)
        await controller.aclose()

    async def test_stop_overtakes_a_slow_brain(self) -> None:
        runtime = SlowBrainRuntime()
        controller = WakeDictationController(MockVoiceInput(), runtime)
        await controller.handle_event(event(VoiceInputEventType.WAKE, "s1"))
        await controller.handle_event(
            event(VoiceInputEventType.FINAL, "s1", "Jackie, explain this exhibit")
        )
        await asyncio.wait_for(runtime.brain_started.wait(), timeout=0.2)

        await controller.handle_event(
            event(VoiceInputEventType.PARTIAL, "ambient", "stop")
        )
        await asyncio.wait_for(runtime.stop_seen.wait(), timeout=0.2)
        self.assertFalse(runtime.release_brain.is_set())

        runtime.release_brain.set()
        self.assertTrue(await controller.wait_idle(timeout_s=0.2))
        self.assertEqual(runtime.brain_calls, ["explain this exhibit"])
        await controller.aclose()

    async def test_fixed_control_overtakes_a_slow_brain(self) -> None:
        runtime = FixedControlRuntime()
        controller = WakeDictationController(MockVoiceInput(), runtime)
        await controller.handle_event(
            event(VoiceInputEventType.WAKE, "question", timestamp=1.0)
        )
        await controller.handle_event(
            event(
                VoiceInputEventType.FINAL,
                "question",
                "Jackie, explain this exhibit",
                timestamp=2.0,
            )
        )
        await asyncio.wait_for(runtime.brain_started.wait(), timeout=0.2)

        await controller.handle_event(
            event(VoiceInputEventType.WAKE, "control", timestamp=3.0)
        )
        await controller.handle_event(
            event(VoiceInputEventType.FINAL, "control", "pause", timestamp=4.0)
        )
        await asyncio.wait_for(runtime.pause_seen.wait(), timeout=0.2)
        self.assertFalse(runtime.release_brain.is_set())

        runtime.release_brain.set()
        self.assertTrue(await controller.wait_idle(timeout_s=0.2))
        await controller.aclose()

    async def test_run_replays_source_and_closes_without_tasks(self) -> None:
        runtime = RecordingRuntime()
        source = CloseTrackingVoiceInput(
            [
                event(VoiceInputEventType.WAKE, "s1", timestamp=1.0),
                event(
                    VoiceInputEventType.FINAL,
                    "s1",
                    "Jackie, hello",
                    timestamp=2.0,
                ),
            ]
        )
        controller = WakeDictationController(source, runtime)
        await controller.run()
        self.assertEqual(runtime.brain_calls, ["hello"])
        self.assertEqual(controller.state, WakeDictationState.CLOSED)
        self.assertEqual(controller.pending_count, 0)
        self.assertEqual(source.close_calls, 1)

    async def test_start_then_immediate_close_has_no_runner_error(self) -> None:
        controller = WakeDictationController(MockVoiceInput(), RecordingRuntime())
        runner = controller.start()
        await controller.aclose()
        await asyncio.gather(runner, return_exceptions=True)
        await asyncio.sleep(0)

        self.assertTrue(runner.done())
        self.assertEqual(controller.task_errors, ())

    async def test_pending_runtime_calls_are_bounded_and_newest_wins(self) -> None:
        runtime = BlockingRuntime()
        controller = WakeDictationController(
            MockVoiceInput(),
            runtime,
            max_pending_normal_calls=2,
            max_pending_safety_calls=2,
        )

        for index in range(10):
            await controller.handle_event(
                event(
                    VoiceInputEventType.PARTIAL,
                    "ambient",
                    f"partial {index}",
                    timestamp=float(index + 1),
                )
            )
            self.assertLessEqual(controller.pending_count, 2)

        for index in range(10):
            wake_time = float(index * 2 + 11)
            await controller.handle_event(
                event(
                    VoiceInputEventType.WAKE,
                    f"normal-{index}",
                    timestamp=wake_time,
                )
            )
            await controller.handle_event(
                event(
                    VoiceInputEventType.FINAL,
                    f"normal-{index}",
                    f"request {index}",
                    timestamp=wake_time + 1.0,
                )
            )
            self.assertLessEqual(controller.pending_count, 4)

        self.assertEqual(controller.superseded_count, 16)
        runtime.release.set()
        self.assertTrue(await controller.wait_idle(timeout_s=0.2))
        self.assertEqual(runtime.started, 20)
        self.assertEqual(runtime.cancelled, 16)
        await controller.aclose()

    async def test_completed_out_of_order_call_does_not_consume_capacity(self) -> None:
        runtime = SelectiveBlockingRuntime()
        controller = WakeDictationController(
            MockVoiceInput(), runtime, max_pending_safety_calls=2
        )
        await controller.handle_event(
            event(VoiceInputEventType.PARTIAL, "ambient", "slow", timestamp=1.0)
        )
        await controller.handle_event(
            event(VoiceInputEventType.PARTIAL, "ambient", "quick", timestamp=2.0)
        )
        await controller.handle_event(
            event(VoiceInputEventType.PARTIAL, "ambient", "new", timestamp=3.0)
        )

        self.assertEqual(controller.superseded_count, 0)
        self.assertFalse(runtime.slow_cancelled.is_set())
        runtime.release_slow.set()
        self.assertTrue(await controller.wait_idle(timeout_s=0.2))
        await controller.aclose()

    async def test_close_is_bounded_and_cancels_slow_brain(self) -> None:
        runtime = SlowBrainRuntime()
        controller = WakeDictationController(
            MockVoiceInput(), runtime, close_timeout_s=0.05
        )
        await controller.handle_event(event(VoiceInputEventType.WAKE, "s1"))
        await controller.handle_event(
            event(VoiceInputEventType.FINAL, "s1", "Jackie, wait forever")
        )
        await asyncio.wait_for(runtime.brain_started.wait(), timeout=0.2)

        await asyncio.wait_for(controller.aclose(), timeout=0.2)
        await asyncio.wait_for(runtime.brain_cancelled.wait(), timeout=0.2)
        self.assertEqual(controller.pending_count, 0)
        self.assertEqual(controller.state, WakeDictationState.CLOSED)


if __name__ == "__main__":
    unittest.main()

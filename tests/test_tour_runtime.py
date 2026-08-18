from __future__ import annotations

import asyncio
import time
import unittest

from longship.brain.base import TourBrainAction, TourBrainProposal
from longship.navigation.mock import MockNavigation
from longship.navigation.base import (
    NavigationAuthority,
    NavigationRequest,
    NavigationResult,
    NavigationStopRequest,
    StopResult,
)
from longship.tour.models import TourPlan, TourState
from longship.tour.ports import RecordingSpeaker
from longship.tour.runtime import VoiceTourRuntime


def make_plan() -> TourPlan:
    return TourPlan.from_mapping(
        {
            "schema_version": "longship.voice-tour.v0",
            "tour_id": "public-demo",
            "title": "Longship public demo",
            "locale": "en-US",
            "map_id": "mock-map",
            "map_version": "v1",
            "route_id": "public-route",
            "stops": [
                {
                    "stop_id": "entrance",
                    "waypoint_id": "wp-entrance",
                    "title": "Entrance",
                    "travel_announcement": "We are moving to the entrance.",
                    "narration": "This is the entrance.",
                },
                {
                    "stop_id": "workshop",
                    "waypoint_id": "wp-workshop",
                    "title": "Workshop",
                    "travel_announcement": "Next is the workshop.",
                    "narration": "This is the workshop.",
                },
            ],
        }
    )


async def wait_for_state(runtime: VoiceTourRuntime, state: TourState) -> None:
    for _ in range(200):
        if runtime.state is state:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"runtime did not reach {state}; current={runtime.state}")


class SpyBrain:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, text, snapshot):
        self.calls += 1
        return TourBrainProposal(TourBrainAction.RESPOND, "brain response")


class HangingBrain:
    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def decide(self, text, snapshot):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return TourBrainProposal(TourBrainAction.START_TOUR, "late start")


class UnverifiedStopNavigation(MockNavigation):
    async def stop(self, request: NavigationStopRequest) -> StopResult:
        await super().stop(request)
        return StopResult(
            request.request_id,
            request.revoke_through_epoch,
            requested=True,
            verified_stopped=False,
            evidence="",
            detail="no fresh velocity observation",
        )


class StubbornStopNavigation(MockNavigation):
    async def stop(self, request: NavigationStopRequest) -> StopResult:
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)
        return StopResult(
            request.request_id,
            request.revoke_through_epoch,
            True,
            True,
            "late.mock.stop",
            "",
        )


class MalformedStopNavigation(MockNavigation):
    async def stop(self, request: NavigationStopRequest) -> StopResult:
        return StopResult(
            request.request_id,
            request.revoke_through_epoch,
            True,
            True,
            None,  # type: ignore[arg-type]
            "",
        )


class WrongWaypointNavigation(MockNavigation):
    async def navigate_to(
        self, request: NavigationRequest, authority: NavigationAuthority
    ) -> NavigationResult:
        return NavigationResult(
            True,
            request.request_id,
            request.authority_epoch,
            request.map_id,
            request.map_version,
            request.route_id,
            "different-waypoint",
            "wrong.arrival",
        )


class RaisingNavigation(MockNavigation):
    def __init__(self) -> None:
        super().__init__()
        self.stop_calls = 0

    async def navigate_to(
        self, request: NavigationRequest, authority: NavigationAuthority
    ) -> NavigationResult:
        raise ConnectionError("mock transport loss")

    async def stop(self, request: NavigationStopRequest) -> StopResult:
        self.stop_calls += 1
        return await super().stop(request)


class ResumeFailureNavigation(MockNavigation):
    async def resume(self, authority: NavigationAuthority) -> None:
        raise ConnectionError("resume acknowledgement lost")


class GatedStopNavigation(MockNavigation):
    def __init__(self) -> None:
        super().__init__()
        self.stop_started = asyncio.Event()
        self.release_stop = asyncio.Event()

    async def stop(self, request: NavigationStopRequest) -> StopResult:
        self.stop_started.set()
        await self.release_stop.wait()
        return await super().stop(request)


class LateResumeNavigation(MockNavigation):
    """Deliberately violates the authority contract to exercise fail-closed recovery."""

    def __init__(self) -> None:
        super().__init__(travel_seconds=0.2)
        self.moving = False
        self.stop_calls = 0

    async def resume(self, authority: NavigationAuthority) -> None:
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            await asyncio.sleep(0.04)
        self.moving = True

    async def stop(self, request: NavigationStopRequest) -> StopResult:
        self.stop_calls += 1
        self.moving = False
        return await super().stop(request)


class WrongMapNavigation(MockNavigation):
    async def navigate_to(
        self, request: NavigationRequest, authority: NavigationAuthority
    ) -> NavigationResult:
        return NavigationResult(
            True,
            request.request_id,
            request.authority_epoch,
            "stale-map",
            request.map_version,
            request.route_id,
            request.waypoint_id,
            "stale.arrival",
        )


class InFlightResumeNavigation(MockNavigation):
    """Returns inside the deadline but actuates after its authority is revoked."""

    def __init__(self) -> None:
        super().__init__(travel_seconds=0.2)
        self.resume_started = asyncio.Event()
        self.release_resume = asyncio.Event()
        self.moving = False
        self.stop_calls = 0

    async def resume(self, authority: NavigationAuthority) -> None:
        self.resume_started.set()
        await self.release_resume.wait()
        self.moving = True

    async def stop(self, request: NavigationStopRequest) -> StopResult:
        self.stop_calls += 1
        self.moving = False
        return await super().stop(request)


class CancelledLateNavigateNavigation(MockNavigation):
    def __init__(self) -> None:
        super().__init__(travel_seconds=0.2)
        self.moving = True
        self.stop_calls = 0

    async def navigate_to(
        self, request: NavigationRequest, authority: NavigationAuthority
    ) -> NavigationResult:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.04)
            self.moving = True
            raise

    async def stop(self, request: NavigationStopRequest) -> StopResult:
        self.stop_calls += 1
        self.moving = False
        return await super().stop(request)


class IdempotentStopNavigation(CancelledLateNavigateNavigation):
    """Caches repeated request IDs like an idempotent remote transport."""

    def __init__(self) -> None:
        super().__init__()
        self.stop_request_ids: list[str] = []
        self._stop_cache: dict[str, StopResult] = {}

    async def stop(self, request: NavigationStopRequest) -> StopResult:
        self.stop_request_ids.append(request.request_id)
        cached = self._stop_cache.get(request.request_id)
        if cached is not None:
            return cached
        result = await super().stop(request)
        self._stop_cache[request.request_id] = result
        return result


class CancelUpgradeNavigation(MockNavigation):
    def __init__(self) -> None:
        super().__init__(travel_seconds=0.2)
        self.moving = True
        self.cancel_received = asyncio.Event()
        self.release_late_motion = asyncio.Event()
        self.second_stop_started = asyncio.Event()
        self.release_second_stop = asyncio.Event()
        self.stop_calls = 0

    async def navigate_to(
        self, request: NavigationRequest, authority: NavigationAuthority
    ) -> NavigationResult:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancel_received.set()
            await self.release_late_motion.wait()
            self.moving = True
            raise

    async def stop(self, request: NavigationStopRequest) -> StopResult:
        self.stop_calls += 1
        if self.stop_calls == 2:
            self.second_stop_started.set()
            await self.release_second_stop.wait()
        self.moving = False
        return await super().stop(request)


class VoiceTourRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_partial_asr_routes_only_reserved_stop(self) -> None:
        brain = SpyBrain()
        runtime = VoiceTourRuntime(
            make_plan(), MockNavigation(), RecordingSpeaker(), brain=brain
        )

        ignored = await runtime.handle_text("开始导", partial=True)
        stopped = await runtime.handle_text("请停止", partial=True)

        self.assertIn("ignored", ignored)
        self.assertIn("Protective stop", stopped)
        self.assertEqual(brain.calls, 0)

    async def test_fixed_tour_controls_bypass_brain(self) -> None:
        brain = SpyBrain()
        runtime = VoiceTourRuntime(
            make_plan(),
            MockNavigation(travel_seconds=0.04),
            RecordingSpeaker(),
            brain=brain,
        )

        await runtime.handle_text("开始导览")
        await wait_for_state(runtime, TourState.MOVING)
        await runtime.handle_text("暂停")
        await runtime.handle_text("恢复")
        await wait_for_state(runtime, TourState.WAITING)
        runtime.status_message()
        await runtime.handle_text("下一站")
        await runtime.wait_finished()

        self.assertEqual(brain.calls, 0)
        self.assertIs(runtime.state, TourState.COMPLETED)

    async def test_reserved_stop_bypasses_brain(self) -> None:
        brain = SpyBrain()
        navigation = MockNavigation(travel_seconds=0.2)
        runtime = VoiceTourRuntime(make_plan(), navigation, RecordingSpeaker(), brain=brain)

        await runtime.start()
        await wait_for_state(runtime, TourState.MOVING)
        result = await asyncio.wait_for(runtime.handle_text("请立即停止"), timeout=0.1)

        self.assertIn("Protective stop", result)
        self.assertEqual(brain.calls, 0)
        self.assertIs(runtime.state, TourState.SAFE_STOPPED)
        self.assertTrue(any(event[0] == "navigation.stopped" for event in navigation.events))

    async def test_stop_remains_live_while_brain_hangs(self) -> None:
        brain = HangingBrain()
        runtime = VoiceTourRuntime(
            make_plan(), MockNavigation(), RecordingSpeaker(), brain=brain, brain_timeout_s=30
        )
        brain_task = asyncio.create_task(runtime.handle_text("Could you explain the design?"))
        await brain.started.wait()

        await asyncio.wait_for(runtime.handle_text("stop"), timeout=0.1)
        self.assertIs(runtime.state, TourState.SAFE_STOPPED)
        self.assertEqual(brain.calls, 1)

        brain_task.cancel()
        await asyncio.gather(brain_task, return_exceptions=True)

    async def test_stop_is_not_claimed_safe_without_target_evidence(self) -> None:
        runtime = VoiceTourRuntime(
            make_plan(), UnverifiedStopNavigation(), RecordingSpeaker()
        )

        result = await runtime.handle_text("停止")

        self.assertIn("unverified", result)
        self.assertIs(runtime.state, TourState.STOP_UNVERIFIED)
        self.assertIn("already stop_unverified", await runtime.start())

    async def test_stop_returns_at_deadline_for_cooperative_async_plugin(self) -> None:
        runtime = VoiceTourRuntime(
            make_plan(),
            StubbornStopNavigation(),
            RecordingSpeaker(),
            control_timeout_s=0.02,
        )

        started = time.monotonic()
        result = await runtime.handle_text("stop")
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.08)
        self.assertIn("unverified", result)
        self.assertIs(runtime.state, TourState.STOP_UNVERIFIED)
        await asyncio.sleep(0.07)  # allow the detached test coroutine to finish

    async def test_malformed_stop_evidence_fails_closed(self) -> None:
        runtime = VoiceTourRuntime(
            make_plan(), MalformedStopNavigation(), RecordingSpeaker()
        )

        result = await runtime.handle_text("停止")

        self.assertIn("unverified", result)
        self.assertIs(runtime.state, TourState.STOP_UNVERIFIED)

    async def test_speech_and_motion_overlap_but_narration_waits_for_arrival(self) -> None:
        navigation = MockNavigation(travel_seconds=0.08)
        speaker = RecordingSpeaker(delay_seconds=0.02)
        runtime = VoiceTourRuntime(make_plan(), navigation, speaker)

        await runtime.start()
        await wait_for_state(runtime, TourState.WAITING)

        nav_started = next(t for kind, _, t in navigation.events if kind == "navigation.started")
        nav_arrived = next(t for kind, _, t in navigation.events if kind == "navigation.arrived")
        travel_started = next(
            t
            for kind, text, t in speaker.events
            if kind == "speech.started" and text == "We are moving to the entrance."
        )
        narration_started = next(
            t
            for kind, text, t in speaker.events
            if kind == "speech.started" and text == "This is the entrance."
        )
        self.assertLessEqual(nav_started, travel_started)
        self.assertLess(travel_started, nav_arrived)
        self.assertGreaterEqual(narration_started, nav_arrived)

        await runtime.continue_tour()
        await runtime.wait_finished()
        self.assertIs(runtime.state, TourState.COMPLETED)

    async def test_navigation_failure_blocks_without_model_recovery(self) -> None:
        brain = SpyBrain()
        navigation = MockNavigation(failing_waypoints={"wp-entrance"})
        runtime = VoiceTourRuntime(make_plan(), navigation, RecordingSpeaker(), brain=brain)

        await runtime.start()
        await runtime.wait_finished()

        self.assertIs(runtime.state, TourState.BLOCKED)
        self.assertEqual(brain.calls, 0)

    async def test_navigation_exception_requests_stop_and_latches_restart(self) -> None:
        navigation = RaisingNavigation()
        runtime = VoiceTourRuntime(make_plan(), navigation, RecordingSpeaker())

        await runtime.start()
        await runtime.wait_finished()

        self.assertEqual(navigation.stop_calls, 1)
        self.assertIs(runtime.state, TourState.SAFE_STOPPED)
        self.assertIn("already safe_stopped", await runtime.start())

    async def test_wrong_waypoint_arrival_is_rejected_before_narration(self) -> None:
        speaker = RecordingSpeaker()
        runtime = VoiceTourRuntime(make_plan(), WrongWaypointNavigation(), speaker)

        await runtime.start()
        await runtime.wait_finished()

        self.assertIs(runtime.state, TourState.SAFE_STOPPED)
        self.assertFalse(
            any(text == "This is the entrance." for _, text, _ in speaker.events)
        )

    async def test_resume_failure_becomes_protective_stop(self) -> None:
        runtime = VoiceTourRuntime(
            make_plan(), ResumeFailureNavigation(travel_seconds=0.2), RecordingSpeaker()
        )

        await runtime.start()
        await wait_for_state(runtime, TourState.MOVING)
        self.assertIn("paused", (await runtime.pause()).lower())
        result = await runtime.resume()

        self.assertIn("Protective stop", result)
        self.assertIs(runtime.state, TourState.SAFE_STOPPED)

    async def test_mock_pause_preserves_remaining_travel(self) -> None:
        navigation = MockNavigation(travel_seconds=0.12)
        runtime = VoiceTourRuntime(make_plan(), navigation, RecordingSpeaker())

        await runtime.start()
        await wait_for_state(runtime, TourState.MOVING)
        await asyncio.sleep(0.02)
        await runtime.pause()
        await asyncio.sleep(0.12)
        resumed_at = time.monotonic()
        await runtime.resume()
        await wait_for_state(runtime, TourState.WAITING)
        arrived_at = next(t for kind, _, t in navigation.events if kind == "navigation.arrived")

        self.assertGreater(arrived_at - resumed_at, 0.05)

    async def test_repeat_cannot_bypass_arrival_gate(self) -> None:
        navigation = MockNavigation(travel_seconds=0.1)
        speaker = RecordingSpeaker()
        runtime = VoiceTourRuntime(make_plan(), navigation, speaker)

        await runtime.start()
        await wait_for_state(runtime, TourState.MOVING)
        result = await runtime.repeat()

        self.assertIn("only after arrival", result)
        self.assertFalse(
            any(text == "This is the entrance." for _, text, _ in speaker.events)
        )

    async def test_stale_brain_decision_has_no_side_effect(self) -> None:
        brain = HangingBrain()
        runtime = VoiceTourRuntime(make_plan(), MockNavigation(), RecordingSpeaker(), brain=brain)
        decision_task = asyncio.create_task(runtime.handle_text("Please begin when ready"))
        await brain.started.wait()
        await runtime.protective_stop("test stop")
        brain.release.set()

        result = await decision_task

        self.assertIn("stale", result)
        self.assertIs(runtime.state, TourState.SAFE_STOPPED)

    async def test_stop_cancels_in_flight_brain_speech(self) -> None:
        speaker = RecordingSpeaker(delay_seconds=0.2)
        runtime = VoiceTourRuntime(
            make_plan(), MockNavigation(), speaker, brain=SpyBrain()
        )
        response_task = asyncio.create_task(runtime.handle_text("Tell me about Longship"))
        for _ in range(100):
            if any(
                kind == "speech.started" and text == "brain response"
                for kind, text, _ in speaker.events
            ):
                break
            await asyncio.sleep(0.002)

        await runtime.handle_text("stop")
        await asyncio.gather(response_task, return_exceptions=True)
        await asyncio.sleep(0.22)

        self.assertFalse(
            any(
                kind == "speech.finished" and text == "brain response"
                for kind, text, _ in speaker.events
            )
        )

    async def test_caller_cancellation_cannot_abort_protective_stop(self) -> None:
        navigation = GatedStopNavigation()
        runtime = VoiceTourRuntime(
            make_plan(), navigation, RecordingSpeaker(), control_timeout_s=0.2
        )

        caller = asyncio.create_task(runtime.protective_stop("test"))
        await navigation.stop_started.wait()
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        navigation.release_stop.set()
        for _ in range(100):
            if runtime.state is TourState.SAFE_STOPPED:
                break
            await asyncio.sleep(0.005)

        self.assertIs(runtime.state, TourState.SAFE_STOPPED)
        self.assertTrue(any(kind == "navigation.stopped" for kind, _, _ in navigation.events))

    async def test_late_resume_is_re_stopped_and_never_claimed_safe_while_pending(self) -> None:
        navigation = LateResumeNavigation()
        runtime = VoiceTourRuntime(
            make_plan(), navigation, RecordingSpeaker(), control_timeout_s=0.01
        )

        await runtime.start()
        await wait_for_state(runtime, TourState.MOVING)
        await runtime.pause()
        first_result = await runtime.resume()

        self.assertIn("unverified", first_result)
        self.assertIs(runtime.state, TourState.STOP_UNVERIFIED)
        for _ in range(100):
            if navigation.stop_calls >= 2 and runtime.state is TourState.SAFE_STOPPED:
                break
            await asyncio.sleep(0.005)

        self.assertGreaterEqual(navigation.stop_calls, 2)
        self.assertFalse(navigation.moving)
        self.assertIs(runtime.state, TourState.SAFE_STOPPED)

    async def test_arrival_must_echo_map_and_request_identity(self) -> None:
        runtime = VoiceTourRuntime(
            make_plan(), WrongMapNavigation(), RecordingSpeaker()
        )

        await runtime.start()
        await runtime.wait_finished()

        self.assertIs(runtime.state, TourState.SAFE_STOPPED)

    async def test_inflight_resume_cannot_resurrect_motion_after_stop(self) -> None:
        navigation = InFlightResumeNavigation()
        runtime = VoiceTourRuntime(
            make_plan(), navigation, RecordingSpeaker(), control_timeout_s=0.2
        )

        await runtime.start()
        await wait_for_state(runtime, TourState.MOVING)
        await runtime.pause()
        resume_task = asyncio.create_task(runtime.resume())
        await navigation.resume_started.wait()

        first_stop = await runtime.protective_stop("concurrent stop")
        self.assertIn("unverified", first_stop)
        self.assertIs(runtime.state, TourState.STOP_UNVERIFIED)
        navigation.release_resume.set()
        await resume_task
        for _ in range(100):
            if navigation.stop_calls >= 2 and runtime.state is TourState.SAFE_STOPPED:
                break
            await asyncio.sleep(0.005)

        self.assertGreaterEqual(navigation.stop_calls, 2)
        self.assertFalse(navigation.moving)
        self.assertIs(runtime.state, TourState.SAFE_STOPPED)

    async def test_cancelled_motion_task_is_stopped_again_after_late_action(self) -> None:
        navigation = IdempotentStopNavigation()
        runtime = VoiceTourRuntime(
            make_plan(), navigation, RecordingSpeaker(), control_timeout_s=0.1
        )

        await runtime.start()
        await wait_for_state(runtime, TourState.MOVING)
        result = await runtime.protective_stop("test cancelled late action")

        self.assertIn("verified", result)
        self.assertGreaterEqual(navigation.stop_calls, 2)
        self.assertEqual(len(navigation.stop_request_ids), 2)
        self.assertEqual(len(set(navigation.stop_request_ids)), 2)
        self.assertFalse(navigation.moving)
        self.assertIs(runtime.state, TourState.SAFE_STOPPED)

    async def test_cancelling_wait_observer_does_not_cancel_tour(self) -> None:
        navigation = MockNavigation(travel_seconds=0.2)
        runtime = VoiceTourRuntime(make_plan(), navigation, RecordingSpeaker())
        await runtime.start()
        await wait_for_state(runtime, TourState.MOVING)

        observer = asyncio.create_task(runtime.wait_finished())
        await asyncio.sleep(0)
        observer.cancel()
        await asyncio.gather(observer, return_exceptions=True)

        self.assertIs(runtime.state, TourState.MOVING)
        self.assertIsNotNone(runtime._run_task)
        self.assertFalse(runtime._run_task.done())
        await runtime.protective_stop("test cleanup")

    async def test_cancelling_resume_caller_launches_owned_fail_stop(self) -> None:
        navigation = LateResumeNavigation()
        runtime = VoiceTourRuntime(
            make_plan(), navigation, RecordingSpeaker(), control_timeout_s=0.2
        )
        await runtime.start()
        await wait_for_state(runtime, TourState.MOVING)
        await runtime.pause()

        caller = asyncio.create_task(runtime.resume())
        await asyncio.sleep(0.005)
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        for _ in range(100):
            if runtime.state is TourState.SAFE_STOPPED:
                break
            await asyncio.sleep(0.005)

        self.assertIs(runtime.state, TourState.SAFE_STOPPED)
        self.assertFalse(navigation.moving)
        self.assertGreaterEqual(navigation.stop_calls, 2)

    async def test_protective_stop_serially_upgrades_inflight_cancel(self) -> None:
        navigation = CancelUpgradeNavigation()
        runtime = VoiceTourRuntime(
            make_plan(), navigation, RecordingSpeaker(), control_timeout_s=0.2
        )
        await runtime.start()
        await wait_for_state(runtime, TourState.MOVING)

        cancel_task = asyncio.create_task(runtime.cancel())
        await navigation.cancel_received.wait()
        protective_task = asyncio.create_task(
            runtime.protective_stop("upgrade cancel")
        )
        navigation.release_late_motion.set()
        await navigation.second_stop_started.wait()

        self.assertTrue(navigation.moving)
        self.assertIs(runtime.state, TourState.STOPPING)
        navigation.release_second_stop.set()
        await asyncio.gather(cancel_task, protective_task)

        self.assertGreaterEqual(navigation.stop_calls, 3)
        self.assertFalse(navigation.moving)
        self.assertIs(runtime.state, TourState.SAFE_STOPPED)


if __name__ == "__main__":
    unittest.main()

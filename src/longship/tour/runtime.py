from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from longship.brain.base import TourBrainAction, TourBrainPort, TourBrainProposal
from longship.navigation.base import (
    NavigationAuthority,
    NavigationPort,
    NavigationRequest,
    NavigationResult,
    NavigationStopRequest,
    StopResult,
)

from .interaction import CommandKind, InteractionRouter
from .models import TourPlan, TourSnapshot, TourState
from .ports import SpeakerPort

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TourRuntimeEvent:
    event_type: str
    revision: int
    state: TourState
    detail: str
    monotonic_s: float


class VoiceTourRuntime:
    """Deterministic, experimental mock-only V0 tour state machine.

    Fixed controls own state. The optional Brain can return only a small
    high-level proposal, which is rejected if runtime revision changes while it
    is thinking. Event-loop-cooperative async calls use bounded local waits;
    late motion tasks are fenced by a revoked authority epoch and re-stopped.
    Blocking transports require process/thread isolation and their own
    target-side deadlines.
    """

    def __init__(
        self,
        plan: TourPlan,
        navigation: NavigationPort,
        speaker: SpeakerPort,
        *,
        brain: TourBrainPort | None = None,
        brain_timeout_s: float = 10.0,
        control_timeout_s: float = 0.5,
        router: InteractionRouter | None = None,
    ) -> None:
        if brain_timeout_s <= 0 or control_timeout_s <= 0:
            raise ValueError("timeouts must be positive")
        self.plan = plan
        self.navigation = navigation
        self.speaker = speaker
        self.brain = brain
        self.brain_timeout_s = brain_timeout_s
        self.control_timeout_s = control_timeout_s
        self.router = router or InteractionRouter()
        self.state = TourState.IDLE
        self.revision = 0
        self.current_stop_index: int | None = None
        self.events: list[TourRuntimeEvent] = []
        self._run_task: asyncio.Task[None] | None = None
        self._run_generation = 0
        self._request_sequence = 0
        self._authority_epoch = 0
        self._motion_authority: NavigationAuthority | None = None
        self._cancel_event = asyncio.Event()
        self._continue_event = asyncio.Event()
        self._pause_gate = asyncio.Event()
        self._pause_gate.set()
        self._state_before_pause = TourState.PREPARING
        self._command_lock = asyncio.Lock()
        self._protective_stop_requested = False
        self._speech_epoch = 0
        self._speaker_lock = asyncio.Lock()
        self._speech_tasks: set[asyncio.Task[Any]] = set()
        self._detached_tasks: set[asyncio.Task[Any]] = set()
        self._detached_motion_tasks: set[asyncio.Task[Any]] = set()
        self._motion_tasks: set[asyncio.Task[Any]] = set()
        self._completed_motion_after_revoke: set[asyncio.Task[Any]] = set()
        self._active_stop_sequences = 0
        self._restop_scheduled = False
        self._owned_control_tasks: set[asyncio.Task[Any]] = set()
        self._protective_stop_task: asyncio.Task[str] | None = None
        self._cancel_task: asyncio.Task[str] | None = None

    def snapshot(self) -> TourSnapshot:
        stop = (
            self.plan.stops[self.current_stop_index]
            if self.current_stop_index is not None
            and 0 <= self.current_stop_index < len(self.plan.stops)
            else None
        )
        return TourSnapshot(
            tour_id=self.plan.tour_id,
            state=self.state,
            revision=self.revision,
            current_stop_index=self.current_stop_index,
            current_stop_id=stop.stop_id if stop else None,
            current_waypoint_id=stop.waypoint_id if stop else None,
            total_stops=len(self.plan.stops),
        )

    async def handle_text(self, text: str, *, partial: bool = False) -> str:
        command = self.router.route(text, partial=partial)
        if command.kind is CommandKind.STOP:
            return await self.protective_stop("reserved voice or keyboard command")
        if partial:
            return "Partial input ignored."
        if command.kind is CommandKind.START:
            return await self.start()
        if command.kind is CommandKind.PAUSE:
            return await self.pause()
        if command.kind is CommandKind.RESUME:
            return await self.resume()
        if command.kind is CommandKind.CONTINUE:
            return await self.continue_tour()
        if command.kind is CommandKind.CANCEL:
            return await self.cancel()
        if command.kind is CommandKind.STATUS:
            return self.status_message()
        if command.kind is CommandKind.REPEAT:
            return await self.repeat()
        return await self._ask_brain(text)

    async def start(self) -> str:
        async with self._command_lock:
            if self.state not in {
                TourState.IDLE,
                TourState.COMPLETED,
                TourState.CANCELLED,
            }:
                return f"Tour is already {self.state.value}."
            self._run_generation += 1
            generation = self._run_generation
            self._authority_epoch += 1
            self._motion_authority = NavigationAuthority(self._authority_epoch)
            self._cancel_event = asyncio.Event()
            self._continue_event = asyncio.Event()
            self._pause_gate.set()
            self._protective_stop_requested = False
            self.current_stop_index = None
            self._set_state(TourState.PREPARING, "tour accepted")
            self._run_task = asyncio.create_task(
                self._run(generation), name=f"tour:{self.plan.tour_id}:{generation}"
            )
        return f"Starting {self.plan.title}."

    async def pause(self) -> str:
        async with self._command_lock:
            if self.state not in {
                TourState.PREPARING,
                TourState.MOVING,
                TourState.NARRATING,
                TourState.WAITING,
            }:
                return f"Cannot pause while tour is {self.state.value}."
            self._state_before_pause = self.state
            self._pause_gate.clear()
            authority = self._require_motion_authority()
            self._set_state(TourState.PAUSING, "deterministic pause requested")
        try:
            completed, _ = await self._bounded(
                lambda: self.navigation.pause(authority),
                self.control_timeout_s,
                "navigation.pause",
            )
        except asyncio.CancelledError:
            self._launch_fail_stop("pause caller was cancelled")
            raise
        if not completed:
            return await self.protective_stop("navigation pause failed or timed out")
        async with self._command_lock:
            if self.state is not TourState.PAUSING:
                return "Pause was superseded by a higher-priority control."
            self._set_state(TourState.PAUSED, "navigation pause acknowledged")
        return "Tour paused."

    async def resume(self) -> str:
        async with self._command_lock:
            if self.state is not TourState.PAUSED:
                return f"Cannot resume while tour is {self.state.value}."
            authority = self._require_motion_authority()
            self._set_state(TourState.RESUMING, "deterministic resume requested")
        try:
            completed, _ = await self._bounded(
                lambda: self.navigation.resume(authority),
                self.control_timeout_s,
                "navigation.resume",
                motion_authority=authority,
            )
        except asyncio.CancelledError:
            self._launch_fail_stop("resume caller was cancelled")
            raise
        if not completed:
            # An acknowledgement may have been lost after motion resumed.
            return await self.protective_stop("navigation resume failed or timed out")
        async with self._command_lock:
            if self.state is not TourState.RESUMING:
                return "Resume was superseded by a higher-priority control."
            self._set_state(self._state_before_pause, "navigation resume acknowledged")
            self._pause_gate.set()
        return "Tour resumed."

    async def continue_tour(self) -> str:
        async with self._command_lock:
            if self.state is not TourState.WAITING:
                return f"No continuation is pending while tour is {self.state.value}."
            self._continue_event.set()
            self._emit("control.continue", "operator confirmation")
        return "Continuing to the next stop."

    async def repeat(self) -> str:
        repeatable = self.state is TourState.WAITING or (
            self.state is TourState.PAUSED and self._state_before_pause is TourState.WAITING
        )
        if self.current_stop_index is None or not repeatable:
            return "Narration can be repeated only after arrival."
        narration = self.plan.stops[self.current_stop_index].narration
        await self._say(narration)
        self._emit("speech.repeated", self.plan.stops[self.current_stop_index].stop_id)
        return "Narration repeated."

    async def cancel(self) -> str:
        async with self._command_lock:
            if self._protective_stop_requested:
                task = self._protective_stop_task
                if task is None:
                    return "Protective stop is already latched."
            elif self._cancel_task is not None and not self._cancel_task.done():
                task = self._cancel_task
            else:
                self._cancel_event.set()
                self._run_generation += 1
                self._revoke_motion_authority_locked()
                request = self._new_stop_request_locked("tour cancelled")
                self._set_state(TourState.STOPPING, "tour cancelled")
                owned_run = (
                    None
                    if self._run_task is asyncio.current_task()
                    else self._run_task
                )
                task = asyncio.create_task(
                    self._execute_stop_sequence(
                        request,
                        owned_run,
                        final_state=TourState.CANCELLED,
                        protective=False,
                    ),
                    name=f"tour-cancel:{request.request_id}",
                )
                self._cancel_task = task
                self._retain_control_task(task)
        return await asyncio.shield(task)

    async def protective_stop(self, reason: str) -> str:
        """Launch a caller-cancellation-resistant, coalesced target stop."""

        return await self._start_protective_stop(reason, force=False)

    async def _start_protective_stop(self, reason: str, *, force: bool) -> str:
        async with self._command_lock:
            current = self._protective_stop_task
            if current is not None and not current.done():
                task = current
            elif self._cancel_task is not None and not self._cancel_task.done():
                self._protective_stop_requested = True
                self._cancel_event.set()
                self._run_generation += 1
                self._revoke_motion_authority_locked()
                self._set_state(TourState.STOPPING, reason)
                task = asyncio.create_task(
                    self._upgrade_cancel_to_protective_stop(
                        self._cancel_task, reason
                    ),
                    name="cancel-to-protective-stop",
                )
                self._protective_stop_task = task
                self._retain_control_task(task)
            elif (
                not force
                and self._protective_stop_requested
                and self.state in {TourState.SAFE_STOPPED, TourState.STOP_UNVERIFIED}
            ):
                verification = (
                    "verified by target evidence"
                    if self.state is TourState.SAFE_STOPPED
                    else "unverified"
                )
                return (
                    f"Protective stop remains latched; target stop is {verification}. "
                    "Physical E-stop remains independent."
                )
            else:
                self._protective_stop_requested = True
                self._cancel_event.set()
                self._run_generation += 1
                self._revoke_motion_authority_locked()
                request = self._new_stop_request_locked(reason)
                self._set_state(TourState.STOPPING, reason)
                owned_run = (
                    None
                    if self._run_task is asyncio.current_task()
                    else self._run_task
                )
                task = asyncio.create_task(
                    self._execute_stop_sequence(
                        request,
                        owned_run,
                        final_state=TourState.SAFE_STOPPED,
                        protective=True,
                    ),
                    name=f"protective-stop:{request.request_id}",
                )
                self._protective_stop_task = task
                self._retain_control_task(task)
        return await asyncio.shield(task)

    async def _upgrade_cancel_to_protective_stop(
        self, cancel_task: asyncio.Task[str], reason: str
    ) -> str:
        try:
            await asyncio.shield(cancel_task)
        except Exception:
            pass
        async with self._command_lock:
            if self._protective_stop_task is asyncio.current_task():
                self._protective_stop_task = None
        return await self._start_protective_stop(reason, force=True)

    async def _execute_stop_sequence(
        self,
        request: NavigationStopRequest,
        owned_run: asyncio.Task[Any] | None,
        *,
        final_state: TourState,
        protective: bool,
    ) -> str:
        self._active_stop_sequences += 1
        try:
            return await self._execute_stop_sequence_inner(
                request,
                owned_run,
                final_state=final_state,
                protective=protective,
            )
        finally:
            self._active_stop_sequences = max(0, self._active_stop_sequences - 1)
            self._schedule_late_motion_restop()

    async def _execute_stop_sequence_inner(
        self,
        request: NavigationStopRequest,
        owned_run: asyncio.Task[Any] | None,
        *,
        final_state: TourState,
        protective: bool,
    ) -> str:
        navigation_stop = asyncio.create_task(self._request_navigation_stop(request))
        # Give the target stop the first event-loop opportunity.
        await asyncio.sleep(0)
        speech_stop = asyncio.create_task(self._stop_all_speech())
        run_cancel = asyncio.create_task(
            self._cancel_owned_task(
                owned_run,
                "tour.stop" if protective else "tour.cancel",
            )
        )
        results = await asyncio.gather(
            navigation_stop, speech_stop, run_cancel, return_exceptions=True
        )
        stop_result = results[0]
        if not isinstance(stop_result, StopResult):
            stop_result = StopResult(
                request.request_id,
                request.revoke_through_epoch,
                True,
                False,
                "",
                "stop sequence failed before typed target evidence",
            )
        # A target stop that raced any revoked motion completion is followed by
        # a second stop after those calls quiesce. This covers normal returns,
        # cancellation suppression, and tasks that actuate before re-raising
        # CancelledError.
        if self._completed_motion_after_revoke and not self._motion_tasks:
            self._completed_motion_after_revoke.clear()
            async with self._command_lock:
                follow_up_request = self._new_stop_request_locked(
                    f"post-motion barrier after {request.request_id}"
                )
            stop_result = await self._request_navigation_stop(follow_up_request)
        unsettled_motion = bool(self._motion_tasks)
        verified = stop_result.verified_stopped and not unsettled_motion
        detail = (
            "a detached motion-producing operation has not quiesced"
            if unsettled_motion
            else stop_result.detail
        )
        async with self._command_lock:
            if protective or not self._protective_stop_requested:
                if verified:
                    self._set_state(final_state, stop_result.evidence)
                else:
                    self._set_state(TourState.STOP_UNVERIFIED, detail)
        if protective:
            verification = "verified by target evidence" if verified else "unverified"
            return (
                f"Protective stop requested; target stop is {verification}. "
                "Physical E-stop remains independent."
            )
        if verified:
            return "Tour cancelled after verified target stop."
        return "Tour cancellation requested; target stop is unverified."

    def status_message(self) -> str:
        snapshot = self.snapshot()
        if snapshot.current_stop_id is None:
            return f"Tour {snapshot.tour_id} is {snapshot.state.value}."
        return (
            f"Tour {snapshot.tour_id} is {snapshot.state.value} at "
            f"{snapshot.current_stop_id} ({snapshot.current_stop_index + 1}/{snapshot.total_stops})."
        )

    async def wait_finished(self) -> None:
        if self._run_task:
            await asyncio.shield(self._run_task)

    async def _run(self, generation: int) -> None:
        try:
            await self._say(f"Welcome. {self.plan.title} is starting.")
            self._ensure_run_active(generation)
            authority = self._require_motion_authority()
            for index, stop in enumerate(self.plan.stops):
                self.current_stop_index = index
                await self._wait_if_paused(generation)
                request = NavigationRequest(
                    request_id=self._new_navigation_request_id(),
                    authority_epoch=authority.epoch,
                    map_id=self.plan.map_id,
                    map_version=self.plan.map_version,
                    route_id=self.plan.route_id,
                    waypoint_id=stop.waypoint_id,
                )
                self._set_state(TourState.MOVING, stop.waypoint_id)
                result = await self._navigate_with_announcement(
                    request, authority, stop.travel_announcement
                )
                self._ensure_run_active(generation)
                self._validate_navigation_result(request, result)
                if not result.arrived:
                    self._set_state(TourState.BLOCKED, result.detail or result.evidence)
                    await self._say(
                        "I could not safely reach the next stop. Please assist or cancel the tour.",
                        priority=10,
                    )
                    return
                self._emit("navigation.arrival_evidence", result.evidence)
                await self._wait_if_paused(generation)
                self._set_state(TourState.NARRATING, stop.stop_id)
                await self._say(stop.narration)
                await self._wait_if_paused(generation)
                if index < len(self.plan.stops) - 1:
                    self._continue_event.clear()
                    self._set_state(TourState.WAITING, "awaiting next/continue")
                    await self._wait_for_continue_or_cancel(generation)
            self._ensure_run_active(generation)
            self._set_state(TourState.COMPLETED, "all stops completed")
            await self._say("The tour is complete. Thank you.")
        except asyncio.CancelledError:
            return
        except Exception as exc:
            if generation == self._run_generation and not self._cancel_event.is_set():
                self._emit("runtime.failure", type(exc).__name__)
                await self.protective_stop(f"runtime failure: {type(exc).__name__}")

    async def _navigate_with_announcement(
        self,
        request: NavigationRequest,
        authority: NavigationAuthority,
        announcement: str,
    ) -> NavigationResult:
        navigation_task = asyncio.create_task(
            self.navigation.navigate_to(request, authority)
        )
        self._track_motion_task(
            navigation_task, "navigation.navigate_to", authority
        )
        announcement_task = (
            asyncio.create_task(self._say(announcement)) if announcement else None
        )
        tasks: list[asyncio.Task[Any]] = [navigation_task]
        if announcement_task is not None:
            tasks.append(announcement_task)
        try:
            result = await navigation_task
            if announcement_task is not None:
                await announcement_task
            return result
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            done, pending = await asyncio.wait(tasks, timeout=self.control_timeout_s)
            for task in pending:
                self._detach(
                    task,
                    "tour.navigation-child",
                    motion_action=task is navigation_task,
                )
            for task in done:
                self._consume_task(task)
            raise

    @staticmethod
    def _validate_navigation_result(
        request: NavigationRequest, result: Any
    ) -> None:
        if not isinstance(result, NavigationResult):
            raise TypeError("navigation returned an unexpected result type")
        if type(result.arrived) is not bool:
            raise TypeError("navigation arrived field must be boolean")
        if not all(
            isinstance(value, str)
            for value in (
                result.request_id,
                result.map_id,
                result.map_version,
                result.route_id,
                result.waypoint_id,
                result.evidence,
                result.detail,
            )
        ):
            raise TypeError("navigation result text fields must be strings")
        if type(result.authority_epoch) is not int:
            raise TypeError("navigation result authority epoch must be an integer")
        if (
            result.request_id != request.request_id
            or result.authority_epoch != request.authority_epoch
            or result.map_id != request.map_id
            or result.map_version != request.map_version
            or result.route_id != request.route_id
            or result.waypoint_id != request.waypoint_id
        ):
            raise ValueError("navigation result does not match the bound request")
        if result.arrived and not result.evidence.strip():
            raise ValueError("arrival requires non-empty target evidence")

    async def _request_navigation_stop(
        self, request: NavigationStopRequest
    ) -> StopResult:
        completed, result = await self._bounded(
            lambda: self.navigation.stop(request),
            self.control_timeout_s,
            "navigation.stop",
        )
        if not completed:
            return StopResult(
                request.request_id,
                request.revoke_through_epoch,
                True,
                False,
                "",
                "target stop failed or timed out",
            )
        if not isinstance(result, StopResult):
            return StopResult(
                request.request_id,
                request.revoke_through_epoch,
                True,
                False,
                "",
                "target returned no typed stop evidence",
            )
        if (
            not isinstance(result.request_id, str)
            or type(result.revoked_through_epoch) is not int
            or
            type(result.requested) is not bool
            or type(result.verified_stopped) is not bool
            or not all(
                isinstance(value, str)
                for value in (result.evidence, result.detail)
            )
        ):
            return StopResult(
                request.request_id,
                request.revoke_through_epoch,
                True,
                False,
                "",
                "target returned malformed stop evidence",
            )
        if (
            result.request_id != request.request_id
            or result.revoked_through_epoch < request.revoke_through_epoch
        ):
            return StopResult(
                request.request_id,
                request.revoke_through_epoch,
                True,
                False,
                "",
                "target stop evidence does not match the stop request",
            )
        if result.verified_stopped and (
            not result.requested or not result.evidence.strip()
        ):
            return StopResult(
                result.request_id,
                result.revoked_through_epoch,
                result.requested,
                False,
                "",
                "target claimed stop without a request and non-empty evidence",
            )
        return result

    async def _say(self, text: str, *, priority: int = 0) -> bool:
        epoch = self._speech_epoch
        async with self._speaker_lock:
            if epoch != self._speech_epoch:
                raise asyncio.CancelledError
            task = asyncio.create_task(self.speaker.say(text, priority=priority))
            self._speech_tasks.add(task)
            try:
                await task
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._emit("speech.failed", type(exc).__name__)
                return False
            finally:
                self._speech_tasks.discard(task)
            if epoch != self._speech_epoch:
                raise asyncio.CancelledError
            return True

    async def _stop_all_speech(self) -> None:
        self._speech_epoch += 1
        active = list(self._speech_tasks)
        for task in active:
            task.cancel()
        speaker_stop = asyncio.create_task(
            self._bounded(
                lambda: self.speaker.stop(),
                self.control_timeout_s,
                "speaker.stop",
            )
        )
        if active:
            done, pending = await asyncio.wait(active, timeout=self.control_timeout_s)
            for task in pending:
                self._detach(task, "speaker.say")
            for task in done:
                self._consume_task(task)
        await speaker_stop

    async def _wait_if_paused(self, generation: int) -> None:
        await self._pause_gate.wait()
        self._ensure_run_active(generation)

    async def _wait_for_continue_or_cancel(self, generation: int) -> None:
        continue_task = asyncio.create_task(self._continue_event.wait())
        cancel_task = asyncio.create_task(self._cancel_event.wait())
        done, pending = await asyncio.wait(
            {continue_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        self._ensure_run_active(generation)
        if cancel_task in done and cancel_task.result():
            raise asyncio.CancelledError

    async def _ask_brain(self, text: str) -> str:
        if self.brain is None:
            message = "I did not understand that command. Try start, pause, next, status, or stop."
            await self._say(message)
            return message
        snapshot = self.snapshot()
        completed, proposal = await self._bounded(
            lambda: self.brain.decide(text, snapshot),
            self.brain_timeout_s,
            "brain.decide",
        )
        if not completed or not self._valid_brain_proposal(proposal):
            self._emit("brain.rejected", "timeout, crash, or invalid output")
            message = "The optional language model is unavailable; deterministic controls still work."
            await self._say(message)
            return message
        if snapshot.revision != self.revision:
            self._emit("brain.stale", f"snapshot revision {snapshot.revision}")
            return "A stale language-model proposal was ignored."
        if proposal.action in {TourBrainAction.RESPOND, TourBrainAction.CLARIFY}:
            await self._say(proposal.message)
            return proposal.message
        if proposal.action is TourBrainAction.START_TOUR:
            return await self.start()
        if proposal.action is TourBrainAction.CONTINUE_TOUR:
            return await self.continue_tour()
        if proposal.action is TourBrainAction.STATUS:
            return self.status_message()
        self._emit("brain.rejected", "unknown action")
        return "The language-model proposal was rejected."

    @staticmethod
    def _valid_brain_proposal(value: Any) -> bool:
        return (
            isinstance(value, TourBrainProposal)
            and isinstance(value.action, TourBrainAction)
            and isinstance(value.message, str)
            and len(value.message) <= 500
            and not any(
                ord(character) < 32 or ord(character) == 127
                for character in value.message
            )
        )

    def _ensure_run_active(self, generation: int) -> None:
        if generation != self._run_generation or self._cancel_event.is_set():
            raise asyncio.CancelledError

    async def _bounded(
        self,
        factory: Callable[[], Awaitable[T]],
        timeout_s: float,
        label: str,
        *,
        motion_authority: NavigationAuthority | None = None,
    ) -> tuple[bool, T | None]:
        task = asyncio.create_task(factory())
        if motion_authority is not None:
            self._track_motion_task(task, label, motion_authority)
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout_s)
        except asyncio.CancelledError:
            task.cancel()
            self._detach(
                task, label, motion_action=motion_authority is not None
            )
            raise
        if not done:
            task.cancel()
            self._detach(
                task, label, motion_action=motion_authority is not None
            )
            self._emit("control.timeout", label)
            return False, None
        try:
            return True, task.result()
        except asyncio.CancelledError:
            self._emit("control.cancelled", label)
        except Exception as exc:
            self._emit("control.failed", f"{label}:{type(exc).__name__}")
        return False, None

    async def _cancel_owned_task(
        self,
        task: asyncio.Task[Any] | None,
        label: str,
    ) -> None:
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=self.control_timeout_s)
        if not done:
            self._detach(task, label)
            self._emit("control.detach", label)
        else:
            self._consume_task(task)

    def _detach(
        self,
        task: asyncio.Task[Any],
        label: str,
        *,
        motion_action: bool = False,
    ) -> None:
        self._detached_tasks.add(task)
        if motion_action:
            self._detached_motion_tasks.add(task)

        def observe(completed: asyncio.Task[Any]) -> None:
            self._detached_tasks.discard(completed)
            self._detached_motion_tasks.discard(completed)
            self._consume_task(completed)

        task.add_done_callback(observe)
        self._emit("control.detached", label)

    def _track_motion_task(
        self,
        task: asyncio.Task[Any],
        label: str,
        authority: NavigationAuthority,
    ) -> None:
        if task in self._motion_tasks:
            return
        self._motion_tasks.add(task)

        def motion_completed(completed: asyncio.Task[Any]) -> None:
            self._motion_tasks.discard(completed)
            if authority.revoked:
                self._completed_motion_after_revoke.add(completed)
                self._schedule_late_motion_restop(label)

        task.add_done_callback(motion_completed)

    def _schedule_late_motion_restop(self, label: str = "motion task") -> None:
        if (
            self._restop_scheduled
            or self._active_stop_sequences
            or self._motion_tasks
            or not self._completed_motion_after_revoke
        ):
            return
        self._completed_motion_after_revoke.clear()
        self._restop_scheduled = True
        follow_up = asyncio.create_task(
            self._restop_after_late_motion(label),
            name=f"late-motion-restop:{label}",
        )
        self._retain_control_task(follow_up)

    async def _restop_after_late_motion(self, label: str) -> None:
        try:
            current = self._protective_stop_task
            if (
                current is not None
                and current is not asyncio.current_task()
                and not current.done()
            ):
                try:
                    await asyncio.shield(current)
                except Exception:
                    pass
            await self._start_protective_stop(
                f"late completion of {label}", force=True
            )
        finally:
            self._restop_scheduled = False
            self._schedule_late_motion_restop(label)

    def _retain_control_task(self, task: asyncio.Task[Any]) -> None:
        self._owned_control_tasks.add(task)

        def release(completed: asyncio.Task[Any]) -> None:
            self._owned_control_tasks.discard(completed)
            self._consume_task(completed)

        task.add_done_callback(release)

    def _launch_fail_stop(self, reason: str) -> None:
        task = asyncio.create_task(
            self._start_protective_stop(reason, force=False),
            name=f"fail-stop:{reason}",
        )
        self._retain_control_task(task)

    def _require_motion_authority(self) -> NavigationAuthority:
        authority = self._motion_authority
        if authority is None or authority.revoked:
            raise RuntimeError("no active navigation authority")
        return authority

    def _revoke_motion_authority_locked(self) -> None:
        if self._motion_authority is not None:
            self._motion_authority.revoke()

    def _new_navigation_request_id(self) -> str:
        self._request_sequence += 1
        return f"nav-{self._authority_epoch}-{self._request_sequence}"

    def _new_stop_request_locked(self, reason: str) -> NavigationStopRequest:
        self._request_sequence += 1
        return NavigationStopRequest(
            request_id=f"stop-{self._authority_epoch}-{self._request_sequence}",
            reason=reason,
            revoke_through_epoch=self._authority_epoch,
        )

    @staticmethod
    def _consume_task(task: asyncio.Task[Any]) -> None:
        if not task.done():
            return
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass

    def _set_state(self, state: TourState, detail: str) -> None:
        self.state = state
        self.revision += 1
        self.events.append(
            TourRuntimeEvent("state.changed", self.revision, state, detail, time.monotonic())
        )

    def _emit(self, event_type: str, detail: str) -> None:
        self.events.append(
            TourRuntimeEvent(event_type, self.revision, self.state, detail, time.monotonic())
        )

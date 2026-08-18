from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from enum import Enum
from typing import Any

from .base import (
    RuntimeTextPort,
    TextHandler,
    VoiceInputEvent,
    VoiceInputEventType,
    VoiceInputPort,
)


class WakeDictationState(str, Enum):
    ARMED = "armed"
    LISTENING = "listening"
    CLOSED = "closed"


_DEFAULT_WAKE_PHRASES = ("hey jackie", "jackie", "你好杰基")
_LEADING_SEPARATORS = " \t\r\n,，.。!！?？:：;；、-—"


class WakeDictationController:
    """Route wake-word/ASR events to a text Runtime without owning a Brain.

    Ordinary FINAL text is accepted only for the currently awakened session.
    PARTIAL text and unawakened FINAL text use ``partial=True`` so a Runtime can
    recognize its local STOP grammar without allowing either path into a Brain.

    Runtime calls run in owned tasks. Runtime classifies and schedules full
    utterances, while the partial safety path remains independent. Consequently
    a fixed control or cooperative STOP handler can overtake a slow Brain.
    """

    def __init__(
        self,
        source: VoiceInputPort,
        runtime: RuntimeTextPort | TextHandler,
        *,
        wake_phrases: Iterable[str] = _DEFAULT_WAKE_PHRASES,
        close_timeout_s: float = 1.0,
        max_pending_normal_calls: int = 8,
        max_pending_safety_calls: int = 8,
        task_factory: (
            Callable[[Awaitable[str], str], asyncio.Task[str]] | None
        ) = None,
    ) -> None:
        if close_timeout_s <= 0:
            raise ValueError("close_timeout_s must be positive")
        if max_pending_normal_calls <= 0 or max_pending_safety_calls <= 0:
            raise ValueError("pending-call limits must be positive")
        configured_phrases = tuple(wake_phrases)
        if any(not isinstance(phrase, str) for phrase in configured_phrases):
            raise TypeError("wake phrases must be strings")
        phrases = tuple(
            dict.fromkeys(
                phrase.strip() for phrase in configured_phrases if phrase.strip()
            )
        )
        if not phrases:
            raise ValueError("at least one wake phrase is required")
        handler = getattr(runtime, "handle_text", None)
        if handler is None:
            if not callable(runtime):
                raise TypeError("runtime must expose handle_text or be callable")
            handler = runtime

        self.source = source
        self._handle_text: TextHandler = handler
        self.wake_phrases = tuple(sorted(phrases, key=len, reverse=True))
        self.close_timeout_s = close_timeout_s
        self.max_pending_normal_calls = max_pending_normal_calls
        self.max_pending_safety_calls = max_pending_safety_calls
        self.state = WakeDictationState.ARMED
        self.active_session_id: str | None = None
        self._active_wake_monotonic_s: float | None = None
        self._event_high_water_s: float | None = None
        self._tasks: set[asyncio.Task[str]] = set()
        self._normal_tasks: deque[asyncio.Task[str]] = deque()
        self._safety_tasks: deque[asyncio.Task[str]] = deque()
        self._superseded_count = 0
        self._task_errors: list[BaseException] = []
        self._run_task: asyncio.Task[None] | None = None
        self._source_close_task: asyncio.Task[None] | None = None
        self._closed = False
        self._task_factory = task_factory or self._default_task_factory

    @staticmethod
    def _default_task_factory(
        awaitable: Awaitable[str], name: str
    ) -> asyncio.Task[str]:
        return asyncio.create_task(awaitable, name=name)

    @property
    def pending_count(self) -> int:
        return sum(not task.done() for task in self._tasks)

    @property
    def task_errors(self) -> tuple[BaseException, ...]:
        return tuple(self._task_errors)

    @property
    def superseded_count(self) -> int:
        """Number of older Runtime calls cancelled by bounded admission."""

        return self._superseded_count

    def start(self) -> asyncio.Task[None]:
        """Start consuming the source once and return the owned runner task."""

        if self._closed:
            raise RuntimeError("controller is closed")
        if self._run_task is not None and not self._run_task.done():
            raise RuntimeError("controller is already running")
        self._run_task = asyncio.create_task(self.run(), name="jackie-voice-input")
        self._run_task.add_done_callback(self._runner_done)
        return self._run_task

    async def run(self) -> None:
        """Consume events until EOF/close, then finish or cancel owned calls."""

        current = asyncio.current_task()
        if self._closed:
            if self._run_task is current:
                return
            raise RuntimeError("controller is closed")
        if self._run_task is not None and self._run_task is not current:
            if not self._run_task.done():
                raise RuntimeError("controller is already running")
        elif self._run_task is None:
            self._run_task = current
        try:
            async for event in self.source:
                if self._closed:
                    break
                await self.handle_event(event)
        finally:
            self._closed = True
            self.state = WakeDictationState.CLOSED
            self.active_session_id = None
            self._active_wake_monotonic_s = None
            deadline = time.monotonic() + self.close_timeout_s
            await self._close_source(self._remaining(deadline))
            await self._finish_owned_tasks(deadline)

    async def handle_event(self, event: VoiceInputEvent) -> None:
        """Apply one event without waiting for Runtime/Brain completion."""

        if self._closed:
            raise RuntimeError("controller is closed")
        if not isinstance(event, VoiceInputEvent):
            raise TypeError("event must be a VoiceInputEvent")

        prior_high_water = self._event_high_water_s
        if prior_high_water is None or event.monotonic_s > prior_high_water:
            self._event_high_water_s = event.monotonic_s

        if event.event_type is VoiceInputEventType.WAKE:
            # Preserve the newest session when an old/replayed event arrives.
            if prior_high_water is not None and event.monotonic_s <= prior_high_water:
                return
            self.state = WakeDictationState.LISTENING
            self.active_session_id = event.session_id
            self._active_wake_monotonic_s = event.monotonic_s
            return

        if event.event_type is VoiceInputEventType.PARTIAL:
            self._dispatch(event.text or "", partial=True, event=event)
            await asyncio.sleep(0)
            return

        if event.event_type is VoiceInputEventType.FINAL:
            if (
                self.state is WakeDictationState.LISTENING
                and event.session_id == self.active_session_id
                and self._active_wake_monotonic_s is not None
                and event.monotonic_s >= self._active_wake_monotonic_s
            ):
                self.state = WakeDictationState.ARMED
                self.active_session_id = None
                self._active_wake_monotonic_s = None
                text = self.strip_leading_wake_phrase(event.text or "")
                if text:
                    self._dispatch(text, partial=False, event=event)
            else:
                # This safety-only path permits an unprefixed STOP while armed,
                # but Runtime's partial contract prevents a Brain invocation.
                self._dispatch(event.text or "", partial=True, event=event)
            await asyncio.sleep(0)
            return

        if event.event_type in {
            VoiceInputEventType.TIMEOUT,
            VoiceInputEventType.ERROR,
        }:
            # Ignore stale terminal events from a superseded wake session.
            if (
                event.session_id == self.active_session_id
                and self._active_wake_monotonic_s is not None
                and event.monotonic_s >= self._active_wake_monotonic_s
            ):
                self.state = WakeDictationState.ARMED
                self.active_session_id = None
                self._active_wake_monotonic_s = None

    def strip_leading_wake_phrase(self, text: str) -> str:
        """Remove one configured, whole leading wake phrase and its separator."""

        candidate = text.lstrip()
        for phrase in self.wake_phrases:
            match = re.match(re.escape(phrase), candidate, flags=re.IGNORECASE)
            if match is None:
                continue
            remainder = candidate[match.end() :]
            if remainder and self._continues_ascii_word(remainder[0]):
                continue
            return remainder.lstrip(_LEADING_SEPARATORS)
        return candidate

    @staticmethod
    def _continues_ascii_word(character: str) -> bool:
        return character.isascii() and (character.isalnum() or character == "_")

    def _dispatch(
        self,
        text: str,
        *,
        partial: bool,
        event: VoiceInputEvent,
    ) -> None:
        if not text.strip() or self._closed:
            return
        lane = self._safety_tasks if partial else self._normal_tasks
        limit = (
            self.max_pending_safety_calls
            if partial
            else self.max_pending_normal_calls
        )
        if lane:
            active = (task for task in lane if not task.done())
            retained = tuple(active)
            lane.clear()
            lane.extend(retained)
        if len(lane) >= limit:
            # Prefer fresh hypotheses and user turns. Runtime implementations
            # must own/shield safety-critical work from caller cancellation.
            lane.popleft().cancel()
            self._superseded_count += 1
        label = "safety" if partial else "utterance"
        coroutine = self._invoke_runtime(text, partial=partial)
        try:
            task = self._task_factory(
                coroutine,
                f"voice-{label}:{event.session_id}:{event.monotonic_s:.6f}",
            )
        except BaseException:
            coroutine.close()  # type: ignore[attr-defined]
            raise
        self._tasks.add(task)
        lane.append(task)
        task.add_done_callback(self._task_done)

    async def _invoke_runtime(self, text: str, *, partial: bool) -> str:
        # Runtime, rather than the audio boundary, classifies complete input.
        # This lets later fixed controls overtake a provider-specific Brain turn.
        return await self._handle_text(text, partial=partial)

    def _task_done(self, task: asyncio.Task[str]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None and error not in self._task_errors:
            # Consume task exceptions so a failed plugin cannot produce an
            # unobserved-task warning. Supervisors may inspect ``task_errors``.
            self._task_errors.append(error)

    def _runner_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None and error not in self._task_errors:
            self._task_errors.append(error)

    async def wait_idle(self, timeout_s: float | None = None) -> bool:
        """Wait for currently owned Runtime calls; return False on timeout."""

        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be non-negative or None")
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while self._tasks:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            if remaining == 0.0:
                return False
            _, pending = await asyncio.wait(tuple(self._tasks), timeout=remaining)
            if pending and deadline is not None and time.monotonic() >= deadline:
                return False
        return True

    async def aclose(self) -> None:
        """Close input and cancel all owned work within the configured bound."""

        if (
            self._closed
            and not self._tasks
            and self._source_close_task is not None
            and self._source_close_task.done()
        ):
            return
        self._closed = True
        self.state = WakeDictationState.CLOSED
        self.active_session_id = None
        self._active_wake_monotonic_s = None
        deadline = time.monotonic() + self.close_timeout_s

        await self._close_source(self._remaining(deadline))

        current = asyncio.current_task()
        runner = self._run_task
        if runner is not None and runner is not current and not runner.done():
            runner.cancel()
            await self._bounded_gather({runner}, self._remaining(deadline))
        await self._cancel_owned_tasks(self._remaining(deadline))

    async def _close_source(self, timeout_s: float) -> None:
        if self._source_close_task is None:
            self._source_close_task = asyncio.create_task(
                self.source.aclose(), name="jackie-voice-source-close"
            )
            self._source_close_task.add_done_callback(self._source_close_done)
        task = self._source_close_task
        if task.done():
            return
        _, pending = await asyncio.wait({task}, timeout=max(0.0, timeout_s))
        if pending:
            task.cancel()
            self._task_errors.append(
                TimeoutError("voice input source did not close within the deadline")
            )

    def _source_close_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None and error not in self._task_errors:
            self._task_errors.append(error)

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())

    async def _finish_owned_tasks(self, deadline: float) -> None:
        # Preserve part of the close budget for cancellation and collection.
        grace_s = self._remaining(deadline) * 0.8
        if await self.wait_idle(grace_s):
            return
        await self._cancel_owned_tasks(self._remaining(deadline))

    async def _cancel_owned_tasks(self, timeout_s: float) -> None:
        pending = {task for task in self._tasks if not task.done()}
        for task in pending:
            task.cancel()
        await self._bounded_gather(pending, timeout_s)

    async def _bounded_gather(
        self, tasks: set[asyncio.Task[Any]], timeout_s: float
    ) -> None:
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=max(0.0, timeout_s))
        for task in done:
            if not task.cancelled():
                # Retrieve exceptions for runner tasks that do not use the
                # normal done callback.
                error = task.exception()
                if error is not None and error not in self._task_errors:
                    self._task_errors.append(error)
        for task in pending:
            task.cancel()
            self._task_errors.append(
                TimeoutError(f"owned task did not close: {task.get_name()}")
            )

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterable

from .base import VoiceInputEvent


class MockVoiceInput:
    """One-shot, deterministic voice event source for tests and demos."""

    def __init__(
        self,
        events: Iterable[VoiceInputEvent] = (),
        *,
        inter_event_delay_s: float = 0.0,
    ) -> None:
        if inter_event_delay_s < 0:
            raise ValueError("inter_event_delay_s must be non-negative")
        self._events = deque(events)
        self._inter_event_delay_s = inter_event_delay_s
        self._closed = False

    @property
    def remaining(self) -> int:
        return len(self._events)

    def __aiter__(self) -> MockVoiceInput:
        return self

    async def __anext__(self) -> VoiceInputEvent:
        if self._closed or not self._events:
            raise StopAsyncIteration
        # Even a zero-delay mock yields once so scheduled Runtime calls can make
        # progress in the same deterministic order as their source events.
        await asyncio.sleep(self._inter_event_delay_s)
        if self._closed or not self._events:
            raise StopAsyncIteration
        return self._events.popleft()

    async def aclose(self) -> None:
        self._closed = True

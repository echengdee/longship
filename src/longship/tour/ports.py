from __future__ import annotations

import asyncio
import time
from typing import Protocol


class SpeakerPort(Protocol):
    async def say(self, text: str, *, priority: int = 0) -> None:
        ...

    async def stop(self) -> None:
        ...


class ConsoleSpeaker:
    def __init__(self, label: str = "Longship") -> None:
        self.label = label

    async def say(self, text: str, *, priority: int = 0) -> None:
        print(f"{self.label}: {text}")

    async def stop(self) -> None:
        return None


class RecordingSpeaker:
    """Small deterministic fake used by tests and downstream examples."""

    def __init__(self, *, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = delay_seconds
        self.events: list[tuple[str, str, float]] = []
        self._tasks: set[asyncio.Task[object]] = set()

    async def say(self, text: str, *, priority: int = 0) -> None:
        self.events.append(("speech.started", text, time.monotonic()))
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        self.events.append(("speech.finished", text, time.monotonic()))

    async def stop(self) -> None:
        self.events.append(("speech.stopped", "", time.monotonic()))

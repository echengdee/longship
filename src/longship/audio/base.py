from __future__ import annotations

import math
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class VoiceInputEventType(str, Enum):
    """Events emitted by a wake-word/VAD/ASR audio front end."""

    WAKE = "wake"
    PARTIAL = "partial"
    FINAL = "final"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class VoiceInputEvent:
    """One timestamped event from a voice-input implementation.

    ``monotonic_s`` must come from a monotonic clock in the producer's process.
    A producer must allocate a fresh ``session_id`` for every wake session and
    preserve the original timestamp when delivery is delayed. PARTIAL and FINAL
    events carry text. ERROR may use ``text`` for a safe, operator-facing
    diagnostic; it must not contain credentials or raw audio.
    """

    event_type: VoiceInputEventType
    session_id: str
    monotonic_s: float
    text: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, VoiceInputEventType):
            raise TypeError("event_type must be a VoiceInputEventType")
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(self.monotonic_s, (int, float)) or not math.isfinite(
            self.monotonic_s
        ):
            raise ValueError("monotonic_s must be finite")
        if self.monotonic_s < 0:
            raise ValueError("monotonic_s must be non-negative")
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("text must be a string or None")
        if self.event_type in {
            VoiceInputEventType.PARTIAL,
            VoiceInputEventType.FINAL,
        } and self.text is None:
            raise ValueError("PARTIAL and FINAL events require text")
        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)) or not math.isfinite(
                self.confidence
            ):
                raise ValueError("confidence must be finite")
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError("confidence must be between 0 and 1")


@runtime_checkable
class VoiceInputPort(Protocol):
    """Async event stream implemented by a local voice-input plugin.

    ``aclose`` must stop capture, release the audio device, and cooperate with
    cancellation. Blocking native integrations need their own bounded worker or
    process boundary; they must not block the Runtime event loop.
    """

    def __aiter__(self) -> AsyncIterator[VoiceInputEvent]: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class RuntimeTextPort(Protocol):
    """Small injection boundary implemented by a Longship runtime.

    ``partial=True`` is the safety-only path. Implementations must never invoke
    a high-level Brain for partial input. Calls must cooperate with
    cancellation, while safety-critical work such as a protective stop must be
    Runtime-owned or shielded so caller cancellation cannot abort it.
    """

    async def handle_text(self, text: str, *, partial: bool = False) -> str: ...


class TextHandler(Protocol):
    """Callback form of :class:`RuntimeTextPort`."""

    def __call__(self, text: str, *, partial: bool = False) -> Awaitable[str]: ...

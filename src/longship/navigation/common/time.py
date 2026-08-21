"""Time primitives shared by navigation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class TimePoint:
    """A timestamp in an explicitly named clock domain."""

    clock_id: str
    nanoseconds: int


@runtime_checkable
class TimeSource(Protocol):
    """Reads time from one explicitly named clock domain."""

    def now(self) -> TimePoint:
        """Returns the current point in the configured clock domain."""
        ...

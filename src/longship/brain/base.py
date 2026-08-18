from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from longship.tour.models import TourSnapshot


class TourBrainAction(str, Enum):
    """High-level actions a brain may propose in the V0 tour runtime."""

    RESPOND = "respond"
    CLARIFY = "clarify"
    START_TOUR = "start_tour"
    CONTINUE_TOUR = "continue_tour"
    STATUS = "status"


@dataclass(frozen=True, slots=True)
class TourBrainProposal:
    action: TourBrainAction
    message: str


class TourBrainPort(Protocol):
    async def decide(self, text: str, snapshot: "TourSnapshot") -> TourBrainProposal:
        """Return a bounded high-level proposal, never an actuator command."""

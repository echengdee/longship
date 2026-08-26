from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
import math
from typing import Protocol

from longship.contracts.skills.follow_person import FollowState


FOLLOW_PERSON_SKILL_ID = "navigation.follow_person"
FOLLOW_PERSON_SKILL_VERSION = "0.1.0"


class FollowBrainAction(str, Enum):
    CALL_SKILL = "call_skill"
    RESPOND = "respond"


class FollowTaskOperation(str, Enum):
    FOLLOW = "follow"
    PAUSE = "pause"
    RESUME = "resume"


@dataclass(frozen=True, slots=True)
class FollowTaskStep:
    operation: FollowTaskOperation
    duration_s: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, FollowTaskOperation):
            raise TypeError("Follow task operation is invalid")
        if self.duration_s is None:
            return
        if (
            not isinstance(self.duration_s, (int, float))
            or isinstance(self.duration_s, bool)
            or not math.isfinite(float(self.duration_s))
            or not 0.1 <= float(self.duration_s) <= 60.0
        ):
            raise ValueError("Follow task duration is outside supported bounds")


@dataclass(frozen=True, slots=True)
class FollowTaskDraft:
    """Untrusted, bounded semantic plan proposed by a Brain provider."""

    steps: tuple[FollowTaskStep, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.steps) <= 7:
            raise ValueError("Follow task draft step count is outside bounds")
        if not all(isinstance(step, FollowTaskStep) for step in self.steps):
            raise TypeError("Follow task draft contains an invalid step")


@dataclass(frozen=True, slots=True)
class FollowBrainContext:
    request_id: str
    runtime_revision: int
    active_skill_call_id: str | None
    skill_state: FollowState
    available_skill_ids: tuple[str, ...] = (FOLLOW_PERSON_SKILL_ID,)


@dataclass(frozen=True, slots=True)
class FollowBrainDecision:
    request_id: str
    based_on_runtime_revision: int
    action: FollowBrainAction
    skill_id: str | None
    summary: str
    confidence: float
    task_draft: FollowTaskDraft | None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("brain decision request_id is required")
        if type(self.based_on_runtime_revision) is not int:
            raise ValueError("brain decision revision must be an integer")
        if not isinstance(self.action, FollowBrainAction):
            raise TypeError("brain decision action is invalid")
        if self.action is FollowBrainAction.CALL_SKILL:
            if self.skill_id != FOLLOW_PERSON_SKILL_ID:
                raise ValueError("brain requested an unavailable skill")
            if not isinstance(self.task_draft, FollowTaskDraft):
                raise ValueError("Skill decisions require a Follow task draft")
        elif self.skill_id is not None:
            raise ValueError("response decisions cannot carry a skill ID")
        elif self.task_draft is not None:
            raise ValueError("response decisions cannot carry a task draft")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("brain decision summary is required")
        if len(self.summary) > 500:
            raise ValueError("brain decision summary is too long")
        if not isinstance(self.confidence, (int, float)) or isinstance(
            self.confidence, bool
        ):
            raise ValueError("brain confidence must be numeric")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("brain confidence must be between zero and one")


class FollowBrainPort(Protocol):
    async def decide(
        self, text: str, context: FollowBrainContext
    ) -> FollowBrainDecision:
        """Return a semantic proposal, never a motion or target command."""


class DeterministicFollowBrain:
    """Offline Brain test double for reproducible system qualification."""

    _ENGLISH_FOLLOW = re.compile(
        r"\b(follow\s+(me|this person)|come with me|walk behind me)\b",
        flags=re.IGNORECASE,
    )
    _CHINESE_FOLLOW = ("跟我走", "跟着我", "跟随我", "跟着这个人", "跟随这个人")
    _NEGATIONS = ("不要", "别", "不用", "not ", "don't", "do not")

    async def decide(
        self, text: str, context: FollowBrainContext
    ) -> FollowBrainDecision:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("brain input text is required")
        normalized = " ".join(text.casefold().strip().split())
        compact = re.sub(r"[\s，。！？,.!?;；:：]", "", normalized)
        negated = any(token in normalized or token in compact for token in self._NEGATIONS)
        requests_follow = bool(self._ENGLISH_FOLLOW.search(normalized)) or any(
            phrase in compact for phrase in self._CHINESE_FOLLOW
        )
        can_start = (
            context.active_skill_call_id is None
            and context.skill_state is FollowState.IDLE
            and FOLLOW_PERSON_SKILL_ID in context.available_skill_ids
        )
        if requests_follow and not negated and can_start:
            return FollowBrainDecision(
                request_id=context.request_id,
                based_on_runtime_revision=context.runtime_revision,
                action=FollowBrainAction.CALL_SKILL,
                skill_id=FOLLOW_PERSON_SKILL_ID,
                summary="operator requested supervised person following",
                confidence=1.0,
                task_draft=FollowTaskDraft(
                    (FollowTaskStep(FollowTaskOperation.FOLLOW, None),)
                ),
            )
        return FollowBrainDecision(
            request_id=context.request_id,
            based_on_runtime_revision=context.runtime_revision,
            action=FollowBrainAction.RESPOND,
            skill_id=None,
            summary="request does not authorize a new FollowPerson Skill call",
            confidence=1.0,
            task_draft=None,
        )

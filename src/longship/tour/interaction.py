from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class CommandKind(str, Enum):
    STOP = "stop"
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    CONTINUE = "continue"
    CANCEL = "cancel"
    STATUS = "status"
    REPEAT = "repeat"
    BRAIN = "brain"


@dataclass(frozen=True, slots=True)
class RoutedCommand:
    kind: CommandKind
    original_text: str


_CHINESE_STOP = ("急停", "停止", "停下", "别动", "不要动")
_ENGLISH_STOP = ("stop", "emergency stop", "halt", "freeze")

_ALIASES: dict[CommandKind, frozenset[str]] = {
    CommandKind.START: frozenset({"start", "start tour", "开始", "开始导览", "开始讲解"}),
    CommandKind.PAUSE: frozenset({"pause", "pause tour", "暂停", "暂停导览"}),
    CommandKind.RESUME: frozenset({"resume", "resume tour", "恢复", "继续导览"}),
    CommandKind.CONTINUE: frozenset({"continue", "next", "next stop", "继续", "下一站"}),
    CommandKind.CANCEL: frozenset({"cancel", "cancel tour", "取消", "取消导览", "结束导览"}),
    CommandKind.STATUS: frozenset({"status", "tour status", "状态", "现在什么状态"}),
    CommandKind.REPEAT: frozenset({"repeat", "say again", "重复", "再说一遍"}),
}


class InteractionRouter:
    """Routes reserved and deterministic phrases without calling a model."""

    def route(self, text: str, *, partial: bool = False) -> RoutedCommand:
        normalized = " ".join(text.casefold().strip().split())
        compact = re.sub(r"[\s，。！？,.!?;；:：]", "", normalized)
        if any(phrase in compact for phrase in _CHINESE_STOP):
            return RoutedCommand(CommandKind.STOP, text)
        if any(
            re.search(rf"\b{re.escape(phrase)}\b", normalized)
            for phrase in _ENGLISH_STOP
        ):
            return RoutedCommand(CommandKind.STOP, text)

        # Partial ASR hypotheses are only trusted for the reserved stop grammar.
        if partial:
            return RoutedCommand(CommandKind.BRAIN, text)
        for kind, aliases in _ALIASES.items():
            if normalized in aliases or compact in aliases:
                return RoutedCommand(kind, text)
        return RoutedCommand(CommandKind.BRAIN, text)

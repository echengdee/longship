from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from longship.brain.follow_person import (
    FOLLOW_PERSON_SKILL_ID,
    DeterministicFollowBrain,
    FollowBrainAction,
    FollowBrainContext,
    FollowBrainDecision,
    FollowBrainPort,
    FollowTaskDraft,
)
from longship.contracts.runtime.task_graph import (
    MissionTaskGraphState,
    MissionTaskNode,
)
from longship.contracts.skills.follow_person import (
    FollowScene,
    FollowSnapshot,
    FollowState,
)
from longship.runtime.follow_person import FollowEventSink, FollowPersonRuntime
from longship.runtime.follow_task_graph import (
    FOLLOW_PAUSE_OPERATION_ID,
    FOLLOW_RESUME_OPERATION_ID,
    FOLLOW_START_OPERATION_ID,
    compile_follow_task_graph,
    describe_follow_task_draft,
)
from longship.runtime.task_graph import (
    MissionTaskGraphSnapshot,
    SequentialMissionTaskGraphRuntime,
    TaskDispatchResult,
)
from longship.tour.interaction import CommandKind, InteractionRouter


class FollowMissionEventSink(Protocol):
    def publish(self, event: Mapping[str, Any]) -> None:
        ...


class _NullMissionEventSink:
    def publish(self, event: Mapping[str, Any]) -> None:
        del event


class FollowMissionRuntime:
    """Admit text/Brain proposals and own one FollowPerson Skill call."""

    def __init__(
        self,
        skill: FollowPersonRuntime,
        *,
        brain: FollowBrainPort | None = None,
        event_sink: FollowMissionEventSink | FollowEventSink | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.skill = skill
        self.brain = brain or DeterministicFollowBrain()
        self.event_sink = event_sink or _NullMissionEventSink()
        self.clock_ns = clock_ns
        self.router = InteractionRouter()
        self.revision = 0
        self.active_skill_call_id: str | None = None
        self.brain_request_count = 0
        self.accepted_skill_calls = 0
        self.task_graph = SequentialMissionTaskGraphRuntime()

    @property
    def snapshot(self) -> FollowSnapshot:
        return self.skill.snapshot

    @property
    def task_graph_snapshot(self) -> MissionTaskGraphSnapshot | None:
        return self.task_graph.snapshot

    async def handle_text(self, text: str, *, partial: bool = False) -> str:
        kind = self.router.route(text, partial=partial).kind
        self._emit(
            "input.partial" if partial else "input.final",
            {"text": text, "route": kind.value},
        )
        if kind is CommandKind.STOP:
            return self.stop("reserved operator stop")
        if partial:
            self._emit("input.ignored", {"reason": "partial input is safety-only"})
            return "Partial input ignored outside the STOP grammar."
        if kind is CommandKind.PAUSE:
            return self.pause()
        if kind in {CommandKind.RESUME, CommandKind.CONTINUE}:
            return self.resume()
        if kind is CommandKind.STATUS:
            return self.status_message()
        return await self._ask_brain(text)

    def tick(self, scene: FollowScene, *, now_ns: int) -> FollowSnapshot:
        before = self.task_graph.snapshot
        before_sequence = before.transition_sequence if before is not None else None
        graph = self.task_graph.advance(
            now_ns=now_ns, dispatch=self._dispatch_graph_node
        )
        if graph is not None and graph.transition_sequence != before_sequence:
            self.revision += 1
            self._emit(
                "task_graph.transitioned",
                {
                    "graph_id": graph.graph_id,
                    "graph_state": graph.state.value,
                    "current_node_id": graph.current_node_id,
                    "current_operation_id": graph.current_operation_id,
                },
            )
        if graph is not None and graph.state is MissionTaskGraphState.FAILED:
            if self.skill.is_active:
                snapshot = self.skill.stop(
                    "task graph transition failed", now_ns=now_ns
                )
            else:
                snapshot = self.skill.snapshot
            self.active_skill_call_id = None
            self.revision += 1
            self._emit(
                "task_graph.failed_closed",
                {"graph_id": graph.graph_id, "reason": graph.detail},
            )
            return snapshot
        snapshot = self.skill.tick(scene, now_ns=now_ns)
        if snapshot.state in {
            FollowState.FAILED,
            FollowState.STOPPED,
            FollowState.STOP_UNVERIFIED,
        }:
            self.task_graph.cancel(f"Skill became {snapshot.state.value}")
            self.active_skill_call_id = None
            self.revision += 1
            self._emit(
                "skill.call.terminal",
                {"state": snapshot.state.value, "reason": snapshot.detail},
            )
        return snapshot

    def pause(self) -> str:
        if not self.skill.is_active or self.skill.state is FollowState.PAUSED:
            return f"FollowPerson cannot pause from {self.skill.state.value}."
        now_ns = self.clock_ns()
        self.task_graph.cancel("manual Runtime pause superseded the task graph")
        snapshot = self.skill.pause(now_ns=now_ns)
        self.revision += 1
        self._emit("skill.pause.accepted", {"state": snapshot.state.value})
        return "FollowPerson paused."

    def resume(self) -> str:
        if self.skill.state is not FollowState.PAUSED:
            return f"FollowPerson cannot resume from {self.skill.state.value}."
        now_ns = self.clock_ns()
        self.task_graph.cancel("manual Runtime resume superseded the task graph")
        snapshot = self.skill.resume(now_ns=now_ns)
        self.revision += 1
        self._emit("skill.resume.accepted", {"state": snapshot.state.value})
        return "FollowPerson is reacquiring the operator."

    def stop(self, reason: str) -> str:
        self.task_graph.cancel(reason)
        if self.skill.state is FollowState.IDLE:
            # Invalidate any Brain proposal that was based on the pre-STOP
            # revision, even though no Skill has acquired motion yet.
            self.revision += 1
            self._emit("safety.stop.noop", {"reason": reason})
            return "FollowPerson is not active."
        snapshot = self.skill.stop(reason, now_ns=self.clock_ns())
        self.active_skill_call_id = None
        self.revision += 1
        self._emit(
            "safety.stop.completed",
            {
                "reason": reason,
                "state": snapshot.state.value,
                "verified_stopped": snapshot.stop_verified,
            },
        )
        return f"FollowPerson stopped ({snapshot.state.value})."

    def status_message(self) -> str:
        call = self.active_skill_call_id or "none"
        graph = self.task_graph.snapshot
        if graph is None:
            graph_status = "task_graph=none"
        else:
            remaining = "open"
            if graph.deadline_monotonic_ns is not None:
                remaining_s = max(
                    0.0,
                    (graph.deadline_monotonic_ns - self.clock_ns())
                    / 1_000_000_000,
                )
                remaining = f"{remaining_s:.1f}s"
            graph_status = (
                f"task_graph={graph.graph_id}:{graph.state.value}, "
                f"node={graph.current_operation_id or 'none'}, remaining={remaining}"
            )
        return (
            f"FollowPerson state={self.skill.state.value}, active_call={call}, "
            f"{graph_status}."
        )

    async def _ask_brain(self, text: str) -> str:
        request_id = f"follow-brain-{self.brain_request_count + 1}"
        context = FollowBrainContext(
            request_id=request_id,
            runtime_revision=self.revision,
            active_skill_call_id=self.active_skill_call_id,
            skill_state=self.skill.state,
        )
        self.brain_request_count += 1
        self._emit(
            "brain.requested",
            {
                "request_id": request_id,
                "runtime_revision": context.runtime_revision,
                "available_skill_ids": list(context.available_skill_ids),
                "text": text,
            },
        )
        try:
            decision = await self.brain.decide(text, context)
        except Exception as exc:
            self._emit("brain.rejected", {"reason": type(exc).__name__})
            return "Brain failed closed; no Skill was started."
        rejection = self._decision_problem(decision, context)
        if rejection is not None:
            self._emit("brain.rejected", {"reason": rejection})
            return "Brain proposal was rejected; no Skill was started."
        self._emit(
            "brain.decision.accepted",
            {
                "request_id": decision.request_id,
                "action": decision.action.value,
                "skill_id": decision.skill_id,
                "summary": decision.summary,
                "steps": self._task_draft_detail(decision.task_draft),
            },
        )
        if decision.action is FollowBrainAction.RESPOND:
            return decision.summary
        assert decision.task_draft is not None
        return self._start_follow_skill(
            request_id,
            decision.task_draft,
            based_on_runtime_revision=context.runtime_revision,
        )

    def _start_follow_skill(
        self,
        request_id: str,
        task_draft: FollowTaskDraft,
        *,
        based_on_runtime_revision: int,
    ) -> str:
        if (
            self.active_skill_call_id is not None
            or self.skill.state is not FollowState.IDLE
        ):
            self._emit("skill.call.rejected", {"reason": "base motion is occupied"})
            return "FollowPerson is already active."
        call_id = f"follow-skill-call-{self.accepted_skill_calls + 1}"
        graph_id = f"follow-task-graph-{self.accepted_skill_calls + 1}"
        try:
            graph = compile_follow_task_graph(
                task_draft,
                graph_id=graph_id,
                based_on_runtime_revision=based_on_runtime_revision,
            )
        except (TypeError, ValueError) as exc:
            self._emit(
                "task_graph.rejected",
                {"graph_id": graph_id, "reason": str(exc)},
            )
            return "Brain task draft was rejected; no Skill was started."
        self._emit(
            "skill.call.requested",
            {
                "call_id": call_id,
                "skill_id": FOLLOW_PERSON_SKILL_ID,
                "brain_request_id": request_id,
                "resources": ["base_motion", "person_tracker"],
                "graph_id": graph_id,
            },
        )
        self.active_skill_call_id = call_id
        try:
            graph_snapshot = self.task_graph.start(
                graph,
                now_ns=self.clock_ns(),
                dispatch=self._dispatch_graph_node,
            )
        except Exception as exc:
            self.active_skill_call_id = None
            self._emit(
                "task_graph.rejected",
                {"graph_id": graph_id, "reason": type(exc).__name__},
            )
            return "FollowPerson task graph failed during admission."
        if graph_snapshot.state is MissionTaskGraphState.FAILED:
            self.active_skill_call_id = None
            self._emit(
                "skill.call.rejected",
                {"call_id": call_id, "reason": graph_snapshot.detail},
            )
            return "FollowPerson failed during admission."
        snapshot = self.skill.snapshot
        self.accepted_skill_calls += 1
        self.revision += 1
        self._emit(
            "skill.call.accepted",
            {
                "call_id": call_id,
                "skill_id": FOLLOW_PERSON_SKILL_ID,
                "state": snapshot.state.value,
                "graph_id": graph_snapshot.graph_id,
                "current_node_id": graph_snapshot.current_node_id,
            },
        )
        return (
            "Mission task graph started: "
            f"{describe_follow_task_draft(task_draft)}."
        )

    def _dispatch_graph_node(
        self, node: MissionTaskNode, now_ns: int
    ) -> TaskDispatchResult:
        if node.operation_id == FOLLOW_START_OPERATION_ID:
            if self.skill.state is not FollowState.IDLE:
                return TaskDispatchResult(False, "FollowPerson is not idle")
            snapshot = self.skill.start(now_ns=now_ns)
            return TaskDispatchResult(
                snapshot.state is not FollowState.FAILED,
                snapshot.detail,
            )
        if node.operation_id == FOLLOW_PAUSE_OPERATION_ID:
            snapshot = self.skill.pause(now_ns=now_ns)
            return TaskDispatchResult(True, snapshot.detail)
        if node.operation_id == FOLLOW_RESUME_OPERATION_ID:
            snapshot = self.skill.resume(now_ns=now_ns)
            return TaskDispatchResult(True, snapshot.detail)
        return TaskDispatchResult(False, "task graph operation is unavailable")

    @staticmethod
    def _task_draft_detail(
        task_draft: FollowTaskDraft | None,
    ) -> list[dict[str, str | float | None]]:
        if task_draft is None:
            return []
        return [
            {
                "operation": step.operation.value,
                "duration_s": step.duration_s,
            }
            for step in task_draft.steps
        ]

    def _decision_problem(
        self, decision: object, context: FollowBrainContext
    ) -> str | None:
        if not isinstance(decision, FollowBrainDecision):
            return "wrong decision type"
        if decision.request_id != context.request_id:
            return "request correlation mismatch"
        if decision.based_on_runtime_revision != context.runtime_revision:
            return "decision used a stale revision"
        if self.revision != context.runtime_revision:
            return "runtime changed while Brain was deciding"
        if (
            decision.action is FollowBrainAction.CALL_SKILL
            and decision.skill_id not in context.available_skill_ids
        ):
            return "skill is not available"
        return None

    def _emit(self, event_type: str, detail: Mapping[str, Any]) -> None:
        event = {
            "schema_version": "longship.follow-system-event.v0",
            "event_type": event_type,
            "runtime_revision": self.revision,
            "active_skill_call_id": self.active_skill_call_id,
            "skill_state": self.skill.state.value,
            "task_graph": (
                self.task_graph.snapshot.to_dict()
                if self.task_graph.snapshot is not None
                else None
            ),
            "detail": dict(detail),
        }
        try:
            self.event_sink.publish(event)
        except Exception:
            pass

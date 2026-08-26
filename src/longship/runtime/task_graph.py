from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from longship.contracts.runtime.task_graph import (
    MissionTaskGraph,
    MissionTaskGraphState,
    MissionTaskNode,
    MissionTaskNodeState,
)


@dataclass(frozen=True, slots=True)
class TaskDispatchResult:
    accepted: bool
    detail: str

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise TypeError("task dispatch acceptance must be a boolean")
        if not isinstance(self.detail, str) or not self.detail.strip():
            raise ValueError("task dispatch detail is required")


TaskDispatcher = Callable[[MissionTaskNode, int], TaskDispatchResult]


@dataclass(frozen=True, slots=True)
class MissionTaskGraphSnapshot:
    graph_id: str
    graph_version: int
    state: MissionTaskGraphState
    transition_sequence: int
    cancellation_epoch: int
    current_node_id: str | None
    current_operation_id: str | None
    current_node_index: int | None
    node_count: int
    deadline_monotonic_ns: int | None
    node_states: tuple[tuple[str, MissionTaskNodeState], ...]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "longship.mission-task-graph-snapshot.v0",
            "graph_id": self.graph_id,
            "graph_version": self.graph_version,
            "state": self.state.value,
            "transition_sequence": self.transition_sequence,
            "cancellation_epoch": self.cancellation_epoch,
            "current_node_id": self.current_node_id,
            "current_operation_id": self.current_operation_id,
            "current_node_index": self.current_node_index,
            "node_count": self.node_count,
            "deadline_monotonic_ns": self.deadline_monotonic_ns,
            "nodes": [
                {"node_id": node_id, "state": state.value}
                for node_id, state in self.node_states
            ],
            "detail": self.detail,
        }


class SequentialMissionTaskGraphRuntime:
    """Execute the single-lane subset without model-owned timers or sleeps."""

    _TERMINAL = frozenset(
        {
            MissionTaskGraphState.SUCCEEDED,
            MissionTaskGraphState.CANCELLED,
            MissionTaskGraphState.FAILED,
        }
    )

    def __init__(self) -> None:
        self._graph: MissionTaskGraph | None = None
        self._state = MissionTaskGraphState.DECLARED
        self._node_states: list[MissionTaskNodeState] = []
        self._current_index: int | None = None
        self._deadline_ns: int | None = None
        self._transition_sequence = 0
        self._cancellation_epoch = 0
        self._detail = "no task graph admitted"

    @property
    def snapshot(self) -> MissionTaskGraphSnapshot | None:
        graph = self._graph
        if graph is None:
            return None
        current = (
            graph.nodes[self._current_index]
            if self._current_index is not None
            else None
        )
        return MissionTaskGraphSnapshot(
            graph_id=graph.graph_id,
            graph_version=graph.graph_version,
            state=self._state,
            transition_sequence=self._transition_sequence,
            cancellation_epoch=self._cancellation_epoch,
            current_node_id=current.node_id if current else None,
            current_operation_id=current.operation_id if current else None,
            current_node_index=self._current_index,
            node_count=len(graph.nodes),
            deadline_monotonic_ns=self._deadline_ns,
            node_states=tuple(
                (node.node_id, state)
                for node, state in zip(graph.nodes, self._node_states, strict=True)
            ),
            detail=self._detail,
        )

    def start(
        self,
        graph: MissionTaskGraph,
        *,
        now_ns: int,
        dispatch: TaskDispatcher,
    ) -> MissionTaskGraphSnapshot:
        if self._graph is not None and self._state not in self._TERMINAL:
            raise RuntimeError("a task graph is already running")
        self._validate_linear_slice(graph)
        self._graph = graph
        self._state = MissionTaskGraphState.DECLARED
        self._node_states = [MissionTaskNodeState.PENDING] * len(graph.nodes)
        self._current_index = None
        self._deadline_ns = None
        self._transition_sequence = 0
        self._detail = "task graph declared"
        self._enter_node(0, now_ns, dispatch)
        assert self.snapshot is not None
        return self.snapshot

    def advance(
        self, *, now_ns: int, dispatch: TaskDispatcher
    ) -> MissionTaskGraphSnapshot | None:
        graph = self._graph
        if graph is None or self._state is not MissionTaskGraphState.RUNNING:
            return self.snapshot
        if self._deadline_ns is None or now_ns < self._deadline_ns:
            return self.snapshot
        assert self._current_index is not None
        self._node_states[self._current_index] = MissionTaskNodeState.SUCCEEDED
        next_index = self._current_index + 1
        if next_index >= len(graph.nodes):
            self._current_index = None
            self._deadline_ns = None
            self._state = MissionTaskGraphState.SUCCEEDED
            self._transition_sequence += 1
            self._detail = "all task graph nodes succeeded"
        else:
            self._enter_node(next_index, now_ns, dispatch)
        return self.snapshot

    def cancel(self, reason: str) -> MissionTaskGraphSnapshot | None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("task graph cancellation reason is required")
        if self._graph is None or self._state in self._TERMINAL:
            return self.snapshot
        for index, state in enumerate(self._node_states):
            if state in {
                MissionTaskNodeState.PENDING,
                MissionTaskNodeState.RUNNING,
            }:
                self._node_states[index] = MissionTaskNodeState.CANCELLED
        self._state = MissionTaskGraphState.CANCELLED
        self._deadline_ns = None
        self._transition_sequence += 1
        self._cancellation_epoch += 1
        self._detail = reason.strip()
        return self.snapshot

    def _enter_node(
        self, index: int, now_ns: int, dispatch: TaskDispatcher
    ) -> None:
        assert self._graph is not None
        node = self._graph.nodes[index]
        self._current_index = index
        try:
            result = dispatch(node, now_ns)
        except Exception as exc:
            result = TaskDispatchResult(
                False, f"dispatcher raised {type(exc).__name__}"
            )
        self._transition_sequence += 1
        if not result.accepted:
            self._node_states[index] = MissionTaskNodeState.FAILED
            for pending_index in range(index + 1, len(self._node_states)):
                self._node_states[pending_index] = MissionTaskNodeState.CANCELLED
            self._state = MissionTaskGraphState.FAILED
            self._deadline_ns = None
            self._detail = result.detail
            return
        self._node_states[index] = MissionTaskNodeState.RUNNING
        self._state = MissionTaskGraphState.RUNNING
        duration_s = node.duration_s
        self._deadline_ns = (
            None
            if duration_s is None
            else now_ns + round(float(duration_s) * 1_000_000_000)
        )
        self._detail = result.detail

    @staticmethod
    def _validate_linear_slice(graph: MissionTaskGraph) -> None:
        expected_edges = {
            (graph.nodes[index].node_id, graph.nodes[index + 1].node_id)
            for index in range(len(graph.nodes) - 1)
        }
        actual_edges = {
            (edge.from_node_id, edge.to_node_id) for edge in graph.edges
        }
        if actual_edges != expected_edges or len(graph.edges) != len(expected_edges):
            raise ValueError("the executable task graph must be one linear lane")
        for node in graph.nodes[:-1]:
            if node.duration_s is None:
                raise ValueError("every non-final task node requires a duration")
        if graph.nodes[-1].duration_s is not None:
            raise ValueError("the final task node must be open-ended")

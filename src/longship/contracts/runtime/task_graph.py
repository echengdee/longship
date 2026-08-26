from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class MissionTaskGraphState(str, Enum):
    DECLARED = "declared"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"
    FAILED = "failed"


class MissionTaskNodeState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"
    FAILED = "failed"


def _validate_id(value: object, label: str) -> None:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")


@dataclass(frozen=True, slots=True)
class MissionTaskNode:
    """Executable semantic operation admitted by Runtime, never raw control."""

    node_id: str
    operation_id: str
    duration_s: float | None
    resource_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_id(self.node_id, "task node ID")
        _validate_id(self.operation_id, "task operation ID")
        if self.duration_s is not None:
            if (
                not isinstance(self.duration_s, (int, float))
                or isinstance(self.duration_s, bool)
                or not math.isfinite(float(self.duration_s))
                or float(self.duration_s) <= 0.0
            ):
                raise ValueError("task node duration must be finite and positive")
        if not self.resource_ids or len(set(self.resource_ids)) != len(
            self.resource_ids
        ):
            raise ValueError("task node resources must be non-empty and unique")
        for resource_id in self.resource_ids:
            _validate_id(resource_id, "task resource ID")


@dataclass(frozen=True, slots=True)
class MissionTaskEdge:
    edge_id: str
    from_node_id: str
    to_node_id: str
    when: str = "after_success"

    def __post_init__(self) -> None:
        _validate_id(self.edge_id, "task edge ID")
        _validate_id(self.from_node_id, "task edge source")
        _validate_id(self.to_node_id, "task edge destination")
        if self.when != "after_success":
            raise ValueError("this executable slice supports after_success edges only")


@dataclass(frozen=True, slots=True)
class MissionTaskGraph:
    """Small executable subset of the draft Longship MissionTaskGraph.

    The contract validates a bounded DAG. The first Runtime implementation
    intentionally admits only a single linear lane; parallel admission,
    barriers, and graph patches remain future contract slices.
    """

    graph_id: str
    graph_version: int
    based_on_runtime_revision: int
    nodes: tuple[MissionTaskNode, ...]
    edges: tuple[MissionTaskEdge, ...]

    def __post_init__(self) -> None:
        _validate_id(self.graph_id, "task graph ID")
        if type(self.graph_version) is not int or self.graph_version < 1:
            raise ValueError("task graph version must be positive")
        if (
            type(self.based_on_runtime_revision) is not int
            or self.based_on_runtime_revision < 0
        ):
            raise ValueError("task graph Runtime revision is invalid")
        if not 1 <= len(self.nodes) <= 64:
            raise ValueError("task graph node count is outside supported bounds")
        node_ids = tuple(node.node_id for node in self.nodes)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("task graph node IDs must be unique")
        edge_ids = tuple(edge.edge_id for edge in self.edges)
        if len(set(edge_ids)) != len(edge_ids):
            raise ValueError("task graph edge IDs must be unique")
        known = set(node_ids)
        adjacency = {node_id: [] for node_id in node_ids}
        indegree = {node_id: 0 for node_id in node_ids}
        for edge in self.edges:
            if edge.from_node_id not in known or edge.to_node_id not in known:
                raise ValueError("task graph edge references an unknown node")
            if edge.from_node_id == edge.to_node_id:
                raise ValueError("task graph cannot contain a self edge")
            adjacency[edge.from_node_id].append(edge.to_node_id)
            indegree[edge.to_node_id] += 1
        ready = [node_id for node_id, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            node_id = ready.pop()
            visited += 1
            for target in adjacency[node_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if visited != len(node_ids):
            raise ValueError("task graph must be acyclic")

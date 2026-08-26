from __future__ import annotations

from longship.brain.follow_person import (
    FollowTaskDraft,
    FollowTaskOperation,
)
from longship.contracts.runtime.task_graph import (
    MissionTaskEdge,
    MissionTaskGraph,
    MissionTaskNode,
)


FOLLOW_START_OPERATION_ID = "navigation.follow_person.start"
FOLLOW_PAUSE_OPERATION_ID = "navigation.follow_person.pause"
FOLLOW_RESUME_OPERATION_ID = "navigation.follow_person.resume"

_OPERATION_IDS = {
    FollowTaskOperation.FOLLOW: FOLLOW_START_OPERATION_ID,
    FollowTaskOperation.PAUSE: FOLLOW_PAUSE_OPERATION_ID,
    FollowTaskOperation.RESUME: FOLLOW_RESUME_OPERATION_ID,
}
_ALLOWED_TRANSITIONS = frozenset(
    {
        (FollowTaskOperation.FOLLOW, FollowTaskOperation.PAUSE),
        (FollowTaskOperation.PAUSE, FollowTaskOperation.RESUME),
        (FollowTaskOperation.RESUME, FollowTaskOperation.PAUSE),
    }
)


def compile_follow_task_graph(
    draft: FollowTaskDraft,
    *,
    graph_id: str,
    based_on_runtime_revision: int,
) -> MissionTaskGraph:
    """Validate an untrusted Follow draft and compile a single motion lane."""

    if not isinstance(draft, FollowTaskDraft):
        raise TypeError("Follow task draft is required")
    steps = draft.steps
    if steps[0].operation is not FollowTaskOperation.FOLLOW:
        raise ValueError("a Follow task graph must begin with follow")
    if steps[-1].duration_s is not None:
        raise ValueError("the final Follow task step must be open-ended")
    timed_duration_s = 0.0
    for index, step in enumerate(steps):
        if index < len(steps) - 1 and step.duration_s is None:
            raise ValueError("every non-final Follow task step needs a duration")
        if step.duration_s is not None:
            timed_duration_s += float(step.duration_s)
        if index and (steps[index - 1].operation, step.operation) not in (
            _ALLOWED_TRANSITIONS
        ):
            raise ValueError("Follow task step transition is invalid")
    if timed_duration_s > 300.0:
        raise ValueError("Follow task timed duration exceeds the mission bound")

    nodes = tuple(
        MissionTaskNode(
            node_id=f"follow-node-{index + 1}",
            operation_id=_OPERATION_IDS[step.operation],
            duration_s=(
                None if step.duration_s is None else float(step.duration_s)
            ),
            resource_ids=("base_motion", "person_tracker"),
        )
        for index, step in enumerate(steps)
    )
    edges = tuple(
        MissionTaskEdge(
            edge_id=f"follow-edge-{index + 1}",
            from_node_id=nodes[index].node_id,
            to_node_id=nodes[index + 1].node_id,
        )
        for index in range(len(nodes) - 1)
    )
    return MissionTaskGraph(
        graph_id=graph_id,
        graph_version=1,
        based_on_runtime_revision=based_on_runtime_revision,
        nodes=nodes,
        edges=edges,
    )


def describe_follow_task_draft(draft: FollowTaskDraft) -> str:
    parts: list[str] = []
    for step in draft.steps:
        label = step.operation.value
        if step.duration_s is not None:
            label += f" {float(step.duration_s):g}s"
        parts.append(label)
    return " -> ".join(parts)

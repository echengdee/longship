from __future__ import annotations

import unittest

from longship.brain.follow_person import (
    FollowTaskDraft,
    FollowTaskOperation,
    FollowTaskStep,
)
from longship.contracts.runtime.task_graph import (
    MissionTaskGraphState,
    MissionTaskNodeState,
)
from longship.runtime.follow_task_graph import (
    FOLLOW_PAUSE_OPERATION_ID,
    FOLLOW_RESUME_OPERATION_ID,
    FOLLOW_START_OPERATION_ID,
    compile_follow_task_graph,
)
from longship.runtime.task_graph import (
    SequentialMissionTaskGraphRuntime,
    TaskDispatchResult,
)


def _timed_draft() -> FollowTaskDraft:
    return FollowTaskDraft(
        (
            FollowTaskStep(FollowTaskOperation.FOLLOW, 3.0),
            FollowTaskStep(FollowTaskOperation.PAUSE, 1.0),
            FollowTaskStep(FollowTaskOperation.RESUME, None),
        )
    )


class FollowTaskGraphCompilerTests(unittest.TestCase):
    def test_compiles_bounded_steps_into_a_linear_motion_lane(self) -> None:
        graph = compile_follow_task_graph(
            _timed_draft(),
            graph_id="follow-graph-1",
            based_on_runtime_revision=4,
        )

        self.assertEqual(graph.based_on_runtime_revision, 4)
        self.assertEqual(
            tuple(node.operation_id for node in graph.nodes),
            (
                FOLLOW_START_OPERATION_ID,
                FOLLOW_PAUSE_OPERATION_ID,
                FOLLOW_RESUME_OPERATION_ID,
            ),
        )
        self.assertEqual(len(graph.edges), 2)

    def test_rejects_invalid_semantic_transition(self) -> None:
        draft = FollowTaskDraft(
            (
                FollowTaskStep(FollowTaskOperation.FOLLOW, 1.0),
                FollowTaskStep(FollowTaskOperation.RESUME, None),
            )
        )

        with self.assertRaisesRegex(ValueError, "transition"):
            compile_follow_task_graph(
                draft,
                graph_id="follow-graph-invalid",
                based_on_runtime_revision=0,
            )

    def test_final_step_must_be_open_ended(self) -> None:
        draft = FollowTaskDraft(
            (FollowTaskStep(FollowTaskOperation.FOLLOW, 2.0),)
        )

        with self.assertRaisesRegex(ValueError, "open-ended"):
            compile_follow_task_graph(
                draft,
                graph_id="follow-graph-timed-final",
                based_on_runtime_revision=0,
            )


class SequentialMissionTaskGraphRuntimeTests(unittest.TestCase):
    def test_runtime_advances_deadlines_without_blocking(self) -> None:
        graph = compile_follow_task_graph(
            _timed_draft(),
            graph_id="follow-graph-runtime",
            based_on_runtime_revision=0,
        )
        dispatched: list[tuple[str, int]] = []

        def dispatch(node, now_ns: int) -> TaskDispatchResult:
            dispatched.append((node.operation_id, now_ns))
            return TaskDispatchResult(True, f"started {node.operation_id}")

        runtime = SequentialMissionTaskGraphRuntime()
        snapshot = runtime.start(graph, now_ns=1_000_000_000, dispatch=dispatch)

        self.assertEqual(snapshot.state, MissionTaskGraphState.RUNNING)
        self.assertEqual(snapshot.current_operation_id, FOLLOW_START_OPERATION_ID)
        self.assertEqual(snapshot.deadline_monotonic_ns, 4_000_000_000)

        runtime.advance(now_ns=3_999_999_999, dispatch=dispatch)
        self.assertEqual(len(dispatched), 1)
        pause = runtime.advance(now_ns=4_000_000_000, dispatch=dispatch)
        assert pause is not None
        self.assertEqual(pause.current_operation_id, FOLLOW_PAUSE_OPERATION_ID)
        self.assertEqual(pause.deadline_monotonic_ns, 5_000_000_000)

        resume = runtime.advance(now_ns=5_000_000_000, dispatch=dispatch)
        assert resume is not None
        self.assertEqual(resume.current_operation_id, FOLLOW_RESUME_OPERATION_ID)
        self.assertIsNone(resume.deadline_monotonic_ns)
        self.assertEqual(len(dispatched), 3)

    def test_cancellation_is_terminal_and_monotonic(self) -> None:
        graph = compile_follow_task_graph(
            _timed_draft(),
            graph_id="follow-graph-cancel",
            based_on_runtime_revision=0,
        )
        runtime = SequentialMissionTaskGraphRuntime()
        runtime.start(
            graph,
            now_ns=1,
            dispatch=lambda node, now_ns: TaskDispatchResult(True, "accepted"),
        )

        cancelled = runtime.cancel("operator stop")
        assert cancelled is not None
        self.assertEqual(cancelled.state, MissionTaskGraphState.CANCELLED)
        self.assertEqual(cancelled.cancellation_epoch, 1)
        self.assertEqual(
            tuple(state for _, state in cancelled.node_states),
            (
                MissionTaskNodeState.CANCELLED,
                MissionTaskNodeState.CANCELLED,
                MissionTaskNodeState.CANCELLED,
            ),
        )
        same = runtime.cancel("duplicate operator stop")
        assert same is not None
        self.assertEqual(same.cancellation_epoch, 1)


if __name__ == "__main__":
    unittest.main()

"""Tests for diagnostic short-step NoMaD trajectory stitching."""

from __future__ import annotations

import math

import pytest

from longship.navigation.common import TimePoint
from longship.navigation.map_engine.models import (
    AnchorId,
    NodeId,
    ResourceId,
    SegmentId,
    SnapshotId,
)
from longship.navigation.ports.trajectory_policy import (
    PolicyNativeWaypoint,
    TrajectoryCandidate,
    TrajectoryCandidateId,
    TrajectoryCandidateSet,
)
from tools.trajectory_stitching import (
    OfficialDemoTrajectoryStitcher,
    ShortStepTrajectoryStitcher,
    median_candidate_path,
)


def test_median_path_rejects_one_outlying_candidate() -> None:
    candidates = _candidate_set(
        (
            ((1.0, 0.0), (2.0, 0.0)),
            ((1.2, 0.2), (2.2, 0.2)),
            ((100.0, -50.0), (100.0, -50.0)),
        )
    )

    path = median_candidate_path(candidates)

    assert tuple((point.x, point.y) for point in path) == (
        (1.2, 0.0),
        (2.2, 0.0),
    )


def test_stitcher_interpolates_fixed_distance_and_composes_heading() -> None:
    stitcher = ShortStepTrajectoryStitcher(step_distance=1.0)
    left_turn = _candidate_set((((0.0, 2.0),),) * 3)

    first = stitcher.append(left_turn, source_timestamp_s=1.0)
    second = stitcher.append(left_turn, source_timestamp_s=2.0)

    assert first is not None
    assert second is not None
    assert first.local_step.x == pytest.approx(0.0)
    assert first.local_step.y == pytest.approx(1.0)
    assert first.stitched_pose.x == pytest.approx(0.0)
    assert first.stitched_pose.y == pytest.approx(1.0)
    assert first.stitched_pose.heading_rad == pytest.approx(math.pi / 2.0)
    assert second.stitched_pose.x == pytest.approx(-1.0)
    assert second.stitched_pose.y == pytest.approx(1.0)
    assert abs(second.stitched_pose.heading_rad) == pytest.approx(math.pi)
    assert len(stitcher.poses) == 3


def test_stitcher_uses_shorter_complete_path_when_needed() -> None:
    stitcher = ShortStepTrajectoryStitcher(step_distance=1.0)

    update = stitcher.append(
        _candidate_set((((0.3, 0.4),),) * 3),
        source_timestamp_s=1.0,
    )

    assert update is not None
    assert update.actual_step_distance == pytest.approx(0.5)
    assert update.local_step.x == pytest.approx(0.3)
    assert update.local_step.y == pytest.approx(0.4)


def test_official_demo_uses_sample_zero_waypoint_two_and_pd_limits() -> None:
    stitcher = OfficialDemoTrajectoryStitcher()
    candidates = _candidate_set(
        (
            ((1.0, 0.0), (2.0, 1.0), (3.0, 3.0)),
            ((1.0, -10.0), (2.0, -10.0), (3.0, -10.0)),
        )
    )

    update = stitcher.append(candidates, source_timestamp_s=1.0)

    assert update.selected_candidate_index == 0
    assert update.selected_waypoint_index == 2
    assert update.control_waypoint is not None
    assert update.control_waypoint.x == pytest.approx(3.0)
    assert update.control_waypoint.y == pytest.approx(3.0)
    assert update.scaled_control_waypoint is not None
    assert update.scaled_control_waypoint.x == pytest.approx(0.15)
    assert update.scaled_control_waypoint.y == pytest.approx(0.15)
    assert update.linear_velocity == pytest.approx(0.2)
    assert update.angular_velocity == pytest.approx(0.4)
    assert update.actual_step_distance == pytest.approx(0.05)
    assert update.stitched_pose.heading_rad == pytest.approx(0.1)
    assert update.stitched_pose.x == pytest.approx(0.5 * math.sin(0.1))
    assert update.stitched_pose.y == pytest.approx(
        0.5 * (1.0 - math.cos(0.1))
    )


def _candidate_set(
    paths: tuple[tuple[tuple[float, float], ...], ...],
) -> TrajectoryCandidateSet:
    timestamp = TimePoint(clock_id="camera", nanoseconds=1_000_000_000)
    return TrajectoryCandidateSet(
        snapshot_id=SnapshotId("snapshot"),
        segment_id=SegmentId("edge"),
        source_node_id=NodeId("node0"),
        target_node_id=NodeId("node1"),
        target_anchor_id=AnchorId("anchor"),
        goal_resource_id=ResourceId("goal"),
        observation_time=timestamp,
        produced_at=timestamp,
        temporal_distance=4.0,
        coordinate_frame="nomad.policy_native.robot_frame.v1",
        coordinate_units="nomad.policy_native.v1",
        sampling_seed=0,
        candidates=tuple(
            TrajectoryCandidate(
                candidate_id=TrajectoryCandidateId(f"sample-{index}"),
                waypoints=tuple(
                    PolicyNativeWaypoint(
                        step_index=step_index,
                        x=x,
                        y=y,
                    )
                    for step_index, (x, y) in enumerate(path)
                ),
            )
            for index, path in enumerate(paths)
        ),
        policy_id="nomad",
        image_profile_id="profile",
        model_artifact_id="model",
        model_artifact_digest="sha256:" + "1" * 64,
    )

"""Tests for the NoMaD video diagnostic overlay."""

from __future__ import annotations

from PIL import Image

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
from tools.trajectory_overlay import (
    TrajectoryOverlayState,
    draw_trajectory_overlay,
)
from tools.trajectory_stitching import (
    ShortStepTrajectoryStitcher,
)


def test_draws_status_goal_and_all_candidates_without_resizing_frame() -> None:
    candidates = tuple(
        TrajectoryCandidate(
            candidate_id=TrajectoryCandidateId(f"sample-{index}"),
            waypoints=tuple(
                PolicyNativeWaypoint(
                    step_index=step,
                    x=float(step + 1),
                    y=(index - 1.5) * 0.3 * (step + 1),
                )
                for step in range(8)
            ),
        )
        for index in range(4)
    )
    timestamp = TimePoint(clock_id="camera", nanoseconds=7_000_000_000)
    candidate_set = TrajectoryCandidateSet(
        snapshot_id=SnapshotId("snapshot"),
        segment_id=SegmentId("edge-0000"),
        source_node_id=NodeId("node-0000"),
        target_node_id=NodeId("node-0001"),
        target_anchor_id=AnchorId("node-0001:visual"),
        goal_resource_id=ResourceId("node-0001:goal-image"),
        observation_time=timestamp,
        produced_at=timestamp,
        temporal_distance=5.0,
        coordinate_frame="nomad.policy_native.robot_frame.v1",
        coordinate_units="nomad.policy_native.v1",
        sampling_seed=3,
        candidates=candidates,
        policy_id="nomad",
        image_profile_id="profile",
        model_artifact_id="model",
        model_artifact_digest="sha256:" + "1" * 64,
    )
    source = Image.new("RGB", (1280, 720), color=(40, 50, 60))
    stitcher = ShortStepTrajectoryStitcher(step_distance=0.15)
    stitch_update = stitcher.append(
        candidate_set,
        source_timestamp_s=7.0,
    )

    output = draw_trajectory_overlay(
        source,
        TrajectoryOverlayState(
            source_timestamp_s=7.0,
            phase="tracking",
            current_node="node-0000",
            target_node="node-0001",
            status_detail="distance=5.0",
            candidate_set=candidate_set,
            goal_image=Image.new("RGB", (320, 180), color="blue"),
            stitched_path=stitcher.poses,
            stitch_update=stitch_update,
            stitch_panel_title=stitcher.panel_title,
            stitch_panel_detail=stitcher.panel_detail,
        ),
    )

    assert output.size == source.size
    assert output.mode == "RGB"
    assert output.tobytes() != source.tobytes()

"""Records live NoMaD trajectories against one fixed topomap goal image.

This diagnostic tool intentionally excludes localization, route progression,
and chassis control.  It is for observing how live camera motion changes NoMaD
trajectories while the source node and visual goal remain fixed.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time

from longship.navigation.map_engine.models import (
    MapId,
    MapSelector,
    MapVersion,
    NodeId,
    TopologyQuery,
)
from longship.navigation.ports.trajectory_policy import (
    TrajectoryPolicyError,
    VisualGoalTrajectoryRequest,
)
from longship.navigation.localization_engine.service import MonotonicTimeSource
from longship_adapter import (
    LocalFileGoalImageLoader,
    NomadTopomapMapConfig,
    NomadTrajectoryPolicyConfig,
    NomadVisualGoalTrajectoryPolicy,
    create_nomad_topomap_engine,
    resolve_visual_target_goal,
)
from nomad_runtime import (
    NomadConfig,
    NomadPolicy,
    NomadTrajectorySession,
    default_checkpoint_path,
)
from tools.live_trajectory_video import (
    LiveTrajectoryVideoWriter,
    ObservationFrameCache,
    as_rgb_image,
)
from tools.ros2_image_source import (
    Ros2ImageFrameSource,
    Ros2ImageFrameSourceConfig,
)
from tools.trajectory_overlay import (
    TrajectoryOverlayState,
    draw_trajectory_overlay,
)


_IMAGE_PROFILE_ID = "nomad.rgb.direct_resize_96x96.imagenet.v1"
_MODEL_ARTIFACT_ID = "nomad.pth"
_OBSERVATION_CLOCK_ID = "monotonic"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record live NoMaD trajectories against one fixed topomap goal "
            "without localization, route progression, or robot commands."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint_path(),
        help="NoMaD checkpoint (default: repository LFS asset).",
    )
    parser.add_argument("--topomap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--color-topic", default="/camera/camera/color/image_raw"
    )
    parser.add_argument("--source-node", default="node-0000")
    parser.add_argument("--goal-node", default="node-0001")
    parser.add_argument("--inference-hz", type=float, default=4.0)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--sampling-seed-base", type=int, default=0)
    parser.add_argument("--max-observation-age-s", type=float, default=2.0)
    parser.add_argument("--maximum-frame-gap-s", type=float, default=0.5)
    parser.add_argument("--source-read-timeout-s", type=float, default=8.0)
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=0.0,
        help="Stop after this duration; use 0 to run until interrupted.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive_values = (
        args.inference_hz,
        args.max_observation_age_s,
        args.maximum_frame_gap_s,
        args.source_read_timeout_s,
    )
    if not all(
        math.isfinite(value) and value > 0.0 for value in positive_values
    ):
        raise ValueError("fixed-goal timing values must be finite and positive")
    if not math.isfinite(args.run_seconds) or args.run_seconds < 0.0:
        raise ValueError("run_seconds must be finite and non-negative")
    if args.num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    if args.sampling_seed_base < 0:
        raise ValueError("sampling_seed_base must be non-negative")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


async def _run(args: argparse.Namespace) -> None:
    _validate_args(args)
    checkpoint = args.checkpoint.expanduser().resolve()
    topomap_root = args.topomap.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not topomap_root.is_dir():
        raise FileNotFoundError(f"topomap not found: {topomap_root}")

    clock = MonotonicTimeSource(clock_id=_OBSERVATION_CLOCK_ID)
    checkpoint_digest = _sha256(checkpoint)
    map_engine = create_nomad_topomap_engine(
        NomadTopomapMapConfig(
            root=topomap_root,
            map_id=MapId("live-nomad-fixed-goal"),
            version=MapVersion("local"),
            published_at=clock.now(),
            model_artifact_id=_MODEL_ARTIFACT_ID,
            model_artifact_digest=checkpoint_digest,
            image_profile_id=_IMAGE_PROFILE_ID,
        )
    )
    snapshot = await map_engine.get_snapshot(
        MapSelector(
            map_id=MapId("live-nomad-fixed-goal"),
            version=MapVersion("local"),
        )
    )
    source_node_id = NodeId(args.source_node)
    goal_node_id = NodeId(args.goal_node)
    topology = await map_engine.query_topology(snapshot, TopologyQuery())
    segments = tuple(
        segment
        for segment in topology.segments
        if (
            segment.source_node_id == source_node_id
            and segment.target_node_id == goal_node_id
        )
    )
    if len(segments) != 1:
        raise ValueError(
            "fixed source/goal pair must resolve to exactly one directed "
            "topomap segment"
        )
    binding = await resolve_visual_target_goal(
        map_engine=map_engine,
        snapshot=snapshot,
        target_node_id=goal_node_id,
    )

    model = NomadPolicy.from_checkpoint(
        checkpoint,
        config=NomadConfig(),
        device=args.device,
        strict=True,
    )
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="nomad-live-fixed-goal",
    )
    source: Ros2ImageFrameSource | None = None
    writer: LiveTrajectoryVideoWriter | None = None
    try:
        goal_loader = LocalFileGoalImageLoader((topomap_root,))
        policy = NomadVisualGoalTrajectoryPolicy(
            session=NomadTrajectorySession(model),
            goal_image_loader=goal_loader,
            inference_executor=executor,
            config=NomadTrajectoryPolicyConfig(
                policy_id="nomad-trajectory-live-fixed-goal",
                image_profile_id=_IMAGE_PROFILE_ID,
                model_artifact_id=_MODEL_ARTIFACT_ID,
                model_artifact_digest=checkpoint_digest,
                observation_clock_id=_OBSERVATION_CLOCK_ID,
                time_source=clock,
            ),
        )
        frame_cache = ObservationFrameCache(policy, maximum_frames=64)
        source = Ros2ImageFrameSource(
            Ros2ImageFrameSourceConfig(
                topic_name=args.color_topic,
                image_profile_id=_IMAGE_PROFILE_ID,
                node_name="nomad_fixed_goal_image_source",
            )
        )
        goal = goal_loader.load(binding.resource)
        goal_image = as_rgb_image(
            goal.image,
            layout=goal.layout,
            channel_order=goal.channel_order,
            value_range=goal.value_range,
        )
        writer = LiveTrajectoryVideoWriter(
            output,
            frames_per_second=args.inference_hz,
        )
        await source.start()
        started_at = time.monotonic()
        next_inference_s: float | None = None
        last_source_timestamp_s: float | None = None
        inference_count = 0
        while (
            args.run_seconds == 0.0
            or time.monotonic() - started_at < args.run_seconds
        ):
            try:
                frame = await asyncio.wait_for(
                    source.read(),
                    timeout=args.source_read_timeout_s,
                )
            except TimeoutError as error:
                raise RuntimeError(
                    "timed out waiting for a ROS 2 RGB frame"
                ) from error
            if frame is None:
                raise RuntimeError("ROS 2 source stopped during fixed-goal run")
            if (
                last_source_timestamp_s is not None
                and frame.source_timestamp_s is not None
                and frame.source_timestamp_s - last_source_timestamp_s
                > args.maximum_frame_gap_s
            ):
                frame_cache.clear_observations()
            if frame.source_timestamp_s is not None:
                last_source_timestamp_s = frame.source_timestamp_s
            frame_cache.submit_observation(
                frame.image,
                frame.timestamp_s,
                layout=frame.layout,
                channel_order=frame.channel_order,
                value_range=frame.value_range,
            )
            frame_cache.attach_source_timestamp(
                frame.timestamp_s,
                frame.source_timestamp_s,
            )
            if (
                next_inference_s is not None
                and frame.timestamp_s < next_inference_s
            ):
                continue
            request_time = clock.now()
            try:
                candidates = await policy.generate_trajectories(
                    VisualGoalTrajectoryRequest(
                        snapshot_id=snapshot.snapshot_id,
                        segment_id=segments[0].segment_id,
                        source_node_id=source_node_id,
                        target_node_id=goal_node_id,
                        target_anchor_id=binding.anchor.anchor_id,
                        goal_resource=binding.resource,
                        requested_at=request_time,
                        max_observation_age_s=args.max_observation_age_s,
                        num_candidates=args.num_candidates,
                        sampling_seed=(
                            args.sampling_seed_base + inference_count
                        ),
                        expected_image_profile_id=_IMAGE_PROFILE_ID,
                        expected_model_artifact_id=_MODEL_ARTIFACT_ID,
                        expected_model_artifact_digest=checkpoint_digest,
                    )
                )
            except TrajectoryPolicyError as error:
                if not error.retryable:
                    raise
                continue
            cached_frame = frame_cache.take(candidates.observation_time)
            if cached_frame is None:
                print(
                    json.dumps(
                        {
                            "kind": "overlay_frame_unavailable",
                            "observation_time_s": (
                                candidates.observation_time.nanoseconds / 1e9
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            rendered = draw_trajectory_overlay(
                as_rgb_image(
                    cached_frame.image,
                    layout=cached_frame.layout,
                    channel_order=cached_frame.channel_order,
                    value_range=cached_frame.value_range,
                ),
                TrajectoryOverlayState(
                    source_timestamp_s=(
                        cached_frame.timestamp_s
                        if cached_frame.source_timestamp_s is None
                        else cached_frame.source_timestamp_s
                    ),
                    phase="fixed_goal",
                    current_node=str(source_node_id),
                    target_node=str(goal_node_id),
                    status_detail=(
                        f"fixed target; distance="
                        f"{candidates.temporal_distance:.3f} native units"
                    ),
                    candidate_set=candidates,
                    goal_image=goal_image,
                ),
            )
            writer.write_rendered(rendered)
            inference_count += 1
            next_inference_s = (
                frame.timestamp_s + 1.0 / args.inference_hz
            )
            print(
                json.dumps(
                    {
                        "kind": "fixed_goal_trajectory",
                        "inference_index": inference_count,
                        "source_node": str(source_node_id),
                        "goal_node": str(goal_node_id),
                        "temporal_distance": candidates.temporal_distance,
                        "video_frames_written": writer.frames_written,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        print(
            json.dumps(
                {
                    "kind": "summary",
                    "source_node": str(source_node_id),
                    "goal_node": str(goal_node_id),
                    "video_output": str(output),
                    "video_frames_written": writer.frames_written,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if source is not None:
            await source.stop()
        if writer is not None:
            writer.close()
        await asyncio.to_thread(
            executor.shutdown,
            wait=True,
            cancel_futures=True,
        )


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()

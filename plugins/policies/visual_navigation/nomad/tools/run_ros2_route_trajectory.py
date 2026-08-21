"""Publishes NoMaD local trajectories from a live ROS 2 RGB camera topic.

This is a Navigation Harness composition root.  It does not create chassis or
controller commands; its only public output is a RoutePlan and JSONL records
from ``LocalTrajectoryStream``.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import time

from PIL import Image

from longship.navigation.common import TimePoint
from longship.navigation.local_trajectory_engine import (
    LocalTrajectoryEngineConfig,
    LocalTrajectoryPublication,
    LocalTrajectoryState,
    LocalTrajectoryStream,
    LocalTrajectoryStreamId,
    LocalTrajectoryUpdateOutcome,
    RouteBoundLocalTrajectoryEngine,
    WaitForLocalTrajectoryRequest,
)
from longship.navigation.localization_engine.fixed_start_visual import (
    FIXED_START_NODE_ID,
    FixedStartVisualLocalizationEngine,
    FixedStartVisualTrackingProfile,
)
from longship.navigation.localization_engine.models import (
    BeliefStreamId,
    BeliefUpdateOutcome,
    LocationBelief,
    LocalizationStatus,
    NodeLocation,
    WaitForUpdateRequest,
)
from longship.navigation.localization_engine.service import (
    ContinuousLocalizationService,
    LocalizationServiceConfig,
    MonotonicTimeSource,
)
from longship.navigation.map_engine.interface import MapEngine
from longship.navigation.map_engine.models import (
    AnchorId,
    AnchorPurpose,
    AnchorQuery,
    MapEntityKind,
    MapId,
    MapSelector,
    MapSnapshot,
    MapVersion,
    NodeId,
)
from longship.navigation.planning_engine import TopologicalPlanningEngine
from longship.navigation.planning_engine.models import (
    PlanningOutcome,
    PlanningRequestId,
    PlanningTarget,
    RoutePlan,
    RoutePlanningRequest,
)
from longship.navigation.runtime import (
    LocalizationDrivenLocalTrajectoryService,
    LocalizationObservationCompletionPolicy,
    LocalizationRuntime,
    LocalizationRuntimeConfig,
    LocalTrajectoryServiceConfig,
)
from longship_adapter import (
    LocalFileGoalImageLoader,
    NomadObservationFanout,
    NomadObservationProducer,
    NomadObservationProducerConfig,
    NomadObservationProducerState,
    NomadTopomapMapConfig,
    NomadTrajectoryPolicyConfig,
    NomadVisualGoalDistancePolicy,
    NomadVisualGoalTrajectoryPolicy,
    NomadVisualPolicyConfig,
    create_nomad_topomap_engine,
    resolve_visual_target_goal,
)
from nomad_runtime import (
    NomadConfig,
    NomadDistanceSession,
    NomadPolicy,
    NomadTrajectorySession,
)
from tools.ros2_image_source import (
    Ros2ImageFrameSource,
    Ros2ImageFrameSourceConfig,
)
from tools.live_trajectory_video import (
    LiveTrajectoryVideoWriter,
    ObservationFrameCache,
    as_rgb_image,
    write_publication_frame,
)


_IMAGE_PROFILE_ID = "nomad.rgb.direct_resize_96x96.imagenet.v1"
_MODEL_ARTIFACT_ID = "nomad.pth"
_OBSERVATION_CLOCK_ID = "monotonic"


@dataclass(frozen=True, slots=True)
class _TrajectoryRecord:
    """One external trajectory-stream record with source timing diagnostics."""

    source_timestamp_s: float | None
    publication: LocalTrajectoryPublication


class _GoalImageCache:
    """Loads one digest-verified target image per map node for rendering."""

    def __init__(
        self,
        *,
        map_engine: MapEngine,
        snapshot: MapSnapshot,
        goal_image_loader: LocalFileGoalImageLoader,
    ) -> None:
        self._map_engine = map_engine
        self._snapshot = snapshot
        self._goal_image_loader = goal_image_loader
        self._images: dict[NodeId, Image.Image] = {}

    async def get(self, publication: LocalTrajectoryPublication) -> Image.Image | None:
        target_node_id = publication.target_node_id
        if target_node_id is None:
            return None
        cached = self._images.get(target_node_id)
        if cached is not None:
            return cached
        binding = await resolve_visual_target_goal(
            map_engine=self._map_engine,
            snapshot=self._snapshot,
            target_node_id=target_node_id,
        )
        decoded = self._goal_image_loader.load(binding.resource)
        image = as_rgb_image(
            decoded.image,
            layout=decoded.layout,
            channel_order=decoded.channel_order,
            value_range=decoded.value_range,
        )
        self._images[target_node_id] = image
        return image


class _TrajectoryCollector:
    """Writes the exact stream consumed by a future controller adapter."""

    def __init__(
        self,
        *,
        stream: LocalTrajectoryStream,
        producer: NomadObservationProducer,
        output_path: Path | None,
        update_timeout_s: float,
        overlay_frame_cache: ObservationFrameCache | None = None,
        overlay_video_writer: LiveTrajectoryVideoWriter | None = None,
        goal_image_cache: _GoalImageCache | None = None,
    ) -> None:
        self._stream = stream
        self._producer = producer
        self._output_path = output_path
        self._update_timeout_s = update_timeout_s
        self._overlay_frame_cache = overlay_frame_cache
        self._overlay_video_writer = overlay_video_writer
        self._goal_image_cache = goal_image_cache
        self.records: list[_TrajectoryRecord] = []

    async def run(self) -> None:
        output = None
        try:
            if self._output_path is not None:
                self._output_path.parent.mkdir(parents=True, exist_ok=True)
                output = self._output_path.open("w", encoding="utf-8")
            revision = self._stream.get_latest().revision
            while True:
                result = await self._stream.wait_for_update(
                    WaitForLocalTrajectoryRequest(
                        after_revision=revision,
                        timeout_s=self._update_timeout_s,
                    )
                )
                if result.outcome != LocalTrajectoryUpdateOutcome.UPDATED:
                    raise RuntimeError(
                        "timed out waiting for a local trajectory publication"
                    )
                publication = result.publication
                record = _TrajectoryRecord(
                    source_timestamp_s=(
                        self._producer.get_status()
                        .last_submitted_source_timestamp_s
                    ),
                    publication=publication,
                )
                self.records.append(record)
                line = json.dumps(
                    {"kind": "local_trajectory", **_jsonable(record)},
                    sort_keys=True,
                )
                print(line, flush=True)
                if output is not None:
                    output.write(line)
                    output.write("\n")
                    output.flush()
                await self._write_overlay(publication)
                revision = publication.revision
                if publication.state in (
                    LocalTrajectoryState.ROUTE_COMPLETED,
                    LocalTrajectoryState.FAULTED,
                ):
                    return
        finally:
            if output is not None:
                output.close()
            if self._overlay_video_writer is not None:
                self._overlay_video_writer.close()

    async def _write_overlay(
        self, publication: LocalTrajectoryPublication
    ) -> None:
        if self._overlay_video_writer is None:
            return
        if self._overlay_frame_cache is None or self._goal_image_cache is None:
            raise RuntimeError("overlay writer dependencies are incomplete")
        goal_image = await self._goal_image_cache.get(publication)
        wrote = write_publication_frame(
            cache=self._overlay_frame_cache,
            writer=self._overlay_video_writer,
            publication=publication,
            goal_image=goal_image,
        )
        if publication.state == LocalTrajectoryState.ACTIVE and not wrote:
            raise RuntimeError(
                "missing cached source frame for active trajectory publication"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Localize against a NoMaD topomap from ROS 2 RGB input and "
            "publish NoMaD LocalTrajectoryStream records without commands."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--topomap", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--color-topic",
        default="/camera/camera/color/image_raw",
    )
    parser.add_argument("--image-profile-id", default=_IMAGE_PROFILE_ID)
    parser.add_argument(
        "--center-crop-aspect",
        type=_aspect_ratio,
        default=None,
        help="Use '4:3', another ratio, or 'none' (default).",
    )
    parser.add_argument("--observation-sample-hz", type=float, default=9.0)
    parser.add_argument(
        "--localization-tick-period-s",
        type=float,
        default=1.0 / 9.0,
    )
    parser.add_argument("--maximum-frame-gap-s", type=float, default=0.5)
    parser.add_argument("--source-read-timeout-s", type=float, default=2.0)
    parser.add_argument("--max-observation-age-s", type=float, default=0.5)
    parser.add_argument("--publication-validity-s", type=float, default=0.5)
    parser.add_argument("--close-threshold", type=float, default=3.0)
    parser.add_argument("--start-close-confirmations", type=int, default=2)
    parser.add_argument("--successor-close-confirmations", type=int, default=1)
    parser.add_argument("--normal-distance-maximum", type=float, default=15.0)
    parser.add_argument(
        "--untrusted-distance-minimum",
        type=float,
        default=18.0,
    )
    parser.add_argument("--lost-confirmations", type=int, default=3)
    parser.add_argument("--tracking-candidate-count", type=int, default=3)
    parser.add_argument("--evidence-window-size", type=int, default=3)
    parser.add_argument("--relative-advantage-minimum", type=float, default=1.0)
    parser.add_argument("--relative-advance-confirmations", type=int, default=2)
    parser.add_argument("--relative-distance-maximum", type=float, default=5.0)
    parser.add_argument("--lookahead-close-confirmations", type=int, default=2)
    parser.add_argument("--relocalization-candidate-count", type=int, default=8)
    parser.add_argument("--relocalization-close-confirmations", type=int, default=2)
    parser.add_argument("--belief-publish-period-s", type=float, default=0.25)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--selected-candidate-index", type=int, default=0)
    parser.add_argument("--sampling-seed-base", type=int, default=0)
    parser.add_argument("--initial-location-timeout-s", type=float, default=30.0)
    parser.add_argument("--trajectory-update-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=30.0,
        help="Stop after this duration; use 0 to run until interrupted.",
    )
    parser.add_argument("--route-plan-output", type=Path)
    parser.add_argument("--trajectory-output", type=Path)
    parser.add_argument(
        "--overlay-video-output",
        type=Path,
        help=(
            "Write one RGB overlay frame per active NoMaD trajectory proposal. "
            "The overlay uses policy-native robot-frame coordinates."
        ),
    )
    parser.add_argument(
        "--overlay-video-fps",
        type=float,
        help=(
            "Encoded frame rate; defaults to the steady trajectory inference "
            "rate (1 / belief-publish-period-s)."
        ),
    )
    return parser.parse_args()


def _aspect_ratio(value: str) -> float | None:
    if value == "none":
        return None
    if ":" in value:
        width, height = value.split(":", maxsplit=1)
        return float(width) / float(height)
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _single_node_location(belief: LocationBelief) -> NodeId | None:
    if belief.status != LocalizationStatus.TRACKING:
        return None
    locations = tuple(
        hypothesis.topological_location
        for hypothesis in belief.hypotheses
        if isinstance(hypothesis.topological_location, NodeLocation)
    )
    if len(locations) != 1:
        return None
    return locations[0].node_id


async def _wait_for_initial_location(
    *,
    engine: FixedStartVisualLocalizationEngine,
    producer: NomadObservationProducer,
    timeout_s: float,
) -> LocationBelief:
    async def wait() -> LocationBelief:
        belief = engine.get_belief()
        revision = belief.revision
        while True:
            node_id = _single_node_location(belief)
            if node_id is not None:
                if node_id != FIXED_START_NODE_ID:
                    raise RuntimeError(
                        "fixed-start localization did not begin at node-0000"
                    )
                return belief
            producer_status = producer.get_status()
            if producer_status.state in (
                NomadObservationProducerState.COMPLETED,
                NomadObservationProducerState.FAULTED,
            ):
                raise RuntimeError(
                    "ROS 2 source stopped before fixed-start localization "
                    "became usable"
                )
            update = await engine.wait_for_update(
                WaitForUpdateRequest(
                    after_revision=revision,
                    timeout_s=1.0,
                )
            )
            if update.outcome == BeliefUpdateOutcome.UPDATED:
                belief = update.belief
                revision = belief.revision

    try:
        return await asyncio.wait_for(wait(), timeout=timeout_s)
    except TimeoutError as error:
        raise RuntimeError(
            "timed out waiting for fixed-start localization"
        ) from error


async def _resolve_route_goal(
    *,
    map_engine: MapEngine,
    snapshot: MapSnapshot,
) -> tuple[NodeId, AnchorId]:
    anchors = await map_engine.query_anchors(
        snapshot,
        AnchorQuery(purposes=frozenset({AnchorPurpose.COMPLETION})),
    )
    node_anchors = tuple(
        (NodeId(anchor.attached_to.entity_id), anchor.anchor_id)
        for anchor in anchors.anchors
        if anchor.attached_to.kind == MapEntityKind.NODE
    )
    if len(node_anchors) != 1:
        raise RuntimeError(
            "live ROS 2 runner requires exactly one completion anchor"
        )
    return node_anchors[0]


async def _plan_route(
    *,
    map_engine: MapEngine,
    snapshot: MapSnapshot,
    belief: LocationBelief,
    requested_at: TimePoint,
) -> RoutePlan:
    goal_node_id, completion_anchor_id = await _resolve_route_goal(
        map_engine=map_engine,
        snapshot=snapshot,
    )
    result = await TopologicalPlanningEngine(map_engine).plan_route(
        RoutePlanningRequest(
            request_id=PlanningRequestId("nomad-live-ros2-route"),
            requested_at=requested_at,
            snapshot=snapshot,
            location_belief=belief,
            target=PlanningTarget(
                target_ref="topomap-completion",
                candidate_node_ids=(goal_node_id,),
                completion_anchor_ids=(completion_anchor_id,),
            ),
        )
    )
    if result.outcome != PlanningOutcome.ROUTE_FOUND:
        raise RuntimeError(
            "live ROS 2 runner could not construct a RoutePlan: "
            f"{result.outcome.value}: {result.failure}"
        )
    if result.route_plan is None:
        raise RuntimeError("planner returned ROUTE_FOUND without a RoutePlan")
    return result.route_plan


def _write_route_plan(path: Path | None, route_plan: RoutePlan) -> None:
    line = json.dumps(
        {"kind": "route_plan", "route_plan": _jsonable(route_plan)},
        indent=2,
        sort_keys=True,
    )
    print(line, flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{line}\n", encoding="utf-8")


def _validate_args(args: argparse.Namespace) -> None:
    positive_values = (
        args.observation_sample_hz,
        args.localization_tick_period_s,
        args.maximum_frame_gap_s,
        args.source_read_timeout_s,
        args.max_observation_age_s,
        args.publication_validity_s,
        args.initial_location_timeout_s,
        args.trajectory_update_timeout_s,
    )
    if not all(
        math.isfinite(value) and value > 0.0 for value in positive_values
    ):
        raise ValueError("live runner timing values must be finite and positive")
    if not math.isfinite(args.run_seconds) or args.run_seconds < 0.0:
        raise ValueError("run_seconds must be finite and non-negative")
    if args.center_crop_aspect is not None and (
        not math.isfinite(args.center_crop_aspect)
        or args.center_crop_aspect <= 0.0
    ):
        raise ValueError("center_crop_aspect must be finite and positive")
    if args.overlay_video_fps is not None and (
        not math.isfinite(args.overlay_video_fps)
        or args.overlay_video_fps <= 0.0
    ):
        raise ValueError("overlay_video_fps must be finite and positive")


async def _run(args: argparse.Namespace) -> None:
    _validate_args(args)
    checkpoint = args.checkpoint.expanduser().resolve()
    topomap_root = args.topomap.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not topomap_root.is_dir():
        raise FileNotFoundError(f"topomap not found: {topomap_root}")

    clock = MonotonicTimeSource(clock_id=_OBSERVATION_CLOCK_ID)
    checkpoint_digest = _sha256(checkpoint)
    map_engine = create_nomad_topomap_engine(
        NomadTopomapMapConfig(
            root=topomap_root,
            map_id=MapId("live-nomad-ros2-route"),
            version=MapVersion("local"),
            published_at=clock.now(),
            model_artifact_id=_MODEL_ARTIFACT_ID,
            model_artifact_digest=checkpoint_digest,
            image_profile_id=args.image_profile_id,
            expected_center_crop_aspect=args.center_crop_aspect,
        )
    )
    snapshot = await map_engine.get_snapshot(
        MapSelector(
            map_id=MapId("live-nomad-ros2-route"),
            version=MapVersion("local"),
        )
    )
    model = NomadPolicy.from_checkpoint(
        checkpoint,
        config=NomadConfig(center_crop_aspect=args.center_crop_aspect),
        device=args.device,
        strict=True,
    )
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="nomad-live-ros2",
    )
    runtime: LocalizationRuntime | None = None
    producer: NomadObservationProducer | None = None
    trajectory_service: LocalizationDrivenLocalTrajectoryService | None = None
    collector_task: asyncio.Task[None] | None = None
    try:
        goal_image_loader = LocalFileGoalImageLoader(
            allowed_roots=(topomap_root,)
        )
        distance_policy = NomadVisualGoalDistancePolicy(
            session=NomadDistanceSession(model),
            goal_image_loader=goal_image_loader,
            inference_executor=executor,
            config=NomadVisualPolicyConfig(
                policy_id="nomad-distance-live-ros2",
                image_profile_id=args.image_profile_id,
                model_artifact_id=_MODEL_ARTIFACT_ID,
                model_artifact_digest=checkpoint_digest,
                observation_clock_id=_OBSERVATION_CLOCK_ID,
                time_source=clock,
            ),
        )
        trajectory_policy = NomadVisualGoalTrajectoryPolicy(
            session=NomadTrajectorySession(model),
            goal_image_loader=goal_image_loader,
            inference_executor=executor,
            config=NomadTrajectoryPolicyConfig(
                policy_id="nomad-trajectory-live-ros2",
                image_profile_id=args.image_profile_id,
                model_artifact_id=_MODEL_ARTIFACT_ID,
                model_artifact_digest=checkpoint_digest,
                observation_clock_id=_OBSERVATION_CLOCK_ID,
                time_source=clock,
            ),
        )
        source = Ros2ImageFrameSource(
            Ros2ImageFrameSourceConfig(
                topic_name=args.color_topic,
                image_profile_id=args.image_profile_id,
            )
        )
        observation_fanout = NomadObservationFanout(
            (distance_policy, trajectory_policy)
        )
        overlay_video_writer = (
            None
            if args.overlay_video_output is None
            else LiveTrajectoryVideoWriter(
                args.overlay_video_output.expanduser().resolve(),
                frames_per_second=(
                    1.0 / args.belief_publish_period_s
                    if args.overlay_video_fps is None
                    else args.overlay_video_fps
                ),
            )
        )
        overlay_frame_cache = (
            None
            if overlay_video_writer is None
            else ObservationFrameCache(observation_fanout)
        )
        localization_engine = await FixedStartVisualLocalizationEngine.create(
            map_engine=map_engine,
            snapshot=snapshot,
            policy=distance_policy,
            profile=FixedStartVisualTrackingProfile(
                image_profile_id=args.image_profile_id,
                close_threshold=args.close_threshold,
                start_close_confirmations=args.start_close_confirmations,
                successor_close_confirmations=(
                    args.successor_close_confirmations
                ),
                normal_distance_maximum=args.normal_distance_maximum,
                untrusted_distance_minimum=(
                    args.untrusted_distance_minimum
                ),
                lost_confirmations=args.lost_confirmations,
                tracking_candidate_count=args.tracking_candidate_count,
                evidence_window_size=args.evidence_window_size,
                relative_advantage_minimum=(
                    args.relative_advantage_minimum
                ),
                relative_advance_confirmations=(
                    args.relative_advance_confirmations
                ),
                relative_distance_maximum=args.relative_distance_maximum,
                lookahead_close_confirmations=(
                    args.lookahead_close_confirmations
                ),
                relocalization_candidate_count=(
                    args.relocalization_candidate_count
                ),
                relocalization_close_confirmations=(
                    args.relocalization_close_confirmations
                ),
                belief_publish_period_s=args.belief_publish_period_s,
                max_observation_age_s=args.max_observation_age_s,
            ),
            stream_id=BeliefStreamId("nomad-live-ros2-localization"),
            started_at=clock.now(),
        )
        producer = NomadObservationProducer(
            source=source,
            policy=(
                observation_fanout
                if overlay_frame_cache is None
                else overlay_frame_cache
            ),
            config=NomadObservationProducerConfig(
                image_profile_id=args.image_profile_id,
                sample_hz=args.observation_sample_hz,
                source_read_timeout_s=args.source_read_timeout_s,
                maximum_frame_gap_s=args.maximum_frame_gap_s,
            ),
        )
        localization_service = ContinuousLocalizationService(
            engine=localization_engine,
            time_source=clock,
            config=LocalizationServiceConfig(
                tick_period_s=args.localization_tick_period_s,
            ),
        )
        runtime = LocalizationRuntime(
            observation_producer=producer,
            localization_service=localization_service,
            config=LocalizationRuntimeConfig(
                observation_completion_policy=(
                    LocalizationObservationCompletionPolicy.ALLOW_UNTIL_STOP
                )
            ),
        )
        await runtime.start()
        initial_belief = await _wait_for_initial_location(
            engine=localization_engine,
            producer=producer,
            timeout_s=args.initial_location_timeout_s,
        )
        route_plan = await _plan_route(
            map_engine=map_engine,
            snapshot=snapshot,
            belief=initial_belief,
            requested_at=clock.now(),
        )
        _write_route_plan(
            None
            if args.route_plan_output is None
            else args.route_plan_output.expanduser().resolve(),
            route_plan,
        )
        trajectory_engine = await RouteBoundLocalTrajectoryEngine.create(
            map_engine=map_engine,
            snapshot=snapshot,
            route_plan=route_plan,
            localization_engine=localization_engine,
            trajectory_policy=trajectory_policy,
            stream_id=LocalTrajectoryStreamId("nomad-live-ros2-trajectories"),
            started_at=clock.now(),
            config=LocalTrajectoryEngineConfig(
                image_profile_id=args.image_profile_id,
                model_artifact_id=_MODEL_ARTIFACT_ID,
                model_artifact_digest=checkpoint_digest,
                time_source=clock,
                num_candidates=args.num_candidates,
                selected_candidate_index=args.selected_candidate_index,
                sampling_seed_base=args.sampling_seed_base,
                max_observation_age_s=args.max_observation_age_s,
                publication_validity_s=args.publication_validity_s,
            ),
        )
        trajectory_service = LocalizationDrivenLocalTrajectoryService(
            engine=trajectory_engine,
            localization_engine=localization_engine,
            time_source=clock,
            config=LocalTrajectoryServiceConfig(),
        )
        collector = _TrajectoryCollector(
            stream=trajectory_engine,
            producer=producer,
            output_path=(
                None
                if args.trajectory_output is None
                else args.trajectory_output.expanduser().resolve()
            ),
            update_timeout_s=args.trajectory_update_timeout_s,
            overlay_frame_cache=overlay_frame_cache,
            overlay_video_writer=overlay_video_writer,
            goal_image_cache=(
                None
                if overlay_video_writer is None
                else _GoalImageCache(
                    map_engine=map_engine,
                    snapshot=snapshot,
                    goal_image_loader=goal_image_loader,
                )
            ),
        )
        collector_task = asyncio.create_task(
            collector.run(),
            name="nomad-live-ros2-trajectory-collector",
        )
        await trajectory_service.start()
        if args.run_seconds > 0.0:
            await asyncio.sleep(args.run_seconds)
            if collector_task.done():
                await collector_task
        else:
            await collector_task
        summary = {
            "kind": "summary",
            "color_topic": args.color_topic,
            "device": args.device,
            "topomap": str(topomap_root),
            "route_id": str(route_plan.route_id),
            "trajectory_publications": len(collector.records),
            "active_trajectories": sum(
                record.publication.state == LocalTrajectoryState.ACTIVE
                for record in collector.records
            ),
            "frames_submitted": producer.get_status().frames_submitted,
            "runtime_state": runtime.get_status().state.value,
            "trajectory_service_state": (
                trajectory_service.get_status().state.value
            ),
            "overlay_video": (
                None
                if overlay_video_writer is None
                else str(args.overlay_video_output.expanduser().resolve())
            ),
            "overlay_frames_written": (
                0
                if overlay_video_writer is None
                else overlay_video_writer.frames_written
            ),
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        if producer is not None:
            await producer.stop()
        if trajectory_service is not None:
            await trajectory_service.stop()
        if runtime is not None:
            await runtime.stop()
        if collector_task is not None and not collector_task.done():
            collector_task.cancel()
        if collector_task is not None:
            await asyncio.gather(collector_task, return_exceptions=True)
        await asyncio.to_thread(
            executor.shutdown,
            wait=True,
            cancel_futures=True,
        )


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()

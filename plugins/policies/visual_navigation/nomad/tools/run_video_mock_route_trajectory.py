"""Runs the Navigation Harness route-to-trajectory loop over a video."""

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
)
from nomad_runtime import (
    NomadConfig,
    NomadDistanceSession,
    NomadPolicy,
    NomadTrajectorySession,
)
from tools.ffmpeg_video_source import (
    FfmpegVideoFrameSource,
    FfmpegVideoSourceConfig,
)


_IMAGE_PROFILE_ID = "nomad.rgb.direct_resize_96x96.imagenet.v1"
_MODEL_ARTIFACT_ID = "nomad.pth"
_OBSERVATION_CLOCK_ID = "recorded-video"


@dataclass(frozen=True, slots=True)
class _TrajectoryReplayRecord:
    source_timestamp_s: float | None
    publication: LocalTrajectoryPublication


class _VideoReplayTimeSource:
    """Exposes recorded-video progress as the offline policy clock."""

    def __init__(
        self,
        source: FfmpegVideoFrameSource,
        clock_id: str,
    ) -> None:
        self._source = source
        self._clock_id = clock_id

    def now(self) -> TimePoint:
        return TimePoint(
            clock_id=self._clock_id,
            nanoseconds=round(
                self._source.latest_source_timestamp_s * 1_000_000_000
            ),
        )


class _TrajectoryCollector:
    """Consumes the same public stream exposed to an external integration."""

    def __init__(
        self,
        *,
        stream: LocalTrajectoryStream,
        producer: NomadObservationProducer,
        output_path: Path | None,
        update_timeout_s: float,
    ) -> None:
        self._stream = stream
        self._producer = producer
        self._output_path = output_path
        self._update_timeout_s = update_timeout_s
        self.records: list[_TrajectoryReplayRecord] = []

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
                record = _TrajectoryReplayRecord(
                    source_timestamp_s=(
                        self._producer.get_status()
                        .last_submitted_source_timestamp_s
                    ),
                    publication=publication,
                )
                self.records.append(record)
                payload = {
                    "kind": "local_trajectory",
                    **_jsonable(record),
                }
                line = json.dumps(payload, sort_keys=True)
                print(line, flush=True)
                if output is not None:
                    output.write(line)
                    output.write("\n")
                    output.flush()
                revision = publication.revision
                if publication.state in (
                    LocalTrajectoryState.ROUTE_COMPLETED,
                    LocalTrajectoryState.FAULTED,
                ):
                    return
        finally:
            if output is not None:
                output.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay video observations through localization, construct a "
            "RoutePlan, and continuously publish complete NoMaD trajectories."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--topomap", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-profile-id", default=_IMAGE_PROFILE_ID)
    parser.add_argument(
        "--center-crop-aspect",
        type=_aspect_ratio,
        default=None,
        help="Use '4:3', another ratio, or 'none' (default).",
    )
    parser.add_argument("--start-time-s", type=float, default=0.0)
    parser.add_argument("--end-time-s", type=float)
    parser.add_argument("--playback-rate", type=float, default=1.0)
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
    parser.add_argument("--lost-confirmations", type=int, default=3)
    parser.add_argument("--tracking-candidate-count", type=int, default=3)
    parser.add_argument("--evidence-window-size", type=int, default=3)
    parser.add_argument(
        "--relative-advantage-minimum",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--relative-advance-confirmations",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--relative-distance-maximum",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--lookahead-close-confirmations",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--relocalization-candidate-count",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--relocalization-close-confirmations",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--belief-publish-period-s",
        type=float,
        default=0.25,
    )
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--selected-candidate-index", type=int, default=0)
    parser.add_argument("--sampling-seed-base", type=int, default=0)
    parser.add_argument(
        "--initial-location-timeout-s",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--trajectory-update-timeout-s",
        type=float,
        default=30.0,
    )
    parser.add_argument("--eof-grace-s", type=float, default=1.0)
    parser.add_argument("--route-plan-output", type=Path)
    parser.add_argument("--trajectory-output", type=Path)
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
        return {
            str(key): _jsonable(item) for key, item in value.items()
        }
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
                    "video ended before fixed-start localization became usable"
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
        AnchorQuery(
            purposes=frozenset({AnchorPurpose.COMPLETION}),
        ),
    )
    node_anchors = tuple(
        (NodeId(anchor.attached_to.entity_id), anchor.anchor_id)
        for anchor in anchors.anchors
        if anchor.attached_to.kind == MapEntityKind.NODE
    )
    if len(node_anchors) != 1:
        raise RuntimeError(
            "video mock requires exactly one node completion anchor"
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
            request_id=PlanningRequestId("nomad-video-mock-route"),
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
            "video mock could not construct a RoutePlan: "
            f"{result.outcome.value}: {result.failure}"
        )
    if result.route_plan is None:
        raise RuntimeError("planner returned ROUTE_FOUND without a RoutePlan")
    return result.route_plan


def _write_route_plan(path: Path | None, route_plan: RoutePlan) -> None:
    payload = {
        "kind": "route_plan",
        "route_plan": _jsonable(route_plan),
    }
    line = json.dumps(payload, indent=2, sort_keys=True)
    print(line, flush=True)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{line}\n", encoding="utf-8")


async def _wait_for_replay_end(
    *,
    producer: NomadObservationProducer,
    collector_task: asyncio.Task[None],
    eof_grace_s: float,
) -> None:
    producer_task = asyncio.create_task(
        producer.wait_stopped(),
        name="nomad-route-video-producer-wait",
    )
    try:
        done, _ = await asyncio.wait(
            (producer_task, collector_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if collector_task in done:
            await collector_task
            return
        producer_status = await producer_task
        if producer_status.state != NomadObservationProducerState.COMPLETED:
            raise RuntimeError(
                "video observation producer stopped unexpectedly: "
                f"{producer_status.state.value}: "
                f"{producer_status.detail_code}: "
                f"{producer_status.last_error}"
            )
        try:
            await asyncio.wait_for(
                asyncio.shield(collector_task),
                timeout=eof_grace_s,
            )
        except TimeoutError:
            pass
    finally:
        if not producer_task.done():
            producer_task.cancel()
        await asyncio.gather(producer_task, return_exceptions=True)


def _validate_args(args: argparse.Namespace) -> None:
    positive_values = (
        args.playback_rate,
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
        math.isfinite(value) and value > 0.0
        for value in positive_values
    ):
        raise ValueError("video mock timing values must be finite and positive")
    if not math.isfinite(args.eof_grace_s) or args.eof_grace_s < 0.0:
        raise ValueError("eof_grace_s must be finite and non-negative")
    if args.center_crop_aspect is not None and (
        not math.isfinite(args.center_crop_aspect)
        or args.center_crop_aspect <= 0.0
    ):
        raise ValueError("center_crop_aspect must be finite and positive")


async def _run(args: argparse.Namespace) -> None:
    _validate_args(args)
    checkpoint = args.checkpoint.expanduser().resolve()
    video = args.video.expanduser().resolve()
    topomap_root = args.topomap.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    if not topomap_root.is_dir():
        raise FileNotFoundError(f"topomap not found: {topomap_root}")

    checkpoint_digest = _sha256(checkpoint)
    map_id = MapId("offline-nomad-route-video-mock")
    map_version = MapVersion("local")
    map_engine = create_nomad_topomap_engine(
        NomadTopomapMapConfig(
            root=topomap_root,
            map_id=map_id,
            version=map_version,
            published_at=TimePoint(clock_id="unix", nanoseconds=0),
            model_artifact_id=_MODEL_ARTIFACT_ID,
            model_artifact_digest=checkpoint_digest,
            image_profile_id=args.image_profile_id,
            expected_center_crop_aspect=args.center_crop_aspect,
        )
    )
    snapshot = await map_engine.get_snapshot(
        MapSelector(map_id=map_id, version=map_version)
    )

    started_at = time.perf_counter()
    model = NomadPolicy.from_checkpoint(
        checkpoint,
        config=NomadConfig(center_crop_aspect=args.center_crop_aspect),
        device=args.device,
        strict=True,
    )
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="nomad-route-video-mock",
    )
    runtime: LocalizationRuntime | None = None
    producer: NomadObservationProducer | None = None
    trajectory_service: LocalizationDrivenLocalTrajectoryService | None = None
    trajectory_collector: _TrajectoryCollector | None = None
    collector_task: asyncio.Task[None] | None = None
    source: FfmpegVideoFrameSource | None = None
    route_plan: RoutePlan | None = None
    try:
        source = FfmpegVideoFrameSource(
            FfmpegVideoSourceConfig(
                video_path=video,
                image_profile_id=args.image_profile_id,
                start_time_s=args.start_time_s,
                end_time_s=args.end_time_s,
                playback_rate=args.playback_rate,
                use_source_time_as_policy_time=True,
            )
        )
        clock = _VideoReplayTimeSource(source, _OBSERVATION_CLOCK_ID)
        goal_image_loader = LocalFileGoalImageLoader(
            allowed_roots=(topomap_root,)
        )
        distance_policy = NomadVisualGoalDistancePolicy(
            session=NomadDistanceSession(model),
            goal_image_loader=goal_image_loader,
            inference_executor=executor,
            config=NomadVisualPolicyConfig(
                policy_id="nomad-distance-video-mock",
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
                policy_id="nomad-trajectory-video-mock",
                image_profile_id=args.image_profile_id,
                model_artifact_id=_MODEL_ARTIFACT_ID,
                model_artifact_digest=checkpoint_digest,
                observation_clock_id=_OBSERVATION_CLOCK_ID,
                time_source=clock,
            ),
        )
        observation_fanout = NomadObservationFanout(
            (distance_policy, trajectory_policy)
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
            stream_id=BeliefStreamId("nomad-route-video-mock-localization"),
            started_at=clock.now(),
        )
        producer = NomadObservationProducer(
            source=source,
            policy=observation_fanout,
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

        local_trajectory_engine = await RouteBoundLocalTrajectoryEngine.create(
            map_engine=map_engine,
            snapshot=snapshot,
            route_plan=route_plan,
            localization_engine=localization_engine,
            trajectory_policy=trajectory_policy,
            stream_id=LocalTrajectoryStreamId(
                "nomad-route-video-mock-trajectories"
            ),
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
            engine=local_trajectory_engine,
            localization_engine=localization_engine,
            time_source=clock,
            config=LocalTrajectoryServiceConfig(),
        )
        trajectory_collector = _TrajectoryCollector(
            stream=local_trajectory_engine,
            producer=producer,
            output_path=(
                None
                if args.trajectory_output is None
                else args.trajectory_output.expanduser().resolve()
            ),
            update_timeout_s=args.trajectory_update_timeout_s,
        )
        collector_task = asyncio.create_task(
            trajectory_collector.run(),
            name="nomad-route-video-trajectory-collector",
        )
        await trajectory_service.start()
        await _wait_for_replay_end(
            producer=producer,
            collector_task=collector_task,
            eof_grace_s=args.eof_grace_s,
        )
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

    if (
        runtime is None
        or producer is None
        or trajectory_service is None
        or trajectory_collector is None
        or source is None
        or route_plan is None
    ):
        raise RuntimeError("route trajectory video mock did not initialize")
    if not trajectory_collector.records:
        raise RuntimeError("route trajectory stream produced no publications")
    active_records = tuple(
        record
        for record in trajectory_collector.records
        if record.publication.state == LocalTrajectoryState.ACTIVE
    )
    final_tracking = localization_engine.get_tracking_state()
    localization_service_status = runtime.get_status().service
    summary = {
        "kind": "summary",
        "device": args.device,
        "video": str(video),
        "topomap": str(topomap_root),
        "image_profile_id": args.image_profile_id,
        "route_id": str(route_plan.route_id),
        "route_traversal_count": len(route_plan.traversals),
        "publication_count": len(trajectory_collector.records),
        "active_trajectory_count": len(active_records),
        "final_publication_state": (
            trajectory_collector.records[-1].publication.state.value
        ),
        "final_localization_phase": final_tracking.phase.value,
        "final_localization_node": (
            None
            if final_tracking.current_node_id is None
            else str(final_tracking.current_node_id)
        ),
        "frames_submitted": producer.get_status().frames_submitted,
        "localization_ticks": localization_service_status.ticks_completed,
        "localization_skipped_tick_slots": (
            localization_service_status.skipped_tick_slots
        ),
        "trajectory_ticks": trajectory_service.get_status().ticks_completed,
        "coalesced_belief_updates": (
            trajectory_service.get_status().coalesced_belief_updates
        ),
        "video_width": source.metadata.width,
        "video_height": source.metadata.height,
        "video_fps": source.metadata.frames_per_second,
        "elapsed_s": time.perf_counter() - started_at,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()

"""Runs fixed-start NoMaD localization from a recorded video."""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

from longship.navigation.common import TimePoint
from longship.navigation.localization_engine.fixed_start_visual import (
    FIXED_START_NODE_ID,
    FixedStartVisualLocalizationEngine,
    FixedStartVisualPhase,
    FixedStartVisualTrackingProfile,
)
from longship.navigation.localization_engine.models import (
    BeliefRevision,
    BeliefStreamId,
    BeliefUpdateOutcome,
    WaitForUpdateRequest,
)
from longship.navigation.localization_engine.service import (
    ContinuousLocalizationService,
    LocalizationServiceConfig,
)
from longship.navigation.map_engine.models import (
    MapId,
    MapSelector,
    MapVersion,
)
from longship.navigation.runtime import (
    LocalizationObservationCompletionPolicy,
    LocalizationRuntime,
    LocalizationRuntimeConfig,
    LocalizationRuntimeState,
)
from nomad_runtime import NomadConfig, NomadDistanceSession, NomadPolicy
from longship_adapter import (
    LocalFileGoalImageLoader,
    NomadObservationProducer,
    NomadObservationProducerConfig,
    NomadObservationProducerState,
    NomadTopomapMapConfig,
    NomadVisualGoalDistancePolicy,
    NomadVisualPolicyConfig,
    create_nomad_topomap_engine,
)
from tools.ffmpeg_video_source import (
    FfmpegVideoFrameSource,
    FfmpegVideoSourceConfig,
)


_IMAGE_PROFILE_ID = "nomad.rgb.direct_resize_96x96.imagenet.v1"
_MODEL_ARTIFACT_ID = "nomad.pth"
_OBSERVATION_CLOCK_ID = "recorded-video"


@dataclass(frozen=True, slots=True)
class _VideoReplayRecord:
    source_timestamp_s: float | None
    observation_timestamp_s: float | None
    phase: str
    current_node: str | None
    target_node: str | None
    temporal_distance: float | None
    candidate_distances: tuple[tuple[str, float], ...]
    belief_status: str
    belief_sequence: int
    detail_code: str | None


class _ExecutorResource:
    def __init__(self, executor: ThreadPoolExecutor) -> None:
        self._executor = executor

    async def close(self) -> None:
        await asyncio.to_thread(
            self._executor.shutdown,
            wait=True,
            cancel_futures=True,
        )


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


class _VideoBeliefCollector:
    def __init__(
        self,
        *,
        engine: FixedStartVisualLocalizationEngine,
        producer: NomadObservationProducer,
        initial_revision: BeliefRevision,
        update_timeout_s: float,
    ) -> None:
        self._engine = engine
        self._producer = producer
        self._initial_revision = initial_revision
        self._update_timeout_s = update_timeout_s
        self.records: list[_VideoReplayRecord] = []

    async def run(self) -> None:
        revision = self._initial_revision
        previous_transition = None
        while True:
            update = await self._engine.wait_for_update(
                WaitForUpdateRequest(
                    after_revision=revision,
                    timeout_s=self._update_timeout_s,
                )
            )
            if update.outcome != BeliefUpdateOutcome.UPDATED:
                raise RuntimeError("timed out waiting for a belief update")

            belief = update.belief
            tracking = self._engine.get_tracking_state()
            producer_status = self._producer.get_status()
            record = _VideoReplayRecord(
                source_timestamp_s=(
                    producer_status.last_submitted_source_timestamp_s
                ),
                observation_timestamp_s=(
                    None
                    if tracking.last_observation_time is None
                    else tracking.last_observation_time.nanoseconds
                    / 1_000_000_000
                ),
                phase=tracking.phase.value,
                current_node=(
                    None
                    if tracking.current_node_id is None
                    else str(tracking.current_node_id)
                ),
                target_node=(
                    None
                    if tracking.target_node_id is None
                    else str(tracking.target_node_id)
                ),
                temporal_distance=tracking.last_temporal_distance,
                candidate_distances=tuple(
                    (str(node_id), distance)
                    for node_id, distance
                    in tracking.last_candidate_distances
                ),
                belief_status=belief.status.value,
                belief_sequence=belief.revision.sequence,
                detail_code=self._engine.get_status().detail_code,
            )
            self.records.append(record)
            transition = _transition_key(record)
            if transition != previous_transition:
                print(json.dumps(asdict(record), sort_keys=True), flush=True)
                previous_transition = transition
            revision = belief.revision
            if tracking.phase in (
                FixedStartVisualPhase.AT_FINAL_NODE,
                FixedStartVisualPhase.FAULT,
            ):
                return


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode a recorded video at source cadence and run fixed-start "
            "NoMaD localization at an independent tick rate."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--topomap", type=Path, required=True)
    parser.add_argument(
        "--image-profile-id",
        default=_IMAGE_PROFILE_ID,
    )
    parser.add_argument(
        "--center-crop-aspect",
        type=_aspect_ratio,
        default=None,
        help="Use '4:3', another ratio, or 'none' (default).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--start-time-s", type=float, default=0.0)
    parser.add_argument("--end-time-s", type=float)
    parser.add_argument("--playback-rate", type=float, default=1.0)
    parser.add_argument("--observation-sample-hz", type=float, default=9.0)
    parser.add_argument(
        "--tick-period-s",
        type=float,
        default=1.0 / 9.0,
    )
    parser.add_argument("--maximum-frame-gap-s", type=float, default=0.5)
    parser.add_argument("--source-read-timeout-s", type=float, default=2.0)
    parser.add_argument("--max-observation-age-s", type=float, default=0.5)
    parser.add_argument("--close-threshold", type=float, default=3.0)
    parser.add_argument("--start-close-confirmations", type=int, default=2)
    parser.add_argument(
        "--successor-close-confirmations",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--normal-distance-maximum",
        type=float,
        default=15.0,
    )
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
    parser.add_argument("--belief-update-timeout-s", type=float, default=30.0)
    parser.add_argument("--eof-grace-s", type=float, default=1.0)
    parser.add_argument("--trace-output", type=Path)
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


def _transition_key(record: _VideoReplayRecord) -> tuple[object, ...]:
    return (
        record.phase,
        record.current_node,
        record.target_node,
        record.belief_status,
        record.detail_code,
    )


def _confirmed_nodes(
    records: list[_VideoReplayRecord],
) -> list[str]:
    result = []
    previous_node = None
    for record in records:
        if (
            record.current_node is not None
            and record.current_node != previous_node
        ):
            result.append(record.current_node)
        previous_node = record.current_node
    return result


def _write_trace(path: Path, records: list[_VideoReplayRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(asdict(record), sort_keys=True))
            output.write("\n")


async def _wait_for_replay_end(
    *,
    producer: NomadObservationProducer,
    collector_task: asyncio.Task[None],
    eof_grace_s: float,
) -> None:
    producer_task = asyncio.create_task(
        producer.wait_stopped(),
        name="nomad-video-observation-producer-wait",
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


async def _run(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint.expanduser().resolve()
    video = args.video.expanduser().resolve()
    topomap_root = args.topomap.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    if not topomap_root.is_dir():
        raise FileNotFoundError(f"topomap not found: {topomap_root}")
    if not math.isfinite(args.eof_grace_s) or args.eof_grace_s < 0.0:
        raise ValueError("eof_grace_s must be finite and non-negative")
    if not args.image_profile_id.strip():
        raise ValueError("image_profile_id must not be empty")
    if args.center_crop_aspect is not None and (
        not math.isfinite(args.center_crop_aspect)
        or args.center_crop_aspect <= 0.0
    ):
        raise ValueError("center_crop_aspect must be finite and positive")
    image_profile_id = args.image_profile_id

    checkpoint_digest = _sha256(checkpoint)
    map_id = MapId("offline-nomad-video-mock")
    map_version = MapVersion("local")
    map_engine = create_nomad_topomap_engine(
        NomadTopomapMapConfig(
            root=topomap_root,
            map_id=map_id,
            version=map_version,
            published_at=TimePoint(clock_id="unix", nanoseconds=0),
            model_artifact_id=_MODEL_ARTIFACT_ID,
            model_artifact_digest=checkpoint_digest,
            image_profile_id=image_profile_id,
            expected_center_crop_aspect=args.center_crop_aspect,
        )
    )
    snapshot = await map_engine.get_snapshot(
        MapSelector(map_id=map_id, version=map_version)
    )

    started_at = time.perf_counter()
    model = NomadPolicy.from_checkpoint(
        checkpoint,
        config=NomadConfig(
            center_crop_aspect=args.center_crop_aspect,
        ),
        device=args.device,
        strict=True,
    )
    session = NomadDistanceSession(model)
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="nomad-video-mock",
    )
    runtime: LocalizationRuntime | None = None
    collector_task: asyncio.Task[None] | None = None
    collector: _VideoBeliefCollector | None = None
    producer: NomadObservationProducer | None = None
    source: FfmpegVideoFrameSource | None = None
    try:
        source = FfmpegVideoFrameSource(
            FfmpegVideoSourceConfig(
                video_path=video,
                image_profile_id=image_profile_id,
                start_time_s=args.start_time_s,
                end_time_s=args.end_time_s,
                playback_rate=args.playback_rate,
                use_source_time_as_policy_time=True,
            )
        )
        clock = _VideoReplayTimeSource(source, _OBSERVATION_CLOCK_ID)
        visual_policy = NomadVisualGoalDistancePolicy(
            session=session,
            goal_image_loader=LocalFileGoalImageLoader(
                allowed_roots=(topomap_root,),
            ),
            inference_executor=executor,
            config=NomadVisualPolicyConfig(
                policy_id="nomad-distance-video-mock",
                image_profile_id=image_profile_id,
                model_artifact_id=_MODEL_ARTIFACT_ID,
                model_artifact_digest=checkpoint_digest,
                observation_clock_id=_OBSERVATION_CLOCK_ID,
                time_source=clock,
            ),
        )
        engine = await FixedStartVisualLocalizationEngine.create(
            map_engine=map_engine,
            snapshot=snapshot,
            policy=visual_policy,
            profile=FixedStartVisualTrackingProfile(
                image_profile_id=image_profile_id,
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
            stream_id=BeliefStreamId("offline-nomad-video-mock"),
            started_at=clock.now(),
        )
        producer = NomadObservationProducer(
            source=source,
            policy=visual_policy,
            config=NomadObservationProducerConfig(
                image_profile_id=image_profile_id,
                sample_hz=args.observation_sample_hz,
                source_read_timeout_s=args.source_read_timeout_s,
                maximum_frame_gap_s=args.maximum_frame_gap_s,
            ),
        )
        service = ContinuousLocalizationService(
            engine=engine,
            time_source=clock,
            config=LocalizationServiceConfig(
                tick_period_s=args.tick_period_s,
            ),
        )
        runtime = LocalizationRuntime(
            observation_producer=producer,
            localization_service=service,
            shutdown_resources=(_ExecutorResource(executor),),
            config=LocalizationRuntimeConfig(
                observation_completion_policy=(
                    LocalizationObservationCompletionPolicy.ALLOW_UNTIL_STOP
                )
            ),
        )
        collector = _VideoBeliefCollector(
            engine=engine,
            producer=producer,
            initial_revision=engine.get_belief().revision,
            update_timeout_s=args.belief_update_timeout_s,
        )
        collector_task = asyncio.create_task(
            collector.run(),
            name="nomad-video-belief-collector",
        )
        await runtime.start()
        await _wait_for_replay_end(
            producer=producer,
            collector_task=collector_task,
            eof_grace_s=args.eof_grace_s,
        )
    finally:
        if runtime is None:
            executor.shutdown(wait=True, cancel_futures=True)
        else:
            await runtime.stop()
        if collector_task is not None and not collector_task.done():
            collector_task.cancel()
            await asyncio.gather(collector_task, return_exceptions=True)

    if (
        runtime is None
        or producer is None
        or source is None
        or collector is None
    ):
        raise RuntimeError("video localization composition did not initialize")
    runtime_status = runtime.get_status()
    producer_status = producer.get_status()
    if runtime_status.state != LocalizationRuntimeState.STOPPED:
        raise RuntimeError(
            f"localization runtime did not stop cleanly: {runtime_status}"
        )
    if not collector.records:
        raise RuntimeError("video localization produced no beliefs")

    confirmed_nodes = _confirmed_nodes(collector.records)
    if confirmed_nodes and confirmed_nodes[0] != str(FIXED_START_NODE_ID):
        raise RuntimeError(
            "fixed-start video replay confirmed a node other than node-0000"
        )
    if args.trace_output is not None:
        _write_trace(
            args.trace_output.expanduser().resolve(),
            collector.records,
        )

    final_record = collector.records[-1]
    summary = {
        "kind": "summary",
        "device": str(args.device),
        "image_profile_id": image_profile_id,
        "center_crop_aspect": args.center_crop_aspect,
        "video": str(video),
        "video_width": source.metadata.width,
        "video_height": source.metadata.height,
        "video_fps": source.metadata.frames_per_second,
        "start_time_s": args.start_time_s,
        "end_time_s": args.end_time_s,
        "playback_rate": args.playback_rate,
        "observation_sample_hz": args.observation_sample_hz,
        "frames_received": producer_status.frames_received,
        "frames_submitted": producer_status.frames_submitted,
        "frames_dropped_by_sampler": (
            producer_status.frames_dropped_by_sampler
        ),
        "context_resets": producer_status.context_resets,
        "producer_state": producer_status.state.value,
        "belief_count": len(collector.records),
        "confirmed_node_count": len(confirmed_nodes),
        "confirmed_nodes": confirmed_nodes,
        "ticks_completed": runtime_status.service.ticks_completed,
        "skipped_tick_slots": runtime_status.service.skipped_tick_slots,
        "runtime_state": runtime_status.state.value,
        "final_phase": final_record.phase,
        "final_current_node": final_record.current_node,
        "final_target_node": final_record.target_node,
        "final_belief_status": final_record.belief_status,
        "elapsed_s": time.perf_counter() - started_at,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()

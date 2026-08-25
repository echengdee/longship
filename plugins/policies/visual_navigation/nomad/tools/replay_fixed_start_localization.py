"""Replays dense images through fixed-start NoMaD localization."""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

from PIL import Image
import torch

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
    LocalizationObservationProducerState,
    LocalizationObservationProducerStatus,
    LocalizationRuntime,
    LocalizationRuntimeConfig,
    LocalizationRuntimeState,
)
from nomad_runtime import (
    NomadDistanceSession,
    NomadPolicy,
    default_checkpoint_path,
)
from longship_adapter import (
    LocalFileGoalImageLoader,
    NomadTopomapMapConfig,
    NomadVisualGoalDistancePolicy,
    NomadVisualPolicyConfig,
    create_nomad_topomap_engine,
)


_IMAGE_PROFILE_ID = "nomad.rgb.direct_resize_96x96.imagenet.v1"
_MODEL_ARTIFACT_ID = "nomad.pth"
_OBSERVATION_CLOCK_ID = "camera"


@dataclass(frozen=True, slots=True)
class _DenseFrame:
    node: int
    filename: str
    timestamp_s: float


@dataclass(frozen=True, slots=True)
class _ReplayRecord:
    dense_node: int
    frame_filename: str
    timestamp_s: float
    phase: str
    current_node: str | None
    target_node: str | None
    distance_target_node: str | None
    temporal_distance: float | None
    belief_status: str
    belief_sequence: int
    detail_code: str | None


@dataclass(frozen=True, slots=True)
class _SubmittedFrame:
    frame: _DenseFrame
    distance_target_node: str | None
    previous_observation_time: TimePoint | None
    is_last: bool


class _ReplayTimeSource:
    def __init__(self, initial_time: TimePoint) -> None:
        self._current_time = initial_time

    def now(self) -> TimePoint:
        return self._current_time

    def advance(self, value: TimePoint) -> None:
        if value.clock_id != self._current_time.clock_id:
            raise ValueError("replay time source changed clock domain")
        if value.nanoseconds < self._current_time.nanoseconds:
            raise ValueError("replay time source moved backward")
        self._current_time = value


class _ExecutorResource:
    def __init__(self, executor: ThreadPoolExecutor) -> None:
        self._executor = executor

    async def close(self) -> None:
        await asyncio.to_thread(
            self._executor.shutdown,
            wait=True,
            cancel_futures=True,
        )


class _DenseReplayObservationProducer:
    def __init__(
        self,
        *,
        dense_root: Path,
        frames: tuple[_DenseFrame, ...],
        visual_policy: NomadVisualGoalDistancePolicy,
        time_source: _ReplayTimeSource,
        engine: FixedStartVisualLocalizationEngine,
    ) -> None:
        self._dense_root = dense_root
        self._frames = frames
        self._visual_policy = visual_policy
        self._time_source = time_source
        self._engine = engine
        self._acknowledgements: asyncio.Queue[bool] = asyncio.Queue()
        self._active_submission: _SubmittedFrame | None = None
        self._task: asyncio.Task[None] | None = None
        self._state = LocalizationObservationProducerState.CREATED
        self._detail_code: str | None = None
        self._last_error: str | None = None
        self._terminated = asyncio.Event()

    def get_status(self) -> LocalizationObservationProducerStatus:
        return LocalizationObservationProducerStatus(
            state=self._state,
            detail_code=self._detail_code,
            last_error=self._last_error,
        )

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("dense replay producer already started")
        self._state = LocalizationObservationProducerState.STARTING
        self._detail_code = "submitting_first_dense_frame"
        self._submit_frame(0)
        self._state = LocalizationObservationProducerState.RUNNING
        self._detail_code = "running"
        self._task = asyncio.create_task(
            self._run_guarded(),
            name="nomad-dense-replay-observation-producer",
        )

    async def stop(self) -> None:
        if self._task is None:
            self._state = LocalizationObservationProducerState.STOPPED
            self._detail_code = "stopped_before_start"
            self._terminated.set()
            return
        terminal_state = self._state
        if terminal_state not in (
            LocalizationObservationProducerState.COMPLETED,
            LocalizationObservationProducerState.FAULTED,
        ):
            self._state = LocalizationObservationProducerState.STOPPING
            self._detail_code = "stop_requested"
        if not self._task.done():
            self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        if terminal_state not in (
            LocalizationObservationProducerState.COMPLETED,
            LocalizationObservationProducerState.FAULTED,
        ):
            self._state = LocalizationObservationProducerState.STOPPED
            self._detail_code = "stopped"
        self._terminated.set()

    async def wait_stopped(
        self,
        timeout_s: float | None = None,
    ) -> LocalizationObservationProducerStatus:
        if timeout_s is None:
            await self._terminated.wait()
        else:
            await asyncio.wait_for(self._terminated.wait(), timeout_s)
        return self.get_status()

    async def wait_finished(self) -> None:
        if self._task is None:
            raise RuntimeError("dense replay producer has not started")
        await self._task

    def get_active_submission(self) -> _SubmittedFrame:
        if self._active_submission is None:
            raise RuntimeError("dense replay has no active frame")
        return self._active_submission

    def acknowledge(self, *, terminal: bool) -> None:
        self._acknowledgements.put_nowait(terminal)

    async def _run_guarded(self) -> None:
        try:
            await self._run()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._state = LocalizationObservationProducerState.FAULTED
            self._detail_code = f"dense_replay_failed:{type(error).__name__}"
            self._last_error = str(error)
            raise
        else:
            self._state = LocalizationObservationProducerState.COMPLETED
            self._detail_code = "dense_replay_completed"
        finally:
            self._terminated.set()

    async def _run(self) -> None:
        for index in range(len(self._frames)):
            if index > 0:
                self._submit_frame(index)
            if await self._acknowledgements.get():
                return

    def _submit_frame(self, index: int) -> None:
        frame = self._frames[index]
        tracking = self._engine.get_tracking_state()
        self._time_source.advance(_time_point(frame.timestamp_s))
        image = _load_rgb_hwc(self._dense_root / frame.filename)
        self._visual_policy.submit_observation(
            image,
            frame.timestamp_s,
            layout="hwc",
            channel_order="rgb",
            value_range="byte",
        )
        self._active_submission = _SubmittedFrame(
            frame=frame,
            distance_target_node=(
                None
                if tracking.target_node_id is None
                else str(tracking.target_node_id)
            ),
            previous_observation_time=tracking.last_observation_time,
            is_last=index == len(self._frames) - 1,
        )


class _ReplayBeliefCollector:
    def __init__(
        self,
        *,
        engine: FixedStartVisualLocalizationEngine,
        producer: _DenseReplayObservationProducer,
        initial_revision: BeliefRevision,
        frame_timeout_s: float,
    ) -> None:
        self._engine = engine
        self._producer = producer
        self._initial_revision = initial_revision
        self._frame_timeout_s = frame_timeout_s
        self.records: list[_ReplayRecord] = []

    async def run(self) -> None:
        revision = self._initial_revision
        while True:
            update = await self._engine.wait_for_update(
                WaitForUpdateRequest(
                    after_revision=revision,
                    timeout_s=self._frame_timeout_s,
                )
            )
            if update.outcome != BeliefUpdateOutcome.UPDATED:
                raise RuntimeError(
                    "timed out waiting for a belief update after replay frame"
                )

            submission = self._producer.get_active_submission()
            belief = update.belief
            tracking = self._engine.get_tracking_state()
            status = self._engine.get_status()
            received_measurement = (
                tracking.last_observation_time
                != submission.previous_observation_time
            )
            record = _ReplayRecord(
                dense_node=submission.frame.node,
                frame_filename=submission.frame.filename,
                timestamp_s=submission.frame.timestamp_s,
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
                distance_target_node=(
                    submission.distance_target_node
                    if received_measurement
                    else None
                ),
                temporal_distance=(
                    tracking.last_temporal_distance
                    if received_measurement
                    else None
                ),
                belief_status=belief.status.value,
                belief_sequence=belief.revision.sequence,
                detail_code=status.detail_code,
            )
            self.records.append(record)
            revision = belief.revision
            terminal = tracking.phase in (
                FixedStartVisualPhase.AT_FINAL_NODE,
                FixedStartVisualPhase.FAULT,
            )
            self._producer.acknowledge(terminal=terminal)
            if terminal or submission.is_last:
                return


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay dense images through the fixed node-0000 localization "
            "state machine."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint_path(),
        help="NoMaD checkpoint (default: repository LFS asset).",
    )
    parser.add_argument("--dense-images", type=Path, required=True)
    parser.add_argument("--topomap", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
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
    parser.add_argument(
        "--tick-period-s",
        type=float,
        default=0.25,
        help="Continuous Localization tick period in wall-clock seconds.",
    )
    parser.add_argument(
        "--frame-timeout-s",
        type=float,
        default=30.0,
        help="Maximum wall time for one replay frame to publish a belief.",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        help="Optional JSONL path receiving one record per dense frame.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _load_dense_frames(root: Path) -> tuple[_DenseFrame, ...]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("dense manifest must contain a non-empty JSON list")

    frames = []
    previous_timestamp = None
    for index, item in enumerate(manifest):
        if not isinstance(item, dict):
            raise ValueError(f"dense manifest row {index} is not an object")
        frame = _DenseFrame(
            node=int(item["node"]),
            filename=str(item["filename"]),
            timestamp_s=float(item["time_s"]),
        )
        image_path = root / frame.filename
        if not image_path.is_file():
            raise FileNotFoundError(f"dense image not found: {image_path}")
        if (
            previous_timestamp is not None
            and frame.timestamp_s <= previous_timestamp
        ):
            raise ValueError("dense timestamps must be strictly increasing")
        previous_timestamp = frame.timestamp_s
        frames.append(frame)
    return tuple(frames)


def _load_rgb_hwc(path: Path) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB")
        storage = bytearray(image.tobytes())
        return (
            torch.frombuffer(storage, dtype=torch.uint8)
            .clone()
            .view(image.height, image.width, 3)
        )


def _time_point(timestamp_s: float) -> TimePoint:
    return TimePoint(
        clock_id=_OBSERVATION_CLOCK_ID,
        nanoseconds=round(timestamp_s * 1_000_000_000),
    )


def _transition_key(record: _ReplayRecord) -> tuple[object, ...]:
    return (
        record.phase,
        record.current_node,
        record.target_node,
        record.belief_status,
        record.detail_code,
    )


async def _replay(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint.expanduser().resolve()
    dense_root = args.dense_images.expanduser().resolve()
    topomap_root = args.topomap.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not dense_root.is_dir():
        raise FileNotFoundError(
            f"dense image directory not found: {dense_root}"
        )

    frames = _load_dense_frames(dense_root)
    checkpoint_digest = _sha256(checkpoint)
    map_id = MapId("offline-nomad-replay")
    map_version = MapVersion("local")
    map_engine = create_nomad_topomap_engine(
        NomadTopomapMapConfig(
            root=topomap_root,
            map_id=map_id,
            version=map_version,
            published_at=TimePoint(clock_id="unix", nanoseconds=0),
            model_artifact_id=_MODEL_ARTIFACT_ID,
            model_artifact_digest=checkpoint_digest,
            image_profile_id=_IMAGE_PROFILE_ID,
        )
    )
    snapshot = await map_engine.get_snapshot(
        MapSelector(map_id=map_id, version=map_version)
    )

    started_at = time.perf_counter()
    model = NomadPolicy.from_checkpoint(
        checkpoint,
        device=args.device,
        strict=True,
    )
    session = NomadDistanceSession(model)
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="nomad-offline-replay",
    )
    runtime: LocalizationRuntime | None = None
    collector_task: asyncio.Task[None] | None = None
    try:
        time_source = _ReplayTimeSource(
            _time_point(frames[0].timestamp_s)
        )
        visual_policy = NomadVisualGoalDistancePolicy(
            session=session,
            goal_image_loader=LocalFileGoalImageLoader(
                allowed_roots=(topomap_root,),
            ),
            inference_executor=executor,
            config=NomadVisualPolicyConfig(
                policy_id="nomad-distance-offline-replay",
                image_profile_id=_IMAGE_PROFILE_ID,
                model_artifact_id=_MODEL_ARTIFACT_ID,
                model_artifact_digest=checkpoint_digest,
                observation_clock_id=_OBSERVATION_CLOCK_ID,
                time_source=time_source,
            ),
        )
        engine = await FixedStartVisualLocalizationEngine.create(
            map_engine=map_engine,
            snapshot=snapshot,
            policy=visual_policy,
            profile=FixedStartVisualTrackingProfile(
                image_profile_id=_IMAGE_PROFILE_ID,
                close_threshold=args.close_threshold,
                start_close_confirmations=args.start_close_confirmations,
                successor_close_confirmations=(
                    args.successor_close_confirmations
                ),
                normal_distance_maximum=args.normal_distance_maximum,
                lost_confirmations=args.lost_confirmations,
                belief_publish_period_s=1.0e-6,
            ),
            stream_id=BeliefStreamId("offline-nomad-replay"),
            started_at=_time_point(frames[0].timestamp_s),
        )
        producer = _DenseReplayObservationProducer(
            dense_root=dense_root,
            frames=frames,
            visual_policy=visual_policy,
            time_source=time_source,
            engine=engine,
        )
        service = ContinuousLocalizationService(
            engine=engine,
            time_source=time_source,
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
        collector = _ReplayBeliefCollector(
            engine=engine,
            producer=producer,
            initial_revision=engine.get_belief().revision,
            frame_timeout_s=args.frame_timeout_s,
        )
        collector_task = asyncio.create_task(
            collector.run(),
            name="nomad-dense-replay-belief-collector",
        )
        await runtime.start()
        producer_wait_task = asyncio.create_task(
            producer.wait_finished(),
            name="nomad-dense-replay-producer-wait",
        )
        try:
            await asyncio.gather(producer_wait_task, collector_task)
        finally:
            for task in (producer_wait_task, collector_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                producer_wait_task,
                collector_task,
                return_exceptions=True,
            )
    finally:
        if runtime is None:
            executor.shutdown(wait=True, cancel_futures=True)
        else:
            await runtime.stop()
        if collector_task is not None and not collector_task.done():
            collector_task.cancel()
            await asyncio.gather(collector_task, return_exceptions=True)

    runtime_status = runtime.get_status()
    if runtime_status.state != LocalizationRuntimeState.STOPPED:
        raise RuntimeError(
            f"localization runtime did not stop cleanly: {runtime_status}"
        )
    trace = collector.records
    if not trace:
        raise RuntimeError("continuous localization replay produced no beliefs")
    transitions = []
    confirmed_nodes = []
    previous_transition_key = None
    previous_current_node = None
    for record in trace:
        if record.current_node != previous_current_node:
            if record.current_node is not None:
                confirmed_nodes.append(record.current_node)
            previous_current_node = record.current_node

        transition_key = _transition_key(record)
        if transition_key != previous_transition_key:
            transitions.append(record)
            print(json.dumps(asdict(record), sort_keys=True))
            previous_transition_key = transition_key

    if confirmed_nodes and confirmed_nodes[0] != str(FIXED_START_NODE_ID):
        raise RuntimeError(
            "fixed-start replay confirmed a node other than node-0000 first"
        )

    if args.trace_output is not None:
        trace_path = args.trace_output.expanduser().resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf-8") as output:
            for record in trace:
                output.write(json.dumps(asdict(record), sort_keys=True))
                output.write("\n")

    final_record = trace[-1]
    summary = {
        "kind": "summary",
        "device": str(args.device),
        "source_dense_frame_count": len(frames),
        "dense_frame_count": len(trace),
        "transition_count": len(transitions),
        "confirmed_node_count": len(confirmed_nodes),
        "confirmed_nodes": confirmed_nodes,
        "close_threshold": args.close_threshold,
        "start_close_confirmations": args.start_close_confirmations,
        "successor_close_confirmations": (
            args.successor_close_confirmations
        ),
        "normal_distance_maximum": args.normal_distance_maximum,
        "lost_confirmations": args.lost_confirmations,
        "tick_period_s": args.tick_period_s,
        "ticks_completed": runtime_status.service.ticks_completed,
        "skipped_tick_slots": runtime_status.service.skipped_tick_slots,
        "runtime_state": runtime_status.state.value,
        "final_phase": final_record.phase,
        "final_current_node": final_record.current_node,
        "final_target_node": final_record.target_node,
        "final_belief_status": final_record.belief_status,
        "elapsed_s": time.perf_counter() - started_at,
    }
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    asyncio.run(_replay(_parse_args()))


if __name__ == "__main__":
    main()

"""Renders Map-bound NoMaD trajectories over every recorded-video frame."""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import BinaryIO, Mapping

from PIL import Image
import torch

from longship.navigation.common import TimePoint
from longship.navigation.map_engine.interface import MapEngine
from longship.navigation.map_engine.models import (
    MapId,
    MapSelector,
    MapVersion,
    MapSnapshot,
    NodeId,
    SegmentDescriptor,
    TopologyQuery,
)
from longship.navigation.ports.trajectory_policy import (
    TrajectoryCandidateSet,
    VisualGoalTrajectoryRequest,
)
from nomad_runtime import NomadConfig, NomadPolicy, NomadTrajectorySession
from longship_adapter import (
    LocalFileGoalImageLoader,
    NomadTopomapMapConfig,
    NomadTrajectoryPolicyConfig,
    NomadVisualGoalTrajectoryPolicy,
    VisualTargetGoalBinding,
    create_nomad_topomap_engine,
    resolve_visual_target_goal,
)
from tools.trajectory_overlay import (
    TrajectoryOverlayState,
    draw_trajectory_overlay,
)
from tools.trajectory_stitching import (
    DiagnosticTrajectoryStitcher,
    OfficialDemoTrajectoryStitcher,
    ShortStepTrajectoryStitcher,
    TrajectoryStitchUpdate,
)


_IMAGE_PROFILE_ID = "nomad.rgb.direct_resize_96x96.imagenet.v1"
_MODEL_ARTIFACT_ID = "nomad.pth"
_OBSERVATION_CLOCK_ID = "camera"
_TERMINAL_PHASES = frozenset(
    {"at_final_node", "localization_lost", "fault"}
)


@dataclass(frozen=True, slots=True)
class _TraceRecord:
    source_timestamp_s: float
    phase: str
    current_node: str | None
    target_node: str | None
    detail_code: str | None


@dataclass(frozen=True, slots=True)
class _VideoMetadata:
    width: int
    height: int
    frames_per_second: float
    duration_s: float


class _FrameTimeSource:
    """Uses the frame currently being processed as the offline policy clock."""

    def __init__(self, initial_time: TimePoint) -> None:
        self._time = initial_time

    def now(self) -> TimePoint:
        return self._time

    def advance(self, value: TimePoint) -> None:
        if value.clock_id != self._time.clock_id:
            raise ValueError("frame time source changed clock domain")
        if value.nanoseconds < self._time.nanoseconds:
            raise ValueError("frame time source moved backward")
        self._time = value


class _TraceTimeline:
    def __init__(self, records: tuple[_TraceRecord, ...]) -> None:
        if not records:
            raise ValueError("localization trace has no timed records")
        self._records = records
        self._index = -1

    def at(self, timestamp_s: float) -> _TraceRecord | None:
        while (
            self._index + 1 < len(self._records)
            and self._records[self._index + 1].source_timestamp_s
            <= timestamp_s + 1.0e-6
        ):
            self._index += 1
        if self._index < 0:
            return None
        return self._records[self._index]


class _RawVideoReader:
    def __init__(
        self,
        video: Path,
        metadata: _VideoMetadata,
        start_time_s: float,
        end_time_s: float,
    ) -> None:
        duration_s = end_time_s - start_time_s
        arguments = (
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-ss",
            f"{start_time_s:.9f}",
            "-t",
            f"{duration_s:.9f}",
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        )
        self._frame_size = metadata.width * metadata.height * 3
        self._process = subprocess.Popen(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def read(self) -> bytes | None:
        stdout = self._require_stdout()
        payload = bytearray()
        while len(payload) < self._frame_size:
            chunk = stdout.read(self._frame_size - len(payload))
            if not chunk:
                break
            payload.extend(chunk)
        if not payload:
            return None
        if len(payload) != self._frame_size:
            raise RuntimeError("FFmpeg returned a truncated input frame")
        return bytes(payload)

    def close(self) -> None:
        stdout = self._require_stdout()
        stdout.close()
        return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(
                "FFmpeg input failed: " + self._read_stderr(return_code)
            )

    def abort(self) -> None:
        if self._process.stdout is not None:
            self._process.stdout.close()
        if self._process.poll() is None:
            self._process.terminate()
        self._process.wait()

    def _require_stdout(self) -> BinaryIO:
        if self._process.stdout is None:
            raise RuntimeError("FFmpeg input pipe is unavailable")
        return self._process.stdout

    def _read_stderr(self, return_code: int) -> str:
        if self._process.stderr is None:
            return str(return_code)
        message = self._process.stderr.read().decode(
            "utf-8", errors="replace"
        )
        return message.strip() or str(return_code)


class _RawVideoWriter:
    def __init__(
        self,
        output: Path,
        metadata: _VideoMetadata,
        *,
        overwrite: bool,
    ) -> None:
        if output.exists() and not overwrite:
            raise FileExistsError(
                f"output already exists; pass --overwrite: {output}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        arguments = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{metadata.width}x{metadata.height}",
            "-framerate",
            f"{metadata.frames_per_second:.12g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "mpeg4",
            "-q:v",
            "3",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        self._process = subprocess.Popen(
            arguments,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: Image.Image) -> None:
        stdin = self._require_stdin()
        try:
            stdin.write(frame.tobytes())
        except BrokenPipeError as error:
            raise RuntimeError("FFmpeg output pipe closed early") from error

    def close(self) -> None:
        stdin = self._require_stdin()
        try:
            stdin.close()
        except BrokenPipeError:
            pass
        return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(
                "FFmpeg output failed: " + self._read_stderr(return_code)
            )

    def _require_stdin(self) -> BinaryIO:
        if self._process.stdin is None:
            raise RuntimeError("FFmpeg output pipe is unavailable")
        return self._process.stdin

    def _read_stderr(self, return_code: int) -> str:
        if self._process.stderr is None:
            return str(return_code)
        message = self._process.stderr.read().decode(
            "utf-8", errors="replace"
        )
        return message.strip() or str(return_code)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render raw NoMaD route-step trajectories over every source "
            "video frame."
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
    parser.add_argument("--localization-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-jsonl", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-time-s", type=float, default=6.0)
    parser.add_argument("--end-time-s", type=float)
    parser.add_argument("--inference-hz", type=float, default=4.0)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument(
        "--stitch-mode",
        choices=("median_short_step", "official_demo"),
        default="median_short_step",
    )
    parser.add_argument(
        "--stitch-step-distance",
        type=float,
        default=0.15,
        help=(
            "Policy-native arc distance integrated from the median candidate "
            "at each inference; diagnostic only."
        ),
    )
    parser.add_argument("--official-sample-index", type=int, default=0)
    parser.add_argument("--official-waypoint-index", type=int, default=2)
    parser.add_argument(
        "--official-max-linear-velocity",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--official-max-angular-velocity",
        type=float,
        default=0.4,
    )
    parser.add_argument("--overwrite", action="store_true")
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


def _load_trace(path: Path) -> tuple[_TraceRecord, ...]:
    records = []
    previous_timestamp_s = -math.inf
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, Mapping):
                raise ValueError(f"trace line {line_number} is not an object")
            timestamp = item.get("source_timestamp_s")
            if timestamp is None:
                timestamp = item.get("timestamp_s")
            if timestamp is None:
                continue
            timestamp_s = float(timestamp)
            if not math.isfinite(timestamp_s):
                raise ValueError("trace timestamps must be finite")
            if timestamp_s < previous_timestamp_s:
                raise ValueError("trace timestamps must not move backward")
            previous_timestamp_s = timestamp_s
            records.append(
                _TraceRecord(
                    source_timestamp_s=timestamp_s,
                    phase=str(item.get("phase", "unknown")),
                    current_node=_optional_string(item.get("current_node")),
                    target_node=_optional_string(item.get("target_node")),
                    detail_code=_optional_string(item.get("detail_code")),
                )
            )
    return tuple(records)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _probe_video(path: Path) -> _VideoMetadata:
    process = subprocess.run(
        (
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,duration",
            "-of",
            "json",
            str(path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(process.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError("video must contain exactly one video stream")
    stream = streams[0]
    metadata = _VideoMetadata(
        width=int(stream["width"]),
        height=int(stream["height"]),
        frames_per_second=float(Fraction(stream["avg_frame_rate"])),
        duration_s=float(stream["duration"]),
    )
    if min(metadata.width, metadata.height) <= 0:
        raise ValueError("video dimensions must be positive")
    if metadata.frames_per_second <= 0.0 or metadata.duration_s <= 0.0:
        raise ValueError("video timing metadata must be positive")
    return metadata


async def _render(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint.expanduser().resolve()
    video = args.video.expanduser().resolve()
    topomap_root = args.topomap.expanduser().resolve()
    trace_path = args.localization_trace.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path, name in (
        (checkpoint, "checkpoint"),
        (video, "video"),
        (trace_path, "localization trace"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
    if not topomap_root.is_dir():
        raise FileNotFoundError(f"topomap not found: {topomap_root}")
    if not math.isfinite(args.inference_hz) or args.inference_hz <= 0.0:
        raise ValueError("inference_hz must be finite and positive")
    if args.num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    if args.sampling_seed < 0:
        raise ValueError("sampling_seed must be non-negative")
    if (
        not math.isfinite(args.stitch_step_distance)
        or args.stitch_step_distance <= 0.0
    ):
        raise ValueError("stitch_step_distance must be finite and positive")
    if args.official_sample_index < 0 or args.official_waypoint_index < 0:
        raise ValueError("official demo indexes must be non-negative")
    if not args.image_profile_id.strip():
        raise ValueError("image_profile_id must not be empty")
    if args.center_crop_aspect is not None and (
        not math.isfinite(args.center_crop_aspect)
        or args.center_crop_aspect <= 0.0
    ):
        raise ValueError("center_crop_aspect must be finite and positive")
    image_profile_id = args.image_profile_id

    metadata = _probe_video(video)
    start_time_s = float(args.start_time_s)
    end_time_s = (
        metadata.duration_s
        if args.end_time_s is None
        else float(args.end_time_s)
    )
    if (
        not math.isfinite(start_time_s)
        or not math.isfinite(end_time_s)
        or start_time_s < 0.0
        or end_time_s <= start_time_s
        or end_time_s > metadata.duration_s + 1.0e-6
    ):
        raise ValueError("invalid video render interval")

    trace = _load_trace(trace_path)
    timeline = _TraceTimeline(trace)
    checkpoint_digest = _sha256(checkpoint)
    map_id = MapId("offline-nomad-overlay")
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
    topology = await map_engine.query_topology(snapshot, TopologyQuery())
    segments = {
        (str(segment.source_node_id), str(segment.target_node_id)): segment
        for segment in topology.segments
    }

    model = NomadPolicy.from_checkpoint(
        checkpoint,
        config=NomadConfig(
            center_crop_aspect=args.center_crop_aspect,
        ),
        device=args.device,
        strict=True,
    )
    session = NomadTrajectorySession(model)
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="nomad-trajectory-overlay",
    )
    policy_time_source = _FrameTimeSource(
        TimePoint(
            clock_id=_OBSERVATION_CLOCK_ID,
            nanoseconds=round(start_time_s * 1_000_000_000),
        )
    )
    policy = NomadVisualGoalTrajectoryPolicy(
        session=session,
        goal_image_loader=LocalFileGoalImageLoader((topomap_root,)),
        inference_executor=executor,
        config=NomadTrajectoryPolicyConfig(
            policy_id="nomad-trajectory-video-overlay-v1",
            image_profile_id=image_profile_id,
            model_artifact_id=_MODEL_ARTIFACT_ID,
            model_artifact_digest=checkpoint_digest,
            observation_clock_id=_OBSERVATION_CLOCK_ID,
            time_source=policy_time_source,
        ),
    )

    bindings: dict[str, VisualTargetGoalBinding] = {}
    goal_images: dict[str, Image.Image] = {}
    reader = _RawVideoReader(
        video,
        metadata,
        start_time_s,
        end_time_s,
    )
    writer = _RawVideoWriter(
        output,
        metadata,
        overwrite=args.overwrite,
    )
    metadata_output = _open_metadata_output(args.trajectory_jsonl)
    frame_index = 0
    inference_index = 0
    next_inference_s = start_time_s
    latest_candidates: TrajectoryCandidateSet | None = None
    latest_goal: Image.Image | None = None
    latest_stitch_update: TrajectoryStitchUpdate | None = None
    stitcher = _create_stitcher(args)
    try:
        while True:
            payload = reader.read()
            if payload is None:
                break
            timestamp_s = (
                start_time_s + frame_index / metadata.frames_per_second
            )
            record = timeline.at(timestamp_s)
            if timestamp_s + 1.0e-6 >= next_inference_s:
                decoded = torch.frombuffer(
                    bytearray(payload),
                    dtype=torch.uint8,
                ).view(metadata.height, metadata.width, 3)
                session.append_observation(
                    decoded,
                    timestamp_s,
                    layout="hwc",
                    channel_order="rgb",
                    value_range="byte",
                )
                latest_candidates, latest_goal, inference_status = (
                    await _run_inference(
                        record=record,
                        timestamp_s=timestamp_s,
                        inference_index=inference_index,
                        sampling_seed=args.sampling_seed,
                        num_candidates=args.num_candidates,
                        inference_hz=args.inference_hz,
                        context_ready=session.ready,
                        model_artifact_digest=checkpoint_digest,
                        image_profile_id=image_profile_id,
                        policy=policy,
                        policy_time_source=policy_time_source,
                        map_engine=map_engine,
                        snapshot=snapshot,
                        segments=segments,
                        bindings=bindings,
                        goal_images=goal_images,
                    )
                )
                if latest_candidates is not None:
                    if record is None:
                        raise AssertionError(
                            "trajectory inference requires localization"
                        )
                    latest_stitch_update = stitcher.append(
                        latest_candidates,
                        timestamp_s,
                    )
                    _write_metadata_candidate(
                        metadata_output,
                        inference_index,
                        timestamp_s,
                        record,
                        latest_candidates,
                        latest_stitch_update,
                        stitcher,
                    )
                else:
                    _write_metadata_status(
                        metadata_output,
                        inference_index,
                        timestamp_s,
                        record,
                        inference_status,
                    )
                inference_index += 1
                next_inference_s = (
                    start_time_s + inference_index / args.inference_hz
                )

            candidates = _fresh_candidates(latest_candidates, record)
            goal = latest_goal if candidates is not None else None
            state = _overlay_state(
                timestamp_s,
                record,
                candidates,
                goal,
                session.ready,
                args.center_crop_aspect,
                stitcher,
                latest_stitch_update,
            )
            source_frame = Image.frombytes(
                "RGB",
                (metadata.width, metadata.height),
                payload,
            )
            writer.write(draw_trajectory_overlay(source_frame, state))
            frame_index += 1
            if frame_index % round(metadata.frames_per_second * 5) == 0:
                print(
                    json.dumps(
                        {
                            "kind": "progress",
                            "source_timestamp_s": timestamp_s,
                            "frames_written": frame_index,
                            "inferences": inference_index,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    except Exception as render_error:
        reader.abort()
        try:
            writer.close()
        except Exception as encode_error:
            raise encode_error from render_error
        raise
    else:
        writer.close()
        reader.close()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        if metadata_output is not None:
            metadata_output.close()

    print(
        json.dumps(
            {
                "kind": "summary",
                "output": str(output),
                "frames_written": frame_index,
                "inferences": inference_index,
                "metadata_rows": _count_metadata_rows(
                    args.trajectory_jsonl
                ),
                "device": str(args.device),
                "image_profile_id": image_profile_id,
                "center_crop_aspect": args.center_crop_aspect,
                "stitch_mode": args.stitch_mode,
                "stitching": stitcher.metadata,
                "stitched_steps": len(stitcher.poses) - 1,
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
            },
            sort_keys=True,
        )
    )


async def _run_inference(
    *,
    record: _TraceRecord | None,
    timestamp_s: float,
    inference_index: int,
    sampling_seed: int,
    num_candidates: int,
    inference_hz: float,
    context_ready: bool,
    model_artifact_digest: str,
    image_profile_id: str,
    policy: NomadVisualGoalTrajectoryPolicy,
    policy_time_source: _FrameTimeSource,
    map_engine: MapEngine,
    snapshot: MapSnapshot,
    segments: Mapping[tuple[str, str], SegmentDescriptor],
    bindings: dict[str, VisualTargetGoalBinding],
    goal_images: dict[str, Image.Image],
) -> tuple[TrajectoryCandidateSet | None, Image.Image | None, str]:
    if (
        record is None
        or not context_ready
        or record.current_node is None
        or record.target_node is None
        or record.current_node == record.target_node
        or record.phase in _TERMINAL_PHASES
    ):
        return None, None, "no_active_route_step"
    pair = (record.current_node, record.target_node)
    segment = segments.get(pair)
    if segment is None:
        raise RuntimeError(f"Map has no directed segment for {pair}")
    binding = bindings.get(record.target_node)
    if binding is None:
        binding = await resolve_visual_target_goal(
            map_engine=map_engine,
            snapshot=snapshot,
            target_node_id=NodeId(record.target_node),
        )
        bindings[record.target_node] = binding
    goal_image = goal_images.get(record.target_node)
    if goal_image is None:
        with Image.open(binding.resource.locator) as source:
            goal_image = source.convert("RGB").copy()
        goal_images[record.target_node] = goal_image

    requested_at = TimePoint(
        clock_id=_OBSERVATION_CLOCK_ID,
        nanoseconds=round(timestamp_s * 1_000_000_000),
    )
    policy_time_source.advance(requested_at)
    candidate_set = await policy.generate_trajectories(
        VisualGoalTrajectoryRequest(
            snapshot_id=snapshot.snapshot_id,
            segment_id=segment.segment_id,
            source_node_id=segment.source_node_id,
            target_node_id=segment.target_node_id,
            target_anchor_id=binding.anchor.anchor_id,
            goal_resource=binding.resource,
            requested_at=requested_at,
            max_observation_age_s=1.0 / inference_hz + 1.0e-3,
            num_candidates=num_candidates,
            sampling_seed=sampling_seed + inference_index,
            expected_image_profile_id=image_profile_id,
            expected_model_artifact_id=_MODEL_ARTIFACT_ID,
            expected_model_artifact_digest=model_artifact_digest,
        )
    )
    return candidate_set, goal_image, "trajectory_generated"


def _fresh_candidates(
    candidates: TrajectoryCandidateSet | None,
    record: _TraceRecord | None,
) -> TrajectoryCandidateSet | None:
    if candidates is None or record is None:
        return None
    if record.phase in _TERMINAL_PHASES:
        return None
    if (
        str(candidates.source_node_id) != record.current_node
        or str(candidates.target_node_id) != record.target_node
    ):
        return None
    return candidates


def _overlay_state(
    timestamp_s: float,
    record: _TraceRecord | None,
    candidates: TrajectoryCandidateSet | None,
    goal: Image.Image | None,
    context_ready: bool,
    center_crop_aspect: float | None,
    stitcher: DiagnosticTrajectoryStitcher,
    stitch_update: TrajectoryStitchUpdate | None,
) -> TrajectoryOverlayState:
    if record is None:
        return TrajectoryOverlayState(
            source_timestamp_s=timestamp_s,
            phase="localizing",
            current_node=None,
            target_node=None,
            status_detail="Waiting for localization timeline",
            model_crop_aspect=center_crop_aspect,
            stitched_path=stitcher.poses,
            stitch_update=stitch_update,
            stitch_panel_title=stitcher.panel_title,
            stitch_panel_detail=stitcher.panel_detail,
        )
    if record.phase == "localization_lost":
        detail = "LOCALIZATION LOST / HOLD - no trajectory generated"
    elif record.phase == "fault":
        detail = "LOCALIZATION FAULT / HOLD - no trajectory generated"
    elif record.target_node is None:
        detail = "ARRIVED / HOLD - no trajectory generated"
    elif not context_ready:
        detail = "Warming four-frame observation context"
    elif candidates is None:
        detail = "Waiting for fresh route-step trajectory"
    else:
        forward_endpoints = sum(
            candidate.waypoints[-1].x > 0.0
            for candidate in candidates.candidates
            if candidate.waypoints
        )
        candidate_count = len(candidates.candidates)
        warning = " CHECK" if forward_endpoints == 0 else ""
        detail = (
            f"distance={candidates.temporal_distance:.3f}  "
            f"end_x>0={forward_endpoints}/{candidate_count}{warning}  "
            "RAW / NOT COMMANDS"
        )
    return TrajectoryOverlayState(
        source_timestamp_s=timestamp_s,
        phase=record.phase,
        current_node=record.current_node,
        target_node=record.target_node,
        status_detail=detail,
        candidate_set=candidates,
        goal_image=goal,
        model_crop_aspect=center_crop_aspect,
        stitched_path=stitcher.poses,
        stitch_update=stitch_update,
        stitch_panel_title=stitcher.panel_title,
        stitch_panel_detail=stitcher.panel_detail,
    )


def _open_metadata_output(path: Path | None) -> BinaryIO | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved.open("wb")


def _write_metadata_status(
    output: BinaryIO | None,
    inference_index: int,
    timestamp_s: float,
    record: _TraceRecord | None,
    status: str,
) -> None:
    if output is None:
        return
    item = {
        "inference_index": inference_index,
        "source_timestamp_s": timestamp_s,
        "status": status,
        "localization": None if record is None else asdict(record),
    }
    output.write((json.dumps(item, sort_keys=True) + "\n").encode())


def _write_metadata_candidate(
    output: BinaryIO | None,
    inference_index: int,
    timestamp_s: float,
    record: _TraceRecord,
    candidates: TrajectoryCandidateSet,
    stitch_update: TrajectoryStitchUpdate | None,
    stitcher: DiagnosticTrajectoryStitcher,
) -> None:
    if output is None:
        return
    item = {
        "inference_index": inference_index,
        "source_timestamp_s": timestamp_s,
        "status": "trajectory_generated",
        "localization": asdict(record),
        "trajectory": asdict(candidates),
        "stitching": {
            "method": stitcher.method_id,
            "configuration": stitcher.metadata,
            "update": (
                None if stitch_update is None else asdict(stitch_update)
            ),
        },
    }
    output.write((json.dumps(item, sort_keys=True) + "\n").encode())


def _create_stitcher(
    args: argparse.Namespace,
) -> DiagnosticTrajectoryStitcher:
    if args.stitch_mode == "official_demo":
        return OfficialDemoTrajectoryStitcher(
            sample_index=args.official_sample_index,
            waypoint_index=args.official_waypoint_index,
            frame_rate_hz=args.inference_hz,
            max_linear_velocity=args.official_max_linear_velocity,
            max_angular_velocity=args.official_max_angular_velocity,
        )
    return ShortStepTrajectoryStitcher(args.stitch_step_distance)


def _count_metadata_rows(path: Path | None) -> int | None:
    if path is None:
        return None
    with path.expanduser().resolve().open("r", encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def main() -> None:
    asyncio.run(_render(_parse_args()))


if __name__ == "__main__":
    main()

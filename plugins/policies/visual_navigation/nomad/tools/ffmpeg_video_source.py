"""FFmpeg-backed decoded observation source for offline video mocking."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import time

import torch

from longship_adapter import (
    DecodedObservationFrame,
)


@dataclass(frozen=True, slots=True)
class FfmpegVideoSourceConfig:
    video_path: Path
    image_profile_id: str
    start_time_s: float = 0.0
    end_time_s: float | None = None
    playback_rate: float = 1.0
    use_source_time_as_policy_time: bool = False
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"

    def validate(self) -> None:
        if not self.video_path.is_file():
            raise FileNotFoundError(f"video not found: {self.video_path}")
        if not self.image_profile_id.strip():
            raise ValueError("image_profile_id must not be empty")
        if not math.isfinite(self.start_time_s) or self.start_time_s < 0.0:
            raise ValueError("start_time_s must be finite and non-negative")
        if self.end_time_s is not None:
            if not math.isfinite(self.end_time_s):
                raise ValueError("end_time_s must be finite")
            if self.end_time_s <= self.start_time_s:
                raise ValueError("end_time_s must follow start_time_s")
        if not math.isfinite(self.playback_rate) or self.playback_rate <= 0.0:
            raise ValueError("playback_rate must be finite and positive")
        if not self.ffmpeg_binary.strip() or not self.ffprobe_binary.strip():
            raise ValueError("FFmpeg binary names must not be empty")


@dataclass(frozen=True, slots=True)
class VideoStreamMetadata:
    width: int
    height: int
    frames_per_second: float
    duration_s: float


class FfmpegVideoFrameSource:
    """Decodes RGB frames and paces them according to recorded video time."""

    def __init__(self, config: FfmpegVideoSourceConfig) -> None:
        config.validate()
        self._config = config
        self._metadata: VideoStreamMetadata | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._sequence_id = 0
        self._latest_source_timestamp_s = config.start_time_s
        self._playback_started_at: float | None = None
        self._stopping = False

    @property
    def metadata(self) -> VideoStreamMetadata:
        if self._metadata is None:
            raise RuntimeError("video source has not started")
        return self._metadata

    @property
    def latest_source_timestamp_s(self) -> float:
        """Returns the latest delivered timestamp in recorded-video time."""

        return self._latest_source_timestamp_s

    async def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("video source already started")
        self._metadata = await _probe_video(self._config)
        arguments = [
            self._config.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(self._config.video_path),
            "-ss",
            f"{self._config.start_time_s:.9f}",
        ]
        if self._config.end_time_s is not None:
            duration_s = (
                self._config.end_time_s - self._config.start_time_s
            )
            arguments.extend(("-t", f"{duration_s:.9f}"))
        arguments.extend(
            (
                "-an",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            )
        )
        self._process = await asyncio.create_subprocess_exec(
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def read(self) -> DecodedObservationFrame | None:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("video source is not running")
        frame_size = self.metadata.width * self.metadata.height * 3
        try:
            payload = await process.stdout.readexactly(frame_size)
        except asyncio.IncompleteReadError as error:
            if error.partial:
                raise RuntimeError(
                    "FFmpeg returned a truncated video frame"
                ) from error
            return_code = await process.wait()
            if return_code != 0 and not self._stopping:
                message = await self._read_stderr()
                raise RuntimeError(
                    f"FFmpeg video decode failed: {message or return_code}"
                )
            return None

        sequence_id = self._sequence_id
        source_timestamp_s = (
            self._config.start_time_s
            + sequence_id / self.metadata.frames_per_second
        )
        if self._playback_started_at is None:
            self._playback_started_at = time.monotonic()
        playback_offset_s = (
            source_timestamp_s - self._config.start_time_s
        ) / self._config.playback_rate
        deadline = self._playback_started_at + playback_offset_s
        delay_s = deadline - time.monotonic()
        if delay_s > 0.0:
            await asyncio.sleep(delay_s)

        self._sequence_id += 1
        self._latest_source_timestamp_s = source_timestamp_s
        image = torch.frombuffer(
            bytearray(payload),
            dtype=torch.uint8,
        ).view(self.metadata.height, self.metadata.width, 3)
        return DecodedObservationFrame(
            image=image,
            timestamp_s=(
                source_timestamp_s
                if self._config.use_source_time_as_policy_time
                else time.monotonic()
            ),
            source_timestamp_s=source_timestamp_s,
            sequence_id=sequence_id,
            image_profile_id=self._config.image_profile_id,
            layout="hwc",
            channel_order="rgb",
            value_range="byte",
        )

    async def stop(self) -> None:
        process = self._process
        if process is None:
            return
        self._stopping = True
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()

    async def _read_stderr(self) -> str:
        process = self._process
        if process is None or process.stderr is None:
            return ""
        payload = await process.stderr.read()
        return payload.decode("utf-8", errors="replace").strip()


async def _probe_video(
    config: FfmpegVideoSourceConfig,
) -> VideoStreamMetadata:
    process = await asyncio.create_subprocess_exec(
        config.ffprobe_binary,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,duration",
        "-of",
        "json",
        str(config.video_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"FFprobe video inspection failed: {message or process.returncode}"
        )
    document = json.loads(stdout.decode("utf-8"))
    streams = document.get("streams", [])
    if len(streams) != 1:
        raise ValueError("video must contain exactly one selected video stream")
    stream = streams[0]
    frames_per_second = float(Fraction(stream["avg_frame_rate"]))
    metadata = VideoStreamMetadata(
        width=int(stream["width"]),
        height=int(stream["height"]),
        frames_per_second=frames_per_second,
        duration_s=float(stream["duration"]),
    )
    if metadata.width <= 0 or metadata.height <= 0:
        raise ValueError("video dimensions must be positive")
    if not math.isfinite(frames_per_second) or frames_per_second <= 0.0:
        raise ValueError("video frame rate must be finite and positive")
    if not math.isfinite(metadata.duration_s) or metadata.duration_s <= 0.0:
        raise ValueError("video duration must be finite and positive")
    if config.start_time_s >= metadata.duration_s:
        raise ValueError("start_time_s lies beyond the video duration")
    if (
        config.end_time_s is not None
        and config.end_time_s > metadata.duration_s + 1.0e-6
    ):
        raise ValueError("end_time_s lies beyond the video duration")
    return metadata

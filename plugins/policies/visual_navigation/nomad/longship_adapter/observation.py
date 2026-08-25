"""Decoded-frame ingress for continuous NoMaD localization."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

from longship.navigation.runtime import (
    LocalizationObservationProducerState,
    LocalizationObservationProducerStatus,
)


@dataclass(frozen=True, slots=True)
class DecodedObservationFrame:
    """One decoded source frame with policy-clock and source timestamps."""

    image: object
    timestamp_s: float
    sequence_id: int
    image_profile_id: str
    layout: str = "hwc"
    channel_order: str = "rgb"
    value_range: str = "byte"
    source_timestamp_s: float | None = None


@runtime_checkable
class DecodedObservationSource(Protocol):
    """Camera, video, or replay source that owns capture and decoding."""

    async def start(self) -> None: ...

    async def read(self) -> DecodedObservationFrame | None: ...

    async def stop(self) -> None: ...


@runtime_checkable
class NomadObservationSink(Protocol):
    """Narrow observation ingress implemented by the NoMaD policy adapter."""

    def submit_observation(
        self,
        image: object,
        timestamp_s: float,
        *,
        layout: str = "chw",
        channel_order: str = "rgb",
        value_range: str = "auto",
    ) -> None: ...

    def clear_observations(self) -> None: ...


class NomadObservationFanout:
    """Replicates each accepted frame into synchronized NoMaD contexts."""

    def __init__(self, sinks: tuple[NomadObservationSink, ...]) -> None:
        if not sinks:
            raise ValueError("observation fanout requires at least one sink")
        self._sinks = sinks

    def submit_observation(
        self,
        image: object,
        timestamp_s: float,
        *,
        layout: str = "chw",
        channel_order: str = "rgb",
        value_range: str = "auto",
    ) -> None:
        try:
            for sink in self._sinks:
                sink.submit_observation(
                    image,
                    timestamp_s,
                    layout=layout,
                    channel_order=channel_order,
                    value_range=value_range,
                )
        except Exception:
            self.clear_observations()
            raise

    def clear_observations(self) -> None:
        first_error: Exception | None = None
        for sink in self._sinks:
            try:
                sink.clear_observations()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


NomadObservationProducerState = LocalizationObservationProducerState


@dataclass(frozen=True, slots=True)
class NomadObservationProducerConfig:
    image_profile_id: str
    sample_hz: float = 9.0
    source_read_timeout_s: float = 2.0
    maximum_frame_gap_s: float = 0.5

    def validate(self) -> None:
        if not self.image_profile_id.strip():
            raise ValueError("image_profile_id must not be empty")
        values = (
            self.sample_hz,
            self.source_read_timeout_s,
            self.maximum_frame_gap_s,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("observation producer timing must be finite")
        if self.sample_hz <= 0.0:
            raise ValueError("sample_hz must be positive")
        if self.source_read_timeout_s <= 0.0:
            raise ValueError("source_read_timeout_s must be positive")
        if self.maximum_frame_gap_s <= 0.0:
            raise ValueError("maximum_frame_gap_s must be positive")


@dataclass(frozen=True, slots=True)
class NomadObservationProducerStatus(
    LocalizationObservationProducerStatus
):
    frames_received: int
    frames_submitted: int
    frames_dropped_by_sampler: int
    context_resets: int
    last_received_timestamp_s: float | None
    last_submitted_timestamp_s: float | None
    last_source_timestamp_s: float | None
    last_submitted_source_timestamp_s: float | None


class NomadObservationProducer:
    """Samples decoded frames into one NoMaD observation context.

    The source owns camera or video decoding and delivery cadence. This class
    validates identity and ordering, applies context sampling, and submits only
    accepted frames to ``NomadVisualGoalDistancePolicy``.
    """

    def __init__(
        self,
        *,
        source: DecodedObservationSource,
        policy: NomadObservationSink,
        config: NomadObservationProducerConfig,
    ) -> None:
        config.validate()
        self._source = source
        self._policy = policy
        self._config = config
        self._state = NomadObservationProducerState.CREATED
        self._frames_received = 0
        self._frames_submitted = 0
        self._frames_dropped_by_sampler = 0
        self._context_resets = 0
        self._last_received_timestamp_s: float | None = None
        self._last_submitted_timestamp_s: float | None = None
        self._last_source_timestamp_s: float | None = None
        self._last_sampling_timestamp_s: float | None = None
        self._last_submitted_source_timestamp_s: float | None = None
        self._last_sequence_id: int | None = None
        self._next_sample_timestamp_s: float | None = None
        self._detail_code: str | None = None
        self._last_error: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._terminated = asyncio.Event()
        self._lifecycle_lock = asyncio.Lock()

    def get_status(self) -> NomadObservationProducerStatus:
        return NomadObservationProducerStatus(
            state=self._state,
            frames_received=self._frames_received,
            frames_submitted=self._frames_submitted,
            frames_dropped_by_sampler=self._frames_dropped_by_sampler,
            context_resets=self._context_resets,
            last_received_timestamp_s=self._last_received_timestamp_s,
            last_submitted_timestamp_s=self._last_submitted_timestamp_s,
            last_source_timestamp_s=self._last_source_timestamp_s,
            last_submitted_source_timestamp_s=(
                self._last_submitted_source_timestamp_s
            ),
            detail_code=self._detail_code,
            last_error=self._last_error,
        )

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._state != NomadObservationProducerState.CREATED:
                raise RuntimeError(
                    "NoMaD observation producer can only start from CREATED"
                )
            self._state = NomadObservationProducerState.STARTING
            self._detail_code = "clearing_observation_context"
            self._policy.clear_observations()
            try:
                self._detail_code = "starting_decoded_observation_source"
                await self._source.start()
            except BaseException as error:
                self._state = NomadObservationProducerState.FAULTED
                error_name = type(error).__name__
                self._detail_code = f"source_start_failed:{error_name}"
                self._last_error = str(error)
                self._terminated.set()
                raise
            self._state = NomadObservationProducerState.RUNNING
            self._detail_code = "running"
            self._task = asyncio.create_task(
                self._run(),
                name="nomad-observation-producer",
            )

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            if self._state == NomadObservationProducerState.CREATED:
                self._state = NomadObservationProducerState.STOPPED
                self._detail_code = "stopped_before_start"
                self._terminated.set()
                return
            if self._state == NomadObservationProducerState.STOPPED:
                return
            terminal_state = self._state
            if terminal_state not in (
                NomadObservationProducerState.COMPLETED,
                NomadObservationProducerState.FAULTED,
            ):
                self._state = NomadObservationProducerState.STOPPING
                self._detail_code = "stop_requested"
            task = self._task
            try:
                await self._source.stop()
            finally:
                if task is not None and not task.done():
                    task.cancel()
                if task is not None:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            if terminal_state not in (
                NomadObservationProducerState.COMPLETED,
                NomadObservationProducerState.FAULTED,
            ):
                self._state = NomadObservationProducerState.STOPPED
                self._detail_code = "stopped"
            self._terminated.set()

    async def wait_stopped(
        self,
        timeout_s: float | None = None,
    ) -> NomadObservationProducerStatus:
        if timeout_s is not None:
            if not math.isfinite(timeout_s) or timeout_s < 0.0:
                raise ValueError("timeout_s must be finite and non-negative")
            await asyncio.wait_for(self._terminated.wait(), timeout=timeout_s)
        else:
            await self._terminated.wait()
        return self.get_status()

    async def _run(self) -> None:
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(
                        self._source.read(),
                        timeout=self._config.source_read_timeout_s,
                    )
                except TimeoutError as error:
                    raise RuntimeError(
                        "decoded observation source timed out"
                    ) from error
                if frame is None:
                    self._state = NomadObservationProducerState.COMPLETED
                    self._detail_code = "source_exhausted"
                    return
                self._accept_frame(frame)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._state = NomadObservationProducerState.FAULTED
            self._detail_code = f"observation_failed:{type(error).__name__}"
            self._last_error = str(error)
        finally:
            if self._state == NomadObservationProducerState.RUNNING:
                self._state = NomadObservationProducerState.STOPPED
                self._detail_code = "producer_stopped_unexpectedly"
            self._terminated.set()

    def _accept_frame(self, frame: DecodedObservationFrame) -> None:
        self._validate_frame(frame)
        self._frames_received += 1
        previous_sampling = self._last_sampling_timestamp_s
        sampling_timestamp_s = (
            frame.timestamp_s
            if frame.source_timestamp_s is None
            else frame.source_timestamp_s
        )
        self._last_received_timestamp_s = frame.timestamp_s
        self._last_source_timestamp_s = frame.source_timestamp_s
        self._last_sampling_timestamp_s = sampling_timestamp_s
        self._last_sequence_id = frame.sequence_id

        if (
            previous_sampling is not None
            and sampling_timestamp_s - previous_sampling
            > self._config.maximum_frame_gap_s
        ):
            self._policy.clear_observations()
            self._context_resets += 1
            self._last_submitted_timestamp_s = None
            self._last_submitted_source_timestamp_s = None
            self._next_sample_timestamp_s = None

        minimum_interval_s = 1.0 / self._config.sample_hz
        if (
            self._next_sample_timestamp_s is not None
            and sampling_timestamp_s
            < self._next_sample_timestamp_s - 1.0e-9
        ):
            self._frames_dropped_by_sampler += 1
            return

        self._policy.submit_observation(
            frame.image,
            frame.timestamp_s,
            layout=frame.layout,
            channel_order=frame.channel_order,
            value_range=frame.value_range,
        )
        self._frames_submitted += 1
        self._last_submitted_timestamp_s = frame.timestamp_s
        self._last_submitted_source_timestamp_s = frame.source_timestamp_s
        if self._next_sample_timestamp_s is None:
            self._next_sample_timestamp_s = (
                sampling_timestamp_s + minimum_interval_s
            )
        else:
            elapsed_intervals = max(
                0,
                math.floor(
                    (
                        sampling_timestamp_s
                        - self._next_sample_timestamp_s
                    )
                    / minimum_interval_s
                ),
            )
            self._next_sample_timestamp_s += (
                elapsed_intervals + 1
            ) * minimum_interval_s

    def _validate_frame(self, frame: DecodedObservationFrame) -> None:
        if frame.image_profile_id != self._config.image_profile_id:
            raise ValueError(
                "decoded frame image profile does not match policy"
            )
        if not math.isfinite(frame.timestamp_s):
            raise ValueError("decoded frame timestamp must be finite")
        if frame.source_timestamp_s is not None:
            if not math.isfinite(frame.source_timestamp_s):
                raise ValueError("source frame timestamp must be finite")
            if (
                self._last_source_timestamp_s is not None
                and frame.source_timestamp_s
                <= self._last_source_timestamp_s
            ):
                raise ValueError("source frame timestamps must increase")
        if self._last_received_timestamp_s is not None:
            if frame.timestamp_s <= self._last_received_timestamp_s:
                raise ValueError("decoded frame timestamps must increase")
        if self._last_sequence_id is not None:
            if frame.sequence_id <= self._last_sequence_id:
                raise ValueError("decoded frame sequence ids must increase")
        if frame.layout not in ("chw", "hwc"):
            raise ValueError("decoded frame layout is unsupported")
        if frame.channel_order not in ("rgb", "bgr"):
            raise ValueError("decoded frame channel order is unsupported")
        if frame.value_range not in ("auto", "unit", "byte"):
            raise ValueError("decoded frame value range is unsupported")

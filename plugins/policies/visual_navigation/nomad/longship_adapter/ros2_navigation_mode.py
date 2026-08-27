"""NoMaD Navigation Mode driver for an externally published ROS 2 RGB topic.

The driver consumes the color image from an RGBD camera. Depth remains owned by
the external ROS graph for future safety and geometry integrations; NoMaD's
current localization and trajectory policies consume RGB only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import TypeVar

from longship.navigation import (
    NavigationAuthority,
    NavigationModeDriver,
    NavigationRequest,
    NavigationResult,
    NavigationSession,
    NavigationStopRequest,
    StopResult,
)
from longship.navigation.common import TimePoint
from longship.navigation.local_trajectory_engine import (
    LocalTrajectoryEngineConfig,
    LocalTrajectoryHoldReason,
    LocalTrajectoryPublication,
    LocalTrajectoryRevision,
    LocalTrajectoryState,
    LocalTrajectoryStream,
    LocalTrajectoryStreamError,
    LocalTrajectoryStreamErrorCode,
    LocalTrajectoryStreamId,
    LocalTrajectoryUpdateOutcome,
    LocalTrajectoryUpdateResult,
    RouteBoundLocalTrajectoryEngine,
    WaitForLocalTrajectoryRequest,
)
from longship.navigation.localization_engine.fixed_start_visual import (
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
from longship.navigation.map_engine import MapEngine
from longship.navigation.map_engine.models import (
    MapId,
    MapSelector,
    MapSnapshot,
    MapVersion,
    NodeId,
    SnapshotId,
)
from longship.navigation.planning_engine import TopologicalPlanningEngine
from longship.navigation.planning_engine.models import (
    PlanningOutcome,
    PlanningRequestId,
    PlanningTarget,
    RouteId,
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

from .image_resource import LocalFileGoalImageLoader
from .observation import (
    DecodedObservationSource,
    NomadObservationFanout,
    NomadObservationProducer,
    NomadObservationProducerConfig,
)
from .topomap import NomadTopomapMapConfig, create_nomad_topomap_engine
from .trajectory_policy import (
    NomadTrajectoryPolicyConfig,
    NomadVisualGoalTrajectoryPolicy,
)
from .visual_policy import (
    NomadVisualGoalDistancePolicy,
    NomadVisualPolicyConfig,
)


_DEFAULT_IMAGE_PROFILE_ID = "nomad.rgb.direct_resize_96x96.imagenet.v1"
_DEFAULT_MODEL_ARTIFACT_ID = "nomad.pth"
_DEFAULT_CLOCK_ID = "monotonic"


DecodedObservationSourceFactory = Callable[[str, str], DecodedObservationSource]


@dataclass(frozen=True, slots=True)
class NomadRos2NavigationModeConfig:
    """Static configuration for one ROS 2 NoMaD Navigation Mode deployment."""

    topomap_root: Path
    color_topic: str
    checkpoint_path: Path
    device: str = "cuda:0"
    map_id: str = "nomad-ros2"
    map_version: str = "local"
    image_profile_id: str = _DEFAULT_IMAGE_PROFILE_ID
    center_crop_aspect: float | None = None
    observation_sample_hz: float = 9.0
    localization_tick_period_s: float = 1.0 / 9.0
    source_read_timeout_s: float = 5.0
    maximum_frame_gap_s: float = 0.5
    max_observation_age_s: float = 0.5
    publication_validity_s: float = 0.5
    initial_location_timeout_s: float = 30.0
    num_candidates: int = 8
    selected_candidate_index: int = 0
    sampling_seed_base: int = 0

    def validate(self) -> None:
        if not self.color_topic.strip():
            raise ValueError("color_topic must not be empty")
        if not self.map_id.strip() or not self.map_version.strip():
            raise ValueError("map identity must not be empty")
        if not self.image_profile_id.strip():
            raise ValueError("image_profile_id must not be empty")
        if not self.device.strip():
            raise ValueError("device must not be empty")
        values = (
            self.observation_sample_hz,
            self.localization_tick_period_s,
            self.source_read_timeout_s,
            self.maximum_frame_gap_s,
            self.max_observation_age_s,
            self.publication_validity_s,
            self.initial_location_timeout_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("Navigation Mode timing values must be positive")
        if self.center_crop_aspect is not None and (
            not math.isfinite(self.center_crop_aspect)
            or self.center_crop_aspect <= 0.0
        ):
            raise ValueError("center_crop_aspect must be finite and positive")
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if not 0 <= self.selected_candidate_index < self.num_candidates:
            raise ValueError("selected_candidate_index is outside candidate range")


class NomadRos2NavigationModeDriverFactory:
    """Creates fresh drivers; each driver owns one entered Navigation Mode."""

    def __init__(
        self,
        config: NomadRos2NavigationModeConfig,
        *,
        source_factory: DecodedObservationSourceFactory | None = None,
    ) -> None:
        config.validate()
        self._config = config
        self._source_factory = source_factory

    def create_driver(self) -> NomadRos2NavigationModeDriver:
        return NomadRos2NavigationModeDriver(
            self._config,
            source_factory=self._source_factory,
        )


class NomadRos2NavigationModeDriver(NavigationModeDriver):
    """Shares one NoMaD model, ROS 2 RGB source, and localizer per Nav mode."""

    def __init__(
        self,
        config: NomadRos2NavigationModeConfig,
        *,
        source_factory: DecodedObservationSourceFactory | None = None,
    ) -> None:
        config.validate()
        self._config = config
        self._source_factory = source_factory or _create_ros2_color_source
        self._clock = MonotonicTimeSource(clock_id=_DEFAULT_CLOCK_ID)
        self._map_engine: MapEngine | None = None
        self._snapshot: MapSnapshot | None = None
        self._localization_engine: FixedStartVisualLocalizationEngine | None = None
        self._localization_runtime: LocalizationRuntime | None = None
        self._trajectory_policy: NomadVisualGoalTrajectoryPolicy | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._stream: _ModeTrajectoryStream | None = None
        self._active_session: _NomadRos2NavigationSession | None = None
        self._checkpoint_digest: str | None = None
        self._entered = False
        self._lock = asyncio.Lock()

    @property
    def trajectory_stream(self) -> LocalTrajectoryStream:
        if self._stream is None:
            raise RuntimeError("NoMaD Navigation Mode has not entered")
        return self._stream

    async def enter(self) -> None:
        """Starts shared RGB ingestion, NoMaD policies, and start localization."""
        async with self._lock:
            if self._entered:
                return
            checkpoint = self._config.checkpoint_path.expanduser().resolve()
            topomap_root = self._config.topomap_root.expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"NoMaD checkpoint not found: {checkpoint}")
            if not topomap_root.is_dir():
                raise FileNotFoundError(f"NoMaD topomap not found: {topomap_root}")

            checkpoint_digest = _sha256(checkpoint)
            map_engine = create_nomad_topomap_engine(
                NomadTopomapMapConfig(
                    root=topomap_root,
                    map_id=MapId(self._config.map_id),
                    version=MapVersion(self._config.map_version),
                    published_at=self._clock.now(),
                    model_artifact_id=_DEFAULT_MODEL_ARTIFACT_ID,
                    model_artifact_digest=checkpoint_digest,
                    image_profile_id=self._config.image_profile_id,
                    expected_center_crop_aspect=self._config.center_crop_aspect,
                )
            )
            snapshot = await map_engine.get_snapshot(
                MapSelector(
                    map_id=MapId(self._config.map_id),
                    version=MapVersion(self._config.map_version),
                )
            )
            from nomad_runtime import NomadConfig, NomadPolicy

            model = NomadPolicy.from_checkpoint(
                checkpoint,
                config=NomadConfig(
                    center_crop_aspect=self._config.center_crop_aspect
                ),
                device=self._config.device,
                strict=True,
            )
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="nomad-ros2-navigation",
            )
            runtime: LocalizationRuntime | None = None
            stream: _ModeTrajectoryStream | None = None
            try:
                runtime, localization_engine, trajectory_policy = (
                    await self._create_localization_runtime(
                        map_engine=map_engine,
                        snapshot=snapshot,
                        topomap_root=topomap_root,
                        model=model,
                        executor=executor,
                        checkpoint_digest=checkpoint_digest,
                    )
                )
                stream = _ModeTrajectoryStream(
                    stream_id=LocalTrajectoryStreamId("nomad-ros2-nav-mode"),
                    snapshot_id=snapshot.snapshot_id,
                    clock=self._clock,
                )
                await runtime.start()
                await self._wait_for_location(localization_engine)
                await stream.publish_holding("waiting_for_navigation_target")
            except BaseException:
                if runtime is not None:
                    await runtime.stop()
                if stream is not None:
                    await stream.close()
                await asyncio.to_thread(
                    executor.shutdown,
                    wait=True,
                    cancel_futures=True,
                )
                raise

            self._map_engine = map_engine
            self._snapshot = snapshot
            self._localization_engine = localization_engine
            self._localization_runtime = runtime
            self._trajectory_policy = trajectory_policy
            self._executor = executor
            self._stream = stream
            self._checkpoint_digest = checkpoint_digest
            self._entered = True

    async def exit(self) -> None:
        """Stops target generation first, then the shared localization runtime."""
        async with self._lock:
            active = self._active_session
            if active is not None:
                await active.shutdown()
            if self._stream is not None:
                await self._stream.publish_stopped()
                await self._stream.close()
            if self._localization_runtime is not None:
                await self._localization_runtime.stop()
            if self._executor is not None:
                await asyncio.to_thread(
                    self._executor.shutdown,
                    wait=True,
                    cancel_futures=True,
                )
            self._active_session = None
            self._entered = False

    async def start_session(
        self,
        request: NavigationRequest,
        authority: NavigationAuthority,
    ) -> NavigationSession:
        """Plans one target route without rebuilding entered mode resources."""
        async with self._lock:
            self._ensure_entered()
            if self._active_session is not None:
                raise RuntimeError("NoMaD Navigation Mode already has an active target")
            snapshot = _required(self._snapshot, "map snapshot")
            map_engine = _required(self._map_engine, "map engine")
            localization_engine = _required(
                self._localization_engine,
                "localization engine",
            )
            trajectory_policy = _required(
                self._trajectory_policy,
                "trajectory policy",
            )
            stream = _required(self._stream, "trajectory stream")
            self._validate_request_map(request, snapshot)
            belief = await self._wait_for_location(localization_engine)
            route_result = await TopologicalPlanningEngine(map_engine).plan_route(
                RoutePlanningRequest(
                    request_id=PlanningRequestId(request.request_id),
                    requested_at=self._clock.now(),
                    snapshot=snapshot,
                    location_belief=belief,
                    target=PlanningTarget(
                        target_ref=request.waypoint_id,
                        candidate_node_ids=(NodeId(request.waypoint_id),),
                    ),
                )
            )
            if route_result.outcome == PlanningOutcome.ALREADY_AT_GOAL:
                await stream.publish_completed(
                    route_id=RouteId(request.route_id),
                    snapshot_id=snapshot.snapshot_id,
                )
                return _CompletedNavigationSession(
                    request=request,
                    stream=stream,
                    arrived=True,
                    evidence="nomad.navigation.already_at_goal",
                )
            if (
                route_result.outcome != PlanningOutcome.ROUTE_FOUND
                or route_result.route_plan is None
            ):
                await stream.publish_holding("route_not_found")
                return _CompletedNavigationSession(
                    request=request,
                    stream=stream,
                    arrived=False,
                    evidence="nomad.navigation.route_not_found",
                    detail=str(route_result.failure),
                )

            route_plan = route_result.route_plan
            trajectory_engine = await RouteBoundLocalTrajectoryEngine.create(
                map_engine=map_engine,
                snapshot=snapshot,
                route_plan=route_plan,
                localization_engine=localization_engine,
                trajectory_policy=trajectory_policy,
                stream_id=LocalTrajectoryStreamId(
                    f"nomad-route:{request.request_id}"
                ),
                started_at=self._clock.now(),
                config=LocalTrajectoryEngineConfig(
                    image_profile_id=self._config.image_profile_id,
                    model_artifact_id=_DEFAULT_MODEL_ARTIFACT_ID,
                    model_artifact_digest=_required(
                        self._checkpoint_digest,
                        "checkpoint digest",
                    ),
                    time_source=self._clock,
                    num_candidates=self._config.num_candidates,
                    selected_candidate_index=(
                        self._config.selected_candidate_index
                    ),
                    sampling_seed_base=self._config.sampling_seed_base,
                    max_observation_age_s=self._config.max_observation_age_s,
                    publication_validity_s=(
                        self._config.publication_validity_s
                    ),
                ),
            )
            trajectory_service = LocalizationDrivenLocalTrajectoryService(
                engine=trajectory_engine,
                localization_engine=localization_engine,
                time_source=self._clock,
                config=LocalTrajectoryServiceConfig(),
            )
            await stream.attach(trajectory_engine)
            await trajectory_service.start()
            session = _NomadRos2NavigationSession(
                request=request,
                route_plan=route_plan,
                stream=stream,
                route_stream=trajectory_engine,
                trajectory_service=trajectory_service,
                on_terminal=self._clear_active_session,
            )
            self._active_session = session
            return session

    async def _create_localization_runtime(
        self,
        *,
        map_engine: MapEngine,
        snapshot: MapSnapshot,
        topomap_root: Path,
        model: object,
        executor: ThreadPoolExecutor,
        checkpoint_digest: str,
    ) -> tuple[
        LocalizationRuntime,
        FixedStartVisualLocalizationEngine,
        NomadVisualGoalTrajectoryPolicy,
    ]:
        from nomad_runtime import NomadDistanceSession, NomadTrajectorySession

        goal_image_loader = LocalFileGoalImageLoader(
            allowed_roots=(topomap_root,)
        )
        distance_policy = NomadVisualGoalDistancePolicy(
            session=NomadDistanceSession(model),
            goal_image_loader=goal_image_loader,
            inference_executor=executor,
            config=NomadVisualPolicyConfig(
                policy_id="nomad-distance-ros2",
                image_profile_id=self._config.image_profile_id,
                model_artifact_id=_DEFAULT_MODEL_ARTIFACT_ID,
                model_artifact_digest=checkpoint_digest,
                observation_clock_id=_DEFAULT_CLOCK_ID,
                time_source=self._clock,
            ),
        )
        trajectory_policy = NomadVisualGoalTrajectoryPolicy(
            session=NomadTrajectorySession(model),
            goal_image_loader=goal_image_loader,
            inference_executor=executor,
            config=NomadTrajectoryPolicyConfig(
                policy_id="nomad-trajectory-ros2",
                image_profile_id=self._config.image_profile_id,
                model_artifact_id=_DEFAULT_MODEL_ARTIFACT_ID,
                model_artifact_digest=checkpoint_digest,
                observation_clock_id=_DEFAULT_CLOCK_ID,
                time_source=self._clock,
            ),
        )
        localization_engine = await FixedStartVisualLocalizationEngine.create(
            map_engine=map_engine,
            snapshot=snapshot,
            policy=distance_policy,
            profile=FixedStartVisualTrackingProfile(
                image_profile_id=self._config.image_profile_id,
                max_observation_age_s=self._config.max_observation_age_s,
            ),
            stream_id=BeliefStreamId("nomad-ros2-nav-localization"),
            started_at=self._clock.now(),
        )
        source = self._source_factory(
            self._config.color_topic,
            self._config.image_profile_id,
        )
        observation_producer = NomadObservationProducer(
            source=source,
            policy=NomadObservationFanout((distance_policy, trajectory_policy)),
            config=NomadObservationProducerConfig(
                image_profile_id=self._config.image_profile_id,
                sample_hz=self._config.observation_sample_hz,
                source_read_timeout_s=self._config.source_read_timeout_s,
                maximum_frame_gap_s=self._config.maximum_frame_gap_s,
            ),
        )
        localization_service = ContinuousLocalizationService(
            engine=localization_engine,
            time_source=self._clock,
            config=LocalizationServiceConfig(
                tick_period_s=self._config.localization_tick_period_s,
            ),
        )
        return (
            LocalizationRuntime(
                observation_producer=observation_producer,
                localization_service=localization_service,
                config=LocalizationRuntimeConfig(
                    observation_completion_policy=(
                        LocalizationObservationCompletionPolicy.ALLOW_UNTIL_STOP
                    )
                ),
            ),
            localization_engine,
            trajectory_policy,
        )

    async def _wait_for_location(
        self,
        engine: FixedStartVisualLocalizationEngine,
    ) -> LocationBelief:
        async def wait() -> LocationBelief:
            belief = engine.get_belief()
            revision = belief.revision
            while _single_node_location(belief) is None:
                update = await engine.wait_for_update(
                    WaitForUpdateRequest(after_revision=revision, timeout_s=1.0)
                )
                if update.outcome == BeliefUpdateOutcome.UPDATED:
                    belief = update.belief
                    revision = belief.revision
            return belief

        try:
            return await asyncio.wait_for(
                wait(),
                timeout=self._config.initial_location_timeout_s,
            )
        except TimeoutError as error:
            raise RuntimeError("timed out waiting for visual localization") from error

    async def _clear_active_session(
        self,
        session: _NomadRos2NavigationSession,
    ) -> None:
        if self._active_session is session:
            self._active_session = None

    def _ensure_entered(self) -> None:
        if not self._entered:
            raise RuntimeError("NoMaD Navigation Mode is not entered")

    def _validate_request_map(
        self,
        request: NavigationRequest,
        snapshot: MapSnapshot,
    ) -> None:
        if request.map_id != str(snapshot.map_id):
            raise ValueError("navigation request map ID does not match mode map")
        if request.map_version != str(snapshot.version):
            raise ValueError("navigation request map version does not match mode")


class _NomadRos2NavigationSession:
    """One planned target sharing the enclosing mode's observation resources."""

    def __init__(
        self,
        *,
        request: NavigationRequest,
        route_plan: RoutePlan,
        stream: _ModeTrajectoryStream,
        route_stream: RouteBoundLocalTrajectoryEngine,
        trajectory_service: LocalizationDrivenLocalTrajectoryService,
        on_terminal: Callable[[_NomadRos2NavigationSession], object],
    ) -> None:
        self._request = request
        self._route_plan = route_plan
        self._stream = stream
        self._route_stream = route_stream
        self._trajectory_service = trajectory_service
        self._on_terminal = on_terminal
        self._closed = False
        self._close_lock = asyncio.Lock()

    @property
    def trajectory_stream(self) -> LocalTrajectoryStream:
        return self._stream

    async def wait_result(self) -> NavigationResult:
        publication = self._route_stream.get_latest()
        try:
            while publication.state not in _TERMINAL_TRAJECTORY_STATES:
                update = await self._route_stream.wait_for_update(
                    WaitForLocalTrajectoryRequest(
                        after_revision=publication.revision,
                        timeout_s=None,
                    )
                )
                publication = update.publication
            return _navigation_result_from_publication(
                request=self._request,
                publication=publication,
            )
        finally:
            await self.shutdown()

    async def pause(self, authority: NavigationAuthority) -> None:
        authority.ensure_active()
        await self._stream.set_paused(True)

    async def resume(self, authority: NavigationAuthority) -> None:
        authority.ensure_active()
        await self._stream.set_paused(False)

    async def stop(self, request: NavigationStopRequest) -> StopResult:
        if request.request_id != self._request.request_id:
            raise ValueError("stop request does not match active navigation")
        await self.shutdown()
        return StopResult(
            request_id=request.request_id,
            revoked_through_epoch=request.revoke_through_epoch,
            requested=True,
            verified_stopped=True,
            evidence="nomad.navigation.trajectory_service_stopped",
        )

    async def shutdown(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            await self._trajectory_service.stop()
            await self._on_terminal(self)


class _CompletedNavigationSession:
    """Terminal session used when planning finishes without a trajectory loop."""

    def __init__(
        self,
        *,
        request: NavigationRequest,
        stream: LocalTrajectoryStream,
        arrived: bool,
        evidence: str,
        detail: str = "",
    ) -> None:
        self._request = request
        self._stream = stream
        self._arrived = arrived
        self._evidence = evidence
        self._detail = detail

    @property
    def trajectory_stream(self) -> LocalTrajectoryStream:
        return self._stream

    async def wait_result(self) -> NavigationResult:
        return NavigationResult(
            arrived=self._arrived,
            request_id=self._request.request_id,
            authority_epoch=self._request.authority_epoch,
            map_id=self._request.map_id,
            map_version=self._request.map_version,
            route_id=self._request.route_id,
            waypoint_id=self._request.waypoint_id,
            evidence=self._evidence,
            detail=self._detail,
        )

    async def pause(self, authority: NavigationAuthority) -> None:
        authority.ensure_active()

    async def resume(self, authority: NavigationAuthority) -> None:
        authority.ensure_active()

    async def stop(self, request: NavigationStopRequest) -> StopResult:
        return StopResult(
            request_id=request.request_id,
            revoked_through_epoch=request.revoke_through_epoch,
            requested=True,
            verified_stopped=True,
            evidence="nomad.navigation.already_terminal",
        )


class _ModeTrajectoryStream:
    """Stable mode-level stream forwarding publications from one route stream."""

    def __init__(
        self,
        *,
        stream_id: LocalTrajectoryStreamId,
        snapshot_id: SnapshotId,
        clock: MonotonicTimeSource,
    ) -> None:
        self._stream_id = stream_id
        self._clock = clock
        self._sequence = 0
        self._latest = LocalTrajectoryPublication(
            revision=LocalTrajectoryRevision(stream_id=stream_id, sequence=0),
            route_id=RouteId("navigation-mode"),
            snapshot_id=snapshot_id,
            state=LocalTrajectoryState.INITIALIZING,
            published_at=clock.now(),
            detail_code="waiting_for_default_start_localization",
        )
        self._condition = asyncio.Condition()
        self._source_task: asyncio.Task[None] | None = None
        self._source_latest: LocalTrajectoryPublication | None = None
        self._paused = False
        self._closed = False

    def get_latest(self) -> LocalTrajectoryPublication:
        return self._latest

    async def wait_for_update(
        self,
        request: WaitForLocalTrajectoryRequest,
    ) -> LocalTrajectoryUpdateResult:
        if request.after_revision.stream_id != self._stream_id:
            return LocalTrajectoryUpdateResult(
                outcome=LocalTrajectoryUpdateOutcome.STREAM_RESET,
                publication=self._latest,
            )
        if request.after_revision.sequence > self._latest.revision.sequence:
            raise LocalTrajectoryStreamError(
                LocalTrajectoryStreamErrorCode.INVALID_REQUEST,
                "trajectory revision is ahead of the mode stream",
            )
        if request.after_revision.sequence < self._latest.revision.sequence:
            return LocalTrajectoryUpdateResult(
                outcome=LocalTrajectoryUpdateOutcome.UPDATED,
                publication=self._latest,
            )
        try:
            async with self._condition:
                await asyncio.wait_for(
                    self._condition.wait_for(
                        lambda: request.after_revision.sequence
                        < self._latest.revision.sequence
                    ),
                    timeout=request.timeout_s,
                )
        except TimeoutError:
            return LocalTrajectoryUpdateResult(
                outcome=LocalTrajectoryUpdateOutcome.TIMED_OUT,
                publication=self._latest,
            )
        return LocalTrajectoryUpdateResult(
            outcome=LocalTrajectoryUpdateOutcome.UPDATED,
            publication=self._latest,
        )

    async def attach(self, source: LocalTrajectoryStream) -> None:
        if self._closed:
            raise RuntimeError("mode trajectory stream is closed")
        await self._cancel_source_task()
        self._source_latest = source.get_latest()
        await self._publish(self._source_latest)
        self._source_task = asyncio.create_task(
            self._forward(source),
            name="nomad-navigation-mode-trajectory-forwarder",
        )

    async def set_paused(self, paused: bool) -> None:
        self._paused = paused
        publication = self._source_latest or self._latest
        await self._publish(publication, preserve_publication=False)

    async def publish_holding(self, detail_code: str) -> None:
        await self._publish(
            replace(
                self._latest,
                state=LocalTrajectoryState.HOLDING,
                published_at=self._clock.now(),
                trajectory=None,
                valid_until=None,
                hold_reason=LocalTrajectoryHoldReason.ROUTE_POSITION_UNRESOLVED,
                detail_code=detail_code,
            )
        )

    async def publish_completed(
        self,
        *,
        route_id: RouteId,
        snapshot_id: SnapshotId,
    ) -> None:
        await self._publish(
            replace(
                self._latest,
                route_id=route_id,
                snapshot_id=snapshot_id,
                state=LocalTrajectoryState.ROUTE_COMPLETED,
                published_at=self._clock.now(),
                trajectory=None,
                valid_until=None,
                hold_reason=None,
                detail_code="already_at_goal",
            )
        )

    async def publish_stopped(self) -> None:
        await self._publish(
            replace(
                self._latest,
                state=LocalTrajectoryState.STOPPED,
                published_at=self._clock.now(),
                trajectory=None,
                valid_until=None,
                hold_reason=LocalTrajectoryHoldReason.SERVICE_STOPPED,
                detail_code="navigation_mode_exited",
            )
        )

    async def close(self) -> None:
        self._closed = True
        await self._cancel_source_task()

    async def _forward(self, source: LocalTrajectoryStream) -> None:
        publication = source.get_latest()
        try:
            while True:
                result = await source.wait_for_update(
                    WaitForLocalTrajectoryRequest(
                        after_revision=publication.revision,
                        timeout_s=None,
                    )
                )
                publication = result.publication
                self._source_latest = publication
                await self._publish(publication)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._publish(
                replace(
                    self._latest,
                    state=LocalTrajectoryState.FAULTED,
                    published_at=self._clock.now(),
                    trajectory=None,
                    valid_until=None,
                    hold_reason=None,
                    detail_code=f"stream_forward_failed:{type(error).__name__}",
                )
            )

    async def _publish(
        self,
        publication: LocalTrajectoryPublication,
        *,
        preserve_publication: bool = True,
    ) -> None:
        if self._paused and publication.state not in _TERMINAL_TRAJECTORY_STATES:
            publication = replace(
                publication,
                state=LocalTrajectoryState.HOLDING,
                trajectory=None,
                valid_until=None,
                hold_reason=None,
                detail_code="paused",
            )
        elif not preserve_publication:
            publication = replace(
                publication,
                published_at=self._clock.now(),
            )
        async with self._condition:
            self._sequence += 1
            self._latest = replace(
                publication,
                revision=LocalTrajectoryRevision(
                    stream_id=self._stream_id,
                    sequence=self._sequence,
                ),
            )
            self._condition.notify_all()

    async def _cancel_source_task(self) -> None:
        task = self._source_task
        self._source_task = None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


_TERMINAL_TRAJECTORY_STATES = frozenset(
    {
        LocalTrajectoryState.ROUTE_COMPLETED,
        LocalTrajectoryState.FAULTED,
        LocalTrajectoryState.STOPPED,
    }
)


def _navigation_result_from_publication(
    *,
    request: NavigationRequest,
    publication: LocalTrajectoryPublication,
) -> NavigationResult:
    arrived = publication.state == LocalTrajectoryState.ROUTE_COMPLETED
    return NavigationResult(
        arrived=arrived,
        request_id=request.request_id,
        authority_epoch=request.authority_epoch,
        map_id=request.map_id,
        map_version=request.map_version,
        route_id=request.route_id,
        waypoint_id=request.waypoint_id,
        evidence=f"nomad.navigation.{publication.state.value}",
        detail=publication.detail_code or "",
    )


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


def _create_ros2_color_source(
    topic_name: str,
    image_profile_id: str,
) -> DecodedObservationSource:
    try:
        from tools.ros2_image_source import (
            Ros2ImageFrameSource,
            Ros2ImageFrameSourceConfig,
        )
    except ImportError as error:
        raise RuntimeError(
            "ROS 2 NoMaD mode needs the source checkout's tools package; "
            "pass source_factory when embedding it elsewhere"
        ) from error
    return Ros2ImageFrameSource(
        Ros2ImageFrameSourceConfig(
            topic_name=topic_name,
            image_profile_id=image_profile_id,
            node_name="nomad_navigation_mode_color_source",
        )
    )


_Value = TypeVar("_Value")


def _required(value: _Value | None, label: str) -> _Value:
    if value is None:
        raise RuntimeError(f"NoMaD Navigation Mode has no {label}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"

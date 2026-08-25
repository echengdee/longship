"""NoMaD trajectory candidates behind the executor-side policy SPI."""

from __future__ import annotations

import asyncio
from concurrent.futures import Executor
from dataclasses import dataclass
from functools import partial
import math
from typing import Protocol, runtime_checkable

import torch

from longship.navigation.common import TimePoint, TimeSource
from longship.navigation.map_engine.interface import MapEngine
from longship.navigation.map_engine.models import (
    AnchorDescriptor,
    AnchorKind,
    AnchorPurpose,
    AnchorQuery,
    MapEntityKind,
    MapEntityRef,
    MapSnapshot,
    NodeId,
    ResourceDescriptor,
    ResourceKind,
)
from longship.navigation.ports.trajectory_policy import (
    PolicyNativeWaypoint,
    TrajectoryCandidate,
    TrajectoryCandidateId,
    TrajectoryCandidateSet,
    TrajectoryPolicyError,
    TrajectoryPolicyErrorCode,
    VisualGoalTrajectoryRequest,
)

from .image_resource import GoalImageLoader


class _TrajectoryResult(Protocol):
    temporal_distance: float
    trajectories: torch.Tensor
    observation_timestamp_s: float
    sampling_seed: int | None


@runtime_checkable
class NomadTrajectorySessionPort(Protocol):
    def append_observation(
        self,
        image: object,
        timestamp_s: float,
        *,
        layout: str = "chw",
        channel_order: str = "rgb",
        value_range: str = "auto",
    ) -> None: ...

    def clear_observations(self) -> None: ...

    def predict_goal_trajectories(
        self,
        goal_image: object,
        *,
        goal_layout: str = "chw",
        goal_channel_order: str = "rgb",
        goal_value_range: str = "auto",
        now_s: float,
        max_observation_age_s: float,
        num_candidates: int,
        sampling_seed: int | None,
    ) -> _TrajectoryResult: ...


@dataclass(frozen=True, slots=True)
class NomadTrajectoryPolicyConfig:
    policy_id: str
    image_profile_id: str
    model_artifact_id: str
    model_artifact_digest: str
    observation_clock_id: str
    time_source: TimeSource

    def validate(self) -> None:
        fields = (
            self.policy_id,
            self.image_profile_id,
            self.model_artifact_id,
            self.observation_clock_id,
        )
        if not all(value.strip() for value in fields):
            raise ValueError("NoMaD trajectory policy ids must not be empty")
        if not isinstance(self.time_source, TimeSource):
            raise ValueError("NoMaD trajectory policy needs a time source")
        _canonical_sha256(self.model_artifact_digest)


@dataclass(frozen=True, slots=True)
class VisualTargetGoalBinding:
    node_id: NodeId
    anchor: AnchorDescriptor
    resource: ResourceDescriptor


class NomadVisualGoalTrajectoryPolicy:
    """Generates raw candidates without selection, scaling, or commands."""

    def __init__(
        self,
        *,
        session: NomadTrajectorySessionPort,
        goal_image_loader: GoalImageLoader,
        inference_executor: Executor,
        config: NomadTrajectoryPolicyConfig,
    ) -> None:
        config.validate()
        self._session = session
        self._goal_image_loader = goal_image_loader
        self._inference_executor = inference_executor
        self._config = config

    def submit_observation(
        self,
        image: object,
        timestamp_s: float,
        *,
        layout: str = "chw",
        channel_order: str = "rgb",
        value_range: str = "auto",
    ) -> None:
        """Appends one decoded frame to the trajectory observation context."""

        self._session.append_observation(
            image,
            timestamp_s,
            layout=layout,
            channel_order=channel_order,
            value_range=value_range,
        )

    def clear_observations(self) -> None:
        """Clears context after a camera discontinuity or runtime restart."""

        self._session.clear_observations()

    async def generate_trajectories(
        self,
        request: VisualGoalTrajectoryRequest,
    ) -> TrajectoryCandidateSet:
        self._validate_request(request)
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                self._inference_executor,
                partial(self._load_and_predict, request),
            )
        except _GoalImageLoadError as error:
            raise TrajectoryPolicyError(
                TrajectoryPolicyErrorCode.GOAL_UNAVAILABLE,
                str(error.__cause__ or error),
                retryable=False,
            ) from error
        except Exception as error:
            raise _translate_session_error(error) from error

        produced_at = self._config.time_source.now()
        self._validate_completion_time(request, produced_at)
        return self._candidate_set(request, result, produced_at)

    def _load_and_predict(
        self,
        request: VisualGoalTrajectoryRequest,
    ) -> _TrajectoryResult:
        try:
            goal = self._goal_image_loader.load(request.goal_resource)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise _GoalImageLoadError from error
        now_s = request.requested_at.nanoseconds / 1_000_000_000.0
        return self._session.predict_goal_trajectories(
            goal.image,
            goal_layout=goal.layout,
            goal_channel_order=goal.channel_order,
            goal_value_range=goal.value_range,
            now_s=now_s,
            max_observation_age_s=request.max_observation_age_s,
            num_candidates=request.num_candidates,
            sampling_seed=request.sampling_seed,
        )

    def _candidate_set(
        self,
        request: VisualGoalTrajectoryRequest,
        result: _TrajectoryResult,
        produced_at: TimePoint,
    ) -> TrajectoryCandidateSet:
        distance = float(result.temporal_distance)
        timestamp_s = float(result.observation_timestamp_s)
        trajectories = result.trajectories
        if not math.isfinite(distance) or not math.isfinite(timestamp_s):
            raise TrajectoryPolicyError(
                TrajectoryPolicyErrorCode.INFERENCE_FAILED,
                "NoMaD trajectory session returned non-finite metadata",
                retryable=False,
            )
        if result.sampling_seed != request.sampling_seed:
            raise TrajectoryPolicyError(
                TrajectoryPolicyErrorCode.INFERENCE_FAILED,
                "NoMaD trajectory result changed the requested seed",
                retryable=False,
            )
        requested_s = request.requested_at.nanoseconds / 1_000_000_000.0
        observation_age_s = requested_s - timestamp_s
        if (
            observation_age_s < -1.0e-6
            or observation_age_s > request.max_observation_age_s + 1.0e-6
        ):
            raise TrajectoryPolicyError(
                TrajectoryPolicyErrorCode.CONTEXT_STALE,
                "NoMaD trajectory result used an observation outside the "
                "request freshness window",
                retryable=True,
            )
        if not isinstance(trajectories, torch.Tensor):
            raise TrajectoryPolicyError(
                TrajectoryPolicyErrorCode.INFERENCE_FAILED,
                "NoMaD trajectories must be a torch.Tensor",
                retryable=False,
            )
        if (
            trajectories.ndim != 3
            or trajectories.shape[0] != request.num_candidates
            or trajectories.shape[1] <= 0
            or trajectories.shape[2] != 2
        ):
            raise TrajectoryPolicyError(
                TrajectoryPolicyErrorCode.INFERENCE_FAILED,
                "NoMaD trajectories have an incompatible shape",
                retryable=False,
            )
        if not bool(torch.isfinite(trajectories).all().item()):
            raise TrajectoryPolicyError(
                TrajectoryPolicyErrorCode.INFERENCE_FAILED,
                "NoMaD trajectories contain non-finite values",
                retryable=False,
            )

        samples = trajectories.detach().to(device="cpu").tolist()
        candidates = tuple(
            TrajectoryCandidate(
                candidate_id=TrajectoryCandidateId(
                    f"{request.segment_id}:sample-{sample_index:04d}"
                ),
                waypoints=tuple(
                    PolicyNativeWaypoint(
                        step_index=step_index,
                        x=float(point[0]),
                        y=float(point[1]),
                    )
                    for step_index, point in enumerate(sample)
                ),
            )
            for sample_index, sample in enumerate(samples)
        )
        return TrajectoryCandidateSet(
            snapshot_id=request.snapshot_id,
            segment_id=request.segment_id,
            source_node_id=request.source_node_id,
            target_node_id=request.target_node_id,
            target_anchor_id=request.target_anchor_id,
            goal_resource_id=request.goal_resource.resource_id,
            observation_time=TimePoint(
                clock_id=self._config.observation_clock_id,
                nanoseconds=round(timestamp_s * 1_000_000_000),
            ),
            produced_at=produced_at,
            temporal_distance=distance,
            coordinate_frame="nomad.policy_native.robot_frame.v1",
            coordinate_units="nomad.policy_native.v1",
            sampling_seed=result.sampling_seed,
            candidates=candidates,
            policy_id=self._config.policy_id,
            image_profile_id=self._config.image_profile_id,
            model_artifact_id=self._config.model_artifact_id,
            model_artifact_digest=_canonical_sha256(
                self._config.model_artifact_digest
            ),
        )

    def _validate_request(
        self,
        request: VisualGoalTrajectoryRequest,
    ) -> None:
        if (
            not math.isfinite(request.max_observation_age_s)
            or request.max_observation_age_s < 0.0
        ):
            raise TrajectoryPolicyError(
                TrajectoryPolicyErrorCode.INVALID_REQUEST,
                "maximum observation age must be finite and non-negative",
                retryable=False,
            )
        if request.num_candidates <= 0:
            raise TrajectoryPolicyError(
                TrajectoryPolicyErrorCode.INVALID_REQUEST,
                "num_candidates must be positive",
                retryable=False,
            )
        if request.sampling_seed is not None and request.sampling_seed < 0:
            raise TrajectoryPolicyError(
                TrajectoryPolicyErrorCode.INVALID_REQUEST,
                "sampling_seed must be non-negative",
                retryable=False,
            )
        if request.source_node_id == request.target_node_id:
            raise TrajectoryPolicyError(
                TrajectoryPolicyErrorCode.INVALID_REQUEST,
                "trajectory route step must connect different nodes",
                retryable=False,
            )
        if request.requested_at.clock_id != self._config.observation_clock_id:
            raise TrajectoryPolicyError(
                TrajectoryPolicyErrorCode.INVALID_REQUEST,
                "request and observation clocks do not match",
                retryable=False,
            )
        expected = (
            request.expected_image_profile_id,
            request.expected_model_artifact_id,
            _canonical_sha256(request.expected_model_artifact_digest),
        )
        configured = (
            self._config.image_profile_id,
            self._config.model_artifact_id,
            _canonical_sha256(self._config.model_artifact_digest),
        )
        if expected != configured:
            raise TrajectoryPolicyError(
                TrajectoryPolicyErrorCode.PROFILE_MISMATCH,
                "route step and NoMaD policy profiles differ",
                retryable=False,
            )
        _validate_resource_profile(request.goal_resource, configured)

    @staticmethod
    def _validate_completion_time(
        request: VisualGoalTrajectoryRequest,
        produced_at: TimePoint,
    ) -> None:
        if produced_at.clock_id != request.requested_at.clock_id:
            raise TrajectoryPolicyError(
                TrajectoryPolicyErrorCode.INVALID_REQUEST,
                "NoMaD policy time source changed clock domain",
                retryable=False,
            )
        if produced_at.nanoseconds < request.requested_at.nanoseconds:
            raise TrajectoryPolicyError(
                TrajectoryPolicyErrorCode.INVALID_REQUEST,
                "NoMaD policy completion predates its request",
                retryable=False,
            )


async def resolve_visual_target_goal(
    *,
    map_engine: MapEngine,
    snapshot: MapSnapshot,
    target_node_id: NodeId,
) -> VisualTargetGoalBinding:
    """Resolves one Map-owned visual TARGET anchor and image resource."""

    anchors = await map_engine.query_anchors(
        snapshot,
        AnchorQuery(
            attached_to=(
                MapEntityRef(
                    kind=MapEntityKind.NODE,
                    entity_id=str(target_node_id),
                ),
            ),
            kinds=frozenset({AnchorKind.VISUAL}),
            purposes=frozenset({AnchorPurpose.TARGET}),
            limit=10,
        ),
    )
    if len(anchors.anchors) != 1:
        raise TrajectoryPolicyError(
            TrajectoryPolicyErrorCode.GOAL_UNAVAILABLE,
            f"node {target_node_id} must have one visual TARGET anchor",
            retryable=False,
        )
    anchor = anchors.anchors[0]
    if anchor.attached_to != MapEntityRef(
        kind=MapEntityKind.NODE,
        entity_id=str(target_node_id),
    ):
        raise TrajectoryPolicyError(
            TrajectoryPolicyErrorCode.GOAL_UNAVAILABLE,
            "Map Engine returned a TARGET anchor for another node",
            retryable=False,
        )
    if len(anchor.resource_ids) != 1:
        raise TrajectoryPolicyError(
            TrajectoryPolicyErrorCode.GOAL_UNAVAILABLE,
            f"TARGET anchor {anchor.anchor_id} must reference one resource",
            retryable=False,
        )
    resources = await map_engine.resolve_resources(
        snapshot,
        anchor.resource_ids,
    )
    if resources.missing_resource_ids or len(resources.resources) != 1:
        raise TrajectoryPolicyError(
            TrajectoryPolicyErrorCode.GOAL_UNAVAILABLE,
            f"TARGET resource for node {target_node_id} is unavailable",
            retryable=False,
        )
    resource = resources.resources[0]
    if resource.kind != ResourceKind.IMAGE or resource.content_digest is None:
        raise TrajectoryPolicyError(
            TrajectoryPolicyErrorCode.GOAL_UNAVAILABLE,
            "visual TARGET resource must be a digest-bound image",
            retryable=False,
        )
    return VisualTargetGoalBinding(
        node_id=target_node_id,
        anchor=anchor,
        resource=resource,
    )


class _GoalImageLoadError(RuntimeError):
    """Separates resource loading failures from inference failures."""


def _translate_session_error(error: Exception) -> TrajectoryPolicyError:
    raw_code = getattr(error, "code", "inference_failed")
    code_value = getattr(raw_code, "value", raw_code)
    codes = {
        "context_not_ready": TrajectoryPolicyErrorCode.CONTEXT_NOT_READY,
        "context_stale": TrajectoryPolicyErrorCode.CONTEXT_STALE,
        "invalid_image": TrajectoryPolicyErrorCode.INVALID_REQUEST,
        "invalid_request": TrajectoryPolicyErrorCode.INVALID_REQUEST,
        "inference_failed": TrajectoryPolicyErrorCode.INFERENCE_FAILED,
    }
    code = codes.get(
        str(code_value),
        TrajectoryPolicyErrorCode.INFERENCE_FAILED,
    )
    return TrajectoryPolicyError(
        code,
        str(error),
        retryable=bool(getattr(error, "retryable", False)),
    )


def _validate_resource_profile(
    resource: ResourceDescriptor,
    configured: tuple[str, str, str],
) -> None:
    actual = (
        resource.attributes.get("image_profile_id"),
        resource.attributes.get("model_artifact_id"),
        resource.attributes.get("model_artifact_digest"),
    )
    if actual != configured:
        raise TrajectoryPolicyError(
            TrajectoryPolicyErrorCode.PROFILE_MISMATCH,
            "goal resource and NoMaD policy profiles differ",
            retryable=False,
        )


def _canonical_sha256(value: str) -> str:
    digest = value.removeprefix("sha256:").casefold()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("model artifact digest must be SHA-256")
    return f"sha256:{digest}"

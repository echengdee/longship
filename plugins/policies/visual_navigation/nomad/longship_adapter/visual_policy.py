"""NoMaD implementation of Longship's visual goal-distance policy SPI."""

from __future__ import annotations

import asyncio
from concurrent.futures import Executor
from dataclasses import dataclass
from functools import partial
import math
from typing import Protocol, runtime_checkable

from longship.navigation.common import TimePoint, TimeSource
from longship.navigation.map_engine.models import ResourceDescriptor
from longship.navigation.localization_engine.visual_policy import (
    VisualGoalCandidate,
    VisualGoalCandidateDistance,
    VisualGoalDistanceBatchMeasurement,
    VisualGoalDistanceBatchRequest,
    VisualGoalDistanceMeasurement,
    VisualGoalDistanceRequest,
    VisualPolicyError,
    VisualPolicyErrorCode,
)

from .image_resource import GoalImageLoader


class _DistanceResult(Protocol):
    temporal_distance: float
    observation_timestamp_s: float


class _DistanceBatchResult(Protocol):
    temporal_distances: tuple[float, ...]
    observation_timestamp_s: float


@runtime_checkable
class NomadDistanceSessionPort(Protocol):
    """Narrow surface implemented by ``nomad_runtime.NomadDistanceSession``."""

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

    def predict_goal_distance(
        self,
        goal_image: object,
        *,
        goal_layout: str = "chw",
        goal_channel_order: str = "rgb",
        goal_value_range: str = "auto",
        now_s: float,
        max_observation_age_s: float,
    ) -> _DistanceResult: ...

    def predict_goal_distances(
        self,
        goal_images: tuple[object, ...],
        *,
        goal_layout: str = "chw",
        goal_channel_order: str = "rgb",
        goal_value_range: str = "auto",
        now_s: float,
        max_observation_age_s: float,
    ) -> _DistanceBatchResult: ...


@dataclass(frozen=True, slots=True)
class NomadVisualPolicyConfig:
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
            raise ValueError("NoMaD visual policy ids must not be empty")
        if not isinstance(self.time_source, TimeSource):
            raise ValueError("NoMaD visual policy needs a time source")
        _canonical_sha256(self.model_artifact_digest)


class NomadVisualGoalDistancePolicy:
    """Maps Longship goals to distance-only NoMaD session inference."""

    def __init__(
        self,
        *,
        session: NomadDistanceSessionPort,
        goal_image_loader: GoalImageLoader,
        inference_executor: Executor,
        config: NomadVisualPolicyConfig,
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
        """Submits one decoded camera image to the NoMaD policy context."""

        self._session.append_observation(
            image,
            timestamp_s,
            layout=layout,
            channel_order=channel_order,
            value_range=value_range,
        )

    def clear_observations(self) -> None:
        """Clears the policy context after camera/profile reconfiguration."""

        self._session.clear_observations()

    async def compare_goal(
        self,
        request: VisualGoalDistanceRequest,
    ) -> VisualGoalDistanceMeasurement:
        batch_request = VisualGoalDistanceBatchRequest(
            snapshot_id=request.snapshot_id,
            candidates=(
                VisualGoalCandidate(
                    target_node_id=request.target_node_id,
                    target_anchor_id=request.target_anchor_id,
                    goal_resource=request.goal_resource,
                ),
            ),
            requested_at=request.requested_at,
            max_observation_age_s=request.max_observation_age_s,
            expected_image_profile_id=request.expected_image_profile_id,
            expected_model_artifact_id=request.expected_model_artifact_id,
            expected_model_artifact_digest=(
                request.expected_model_artifact_digest
            ),
        )
        batch = await self.compare_goals(batch_request)
        candidate = batch.candidate_distances[0]
        return VisualGoalDistanceMeasurement(
            snapshot_id=batch.snapshot_id,
            target_node_id=candidate.target_node_id,
            target_anchor_id=candidate.target_anchor_id,
            goal_resource_id=candidate.goal_resource_id,
            observation_time=batch.observation_time,
            produced_at=batch.produced_at,
            temporal_distance=candidate.temporal_distance,
            policy_id=batch.policy_id,
            image_profile_id=batch.image_profile_id,
            model_artifact_id=batch.model_artifact_id,
            model_artifact_digest=batch.model_artifact_digest,
        )

    async def compare_goals(
        self,
        request: VisualGoalDistanceBatchRequest,
    ) -> VisualGoalDistanceBatchMeasurement:
        """Evaluates local map candidates in one NoMaD encoder batch."""

        self._validate_batch_request(request)
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                self._inference_executor,
                partial(self._load_and_predict_batch, request),
            )
        except _GoalImageLoadError as error:
            raise VisualPolicyError(
                VisualPolicyErrorCode.GOAL_UNAVAILABLE,
                str(error.__cause__ or error),
                retryable=False,
            ) from error
        except Exception as error:
            raise _translate_session_error(error) from error

        produced_at = self._config.time_source.now()
        self._validate_completion_time(request.requested_at, produced_at)
        distances = tuple(float(value) for value in result.temporal_distances)
        observation_timestamp_s = float(result.observation_timestamp_s)
        if len(distances) != len(request.candidates):
            raise VisualPolicyError(
                VisualPolicyErrorCode.INFERENCE_FAILED,
                "NoMaD distance session returned a mismatched batch size",
                retryable=False,
            )
        distances_are_finite = all(
            math.isfinite(value) for value in distances
        )
        if not distances_are_finite or not math.isfinite(
            observation_timestamp_s
        ):
            raise VisualPolicyError(
                VisualPolicyErrorCode.INFERENCE_FAILED,
                "NoMaD distance session returned non-finite output",
                retryable=False,
            )
        return VisualGoalDistanceBatchMeasurement(
            snapshot_id=request.snapshot_id,
            candidate_distances=tuple(
                VisualGoalCandidateDistance(
                    target_node_id=candidate.target_node_id,
                    target_anchor_id=candidate.target_anchor_id,
                    goal_resource_id=candidate.goal_resource.resource_id,
                    temporal_distance=distance,
                )
                for candidate, distance in zip(
                    request.candidates,
                    distances,
                    strict=True,
                )
            ),
            observation_time=TimePoint(
                clock_id=self._config.observation_clock_id,
                nanoseconds=round(observation_timestamp_s * 1_000_000_000),
            ),
            produced_at=produced_at,
            policy_id=self._config.policy_id,
            image_profile_id=self._config.image_profile_id,
            model_artifact_id=self._config.model_artifact_id,
            model_artifact_digest=_canonical_sha256(
                self._config.model_artifact_digest
            ),
        )

    def _load_and_predict_batch(
        self,
        request: VisualGoalDistanceBatchRequest,
    ) -> _DistanceBatchResult:
        try:
            goals = tuple(
                self._goal_image_loader.load(candidate.goal_resource)
                for candidate in request.candidates
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise _GoalImageLoadError from error
        representations = {
            (goal.layout, goal.channel_order, goal.value_range)
            for goal in goals
        }
        if len(representations) != 1:
            raise _GoalImageLoadError(
                "batched goal images use incompatible representations"
            )
        goal_layout, goal_channel_order, goal_value_range = representations.pop()
        now_s = request.requested_at.nanoseconds / 1_000_000_000.0
        return self._session.predict_goal_distances(
            tuple(goal.image for goal in goals),
            goal_layout=goal_layout,
            goal_channel_order=goal_channel_order,
            goal_value_range=goal_value_range,
            now_s=now_s,
            max_observation_age_s=request.max_observation_age_s,
        )

    def _validate_batch_request(
        self,
        request: VisualGoalDistanceBatchRequest,
    ) -> None:
        if not request.candidates:
            raise VisualPolicyError(
                VisualPolicyErrorCode.INVALID_REQUEST,
                "visual goal candidate batch must not be empty",
                retryable=False,
            )
        if (
            not math.isfinite(request.max_observation_age_s)
            or request.max_observation_age_s < 0.0
        ):
            raise VisualPolicyError(
                VisualPolicyErrorCode.INVALID_REQUEST,
                "maximum observation age must be finite and non-negative",
                retryable=False,
            )
        if request.requested_at.clock_id != self._config.observation_clock_id:
            raise VisualPolicyError(
                VisualPolicyErrorCode.INVALID_REQUEST,
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
            raise VisualPolicyError(
                VisualPolicyErrorCode.PROFILE_MISMATCH,
                "map goal and NoMaD policy compatibility profiles differ",
                retryable=False,
            )
        identities = set()
        for candidate in request.candidates:
            identity = (
                candidate.target_node_id,
                candidate.target_anchor_id,
                candidate.goal_resource.resource_id,
            )
            if identity in identities:
                raise VisualPolicyError(
                    VisualPolicyErrorCode.INVALID_REQUEST,
                    "visual goal candidate identities must be unique",
                    retryable=False,
                )
            identities.add(identity)
            _validate_resource_profile(candidate.goal_resource, configured)

    @staticmethod
    def _validate_completion_time(
        requested_at: TimePoint,
        produced_at: TimePoint,
    ) -> None:
        if produced_at.clock_id != requested_at.clock_id:
            raise VisualPolicyError(
                VisualPolicyErrorCode.INVALID_REQUEST,
                "NoMaD policy time source changed clock domain",
                retryable=False,
            )
        if produced_at.nanoseconds < requested_at.nanoseconds:
            raise VisualPolicyError(
                VisualPolicyErrorCode.INVALID_REQUEST,
                "NoMaD policy completion predates its request",
                retryable=False,
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
        raise VisualPolicyError(
            VisualPolicyErrorCode.PROFILE_MISMATCH,
            "goal resource and NoMaD policy compatibility profiles differ",
            retryable=False,
        )


class _GoalImageLoadError(RuntimeError):
    """Separates resource loading failures from inference failures."""


def _translate_session_error(error: Exception) -> VisualPolicyError:
    raw_code = getattr(error, "code", "inference_failed")
    code_value = getattr(raw_code, "value", raw_code)
    codes = {
        "context_not_ready": VisualPolicyErrorCode.CONTEXT_NOT_READY,
        "context_stale": VisualPolicyErrorCode.CONTEXT_STALE,
        "invalid_image": VisualPolicyErrorCode.INVALID_REQUEST,
        "inference_failed": VisualPolicyErrorCode.INFERENCE_FAILED,
    }
    code = codes.get(str(code_value), VisualPolicyErrorCode.INFERENCE_FAILED)
    retryable = bool(getattr(error, "retryable", False))
    return VisualPolicyError(
        code,
        str(error),
        retryable=retryable,
    )


def _canonical_sha256(value: str) -> str:
    digest = value.removeprefix("sha256:").casefold()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("model artifact digest must be SHA-256")
    return f"sha256:{digest}"

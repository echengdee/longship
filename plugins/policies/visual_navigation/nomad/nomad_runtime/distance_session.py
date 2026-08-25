"""Distance-only NoMaD policy session for localization adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import math

import torch

from nomad_runtime.image_input import (
    ImageTensorSpec,
    ObservationBuffer,
    ObservationContextNotReadyError,
    StaleObservationContextError,
    canonicalize_image,
)
from nomad_runtime.policy import NomadPolicy


class NomadDistanceErrorCode(str, Enum):
    """Stable failures exposed to an external localization adapter."""

    CONTEXT_NOT_READY = "context_not_ready"
    CONTEXT_STALE = "context_stale"
    INVALID_IMAGE = "invalid_image"
    INFERENCE_FAILED = "inference_failed"


class NomadDistanceSessionError(RuntimeError):
    """Structured failure from a distance-only policy session."""

    def __init__(
        self,
        code: NomadDistanceErrorCode,
        message: str,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class NomadDistanceResult:
    """One scalar distance tied to the latest observation timestamp."""

    temporal_distance: float
    observation_timestamp_s: float


@dataclass(frozen=True)
class NomadDistanceBatchResult:
    """Temporal distances evaluated against one observation context."""

    temporal_distances: tuple[float, ...]
    observation_timestamp_s: float


class NomadDistanceSession:
    """Owns NoMaD observation context and distance-head inference only."""

    def __init__(
        self,
        policy: NomadPolicy,
        observation_buffer: ObservationBuffer | None = None,
    ) -> None:
        self._policy = policy
        self._observations = observation_buffer or ObservationBuffer(
            context_frames=policy.config.observation_frames,
            history_frames=policy.config.observation_frames * 2,
        )
        if (
            self._observations.context_frames
            != policy.config.observation_frames
        ):
            raise ValueError(
                "observation buffer size must match the NoMaD model context"
            )

    @property
    def ready(self) -> bool:
        """Whether a complete chronological observation context exists."""

        return self._observations.ready

    @property
    def latest_observation_timestamp_s(self) -> float | None:
        return self._observations.latest_timestamp_s

    def clear_observations(self) -> None:
        """Clears context after a camera restart or profile change."""

        self._observations.clear()

    def append_observation(
        self,
        image: torch.Tensor,
        timestamp_s: int | float,
        *,
        layout: str = "chw",
        channel_order: str = "rgb",
        value_range: str = "auto",
    ) -> None:
        """Canonicalizes and appends one decoded camera frame."""

        try:
            self._observations.append(
                image,
                timestamp_s=timestamp_s,
                spec=ImageTensorSpec(
                    layout=layout,
                    channel_order=channel_order,
                    value_range=value_range,
                ),
            )
        except (TypeError, ValueError) as error:
            raise NomadDistanceSessionError(
                NomadDistanceErrorCode.INVALID_IMAGE,
                str(error),
                retryable=False,
            ) from error

    def predict_goal_distance(
        self,
        goal_image: torch.Tensor,
        *,
        goal_layout: str = "chw",
        goal_channel_order: str = "rgb",
        goal_value_range: str = "auto",
        now_s: int | float,
        max_observation_age_s: int | float,
    ) -> NomadDistanceResult:
        """Runs the distance head without sampling diffusion trajectories."""

        result = self.predict_goal_distances(
            (goal_image,),
            goal_layout=goal_layout,
            goal_channel_order=goal_channel_order,
            goal_value_range=goal_value_range,
            now_s=now_s,
            max_observation_age_s=max_observation_age_s,
        )
        return NomadDistanceResult(
            temporal_distance=result.temporal_distances[0],
            observation_timestamp_s=result.observation_timestamp_s,
        )

    def predict_goal_distances(
        self,
        goal_images: Sequence[torch.Tensor],
        *,
        goal_layout: str = "chw",
        goal_channel_order: str = "rgb",
        goal_value_range: str = "auto",
        now_s: int | float,
        max_observation_age_s: int | float,
    ) -> NomadDistanceBatchResult:
        """Runs one batched distance-head inference for local map goals."""

        if not goal_images:
            raise NomadDistanceSessionError(
                NomadDistanceErrorCode.INVALID_IMAGE,
                "goal_images must not be empty",
                retryable=False,
            )
        try:
            context = self._observations.snapshot(
                now_s=now_s,
                max_age_s=max_observation_age_s,
            )
        except StaleObservationContextError as error:
            raise NomadDistanceSessionError(
                NomadDistanceErrorCode.CONTEXT_STALE,
                str(error),
                retryable=True,
            ) from error
        except ObservationContextNotReadyError as error:
            raise NomadDistanceSessionError(
                NomadDistanceErrorCode.CONTEXT_NOT_READY,
                str(error),
                retryable=True,
            ) from error
        except ValueError as error:
            raise NomadDistanceSessionError(
                NomadDistanceErrorCode.INVALID_IMAGE,
                str(error),
                retryable=False,
            ) from error

        try:
            goal_spec = ImageTensorSpec(
                layout=goal_layout,
                channel_order=goal_channel_order,
                value_range=goal_value_range,
            )
            goals = tuple(
                canonicalize_image(goal_image, goal_spec)
                for goal_image in goal_images
            )
            goal_shape = goals[0].shape
            if any(goal.shape != goal_shape for goal in goals[1:]):
                raise ValueError(
                    "batched goal images must have one canonical shape"
                )
            goal_batch = torch.stack(goals)
            observation_batch = context.images.unsqueeze(0).repeat(
                len(goals),
                1,
                1,
                1,
                1,
            )
            condition = self._policy.encode_condition(
                observation_batch,
                goal_batch,
            )
            distance_tensor = self._policy.predict_distance(condition)
            if distance_tensor.numel() != len(goals):
                raise RuntimeError(
                    "distance session returned a mismatched batch size"
                )
            distances = tuple(
                float(value)
                for value in distance_tensor.reshape(-1).tolist()
            )
            if not all(math.isfinite(distance) for distance in distances):
                raise RuntimeError("NoMaD returned a non-finite distance")
        except NomadDistanceSessionError:
            raise
        except (TypeError, ValueError) as error:
            raise NomadDistanceSessionError(
                NomadDistanceErrorCode.INVALID_IMAGE,
                str(error),
                retryable=False,
            ) from error
        except Exception as error:
            raise NomadDistanceSessionError(
                NomadDistanceErrorCode.INFERENCE_FAILED,
                str(error),
                retryable=False,
            ) from error

        return NomadDistanceBatchResult(
            temporal_distances=distances,
            observation_timestamp_s=context.latest_timestamp_s,
        )

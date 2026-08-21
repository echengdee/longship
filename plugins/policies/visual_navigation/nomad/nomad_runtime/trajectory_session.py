"""Goal-conditioned NoMaD trajectory inference over a frame context."""

from __future__ import annotations

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


class NomadTrajectoryErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    CONTEXT_NOT_READY = "context_not_ready"
    CONTEXT_STALE = "context_stale"
    INVALID_IMAGE = "invalid_image"
    INFERENCE_FAILED = "inference_failed"


class NomadTrajectorySessionError(RuntimeError):
    """Structured failure from raw NoMaD trajectory inference."""

    def __init__(
        self,
        code: NomadTrajectoryErrorCode,
        message: str,
        *,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class NomadTrajectoryResult:
    temporal_distance: float
    trajectories: torch.Tensor
    observation_timestamp_s: float
    sampling_seed: int | None


class NomadTrajectorySession:
    """Owns observation context and returns unselected policy-native samples."""

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
        return self._observations.ready

    def clear_observations(self) -> None:
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
            raise NomadTrajectorySessionError(
                NomadTrajectoryErrorCode.INVALID_IMAGE,
                str(error),
                retryable=False,
            ) from error

    def predict_goal_trajectories(
        self,
        goal_image: torch.Tensor,
        *,
        goal_layout: str = "chw",
        goal_channel_order: str = "rgb",
        goal_value_range: str = "auto",
        now_s: int | float,
        max_observation_age_s: int | float,
        num_candidates: int,
        sampling_seed: int | None,
    ) -> NomadTrajectoryResult:
        """Returns distance plus every raw diffusion trajectory sample."""

        if num_candidates <= 0:
            raise NomadTrajectorySessionError(
                NomadTrajectoryErrorCode.INVALID_REQUEST,
                "num_candidates must be positive",
                retryable=False,
            )
        if sampling_seed is not None and sampling_seed < 0:
            raise NomadTrajectorySessionError(
                NomadTrajectoryErrorCode.INVALID_REQUEST,
                "sampling_seed must be non-negative",
                retryable=False,
            )
        try:
            context = self._observations.snapshot(
                now_s=now_s,
                max_age_s=max_observation_age_s,
            )
        except StaleObservationContextError as error:
            raise NomadTrajectorySessionError(
                NomadTrajectoryErrorCode.CONTEXT_STALE,
                str(error),
                retryable=True,
            ) from error
        except ObservationContextNotReadyError as error:
            raise NomadTrajectorySessionError(
                NomadTrajectoryErrorCode.CONTEXT_NOT_READY,
                str(error),
                retryable=True,
            ) from error
        except ValueError as error:
            raise NomadTrajectorySessionError(
                NomadTrajectoryErrorCode.INVALID_IMAGE,
                str(error),
                retryable=False,
            ) from error

        try:
            goal = canonicalize_image(
                goal_image,
                ImageTensorSpec(
                    layout=goal_layout,
                    channel_order=goal_channel_order,
                    value_range=goal_value_range,
                ),
            )
            condition = self._policy.encode_condition(
                context.images,
                goal,
            )
            distance_tensor = self._policy.predict_distance(condition)
            if distance_tensor.numel() != 1:
                raise RuntimeError(
                    "trajectory session requires one observation-goal pair"
                )
            temporal_distance = float(distance_tensor.reshape(-1)[0].item())
            generator = _generator_for(self._policy, sampling_seed)
            trajectories = self._policy.sample_actions(
                condition,
                num_samples=num_candidates,
                generator=generator,
            )
            if trajectories.shape[0] != 1:
                raise RuntimeError(
                    "trajectory session requires a single output batch"
                )
            trajectories = trajectories[0].detach()
            if not math.isfinite(temporal_distance):
                raise RuntimeError("NoMaD returned a non-finite distance")
            if not bool(torch.isfinite(trajectories).all().item()):
                raise RuntimeError("NoMaD returned non-finite trajectories")
        except NomadTrajectorySessionError:
            raise
        except Exception as error:
            raise NomadTrajectorySessionError(
                NomadTrajectoryErrorCode.INFERENCE_FAILED,
                str(error),
                retryable=False,
            ) from error

        return NomadTrajectoryResult(
            temporal_distance=temporal_distance,
            trajectories=trajectories,
            observation_timestamp_s=context.latest_timestamp_s,
            sampling_seed=sampling_seed,
        )


def _generator_for(
    policy: NomadPolicy,
    sampling_seed: int | None,
) -> torch.Generator | None:
    if sampling_seed is None:
        return None
    generator = torch.Generator(device=policy.device)
    generator.manual_seed(sampling_seed)
    return generator

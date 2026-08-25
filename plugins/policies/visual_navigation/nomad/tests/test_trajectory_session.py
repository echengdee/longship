"""Tests for goal-conditioned NoMaD trajectory sessions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nomad_runtime import (
    NomadTrajectoryErrorCode,
    NomadTrajectorySession,
    NomadTrajectorySessionError,
)


class _FakePolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(observation_frames=4)
        self.device = torch.device("cpu")
        self.sample_calls: list[tuple[int, torch.Generator | None]] = []

    def encode_condition(
        self,
        observations: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        assert observations.shape == (4, 3, 6, 8)
        assert goal.shape == (3, 6, 8)
        return torch.ones((1, 4), dtype=torch.float32)

    def predict_distance(self, condition: torch.Tensor) -> torch.Tensor:
        assert condition.shape == (1, 4)
        return torch.tensor([4.25])

    def sample_actions(
        self,
        condition: torch.Tensor,
        *,
        num_samples: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        self.sample_calls.append((num_samples, generator))
        values = torch.arange(num_samples * 8 * 2, dtype=torch.float32)
        return values.view(1, num_samples, 8, 2)


def _image(value: int) -> torch.Tensor:
    return torch.full((6, 8, 3), value, dtype=torch.uint8)


def test_returns_every_seeded_policy_native_candidate() -> None:
    policy = _FakePolicy()
    session = NomadTrajectorySession(policy)  # type: ignore[arg-type]
    for index in range(4):
        session.append_observation(
            _image(index),
            index + 1.0,
            layout="hwc",
            value_range="byte",
        )

    result = session.predict_goal_trajectories(
        _image(9),
        goal_layout="hwc",
        goal_value_range="byte",
        now_s=4.1,
        max_observation_age_s=0.2,
        num_candidates=4,
        sampling_seed=17,
    )

    assert result.temporal_distance == pytest.approx(4.25)
    assert result.trajectories.shape == (4, 8, 2)
    assert result.observation_timestamp_s == 4.0
    assert result.sampling_seed == 17
    assert policy.sample_calls[0][0] == 4
    assert policy.sample_calls[0][1] is not None
    assert policy.sample_calls[0][1].initial_seed() == 17


def test_reports_context_not_ready_before_four_frames() -> None:
    session = NomadTrajectorySession(_FakePolicy())  # type: ignore[arg-type]
    session.append_observation(
        _image(1),
        1.0,
        layout="hwc",
        value_range="byte",
    )

    with pytest.raises(NomadTrajectorySessionError) as captured:
        session.predict_goal_trajectories(
            _image(9),
            goal_layout="hwc",
            goal_value_range="byte",
            now_s=1.0,
            max_observation_age_s=0.2,
            num_candidates=4,
            sampling_seed=17,
        )

    assert captured.value.code == NomadTrajectoryErrorCode.CONTEXT_NOT_READY
    assert captured.value.retryable

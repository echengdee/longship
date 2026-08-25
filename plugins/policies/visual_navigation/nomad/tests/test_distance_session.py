"""Tests for distance-only NoMaD localization inference."""

from types import SimpleNamespace

import pytest
import torch

from nomad_runtime import (
    NomadDistanceErrorCode,
    NomadDistanceSession,
    NomadDistanceSessionError,
)


class _FakePolicy:
    def __init__(self, distance: float = 2.5) -> None:
        self.config = SimpleNamespace(observation_frames=4)
        self.distance = distance
        self.encode_calls = 0
        self.distance_calls = 0

    def encode_condition(
        self,
        observations: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        self.encode_calls += 1
        if observations.ndim == 4:
            assert observations.shape == (4, 3, 2, 3)
            assert goal.shape == (3, 2, 3)
            batch_size = 1
        else:
            assert observations.shape[1:] == (4, 3, 2, 3)
            assert goal.shape == (observations.shape[0], 3, 2, 3)
            batch_size = observations.shape[0]
        return torch.ones((batch_size, 8))

    def predict_distance(self, condition: torch.Tensor) -> torch.Tensor:
        self.distance_calls += 1
        assert condition.shape[1:] == (8,)
        return torch.full((condition.shape[0],), self.distance)


def test_predicts_distance_without_sampling_actions() -> None:
    policy = _FakePolicy()
    session = NomadDistanceSession(policy)  # type: ignore[arg-type]
    for index in range(4):
        session.append_observation(
            torch.full((2, 3, 3), index, dtype=torch.uint8),
            timestamp_s=1.0 + index,
            layout="hwc",
        )

    result = session.predict_goal_distance(
        torch.zeros((2, 3, 3), dtype=torch.uint8),
        goal_layout="hwc",
        now_s=4.1,
        max_observation_age_s=0.2,
    )

    assert result.temporal_distance == pytest.approx(2.5)
    assert result.observation_timestamp_s == 4.0
    assert policy.encode_calls == 1
    assert policy.distance_calls == 1


def test_reports_context_not_ready_and_stale_separately() -> None:
    session = NomadDistanceSession(_FakePolicy())  # type: ignore[arg-type]
    goal = torch.zeros((3, 2, 3))

    with pytest.raises(NomadDistanceSessionError) as not_ready:
        session.predict_goal_distance(
            goal,
            now_s=1.0,
            max_observation_age_s=0.2,
        )
    assert not_ready.value.code == NomadDistanceErrorCode.CONTEXT_NOT_READY
    assert not_ready.value.retryable

    for index in range(4):
        session.append_observation(
            torch.zeros((3, 2, 3)),
            timestamp_s=1.0 + index,
        )
    with pytest.raises(NomadDistanceSessionError) as stale:
        session.predict_goal_distance(
            goal,
            now_s=5.0,
            max_observation_age_s=0.2,
        )
    assert stale.value.code == NomadDistanceErrorCode.CONTEXT_STALE
    assert stale.value.retryable


def test_predicts_multiple_goals_from_one_context_batch() -> None:
    policy = _FakePolicy(distance=4.5)
    session = NomadDistanceSession(policy)  # type: ignore[arg-type]
    for index in range(4):
        session.append_observation(
            torch.full((2, 3, 3), index, dtype=torch.uint8),
            timestamp_s=1.0 + index,
            layout="hwc",
        )

    result = session.predict_goal_distances(
        (
            torch.zeros((2, 3, 3), dtype=torch.uint8),
            torch.ones((2, 3, 3), dtype=torch.uint8),
            torch.full((2, 3, 3), 2, dtype=torch.uint8),
        ),
        goal_layout="hwc",
        now_s=4.1,
        max_observation_age_s=0.2,
    )

    assert result.temporal_distances == pytest.approx((4.5, 4.5, 4.5))
    assert result.observation_timestamp_s == 4.0
    assert policy.encode_calls == 1
    assert policy.distance_calls == 1

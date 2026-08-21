"""Tests for tensor-only image preprocessing."""

import pytest
import torch

from nomad_runtime import NomadConfig
from nomad_runtime.preprocessing import TensorPreprocessor


def test_prepares_observation_and_goal_shapes() -> None:
    preprocessor = TensorPreprocessor(NomadConfig())
    observations = torch.zeros((2, 4, 3, 48, 64))
    goal = torch.ones((2, 3, 48, 64))

    prepared_observations, prepared_goal = preprocessor.prepare(
        observations, goal
    )

    assert prepared_observations.shape == (2, 12, 96, 96)
    assert prepared_goal.shape == (2, 3, 96, 96)
    expected_red = -0.485 / 0.229
    assert prepared_observations[0, 0, 0, 0].item() == pytest.approx(
        expected_red
    )


def test_rejects_wrong_context_length() -> None:
    preprocessor = TensorPreprocessor(NomadConfig())

    with pytest.raises(ValueError, match="4 RGB frames"):
        preprocessor.prepare_observations(torch.zeros((3, 3, 96, 96)))


def test_rejects_integer_images() -> None:
    preprocessor = TensorPreprocessor(NomadConfig())

    with pytest.raises(TypeError, match="floating-point"):
        preprocessor.prepare_goal(
            torch.zeros((3, 96, 96), dtype=torch.uint8)
        )


def test_center_crops_to_four_by_three_before_resize() -> None:
    config = NomadConfig(center_crop_aspect=4.0 / 3.0)
    preprocessor = TensorPreprocessor(config)
    goal = torch.zeros((3, 6, 12))
    goal[:, :, :2] = 1.0
    goal[:, :, 10:] = 1.0

    prepared = preprocessor.prepare_goal(goal)

    expected_red = -0.485 / 0.229
    assert prepared.shape == (1, 3, 96, 96)
    assert prepared[0, 0].amin().item() == pytest.approx(expected_red)
    assert prepared[0, 0].amax().item() == pytest.approx(expected_red)

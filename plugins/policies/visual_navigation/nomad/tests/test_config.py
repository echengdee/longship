"""Tests for NoMaD runtime configuration."""

from dataclasses import replace

import pytest

from nomad_runtime import NomadConfig


def test_default_configuration_matches_released_checkpoint() -> None:
    config = NomadConfig()

    config.validate()

    assert config.observation_frames == 4
    assert config.image_size == (96, 96)
    assert config.encoding_size == 256
    assert config.diffusion_iterations == 10
    assert config.trajectory_length == 8


def test_configuration_rejects_invalid_attention_shape() -> None:
    config = replace(NomadConfig(), encoding_size=255)

    with pytest.raises(ValueError, match="divisible"):
        config.validate()


def test_configuration_rejects_invalid_center_crop_aspect() -> None:
    config = replace(NomadConfig(), center_crop_aspect=0.0)

    with pytest.raises(ValueError, match="center_crop_aspect"):
        config.validate()

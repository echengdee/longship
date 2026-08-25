"""Optional integration test using the released NoMaD checkpoint."""

import os
from pathlib import Path

import pytest
import torch

from nomad_runtime import NomadConfig, NomadPolicy


_CHECKPOINT = os.environ.get("NOMAD_CHECKPOINT")


@pytest.mark.skipif(not _CHECKPOINT, reason="NOMAD_CHECKPOINT is not set")
def test_released_checkpoint_loads_and_infers() -> None:
    checkpoint = Path(_CHECKPOINT)
    config = NomadConfig()
    policy = NomadPolicy.from_checkpoint(
        checkpoint, device="cpu", strict=True
    )
    generator = torch.Generator().manual_seed(5)
    observations = torch.rand(
        (1, config.observation_frames, 3, 96, 96), generator=generator
    )
    goal = torch.rand((1, 3, 96, 96), generator=generator)

    output = policy.infer(
        observations, goal, num_samples=2, generator=generator
    )

    assert sum(
        parameter.numel() for parameter in policy.model.parameters()
    ) == 19_049_675
    assert output.distance.shape == (1,)
    assert output.actions.shape == (1, 2, 8, 2)
    assert torch.isfinite(output.actions).all()

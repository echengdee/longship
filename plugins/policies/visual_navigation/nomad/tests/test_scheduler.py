"""Tests for the embedded PyTorch DDPM scheduler."""

import pytest
import torch

from nomad_runtime.scheduler import DdpScheduler


def test_cosine_schedule_matches_released_scheduler() -> None:
    scheduler = DdpScheduler(10)

    assert scheduler.betas[0].item() == pytest.approx(0.02790726)
    assert scheduler.betas[-1].item() == pytest.approx(0.999)
    assert list(scheduler.timesteps) == list(range(9, -1, -1))


def test_final_step_is_deterministic_and_finite() -> None:
    scheduler = DdpScheduler(10)
    sample = torch.tensor([[[0.25, -0.50]]])
    predicted_noise = torch.tensor([[[0.10, -0.20]]])

    result = scheduler.step(predicted_noise, 0, sample)

    alpha_product = scheduler.alphas_cumprod[0]
    expected = (
        sample - (1 - alpha_product).sqrt() * predicted_noise
    ) / alpha_product.sqrt()
    assert torch.allclose(result, expected.clamp(-1, 1))
    assert torch.isfinite(result).all()

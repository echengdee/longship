"""Small PyTorch-only DDPM scheduler matching diffusers 0.11 defaults."""

from __future__ import annotations

import math

import torch


def _squared_cosine_betas(
    diffusion_steps: int, max_beta: float = 0.999
) -> torch.Tensor:
    """Builds the cosine beta schedule used by the released NoMaD config."""

    def alpha_bar(time_step: float) -> float:
        return math.cos(
            (time_step + 0.008) / 1.008 * math.pi / 2
        ) ** 2

    betas = []
    for index in range(diffusion_steps):
        first_time = index / diffusion_steps
        second_time = (index + 1) / diffusion_steps
        beta = 1 - alpha_bar(second_time) / alpha_bar(first_time)
        betas.append(min(beta, max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class DdpScheduler:
    """Performs epsilon-prediction DDPM reverse diffusion steps."""

    def __init__(self, diffusion_steps: int) -> None:
        if diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive")
        self.diffusion_steps = diffusion_steps
        self.betas = _squared_cosine_betas(diffusion_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    @property
    def timesteps(self) -> range:
        """Returns all reverse timesteps from noisy to denoised."""
        return range(self.diffusion_steps - 1, -1, -1)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Computes the previous noisy sample for one reverse DDPM step."""
        if timestep < 0 or timestep >= self.diffusion_steps:
            raise ValueError("timestep is outside the configured range")
        device = sample.device
        dtype = sample.dtype
        betas = self.betas.to(device=device, dtype=dtype)
        alphas = self.alphas.to(device=device, dtype=dtype)
        cumulative = self.alphas_cumprod.to(device=device, dtype=dtype)
        one = torch.ones((), device=device, dtype=dtype)

        alpha_product = cumulative[timestep]
        previous_alpha_product = (
            cumulative[timestep - 1] if timestep > 0 else one
        )
        beta_product = 1 - alpha_product
        previous_beta_product = 1 - previous_alpha_product

        predicted_original = (
            sample - beta_product.sqrt() * model_output
        ) / alpha_product.sqrt()
        predicted_original = predicted_original.clamp(-1, 1)

        original_coefficient = (
            previous_alpha_product.sqrt() * betas[timestep]
        ) / beta_product
        sample_coefficient = (
            alphas[timestep].sqrt() * previous_beta_product
        ) / beta_product
        previous_sample = (
            original_coefficient * predicted_original
            + sample_coefficient * sample
        )

        if timestep > 0:
            variance = (
                previous_beta_product / beta_product * betas[timestep]
            ).clamp(min=1e-20)
            noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            previous_sample = previous_sample + variance.sqrt() * noise
        return previous_sample

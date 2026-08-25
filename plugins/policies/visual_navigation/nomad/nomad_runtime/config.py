"""Configuration for the released NoMaD checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NomadConfig:
    """Architecture and action normalization constants for NoMaD."""

    context_size: int = 3
    image_size: tuple[int, int] = (96, 96)
    encoding_size: int = 256
    attention_heads: int = 4
    attention_layers: int = 4
    feed_forward_factor: int = 4
    diffusion_down_dims: tuple[int, ...] = (64, 128, 256)
    condition_predicts_scale: bool = False
    diffusion_iterations: int = 10
    trajectory_length: int = 8
    action_min: tuple[float, float] = (-2.5, -4.0)
    action_max: tuple[float, float] = (5.0, 4.0)
    center_crop_aspect: float | None = None

    @property
    def observation_frames(self) -> int:
        """Returns the number of images in one observation context."""
        return self.context_size + 1

    def validate(self) -> None:
        """Raises ValueError when a configuration is internally invalid."""
        if self.context_size < 0:
            raise ValueError("context_size must be non-negative")
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must contain two positive dimensions")
        if self.encoding_size <= 0:
            raise ValueError("encoding_size must be positive")
        if self.encoding_size % self.attention_heads != 0:
            raise ValueError("encoding_size must be divisible by attention_heads")
        if self.diffusion_iterations <= 0:
            raise ValueError("diffusion_iterations must be positive")
        if self.trajectory_length <= 0:
            raise ValueError("trajectory_length must be positive")
        if len(self.action_min) != 2 or len(self.action_max) != 2:
            raise ValueError("NoMaD actions must have exactly two dimensions")
        for minimum, maximum in zip(self.action_min, self.action_max):
            if minimum >= maximum:
                raise ValueError("Each action minimum must be below its maximum")
        if self.center_crop_aspect is not None and (
            not math.isfinite(self.center_crop_aspect)
            or self.center_crop_aspect <= 0.0
        ):
            raise ValueError("center_crop_aspect must be finite and positive")

"""Tensor preprocessing for NoMaD inference."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from nomad_runtime.config import NomadConfig


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class TensorPreprocessor:
    """Converts float RGB tensors into NoMaD model inputs."""

    def __init__(self, config: NomadConfig) -> None:
        self._config = config

    def _resize_and_normalize(self, images: torch.Tensor) -> torch.Tensor:
        if not images.is_floating_point():
            raise TypeError("Image tensors must use a floating-point dtype")
        images = self._center_crop(images)
        target_width, target_height = self._config.image_size
        images = F.interpolate(
            images,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        mean = images.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        return (images - mean) / std

    def _center_crop(self, images: torch.Tensor) -> torch.Tensor:
        aspect = self._config.center_crop_aspect
        if aspect is None:
            return images
        height, width = images.shape[-2:]
        if width / height > aspect:
            crop_width = min(width, max(1, int(height * aspect)))
            left = (width - crop_width) // 2
            return images[..., :, left : left + crop_width]
        crop_height = min(height, max(1, int(width / aspect)))
        top = (height - crop_height) // 2
        return images[..., top : top + crop_height, :]

    def prepare_observations(
        self, observations: torch.Tensor
    ) -> torch.Tensor:
        """Prepares [B,T,3,H,W] RGB values in [0,1] for the encoder."""
        if observations.ndim == 4:
            observations = observations.unsqueeze(0)
        if observations.ndim != 5:
            raise ValueError(
                "observations must have shape [T,3,H,W] or [B,T,3,H,W]"
            )
        batch_size, frames, channels, height, width = observations.shape
        if frames != self._config.observation_frames or channels != 3:
            raise ValueError(
                "observations must contain "
                f"{self._config.observation_frames} RGB frames"
            )
        flattened = observations.reshape(
            batch_size * frames, channels, height, width
        )
        normalized = self._resize_and_normalize(flattened)
        return normalized.reshape(
            batch_size,
            frames * channels,
            normalized.shape[-2],
            normalized.shape[-1],
        )

    def prepare_goal(self, goal: torch.Tensor) -> torch.Tensor:
        """Prepares [B,3,H,W] RGB values in [0,1] for the encoder."""
        if goal.ndim == 3:
            goal = goal.unsqueeze(0)
        if goal.ndim != 4 or goal.shape[1] != 3:
            raise ValueError(
                "goal must have shape [3,H,W] or [B,3,H,W]"
            )
        return self._resize_and_normalize(goal)

    def prepare(
        self, observations: torch.Tensor, goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepares observation context and goal and checks batch sizes."""
        observation_tensor = self.prepare_observations(observations)
        goal_tensor = self.prepare_goal(goal)
        if observation_tensor.shape[0] != goal_tensor.shape[0]:
            raise ValueError("Observation and goal batch sizes must match")
        return observation_tensor, goal_tensor

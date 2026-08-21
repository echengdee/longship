"""High-level model loading and inference interface for NoMaD."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from nomad_runtime.config import NomadConfig
from nomad_runtime.model import NoMaD, build_nomad_model
from nomad_runtime.preprocessing import TensorPreprocessor
from nomad_runtime.scheduler import DdpScheduler


@dataclass(frozen=True)
class NomadOutput:
    """Goal distance and sampled robot-frame waypoint trajectories."""

    distance: torch.Tensor
    actions: torch.Tensor


@dataclass(frozen=True)
class CheckpointLoadResult:
    """Non-strict checkpoint compatibility details."""

    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


class NomadPolicy(nn.Module):
    """Owns a NoMaD model and exposes a tensor-only inference API."""

    def __init__(
        self,
        config: NomadConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config or NomadConfig()
        self.config.validate()
        self.model: NoMaD = build_nomad_model(self.config)
        self.preprocessor = TensorPreprocessor(self.config)
        self.scheduler = DdpScheduler(self.config.diffusion_iterations)
        selected_device = torch.device(device or "cpu")
        self.to(selected_device)
        self.eval()

    @property
    def device(self) -> torch.device:
        """Returns the device holding the model parameters."""
        return next(self.model.parameters()).device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: NomadConfig | None = None,
        device: str | torch.device | None = None,
        strict: bool = True,
    ) -> NomadPolicy:
        """Constructs a policy and loads a raw or wrapped state dictionary."""
        policy = cls(config=config, device=device)
        policy.load_checkpoint(checkpoint_path, strict=strict)
        return policy

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        strict: bool = True,
    ) -> CheckpointLoadResult:
        """Loads model weights without changing the selected runtime device."""
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"NoMaD checkpoint not found: {path}")
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )
        state_dict = self._extract_state_dict(checkpoint)
        result = self.model.load_state_dict(state_dict, strict=strict)
        self.model.eval()
        return CheckpointLoadResult(
            missing_keys=tuple(result.missing_keys),
            unexpected_keys=tuple(result.unexpected_keys),
        )

    @staticmethod
    def _extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Checkpoint must contain a PyTorch state dictionary")
        for key in ("state_dict", "model_state_dict"):
            nested = checkpoint.get(key)
            if isinstance(nested, Mapping):
                return nested
        if checkpoint and all(
            isinstance(value, torch.Tensor) for value in checkpoint.values()
        ):
            return checkpoint
        raise ValueError("Checkpoint does not contain a supported state dictionary")

    def _goal_mask(
        self,
        batch_size: int,
        goal_mask: bool | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(goal_mask, bool):
            value = 1 if goal_mask else 0
            return torch.full(
                (batch_size,), value, dtype=torch.long, device=self.device
            )
        if goal_mask.ndim != 1 or goal_mask.shape[0] != batch_size:
            raise ValueError("goal_mask must have shape [batch]")
        mask = goal_mask.to(device=self.device, dtype=torch.long)
        if torch.any((mask != 0) & (mask != 1)):
            raise ValueError("goal_mask values must be zero or one")
        return mask

    @torch.no_grad()
    def encode_condition(
        self,
        observations: torch.Tensor,
        goal: torch.Tensor,
        goal_mask: bool | torch.Tensor = False,
    ) -> torch.Tensor:
        """Encodes unnormalized RGB tensors into the shared condition."""
        observations = observations.to(self.device)
        goal = goal.to(self.device)
        observation_tensor, goal_tensor = self.preprocessor.prepare(
            observations, goal
        )
        mask = self._goal_mask(observation_tensor.shape[0], goal_mask)
        return self.model.vision_encoder(
            observation_tensor,
            goal_tensor,
            input_goal_mask=mask,
        )

    @torch.no_grad()
    def predict_distance(self, condition: torch.Tensor) -> torch.Tensor:
        """Predicts temporal goal distance for each condition in a batch."""
        condition = condition.to(self.device)
        return self.model.dist_pred_net(condition).reshape(-1)

    @torch.no_grad()
    def sample_actions(
        self,
        condition: torch.Tensor,
        num_samples: int = 4,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Returns actions with shape [batch,samples,horizon,2]."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        condition = condition.to(self.device)
        batch_size = condition.shape[0]
        repeated_condition = condition.repeat_interleave(
            num_samples, dim=0
        )
        actions = torch.randn(
            (
                batch_size * num_samples,
                self.config.trajectory_length,
                2,
            ),
            device=self.device,
            dtype=condition.dtype,
            generator=generator,
        )
        for timestep in self.scheduler.timesteps:
            timestep_tensor = torch.tensor(
                timestep, dtype=torch.long, device=self.device
            )
            predicted_noise = self.model.noise_pred_net(
                sample=actions,
                timestep=timestep_tensor,
                global_cond=repeated_condition,
            )
            actions = self.scheduler.step(
                predicted_noise,
                timestep,
                actions,
                generator=generator,
            )

        action_min = actions.new_tensor(self.config.action_min)
        action_max = actions.new_tensor(self.config.action_max)
        deltas = (actions + 1.0) / 2.0
        deltas = deltas * (action_max - action_min) + action_min
        trajectories = torch.cumsum(deltas, dim=1)
        return trajectories.reshape(
            batch_size,
            num_samples,
            self.config.trajectory_length,
            2,
        )

    @torch.no_grad()
    def infer(
        self,
        observations: torch.Tensor,
        goal: torch.Tensor,
        num_samples: int = 4,
        goal_mask: bool | torch.Tensor = False,
        generator: torch.Generator | None = None,
    ) -> NomadOutput:
        """Runs goal-conditioned distance and diffusion action inference."""
        condition = self.encode_condition(
            observations, goal, goal_mask=goal_mask
        )
        return NomadOutput(
            distance=self.predict_distance(condition),
            actions=self.sample_actions(
                condition, num_samples=num_samples, generator=generator
            ),
        )

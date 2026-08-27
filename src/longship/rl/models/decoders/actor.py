from __future__ import annotations

import torch
from torch import Tensor, nn

from longship.rl.registry import components


@components.register("decoder", "GaussianActorDecoder")
class GaussianActorDecoder(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, init_log_std: float = 0.0) -> None:
        super().__init__()
        if input_dim <= 0 or action_dim <= 0:
            raise ValueError("input_dim and action_dim must be positive")
        self.action_dim = int(action_dim)
        self.mean = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor]:
        mean = self.mean(value)
        return mean, self.log_std.expand_as(mean)

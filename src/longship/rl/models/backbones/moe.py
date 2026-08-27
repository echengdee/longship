from __future__ import annotations

import torch
from torch import Tensor, nn

from longship.rl.models._layers import mlp
from longship.rl.registry import components


@components.register("backbone", "MoEBackbone")
class MoEBackbone(nn.Module):
    """Dense gated mixture of MLP experts used by perceptive locomotion policies."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        num_experts: int,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        experts = [mlp(input_dim, hidden_dims, activation) for _ in range(num_experts)]
        self.experts = nn.ModuleList(network for network, _ in experts)
        self.output_dim = experts[0][1]
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, value: Tensor) -> Tensor:
        weights = torch.softmax(self.gate(value), dim=-1)
        expert_values = torch.stack([expert(value) for expert in self.experts], dim=-2)
        return torch.sum(expert_values * weights.unsqueeze(-1), dim=-2)

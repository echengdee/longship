from __future__ import annotations

from torch import Tensor, nn

from longship.rl.models._layers import mlp
from longship.rl.registry import components


@components.register("encoder", "ProprioceptionEncoder")
class ProprioceptionEncoder(nn.Module):
    """Flatten a fixed observation history and optionally project it with an MLP."""

    def __init__(
        self,
        input_dim: int,
        history_steps: int = 1,
        hidden_dims: list[int] | tuple[int, ...] = (),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        if input_dim <= 0 or history_steps <= 0:
            raise ValueError("input_dim and history_steps must be positive")
        self.input_dim = int(input_dim)
        self.history_steps = int(history_steps)
        flattened_dim = self.input_dim * self.history_steps
        if hidden_dims:
            self.network, self.output_dim = mlp(flattened_dim, hidden_dims, activation)
        else:
            self.network = nn.Identity()
            self.output_dim = flattened_dim

    def forward(self, value: Tensor) -> Tensor:
        if value.shape[-1] != self.input_dim:
            raise ValueError(f"expected final observation dimension {self.input_dim}, got {value.shape[-1]}")
        if self.history_steps > 1 and value.shape[-2] != self.history_steps:
            raise ValueError(f"expected {self.history_steps} history steps, got {value.shape[-2]}")
        flattened = value.reshape(*value.shape[:-2], -1) if self.history_steps > 1 else value
        return self.network(flattened)

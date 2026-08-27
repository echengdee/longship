from __future__ import annotations

from torch import Tensor, nn

from longship.rl.models._layers import mlp
from longship.rl.registry import components


@components.register("backbone", "MLPBackbone")
class MLPBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.network, self.output_dim = mlp(input_dim, hidden_dims, activation)

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)

from __future__ import annotations

from torch import Tensor, nn

from longship.rl.registry import components


@components.register("decoder", "ValueDecoder")
class ValueDecoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        self.value = nn.Linear(input_dim, output_dim)

    def forward(self, value: Tensor) -> Tensor:
        return self.value(value)

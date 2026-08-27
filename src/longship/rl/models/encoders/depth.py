from __future__ import annotations

import math

from torch import Tensor, nn

from longship.rl.models._layers import activation as build_activation
from longship.rl.models._layers import mlp
from longship.rl.registry import components


@components.register("encoder", "DepthConvEncoder")
class DepthConvEncoder(nn.Module):
    """Encode a fixed stack of depth frames into a compact latent vector."""

    def __init__(
        self,
        input_shape: list[int] | tuple[int, int, int],
        channels: list[int] | tuple[int, ...],
        kernel_sizes: list[int] | tuple[int, ...],
        strides: list[int] | tuple[int, ...],
        paddings: list[int] | tuple[int, ...] | None = None,
        hidden_dims: list[int] | tuple[int, ...] = (),
        output_dim: int = 128,
        activation: str = "relu",
        max_pool: bool = True,
    ) -> None:
        super().__init__()
        if len(input_shape) != 3 or min(input_shape) <= 0:
            raise ValueError("input_shape must be [channels, height, width]")
        if not channels or not (len(channels) == len(kernel_sizes) == len(strides)):
            raise ValueError("channels, kernel_sizes, and strides must have equal non-zero length")
        if paddings is None:
            paddings = [0] * len(channels)
        if len(paddings) != len(channels):
            raise ValueError("paddings must match channels")
        layers: list[nn.Module] = []
        in_channels, height, width = (int(value) for value in input_shape)
        for out_channels, kernel, stride, padding in zip(
            channels, kernel_sizes, strides, paddings, strict=True
        ):
            layers.extend(
                (
                    nn.Conv2d(in_channels, int(out_channels), int(kernel), int(stride), int(padding)),
                    build_activation(activation),
                )
            )
            height = math.floor((height + 2 * padding - kernel) / stride + 1)
            width = math.floor((width + 2 * padding - kernel) / stride + 1)
            if max_pool:
                layers.append(nn.MaxPool2d(2))
                height, width = height // 2, width // 2
            if min(height, width) <= 0:
                raise ValueError("convolution configuration collapses the depth image")
            in_channels = int(out_channels)
        self.convolution = nn.Sequential(*layers)
        flattened_dim = in_channels * height * width
        projection_dims = [*hidden_dims, int(output_dim)]
        self.projection, self.output_dim = mlp(flattened_dim, projection_dims, activation)

    def forward(self, value: Tensor) -> Tensor:
        encoded = self.convolution(value)
        return self.projection(encoded.flatten(start_dim=-3))

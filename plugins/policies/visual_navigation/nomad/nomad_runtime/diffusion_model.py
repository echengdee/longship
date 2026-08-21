"""Conditional one-dimensional U-Net used by NoMaD diffusion inference."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn


class SinusoidalPositionEmbedding(nn.Module):
    """Embeds integer diffusion timesteps with sinusoidal features."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dim = dimension

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        half_dimension = self.dim // 2
        scale = math.log(10000) / (half_dimension - 1)
        frequencies = torch.exp(
            torch.arange(half_dimension, device=inputs.device) * -scale
        )
        embeddings = inputs[:, None] * frequencies[None, :]
        return torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)


class Downsample1d(nn.Module):
    """Halves a one-dimensional sequence using a strided convolution."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dimension, dimension, 3, 2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class Upsample1d(nn.Module):
    """Doubles a one-dimensional sequence using a transposed convolution."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dimension, dimension, 4, 2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class Conv1dBlock(nn.Module):
    """Applies Conv1d, GroupNorm, and Mish activation."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(groups, output_channels),
            nn.Mish(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ConditionalResidualBlock1d(nn.Module):
    """Residual Conv1d block modulated by a global condition."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        condition_dimension: int,
        kernel_size: int = 3,
        groups: int = 8,
        condition_predicts_scale: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(
                    input_channels,
                    output_channels,
                    kernel_size,
                    groups,
                ),
                Conv1dBlock(
                    output_channels,
                    output_channels,
                    kernel_size,
                    groups,
                ),
            ]
        )
        condition_channels = output_channels
        if condition_predicts_scale:
            condition_channels *= 2
        self.cond_predict_scale = condition_predicts_scale
        self.out_channels = output_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dimension, condition_channels),
            _UnsqueezeLastDimension(),
        )
        self.residual_conv = (
            nn.Conv1d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(
        self, inputs: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        output = self.blocks[0](inputs)
        embedding = self.cond_encoder(condition)
        if self.cond_predict_scale:
            embedding = embedding.reshape(
                embedding.shape[0], 2, self.out_channels, 1
            )
            scale = embedding[:, 0, ...]
            bias = embedding[:, 1, ...]
            output = scale * output + bias
        else:
            output = output + embedding
        output = self.blocks[1](output)
        return output + self.residual_conv(inputs)


class _UnsqueezeLastDimension(nn.Module):
    """Checkpoint-compatible replacement for einops Rearrange."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.unsqueeze(-1)


class ConditionalUnet1D(nn.Module):
    """Predicts diffusion noise for a waypoint trajectory."""

    def __init__(
        self,
        input_dim: int,
        local_cond_dim: int | None = None,
        global_cond_dim: int | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ) -> None:
        super().__init__()
        all_dimensions = [input_dim] + list(down_dims)
        start_dimension = down_dims[0]
        condition_dimension = diffusion_step_embed_dim
        if global_cond_dim is not None:
            condition_dimension += global_cond_dim

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPositionEmbedding(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        input_output_dimensions = list(
            zip(all_dimensions[:-1], all_dimensions[1:])
        )

        self.local_cond_encoder = None
        if local_cond_dim is not None:
            _, output_dimension = input_output_dimensions[0]
            self.local_cond_encoder = nn.ModuleList(
                [
                    ConditionalResidualBlock1d(
                        local_cond_dim,
                        output_dimension,
                        condition_dimension,
                        kernel_size,
                        n_groups,
                        cond_predict_scale,
                    ),
                    ConditionalResidualBlock1d(
                        local_cond_dim,
                        output_dimension,
                        condition_dimension,
                        kernel_size,
                        n_groups,
                        cond_predict_scale,
                    ),
                ]
            )

        middle_dimension = all_dimensions[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1d(
                    middle_dimension,
                    middle_dimension,
                    condition_dimension,
                    kernel_size,
                    n_groups,
                    cond_predict_scale,
                ),
                ConditionalResidualBlock1d(
                    middle_dimension,
                    middle_dimension,
                    condition_dimension,
                    kernel_size,
                    n_groups,
                    cond_predict_scale,
                ),
            ]
        )

        down_modules = nn.ModuleList()
        for index, (input_dimension, output_dimension) in enumerate(
            input_output_dimensions
        ):
            is_last = index >= len(input_output_dimensions) - 1
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(
                            input_dimension,
                            output_dimension,
                            condition_dimension,
                            kernel_size,
                            n_groups,
                            cond_predict_scale,
                        ),
                        ConditionalResidualBlock1d(
                            output_dimension,
                            output_dimension,
                            condition_dimension,
                            kernel_size,
                            n_groups,
                            cond_predict_scale,
                        ),
                        (
                            nn.Identity()
                            if is_last
                            else Downsample1d(output_dimension)
                        ),
                    ]
                )
            )
        self.down_modules = down_modules

        up_modules = nn.ModuleList()
        reversed_dimensions = reversed(input_output_dimensions[1:])
        for index, (input_dimension, output_dimension) in enumerate(
            reversed_dimensions
        ):
            # This reproduces the released diffusion-policy architecture.
            is_last = index >= len(input_output_dimensions) - 1
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(
                            output_dimension * 2,
                            input_dimension,
                            condition_dimension,
                            kernel_size,
                            n_groups,
                            cond_predict_scale,
                        ),
                        ConditionalResidualBlock1d(
                            input_dimension,
                            input_dimension,
                            condition_dimension,
                            kernel_size,
                            n_groups,
                            cond_predict_scale,
                        ),
                        (
                            nn.Identity()
                            if is_last
                            else Upsample1d(input_dimension)
                        ),
                    ]
                )
            )
        self.up_modules = up_modules
        self.final_conv = nn.Sequential(
            Conv1dBlock(
                start_dimension, start_dimension, kernel_size
            ),
            nn.Conv1d(start_dimension, input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        local_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns predicted noise with the same shape as sample."""
        sample_channels_first = sample.permute(0, 2, 1)
        if not torch.is_tensor(timestep):
            timesteps = torch.tensor(
                [timestep], dtype=torch.long, device=sample.device
            )
        elif timestep.ndim == 0:
            timesteps = timestep[None].to(sample.device)
        else:
            timesteps = timestep.to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_features = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_features = torch.cat(
                [global_features, global_cond], dim=-1
            )

        local_features = []
        if local_cond is not None:
            local_channels_first = local_cond.permute(0, 2, 1)
            first_encoder, second_encoder = self.local_cond_encoder
            local_features.append(
                first_encoder(local_channels_first, global_features)
            )
            local_features.append(
                second_encoder(local_channels_first, global_features)
            )

        hidden = sample_channels_first
        skips = []
        for index, (first_block, second_block, downsample) in enumerate(
            self.down_modules
        ):
            hidden = first_block(hidden, global_features)
            if index == 0 and local_features:
                hidden = hidden + local_features[0]
            hidden = second_block(hidden, global_features)
            skips.append(hidden)
            hidden = downsample(hidden)

        for middle_block in self.mid_modules:
            hidden = middle_block(hidden, global_features)

        for index, (first_block, second_block, upsample) in enumerate(
            self.up_modules
        ):
            hidden = torch.cat([hidden, skips.pop()], dim=1)
            hidden = first_block(hidden, global_features)
            # The original unreachable condition is retained for checkpoint
            # behavior compatibility. Local conditioning is unused by NoMaD.
            if index == len(self.up_modules) and local_features:
                hidden = hidden + local_features[1]
            hidden = second_block(hidden, global_features)
            hidden = upsample(hidden)

        output = self.final_conv(hidden)
        return output.permute(0, 2, 1)

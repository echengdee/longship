"""Goal-conditioned visual encoder used by NoMaD."""

from __future__ import annotations

from collections.abc import Callable

import torch
from efficientnet_pytorch import EfficientNet
from torch import nn

from nomad_runtime.positional_encoding import PositionalEncoding


def _replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    factory: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """Replaces matching submodules while preserving their registered names."""
    if predicate(root_module):
        return factory(root_module)

    names = [
        name
        for name, module in root_module.named_modules(remove_duplicate=True)
        if predicate(module)
    ]
    for name in names:
        parent_name, _, child_name = name.rpartition(".")
        parent = (
            root_module.get_submodule(parent_name)
            if parent_name
            else root_module
        )
        replacement = factory(parent.get_submodule(child_name))
        if isinstance(parent, nn.Sequential):
            parent[int(child_name)] = replacement
        else:
            setattr(parent, child_name, replacement)

    remaining = [
        module
        for module in root_module.modules()
        if predicate(module)
    ]
    if remaining:
        raise RuntimeError("Failed to replace every matching submodule")
    return root_module


def replace_batch_norm_with_group_norm(
    module: nn.Module, features_per_group: int = 16
) -> nn.Module:
    """Matches the GroupNorm conversion used by the released checkpoint."""

    def make_group_norm(batch_norm: nn.Module) -> nn.Module:
        channels = batch_norm.num_features
        return nn.GroupNorm(
            num_groups=channels // features_per_group,
            num_channels=channels,
        )

    return _replace_submodules(
        module,
        lambda child: isinstance(child, nn.BatchNorm2d),
        make_group_norm,
    )


class NomadVisionEncoder(nn.Module):
    """Encodes four observations and one optional goal into one condition."""

    def __init__(
        self,
        context_size: int,
        encoding_size: int,
        attention_heads: int,
        attention_layers: int,
        feed_forward_factor: int,
    ) -> None:
        super().__init__()
        self.obs_encoding_size = encoding_size
        self.goal_encoding_size = encoding_size
        self.context_size = context_size

        self.obs_encoder = EfficientNet.from_name(
            "efficientnet-b0", in_channels=3
        )
        self.obs_encoder = replace_batch_norm_with_group_norm(
            self.obs_encoder
        )
        self.num_obs_features = self.obs_encoder._fc.in_features

        self.goal_encoder = EfficientNet.from_name(
            "efficientnet-b0", in_channels=6
        )
        self.goal_encoder = replace_batch_norm_with_group_norm(
            self.goal_encoder
        )
        self.num_goal_features = self.goal_encoder._fc.in_features

        self.compress_obs_enc = nn.Linear(
            self.num_obs_features, self.obs_encoding_size
        )
        self.compress_goal_enc = nn.Linear(
            self.num_goal_features, self.goal_encoding_size
        )
        sequence_length = self.context_size + 2
        self.positional_encoding = PositionalEncoding(
            self.obs_encoding_size, sequence_length
        )
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size,
            nhead=attention_heads,
            dim_feedforward=feed_forward_factor * self.obs_encoding_size,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sa_encoder = nn.TransformerEncoder(
            self.sa_layer, num_layers=attention_layers
        )

        goal_mask = torch.zeros((1, sequence_length), dtype=torch.bool)
        goal_mask[:, -1] = True
        no_mask = torch.zeros((1, sequence_length), dtype=torch.bool)
        self.goal_mask = goal_mask
        self.no_mask = no_mask
        self.all_masks = torch.cat([no_mask, goal_mask], dim=0)
        self.avg_pool_mask = torch.cat(
            [
                1 - no_mask.float(),
                (1 - goal_mask.float())
                * (sequence_length / (sequence_length - 1)),
            ],
            dim=0,
        )

    @staticmethod
    def _extract_features(
        encoder: EfficientNet, inputs: torch.Tensor
    ) -> torch.Tensor:
        features = encoder.extract_features(inputs)
        features = encoder._avg_pooling(features)
        if encoder._global_params.include_top:
            features = features.flatten(start_dim=1)
            features = encoder._dropout(features)
        return features

    def forward(
        self,
        obs_img: torch.Tensor,
        goal_img: torch.Tensor,
        input_goal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns a 256-dimensional condition for distance and action heads."""
        expected_channels = 3 * (self.context_size + 1)
        if obs_img.ndim != 4 or obs_img.shape[1] != expected_channels:
            raise ValueError(
                "obs_img must have shape [batch, "
                f"{expected_channels}, height, width]"
            )
        if goal_img.ndim != 4 or goal_img.shape[1] != 3:
            raise ValueError(
                "goal_img must have shape [batch, 3, height, width]"
            )
        if obs_img.shape[0] != goal_img.shape[0]:
            raise ValueError("Observation and goal batch sizes must match")

        latest_observation = obs_img[:, 3 * self.context_size :, :, :]
        observation_goal = torch.cat(
            [latest_observation, goal_img], dim=1
        )
        goal_encoding = self._extract_features(
            self.goal_encoder, observation_goal
        )
        goal_encoding = self.compress_goal_enc(goal_encoding).unsqueeze(1)

        observation_frames = torch.cat(
            torch.split(obs_img, 3, dim=1), dim=0
        )
        observation_encoding = self._extract_features(
            self.obs_encoder, observation_frames
        )
        observation_encoding = self.compress_obs_enc(observation_encoding)
        observation_encoding = observation_encoding.unsqueeze(1)
        observation_encoding = observation_encoding.reshape(
            self.context_size + 1,
            -1,
            self.obs_encoding_size,
        ).transpose(0, 1)

        tokens = torch.cat([observation_encoding, goal_encoding], dim=1)
        mask_indices = input_goal_mask.to(
            device=obs_img.device, dtype=torch.long
        )
        padding_mask = torch.index_select(
            self.all_masks.to(obs_img.device), 0, mask_indices
        )
        tokens = self.positional_encoding(tokens)
        tokens = self.sa_encoder(
            tokens, src_key_padding_mask=padding_mask
        )
        average_mask = torch.index_select(
            self.avg_pool_mask.to(obs_img.device), 0, mask_indices
        ).unsqueeze(-1)
        return torch.mean(tokens * average_mask, dim=1)

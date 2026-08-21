"""Positional encoding used by the NoMaD vision transformer."""

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Adds fixed sinusoidal position features to a token sequence."""

    def __init__(self, embedding_size: int, max_sequence_length: int) -> None:
        super().__init__()
        encoding = torch.zeros(max_sequence_length, embedding_size)
        positions = torch.arange(
            max_sequence_length, dtype=torch.float32
        ).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, embedding_size, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embedding_size)
        )
        encoding[:, 0::2] = torch.sin(positions * divisor)
        encoding[:, 1::2] = torch.cos(positions * divisor)
        self.register_buffer("pos_enc", encoding.unsqueeze(0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Adds encodings for the sequence positions present in inputs."""
        return inputs + self.pos_enc[:, : inputs.size(1), :]

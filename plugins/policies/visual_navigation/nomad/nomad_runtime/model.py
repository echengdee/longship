"""Checkpoint-compatible NoMaD model assembly."""

from torch import nn

from nomad_runtime.config import NomadConfig
from nomad_runtime.diffusion_model import ConditionalUnet1D
from nomad_runtime.vision_encoder import NomadVisionEncoder


class DenseNetwork(nn.Module):
    """Predicts temporal distance from the shared visual condition."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 16),
            nn.ReLU(),
            nn.Linear(embedding_dim // 16, 1),
        )

    def forward(self, inputs):
        return self.network(inputs.reshape((-1, self.embedding_dim)))


class NoMaD(nn.Module):
    """Holds the vision, distance, and diffusion action networks."""

    def __init__(
        self,
        vision_encoder: nn.Module,
        noise_pred_net: nn.Module,
        dist_pred_net: nn.Module,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net


def build_nomad_model(config: NomadConfig) -> NoMaD:
    """Builds a model matching the published NoMaD checkpoint layout."""
    config.validate()
    vision_encoder = NomadVisionEncoder(
        context_size=config.context_size,
        encoding_size=config.encoding_size,
        attention_heads=config.attention_heads,
        attention_layers=config.attention_layers,
        feed_forward_factor=config.feed_forward_factor,
    )
    noise_predictor = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config.encoding_size,
        down_dims=config.diffusion_down_dims,
        cond_predict_scale=config.condition_predicts_scale,
    )
    return NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_predictor,
        dist_pred_net=DenseNetwork(config.encoding_size),
    )

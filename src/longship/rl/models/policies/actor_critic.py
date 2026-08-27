from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from longship.rl.registry import components


@components.register("policy", "ActorCriticPolicy")
class ActorCriticPolicy(nn.Module):
    """Composable actor-critic whose data flow is explicit in the experiment."""

    def __init__(
        self,
        encoder: nn.Module,
        backbone: nn.Module,
        actor_decoder: nn.Module,
        critic_decoder: nn.Module,
        critic_backbone: nn.Module | None = None,
        observation_key: str = "proprio",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.critic_backbone = critic_backbone or backbone
        self.actor_decoder = actor_decoder
        self.critic_decoder = critic_decoder
        self.observation_key = observation_key

    def _input(self, observation: Mapping[str, Any] | Tensor) -> Tensor:
        return observation[self.observation_key] if isinstance(observation, Mapping) else observation

    def forward(self, observation: Mapping[str, Any] | Tensor) -> dict[str, Tensor]:
        encoded = self.encoder(self._input(observation))
        actor_features = self.backbone(encoded)
        critic_features = self.critic_backbone(encoded)
        mean, log_std = self.actor_decoder(actor_features)
        return {"mean": mean, "log_std": log_std, "value": self.critic_decoder(critic_features)}

    def act(self, observation: Mapping[str, Any] | Tensor, deterministic: bool = False) -> Tensor:
        outputs = self.forward(observation)
        if deterministic:
            return outputs["mean"]
        return torch.distributions.Normal(outputs["mean"], outputs["log_std"].exp()).rsample()


@components.register("policy", "PerceptiveActorCriticPolicy")
class PerceptiveActorCriticPolicy(ActorCriticPolicy):
    """Concatenate proprioception with an encoded visual observation."""

    def __init__(self, *args: Any, depth_key: str = "depth", proprio_key: str = "proprio", **kwargs: Any):
        super().__init__(*args, observation_key=depth_key, **kwargs)
        self.depth_key = depth_key
        self.proprio_key = proprio_key

    def forward(self, observation: Mapping[str, Any] | Tensor) -> dict[str, Tensor]:
        if not isinstance(observation, Mapping):
            raise TypeError("PerceptiveActorCriticPolicy expects a mapping observation")
        visual = self.encoder(observation[self.depth_key])
        encoded = torch.cat((observation[self.proprio_key], visual), dim=-1)
        actor_features = self.backbone(encoded)
        critic_features = self.critic_backbone(encoded)
        mean, log_std = self.actor_decoder(actor_features)
        return {"mean": mean, "log_std": log_std, "value": self.critic_decoder(critic_features)}

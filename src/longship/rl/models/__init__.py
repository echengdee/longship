"""Reusable policy, encoder, backbone, and decoder implementations."""


def register_builtin_models() -> None:
    """Import model modules lazily so the core package does not require PyTorch."""
    from longship.rl.models.backbones import mlp, moe  # noqa: F401
    from longship.rl.models.decoders import actor, value  # noqa: F401
    from longship.rl.models.encoders import depth, proprioception  # noqa: F401
    from longship.rl.models.policies import actor_critic  # noqa: F401


__all__ = ["register_builtin_models"]

from __future__ import annotations

from collections.abc import Iterable


def activation(name: str):
    """Create a PyTorch activation without exposing framework objects in YAML."""
    from torch import nn

    activations = {
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"unsupported activation {name!r}; choose from {sorted(activations)}") from exc


def mlp(input_dim: int, hidden_dims: Iterable[int], activation_name: str):
    from torch import nn

    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    dims = [input_dim, *(int(value) for value in hidden_dims)]
    if len(dims) == 1 or any(value <= 0 for value in dims):
        raise ValueError("hidden_dims must contain positive dimensions")
    layers: list[nn.Module] = []
    for source, target in zip(dims, dims[1:]):
        layers.extend((nn.Linear(source, target), activation(activation_name)))
    return nn.Sequential(*layers), dims[-1]

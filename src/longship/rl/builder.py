from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from longship.rl.registry import ComponentRegistry, RegistryError, components


_MODEL_PARTS = {
    "encoder": "encoder",
    "backbone": "backbone",
    "actor_decoder": "decoder",
    "critic_decoder": "decoder",
    "q_decoder": "decoder",
    "motion_decoder": "decoder",
}


def build_model(
    config: Mapping[str, Any], *, registry: ComponentRegistry = components
) -> Any:
    """Build a policy and its declared parts from a nested model config.

    The policy class owns the forward graph. This builder deliberately supports
    named architectural slots rather than interpreting a general YAML DAG.
    """

    if not isinstance(config, Mapping):
        raise RegistryError("model config must be a mapping")
    policy_config = dict(config)
    built_parts: dict[str, Any] = {}
    for field, kind in _MODEL_PARTS.items():
        raw_part = policy_config.pop(field, None)
        if raw_part is not None:
            if not isinstance(raw_part, Mapping):
                raise RegistryError(f"model.{field} must be a mapping")
            built_parts[field] = registry.create(kind, raw_part)
    return registry.create("policy", policy_config, **built_parts)

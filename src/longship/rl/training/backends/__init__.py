"""Adapters for Longship's supported upstream training systems."""
from __future__ import annotations

from longship.rl.training.backends.base import TrainingBackendError, TrainingPlan


_REGISTERED = False


def register_builtin_backends() -> None:
    """Register built-in upstream adapters without importing them at package load."""

    global _REGISTERED
    if _REGISTERED:
        return
    from longship.rl.training.backends import (  # noqa: F401
        holosoma,
        instinctlab,
        mimiclite,
        sonic,
    )

    _REGISTERED = True


__all__ = ["TrainingBackendError", "TrainingPlan", "register_builtin_backends"]

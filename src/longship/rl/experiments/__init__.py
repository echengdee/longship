"""Bundled, inspectable RL experiment recipes."""

from importlib.resources import files
from pathlib import Path


def bundled_experiment_path(name: str) -> Path:
    if not name or Path(name).name != name:
        raise ValueError("experiment name must be a plain file name")
    resource = files(__package__).joinpath(name)
    return Path(str(resource))


__all__ = ["bundled_experiment_path"]

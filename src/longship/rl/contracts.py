from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class PolicyModel(Protocol):
    """A framework-neutral policy boundary assembled by an experiment."""

    def act(self, observation: Mapping[str, Any]) -> Any:
        ...


@runtime_checkable
class TrainingBackend(Protocol):
    """Runs an experiment through an upstream RL implementation."""

    def train(self, experiment: Mapping[str, Any], output_dir: Path) -> Path:
        """Train and return the produced checkpoint path."""


@runtime_checkable
class Sim2SimRunner(Protocol):
    """Evaluates an exported policy in a second simulator."""

    def run(self, artifact: Path, config: Mapping[str, Any], output_dir: Path) -> None:
        ...


@runtime_checkable
class DeploymentExporter(Protocol):
    """Converts a checkpoint into a target runtime artifact."""

    def export(self, checkpoint: Path, config: Mapping[str, Any], output_dir: Path) -> Path:
        ...

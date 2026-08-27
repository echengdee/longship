from __future__ import annotations

from pathlib import Path

import yaml

from longship.rl.config import ExperimentConfig
from longship.rl.registry import ComponentRegistry, components
from longship.rl.training.backends import TrainingPlan, register_builtin_backends


class ExperimentRunner:
    """Dispatches a validated experiment to a registered training backend."""

    def __init__(
        self,
        registry: ComponentRegistry = components,
        *,
        workspace: str | Path | None = None,
    ) -> None:
        self.registry = registry
        self.workspace = Path(workspace or Path.cwd()).resolve()

    def _backend(self, experiment: ExperimentConfig):
        if self.registry is components:
            register_builtin_backends()
        backend_config = experiment.values["training"]["backend"]
        backend = self.registry.create("training_backend", backend_config)
        bind_workspace = getattr(backend, "bind_workspace", None)
        if bind_workspace is not None:
            bind_workspace(self.workspace)
        return backend

    def plan(
        self, experiment: ExperimentConfig, output_dir: str | Path
    ) -> TrainingPlan:
        backend = self._backend(experiment)
        plan = getattr(backend, "plan", None)
        if plan is None:
            raise TypeError(
                f"training backend {type(backend).__name__} does not support planning"
            )
        return plan(experiment.values, Path(output_dir).resolve())

    def run(self, experiment: ExperimentConfig, output_dir: str | Path) -> Path:
        resolved_output = Path(output_dir).resolve()
        resolved_output.mkdir(parents=True, exist_ok=False)
        snapshot = resolved_output / "resolved.yaml"
        snapshot.write_text(
            yaml.safe_dump(
                dict(experiment.values), sort_keys=False, allow_unicode=True
            ),
            encoding="utf-8",
        )
        backend = self._backend(experiment)
        return Path(backend.train(experiment.values, resolved_output))

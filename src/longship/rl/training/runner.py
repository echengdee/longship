from __future__ import annotations

from pathlib import Path

import yaml

from longship.rl.config import ExperimentConfig
from longship.rl.registry import ComponentRegistry, components


class ExperimentRunner:
    """Dispatches a validated experiment to a registered training backend."""

    def __init__(self, registry: ComponentRegistry = components) -> None:
        self.registry = registry

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
        backend_config = experiment.values["training"]["backend"]
        backend = self.registry.create("training_backend", backend_config)
        return Path(backend.train(experiment.values, resolved_output))

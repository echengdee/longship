from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from longship.rl.registry import components
from longship.rl.training.backends.base import UpstreamTrainingBackend


@components.register("training_backend", "HoloSomaBackend")
class HoloSomaBackend(UpstreamTrainingBackend):
    """Translate a Longship experiment into HoloSoma's Tyro CLI."""

    backend_name = "holosoma"

    def __init__(
        self,
        *,
        experiment: str = "g1-29dof-fast-sac",
        simulator: str = "isaacgym",
        logger: str = "disabled",
        source_root: str = "third_party/holosoma",
        python_executable: str = "python",
        extra_args: Sequence[str] = (),
        environment: Mapping[str, str] | None = None,
        checkpoint_globs: Sequence[str] = ("**/*.pt", "**/*.pth"),
    ) -> None:
        super().__init__(
            source_root=source_root,
            python_executable=python_executable,
            extra_args=extra_args,
            environment=environment,
            checkpoint_globs=checkpoint_globs,
        )
        self.experiment = experiment
        self.simulator = simulator
        self.logger = logger

    def build_argv(
        self, experiment: Mapping[str, Any], output_dir: Path
    ) -> Sequence[str]:
        seed = int(experiment.get("seed", 0))
        return (
            self.python_executable,
            "src/holosoma/holosoma/train_agent.py",
            f"exp:{self.experiment}",
            f"simulator:{self.simulator}",
            f"logger:{self.logger}",
            "--logger.base-dir",
            str(output_dir / "upstream"),
            "--training.seed",
            str(seed),
            *self.extra_args,
        )

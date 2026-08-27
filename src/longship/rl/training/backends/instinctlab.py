from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from longship.rl.registry import components
from longship.rl.training.backends.base import UpstreamTrainingBackend


@components.register("training_backend", "InstinctLabBackend")
class InstinctLabBackend(UpstreamTrainingBackend):
    """Translate a Longship experiment into InstinctLab's training CLI."""

    backend_name = "instinctlab"

    def __init__(
        self,
        *,
        task: str = "Instinct-Parkour-Target-Amp-G1-v0",
        source_root: str = "third_party/InstinctLab",
        python_executable: str = "python",
        headless: bool = True,
        num_envs: int | None = None,
        max_iterations: int | None = None,
        extra_args: Sequence[str] = (),
        environment: Mapping[str, str] | None = None,
        checkpoint_globs: Sequence[str] = ("**/model_*.pt", "**/*.pth"),
    ) -> None:
        super().__init__(
            source_root=source_root,
            python_executable=python_executable,
            extra_args=extra_args,
            environment=environment,
            checkpoint_globs=checkpoint_globs,
        )
        self.task = task
        self.headless = bool(headless)
        self.num_envs = num_envs
        self.max_iterations = max_iterations

    def build_argv(
        self, experiment: Mapping[str, Any], output_dir: Path
    ) -> Sequence[str]:
        argv = [
            self.python_executable,
            "scripts/instinct_rl/train.py",
            f"--task={self.task}",
            f"--seed={int(experiment.get('seed', 0))}",
            f"--logroot={output_dir / 'upstream'}",
        ]
        if self.headless:
            argv.append("--headless")
        if self.num_envs is not None:
            argv.append(f"--num_envs={int(self.num_envs)}")
        if self.max_iterations is not None:
            argv.append(f"--max_iterations={int(self.max_iterations)}")
        argv.extend(self.extra_args)
        return argv

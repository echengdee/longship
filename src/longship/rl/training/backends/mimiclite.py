from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from longship.rl.registry import components
from longship.rl.training.backends.base import (
    TrainingBackendError,
    TrainingPlan,
    UpstreamTrainingBackend,
)


@components.register("training_backend", "MimicLiteBackend")
class MimicLiteBackend(UpstreamTrainingBackend):
    """Translate a Longship motion-tracking experiment to MimicLite Hydra."""

    backend_name = "mimiclite"

    def __init__(
        self,
        *,
        source_root: str = "third_party/active-adaptation-dev",
        mimic_root: str = "third_party/mimic-lite",
        venv_project: str = "environments/rl/mjlab",
        uv_executable: str = "uv",
        task: str = "tracking-base",
        motion_config: str,
        terrain: str = "flat",
        module: str = "huge",
        num_envs: int = 256,
        total_iters: int = 4000,
        checkpoint_interval: int = 100,
        upload_interval: int = 1000,
        checkpoint: str | None = None,
        wandb_mode: str = "disabled",
        hf_hub_offline: bool = True,
        extra_args: Sequence[str] = (),
        environment: Mapping[str, str] | None = None,
        checkpoint_globs: Sequence[str] = ("**/checkpoint*.pt",),
    ) -> None:
        merged_environment = {
            "HF_HUB_OFFLINE": "1" if hf_hub_offline else "0",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            **dict(environment or {}),
        }
        super().__init__(
            source_root=source_root,
            python_executable=uv_executable,
            extra_args=extra_args,
            environment=merged_environment,
            checkpoint_globs=checkpoint_globs,
        )
        if num_envs < 1 or total_iters < 1:
            raise TrainingBackendError("MimicLite num_envs and total_iters must be positive")
        if checkpoint_interval < 1 or upload_interval < 1:
            raise TrainingBackendError("MimicLite checkpoint intervals must be positive")
        self.mimic_root = mimic_root
        self.venv_project = venv_project
        self.task = task
        self.motion_config = motion_config
        self.terrain = terrain
        self.module = module
        self.num_envs = int(num_envs)
        self.total_iters = int(total_iters)
        self.checkpoint_interval = int(checkpoint_interval)
        self.upload_interval = int(upload_interval)
        self.checkpoint = checkpoint
        self.wandb_mode = wandb_mode

    def _workspace_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.workspace / path
        return path.resolve()

    def plan(
        self, experiment: Mapping[str, Any], output_dir: str | Path
    ) -> TrainingPlan:
        motion_config = (
            self._workspace_path(self.mimic_root)
            / "cfg"
            / "task"
            / "motion"
            / f"{self.motion_config}.yaml"
        )
        if not motion_config.is_file():
            raise TrainingBackendError(
                f"MimicLite motion config does not exist: {motion_config}"
            )
        data = experiment.get("data", {})
        for field in ("motion_file", "manifest"):
            value = data.get(field) if isinstance(data, Mapping) else None
            if value and not self._workspace_path(str(value)).is_file():
                raise TrainingBackendError(
                    f"MimicLite data.{field} does not exist: {self._workspace_path(str(value))}"
                )

        plan = super().plan(experiment, output_dir)
        environment = dict(plan.environment)
        environment.setdefault("UV_CACHE_DIR", str(self.workspace / ".cache" / "uv"))
        environment.setdefault("WANDB_DIR", str(plan.output_dir / "upstream"))
        return replace(plan, environment=environment)

    def build_argv(
        self, experiment: Mapping[str, Any], output_dir: Path
    ) -> Sequence[str]:
        mimic_root = self._workspace_path(self.mimic_root)
        venv_project = self._workspace_path(self.venv_project)
        train_script = mimic_root / "scripts" / "train.py"
        if not train_script.is_file():
            raise TrainingBackendError(f"MimicLite trainer does not exist: {train_script}")
        if not (venv_project / "pyproject.toml").is_file():
            raise TrainingBackendError(
                f"MimicLite environment project does not exist: {venv_project}"
            )

        checkpoint = "null" if self.checkpoint is None else self.checkpoint
        argv = [
            self.python_executable,
            "--project",
            str(venv_project),
            "run",
            str(train_script),
            f"task={self.task}",
            f"task/motion={self.motion_config}",
            "+exp=ppo/train",
            f"algo/ppo/module={self.module}",
            "backend=mjlab",
            f"task.terrain={self.terrain}",
            f"task.num_envs={self.num_envs}",
            f"total_iters={self.total_iters}",
            f"checkpoint_interval={self.checkpoint_interval}",
            f"upload_interval={self.upload_interval}",
            f"checkpoint_path={checkpoint}",
            f"wandb.mode={self.wandb_mode}",
            f"seed={int(experiment.get('seed', 0))}",
            f"hydra.run.dir={output_dir / 'upstream'}",
        ]
        argv.extend(self.extra_args)
        return argv

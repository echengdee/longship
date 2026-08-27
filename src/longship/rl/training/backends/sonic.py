from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from longship.rl.registry import components
from longship.rl.training.backends.base import UpstreamTrainingBackend


@components.register("training_backend", "SonicBackend")
class SonicBackend(UpstreamTrainingBackend):
    """Translate a Longship experiment into SONIC's Hydra training CLI."""

    backend_name = "sonic"

    def __init__(
        self,
        *,
        recipe: str = "manager/universal_token/all_modes/sonic_release",
        source_root: str = "third_party/GR00T-WholeBodyControl",
        python_executable: str = "python",
        headless: bool = True,
        num_envs: int | None = None,
        num_learning_iterations: int | None = None,
        checkpoint: str | None = None,
        motion_file: str | None = None,
        smpl_motion_file: str | None = None,
        extra_args: Sequence[str] = (),
        environment: Mapping[str, str] | None = None,
        checkpoint_globs: Sequence[str] = ("**/last.pt", "**/checkpoint-*/**/*.pt"),
    ) -> None:
        super().__init__(
            source_root=source_root,
            python_executable=python_executable,
            extra_args=extra_args,
            environment=environment,
            checkpoint_globs=checkpoint_globs,
        )
        self.recipe = recipe
        self.headless = bool(headless)
        self.num_envs = num_envs
        self.num_learning_iterations = num_learning_iterations
        self.checkpoint = checkpoint
        self.motion_file = motion_file
        self.smpl_motion_file = smpl_motion_file

    def build_argv(
        self, experiment: Mapping[str, Any], output_dir: Path
    ) -> Sequence[str]:
        argv = [
            self.python_executable,
            "gear_sonic/train_agent_trl.py",
            f"+exp={self.recipe}",
            f"seed={int(experiment.get('seed', 0))}",
            f"headless={str(self.headless)}",
            f"experiment_dir={output_dir / 'upstream'}",
        ]
        if self.num_envs is not None:
            argv.append(f"num_envs={int(self.num_envs)}")
        if self.num_learning_iterations is not None:
            argv.append(
                f"++algo.config.num_learning_iterations={int(self.num_learning_iterations)}"
            )
        if self.checkpoint:
            argv.append(f"+checkpoint={self.checkpoint}")
        if self.motion_file:
            argv.append(
                "++manager_env.commands.motion.motion_lib_cfg.motion_file="
                f"{self.motion_file}"
            )
        if self.smpl_motion_file:
            argv.append(
                "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="
                f"{self.smpl_motion_file}"
            )
        argv.extend(self.extra_args)
        return argv

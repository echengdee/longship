from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class TrainingBackendError(RuntimeError):
    """Raised when an upstream training backend cannot be planned or run."""


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    """A shell-free, inspectable description of one upstream training job."""

    backend: str
    argv: tuple[str, ...]
    cwd: Path
    output_dir: Path
    environment: Mapping[str, str]
    checkpoint_globs: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["cwd"] = str(self.cwd)
        values["output_dir"] = str(self.output_dir)
        values["argv"] = list(self.argv)
        values["checkpoint_globs"] = list(self.checkpoint_globs)
        return values


class UpstreamTrainingBackend:
    """Base class for adapters which launch an upstream trainer as a process."""

    backend_name = "upstream"

    def __init__(
        self,
        *,
        source_root: str,
        python_executable: str = "python",
        extra_args: Sequence[str] = (),
        environment: Mapping[str, str] | None = None,
        checkpoint_globs: Sequence[str] = ("**/*.pt", "**/*.pth"),
    ) -> None:
        self.source_root = source_root
        self.python_executable = python_executable
        self.extra_args = tuple(str(value) for value in extra_args)
        self.environment = dict(environment or {})
        self.checkpoint_globs = tuple(str(value) for value in checkpoint_globs)
        self.workspace = Path.cwd().resolve()

    def bind_workspace(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()

    @property
    def source_path(self) -> Path:
        candidate = Path(self.source_root)
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        return candidate.resolve()

    def plan(
        self, experiment: Mapping[str, Any], output_dir: str | Path
    ) -> TrainingPlan:
        resolved_output = Path(output_dir).resolve()
        source_path = self.source_path
        if not source_path.is_dir():
            raise TrainingBackendError(
                f"{self.backend_name} source root does not exist: {source_path}"
            )
        argv = self.build_argv(experiment, resolved_output)
        if not argv or any(not isinstance(value, str) or not value for value in argv):
            raise TrainingBackendError("training argv must contain non-empty strings")
        return TrainingPlan(
            backend=self.backend_name,
            argv=tuple(argv),
            cwd=source_path,
            output_dir=resolved_output,
            environment=self.environment,
            checkpoint_globs=self.checkpoint_globs,
        )

    def build_argv(
        self, experiment: Mapping[str, Any], output_dir: Path
    ) -> Sequence[str]:
        raise NotImplementedError

    def train(self, experiment: Mapping[str, Any], output_dir: Path) -> Path:
        plan = self.plan(experiment, output_dir)
        plan.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = plan.output_dir / "command.json"
        manifest = plan.as_dict() | {"status": "running"}
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        process_environment = os.environ.copy()
        process_environment.update(plan.environment)
        try:
            subprocess.run(
                plan.argv,
                cwd=plan.cwd,
                env=process_environment,
                check=True,
            )
        except BaseException as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise
        checkpoint = self.latest_checkpoint(plan)
        manifest["status"] = "completed"
        manifest["checkpoint"] = str(checkpoint) if checkpoint else None
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return checkpoint or plan.output_dir

    @staticmethod
    def latest_checkpoint(plan: TrainingPlan) -> Path | None:
        candidates: dict[Path, float] = {}
        for pattern in plan.checkpoint_globs:
            for path in plan.output_dir.glob(pattern):
                if path.is_file():
                    candidates[path] = path.stat().st_mtime
        return max(candidates, key=candidates.get) if candidates else None

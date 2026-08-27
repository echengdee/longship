from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex


@dataclass(frozen=True, slots=True)
class ProcessSpec:
    """A runtime-neutral child-process description shared by sim and hardware."""

    name: str
    cwd: Path
    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...] = ()
    stdin: str | None = None

    def shell_command(self) -> str:
        prefix = tuple(f"{key}={value}" for key, value in self.environment)
        command = shlex.join(("env",) + prefix + self.argv) if prefix else shlex.join(self.argv)
        return f"cd {shlex.quote(str(self.cwd))} && {command}"

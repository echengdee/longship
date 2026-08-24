"""Paths to repository-managed NoMaD model artifacts."""

from __future__ import annotations

from pathlib import Path


def default_checkpoint_path() -> Path:
    """Returns the LFS-managed released checkpoint bundled in this repository."""

    repository_root = Path(__file__).resolve().parents[5]
    return repository_root / "models" / "nomad" / "nomad.pth"

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from .store import ArtifactVerificationError, VerifiedArtifact


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(
    root: Path,
    *,
    maximum_files: int = 2_048,
    maximum_bytes: int = 256 * 1024 * 1024,
) -> tuple[str, int, int]:
    """Hash regular files as relative-name, size, and content records."""

    resolved = root.resolve(strict=True)
    files = sorted(path for path in resolved.rglob("*") if path.is_file())
    if not files or len(files) > maximum_files:
        raise ValueError("artifact directory file count is out of range")
    total_bytes = 0
    digest = hashlib.sha256()
    for path in files:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise ValueError("artifact directory must contain regular files only")
        total_bytes += metadata.st_size
        if total_bytes > maximum_bytes:
            raise ValueError("artifact directory exceeds the byte limit")
        relative = path.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(str(metadata.st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest(), len(files), total_bytes


def read_verified_artifact_bytes(
    artifact: VerifiedArtifact,
    *,
    maximum_size_bytes: int | None = None,
) -> bytes:
    """Read a verified regular file while rechecking its identity and digest."""

    if not isinstance(artifact, VerifiedArtifact):
        raise TypeError("artifact must be a VerifiedArtifact")
    if maximum_size_bytes is not None and (
        type(maximum_size_bytes) is not int or maximum_size_bytes <= 0
    ):
        raise ValueError("maximum_size_bytes must be a positive integer")
    if maximum_size_bytes is not None and artifact.size_bytes > maximum_size_bytes:
        raise ArtifactVerificationError("verified artifact exceeds the read limit")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(artifact.path, flags)
    except OSError as exc:
        raise ArtifactVerificationError(
            "verified artifact cannot be reopened safely"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_dev != artifact.device
            or before.st_ino != artifact.inode
            or before.st_size != artifact.size_bytes
        ):
            raise ArtifactVerificationError("verified artifact identity changed")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise ArtifactVerificationError(
                "verified artifact changed while being read"
            )
        if digest.hexdigest() != artifact.sha256:
            raise ArtifactVerificationError("verified artifact digest changed")
        return b"".join(chunks)
    finally:
        os.close(descriptor)

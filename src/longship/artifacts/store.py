from __future__ import annotations

import fcntl
import hashlib
import math
import os
import shutil
import stat
import tempfile
import time
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterator, Protocol

from .manifest import ArtifactReference, ModelArtifactManifest


class ArtifactStoreError(RuntimeError):
    """Base error for verified external artifact handling."""


class ArtifactVerificationError(ArtifactStoreError):
    """Raised when bytes do not match their immutable manifest identity."""


class ArtifactPolicyError(ArtifactStoreError):
    """Raised when download policy or license state blocks materialization."""


class ArtifactTransport(Protocol):
    """Deployment-supplied, separately qualified byte transport."""

    def open(self, uri: str, *, timeout_s: float) -> BinaryIO:
        ...


@dataclass(frozen=True, slots=True)
class ArtifactApproval:
    """Trusted-registry approval bound to exact manifest bytes.

    A manifest cannot construct or grant its own approval. Deployment code is
    responsible for loading this value from a trusted registry or verified
    signature, never from the artifact manifest being approved.
    """

    approval_id: str
    manifest_sha256: str
    artifact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.approval_id, str) or not self.approval_id.strip():
            raise ValueError("approval_id must be a non-empty string")
        if (
            len(self.manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.manifest_sha256
            )
        ):
            raise ValueError("manifest_sha256 must be lowercase SHA-256")
        if not isinstance(self.artifact_ids, tuple) or not self.artifact_ids:
            raise ValueError("artifact_ids must be a non-empty tuple")
        if len(self.artifact_ids) != len(set(self.artifact_ids)):
            raise ValueError("artifact_ids must not contain duplicates")
        if any(not isinstance(value, str) or not value for value in self.artifact_ids):
            raise ValueError("artifact_ids must contain non-empty strings")

    def allows(self, manifest: ModelArtifactManifest, artifact_id: str) -> bool:
        return (
            self.manifest_sha256 == manifest.source_sha256
            and artifact_id in self.artifact_ids
        )


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    """Identity of bytes checked through one no-follow file descriptor.

    Consumers must still open defensively and revalidate the digest, or load a
    byte snapshot, because a filesystem path is not a durable capability.
    """

    artifact_id: str
    path: Path
    sha256: str
    size_bytes: int
    device: int
    inode: int


class ArtifactStore:
    """Private content-addressed cache with explicit approval and transport."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        max_artifact_size_bytes: int = 16 * 1024 * 1024 * 1024,
    ) -> None:
        if (
            type(max_artifact_size_bytes) is not int
            or max_artifact_size_bytes <= 0
        ):
            raise ValueError("max_artifact_size_bytes must be a positive integer")
        requested_root = Path(cache_root)
        if requested_root.exists() and requested_root.is_symlink():
            raise ArtifactPolicyError("artifact cache root must not be a symlink")
        requested_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._root = requested_root.resolve(strict=True)
        root_stat = self._root.stat(follow_symlinks=False)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ArtifactPolicyError("artifact cache root must be a directory")
        if root_stat.st_uid != os.geteuid():
            raise ArtifactPolicyError("artifact cache root must be owned by this user")
        if stat.S_IMODE(root_stat.st_mode) & 0o022:
            raise ArtifactPolicyError(
                "artifact cache root must not be group- or world-writable"
            )
        self._max_artifact_size_bytes = max_artifact_size_bytes

    def path_for(self, reference: ArtifactReference) -> Path:
        return self._root / "sha256" / reference.sha256[:2] / reference.sha256

    def _ensure_private_directory(self, directory: Path) -> None:
        current = self._root
        relative = directory.relative_to(self._root)
        for component in relative.parts:
            current = current / component
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                pass
            current_stat = current.stat(follow_symlinks=False)
            if not stat.S_ISDIR(current_stat.st_mode):
                raise ArtifactPolicyError("artifact cache path is not a directory")
            if current_stat.st_uid != os.geteuid():
                raise ArtifactPolicyError("artifact cache path has the wrong owner")
            if stat.S_IMODE(current_stat.st_mode) & 0o022:
                raise ArtifactPolicyError("artifact cache path is writable by others")

    @contextmanager
    def _digest_lock(self, destination: Path) -> Iterator[None]:
        self._ensure_private_directory(destination.parent)
        lock_path = destination.with_name(f".{destination.name}.lock")
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise ArtifactPolicyError("cannot open artifact digest lock") from exc
        try:
            lock_stat = os.fstat(descriptor)
            if not stat.S_ISREG(lock_stat.st_mode):
                raise ArtifactPolicyError("artifact digest lock is not regular")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            os.close(descriptor)

    def verify(
        self, path: str | Path, reference: ArtifactReference
    ) -> VerifiedArtifact:
        candidate = Path(path)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(candidate, flags)
        except OSError as exc:
            raise ArtifactVerificationError(
                "artifact must be an accessible non-symlink file"
            ) from exc
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ArtifactVerificationError("artifact must be a regular file")
            if file_stat.st_size != reference.size_bytes:
                raise ArtifactVerificationError(
                    "artifact size does not match manifest"
                )
            digest = hashlib.sha256()
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != reference.sha256:
                raise ArtifactVerificationError(
                    "artifact digest does not match manifest"
                )
            final_stat = os.fstat(descriptor)
            if (
                final_stat.st_dev != file_stat.st_dev
                or final_stat.st_ino != file_stat.st_ino
                or final_stat.st_size != file_stat.st_size
                or final_stat.st_mtime_ns != file_stat.st_mtime_ns
            ):
                raise ArtifactVerificationError("artifact changed during verification")
            return VerifiedArtifact(
                artifact_id=reference.artifact_id,
                path=candidate,
                sha256=reference.sha256,
                size_bytes=reference.size_bytes,
                device=file_stat.st_dev,
                inode=file_stat.st_ino,
            )
        finally:
            os.close(descriptor)

    def prefetch(
        self,
        manifest: ModelArtifactManifest,
        approval: ArtifactApproval,
        artifact_id: str,
        transport: ArtifactTransport,
        *,
        mission_is_active: Callable[[], bool],
        timeout_s: float = 30.0,
    ) -> VerifiedArtifact:
        if not callable(mission_is_active):
            raise TypeError("mission_is_active must be callable")
        if mission_is_active():
            raise ArtifactPolicyError("artifact download is forbidden during a mission")
        if not manifest.prefetch_eligible:
            raise ArtifactPolicyError(
                "artifact license or review state blocks prefetch"
            )
        if not isinstance(approval, ArtifactApproval) or not approval.allows(
            manifest, artifact_id
        ):
            raise ArtifactPolicyError("trusted approval does not match this artifact")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(float(timeout_s))
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a positive finite number")

        reference = manifest.artifact(artifact_id)
        if reference.size_bytes > self._max_artifact_size_bytes:
            raise ArtifactPolicyError("artifact exceeds the configured size limit")
        destination = self.path_for(reference)
        with self._digest_lock(destination):
            if mission_is_active():
                raise ArtifactPolicyError(
                    "artifact download is forbidden during a mission"
                )
            if os.path.lexists(destination):
                try:
                    return self.verify(destination, reference)
                except ArtifactVerificationError:
                    if destination.is_dir():
                        raise ArtifactPolicyError(
                            "invalid cache destination is a directory"
                        )
                    destination.unlink(missing_ok=True)

            if shutil.disk_usage(destination.parent).free < reference.size_bytes:
                raise ArtifactPolicyError("insufficient free space for artifact")

            started = time.monotonic()
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=".download-",
                    dir=destination.parent,
                    delete=False,
                ) as output:
                    temp_path = Path(output.name)
                    digest = hashlib.sha256()
                    size = 0
                    with closing(
                        transport.open(reference.uri, timeout_s=float(timeout_s))
                    ) as stream:
                        while chunk := stream.read(1024 * 1024):
                            if not isinstance(chunk, bytes):
                                raise ArtifactVerificationError(
                                    "artifact transport returned non-byte content"
                                )
                            size += len(chunk)
                            if size > reference.size_bytes:
                                raise ArtifactVerificationError(
                                    "download exceeds declared artifact size"
                                )
                            if time.monotonic() - started > timeout_s:
                                raise ArtifactPolicyError(
                                    "artifact download exceeded its total deadline"
                                )
                            if mission_is_active():
                                raise ArtifactPolicyError(
                                    "mission started during artifact download"
                                )
                            digest.update(chunk)
                            output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())

                    if size != reference.size_bytes:
                        raise ArtifactVerificationError(
                            "download size does not match artifact manifest"
                        )
                    if digest.hexdigest() != reference.sha256:
                        raise ArtifactVerificationError(
                            "download digest does not match artifact manifest"
                        )
                    if mission_is_active():
                        raise ArtifactPolicyError(
                            "mission started before artifact publication"
                        )
                    os.fchmod(output.fileno(), 0o400)
                    os.replace(temp_path, destination)
                    temp_path = None

                directory_fd = os.open(
                    destination.parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                return self.verify(destination, reference)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

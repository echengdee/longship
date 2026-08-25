"""Decoded goal-image boundary for the NoMaD Longship adapter."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

from longship.navigation.map_engine.models import (
    ResourceDescriptor,
    ResourceKind,
)


@dataclass(frozen=True, slots=True)
class DecodedImage:
    """One decoded image plus representation metadata for NoMaD ingress."""

    image: object
    layout: str
    channel_order: str
    value_range: str


@runtime_checkable
class GoalImageLoader(Protocol):
    """Loads a map resource without exposing locator semantics to callers."""

    def load(self, resource: ResourceDescriptor) -> DecodedImage: ...


class LocalFileGoalImageLoader:
    """Loads immutable local PNG/JPEG resources from configured roots."""

    def __init__(self, allowed_roots: tuple[Path, ...]) -> None:
        if not allowed_roots:
            raise ValueError("at least one goal-image root must be allowed")
        self._allowed_roots = tuple(
            root.expanduser().resolve() for root in allowed_roots
        )
        for root in self._allowed_roots:
            if not root.is_dir():
                raise FileNotFoundError(f"goal-image root is not a directory: {root}")
        self._cache: dict[tuple[str, str], DecodedImage] = {}

    def load(self, resource: ResourceDescriptor) -> DecodedImage:
        if resource.kind != ResourceKind.IMAGE:
            raise ValueError(f"resource is not an image: {resource.resource_id}")
        if resource.content_digest is None:
            raise ValueError(f"image resource has no digest: {resource.resource_id}")
        cache_key = (str(resource.resource_id), resource.content_digest)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        path = Path(resource.locator).expanduser().resolve()
        if not any(
            path == root or path.is_relative_to(root) for root in self._allowed_roots
        ):
            raise PermissionError(
                f"image resource is outside configured roots: {resource.resource_id}"
            )
        if not path.is_file():
            raise FileNotFoundError(f"goal image not found: {path}")
        if (
            resource.size_bytes is not None
            and path.stat().st_size != resource.size_bytes
        ):
            raise ValueError(f"goal image size changed: {resource.resource_id}")
        actual_digest = _file_digest(path)
        if actual_digest != resource.content_digest:
            raise ValueError(f"goal image digest changed: {resource.resource_id}")

        try:
            from PIL import Image
            import torch
        except ImportError as error:
            raise RuntimeError(
                "local goal-image loading requires Pillow and PyTorch"
            ) from error

        with Image.open(path) as source:
            image = source.convert("RGB")
            storage = bytearray(image.tobytes())
            tensor = (
                torch.frombuffer(storage, dtype=torch.uint8)
                .clone()
                .view(
                    image.height,
                    image.width,
                    3,
                )
            )
        decoded = DecodedImage(
            image=tensor,
            layout="hwc",
            channel_order="rgb",
            value_range="byte",
        )
        self._cache[cache_key] = decoded
        return decoded


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"

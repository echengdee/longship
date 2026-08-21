"""Public Map Engine facade for the Longship navigation harness."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    AnchorQuery,
    AnchorQueryResult,
    MapErrorCode,
    MapSelector,
    MapSnapshot,
    PlaceQuery,
    PlaceQueryResult,
    ResourceId,
    ResourceQueryResult,
    TopologyQuery,
    TopologyQueryResult,
)


class MapEngineError(RuntimeError):
    """Structured failure raised by a MapEngine implementation."""

    def __init__(
        self,
        code: MapErrorCode,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@runtime_checkable
class MapEngine(Protocol):
    """Read-only, version-consistent access to navigation map knowledge."""

    async def get_snapshot(self, selector: MapSelector) -> MapSnapshot:
        """Resolve and pin one immutable published map version."""
        ...

    async def query_places(
        self,
        snapshot: MapSnapshot,
        query: PlaceQuery,
    ) -> PlaceQueryResult:
        """Return ranked place candidates; zero matches is a valid result."""
        ...

    async def query_topology(
        self,
        snapshot: MapSnapshot,
        query: TopologyQuery,
    ) -> TopologyQueryResult:
        """Return a full graph or bounded topological view."""
        ...

    async def query_anchors(
        self,
        snapshot: MapSnapshot,
        query: AnchorQuery,
    ) -> AnchorQueryResult:
        """Return localization, target, entry, exit, or completion anchors."""
        ...

    async def resolve_resources(
        self,
        snapshot: MapSnapshot,
        resource_ids: tuple[ResourceId, ...],
    ) -> ResourceQueryResult:
        """Resolve stable resource ids into opaque runtime locators."""
        ...

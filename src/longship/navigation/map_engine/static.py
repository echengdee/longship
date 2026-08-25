"""Immutable in-process Map Engine implementation.

This module contains generic query behavior only. Artifact-specific readers
construct a ``StaticMap`` without making the core navigation package depend on
an external map format or plugin.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TypeVar

from .interface import MapEngineError
from .models import (
    AnchorDescriptor,
    AnchorId,
    AnchorQuery,
    AnchorQueryResult,
    MapEntityRef,
    MapErrorCode,
    MapSelector,
    MapSnapshot,
    NodeId,
    PlaceDescriptor,
    PlaceMatch,
    PlaceQuery,
    PlaceQueryResult,
    ResourceDescriptor,
    ResourceId,
    ResourceQueryResult,
    SegmentDescriptor,
    SegmentId,
    TopologyNode,
    TopologyQuery,
    TopologyQueryResult,
)


_Item = TypeVar("_Item")


@dataclass(frozen=True, slots=True)
class StaticMap:
    """One fully loaded, immutable map publication."""

    snapshot: MapSnapshot
    places: tuple[PlaceDescriptor, ...] = ()
    nodes: tuple[TopologyNode, ...] = ()
    segments: tuple[SegmentDescriptor, ...] = ()
    anchors: tuple[AnchorDescriptor, ...] = ()
    resources: tuple[ResourceDescriptor, ...] = ()


class StaticMapEngine:
    """Read-only Map Engine backed by one pinned ``StaticMap``."""

    def __init__(self, map_document: StaticMap) -> None:
        self._map = map_document
        self._places = self._index_unique(
            map_document.places,
            lambda place: str(place.place_id),
            "place",
        )
        self._nodes = self._index_unique(
            map_document.nodes,
            lambda node: str(node.node_id),
            "node",
        )
        self._segments = self._index_unique(
            map_document.segments,
            lambda segment: str(segment.segment_id),
            "segment",
        )
        self._anchors = self._index_unique(
            map_document.anchors,
            lambda anchor: str(anchor.anchor_id),
            "anchor",
        )
        self._resources = self._index_unique(
            map_document.resources,
            lambda resource: str(resource.resource_id),
            "resource",
        )
        self._validate_references()

    @staticmethod
    def _index_unique(
        items: tuple[_Item, ...],
        key: Callable[[_Item], str],
        label: str,
    ) -> dict[str, _Item]:
        indexed: dict[str, _Item] = {}
        for item in items:
            item_id = key(item)
            if item_id in indexed:
                raise ValueError(f"duplicate {label} id: {item_id}")
            indexed[item_id] = item
        return indexed

    def _validate_references(self) -> None:
        for place in self._map.places:
            for node_id in place.target_node_ids:
                if str(node_id) not in self._nodes:
                    raise ValueError(
                        f"place {place.place_id} references missing node "
                        f"{node_id}"
                    )
            self._validate_anchor_ids(
                place.anchor_ids,
                f"place {place.place_id}",
            )
        for node in self._map.nodes:
            for place_id in node.place_ids:
                if str(place_id) not in self._places:
                    raise ValueError(
                        f"node {node.node_id} references missing place "
                        f"{place_id}"
                    )
            self._validate_anchor_ids(
                node.anchor_ids,
                f"node {node.node_id}",
            )
        for segment in self._map.segments:
            for node_id in (
                segment.source_node_id,
                segment.target_node_id,
            ):
                if str(node_id) not in self._nodes:
                    raise ValueError(
                        f"segment {segment.segment_id} references missing "
                        f"node {node_id}"
                    )
            self._validate_anchor_ids(
                segment.anchor_ids,
                f"segment {segment.segment_id}",
            )
            for resource_id in segment.resource_ids:
                if str(resource_id) not in self._resources:
                    raise ValueError(
                        f"segment {segment.segment_id} references missing "
                        f"resource {resource_id}"
                    )
        for anchor in self._map.anchors:
            self._validate_entity_ref(anchor.attached_to)
            for resource_id in anchor.resource_ids:
                if str(resource_id) not in self._resources:
                    raise ValueError(
                        f"anchor {anchor.anchor_id} references missing "
                        f"resource {resource_id}"
                    )

    def _validate_anchor_ids(
        self,
        anchor_ids: tuple[AnchorId, ...],
        owner: str,
    ) -> None:
        for anchor_id in anchor_ids:
            if str(anchor_id) not in self._anchors:
                raise ValueError(
                    f"{owner} references missing anchor {anchor_id}"
                )

    def _validate_entity_ref(self, entity: MapEntityRef) -> None:
        entities = {
            "place": self._places,
            "node": self._nodes,
            "segment": self._segments,
        }[entity.kind.value]
        if entity.entity_id not in entities:
            raise ValueError(
                f"anchor references missing {entity.kind.value} "
                f"{entity.entity_id}"
            )

    def _require_snapshot(self, snapshot: MapSnapshot) -> None:
        expected = self._map.snapshot
        if (
            snapshot.snapshot_id != expected.snapshot_id
            or snapshot.content_digest != expected.content_digest
        ):
            raise MapEngineError(
                MapErrorCode.SNAPSHOT_NOT_FOUND,
                f"map snapshot is not loaded: {snapshot.snapshot_id}",
            )

    async def get_snapshot(self, selector: MapSelector) -> MapSnapshot:
        snapshot = self._map.snapshot
        if selector.map_id != snapshot.map_id:
            raise MapEngineError(
                MapErrorCode.MAP_NOT_FOUND,
                f"map was not found: {selector.map_id}",
            )
        if selector.version is not None and selector.version != snapshot.version:
            raise MapEngineError(
                MapErrorCode.VERSION_NOT_FOUND,
                f"map version was not found: {selector.version}",
            )
        missing_capabilities = (
            selector.required_capabilities - snapshot.capabilities
        )
        if missing_capabilities:
            missing = ", ".join(
                sorted(capability.value for capability in missing_capabilities)
            )
            raise MapEngineError(
                MapErrorCode.CAPABILITY_UNAVAILABLE,
                f"map does not provide required capabilities: {missing}",
            )
        return snapshot

    async def query_places(
        self,
        snapshot: MapSnapshot,
        query: PlaceQuery,
    ) -> PlaceQueryResult:
        self._require_snapshot(snapshot)
        if query.limit <= 0:
            raise MapEngineError(
                MapErrorCode.INVALID_QUERY,
                "place query limit must be positive",
            )

        selected_ids = {str(place_id) for place_id in query.place_ids}
        normalized_text = query.text.casefold().strip() if query.text else None
        matches = []
        for place in self._map.places:
            if selected_ids and str(place.place_id) not in selected_ids:
                continue
            if query.kinds and place.kind not in query.kinds:
                continue
            if not query.required_tags.issubset(place.tags):
                continue

            matched_fields = []
            rank_score = 1.0
            if normalized_text:
                canonical_name = place.canonical_name.casefold()
                aliases = tuple(alias.casefold() for alias in place.aliases)
                if normalized_text == canonical_name:
                    matched_fields.append("canonical_name")
                    rank_score = 1.0
                elif normalized_text in aliases:
                    matched_fields.append("alias")
                    rank_score = 0.95
                elif normalized_text in canonical_name:
                    matched_fields.append("canonical_name")
                    rank_score = 0.8
                elif any(normalized_text in alias for alias in aliases):
                    matched_fields.append("alias")
                    rank_score = 0.75
                else:
                    continue
            matches.append(
                PlaceMatch(
                    place=place,
                    rank_score=rank_score,
                    matched_fields=tuple(matched_fields),
                )
            )

        matches.sort(
            key=lambda match: (
                -match.rank_score,
                match.place.canonical_name.casefold(),
                str(match.place.place_id),
            )
        )
        return PlaceQueryResult(
            snapshot_id=snapshot.snapshot_id,
            matches=tuple(matches[: query.limit]),
            truncated=len(matches) > query.limit,
        )

    async def query_topology(
        self,
        snapshot: MapSnapshot,
        query: TopologyQuery,
    ) -> TopologyQueryResult:
        self._require_snapshot(snapshot)
        if query.expand_hops < 0:
            raise MapEngineError(
                MapErrorCode.INVALID_QUERY,
                "topology expand_hops must be non-negative",
            )

        requested_node_ids = {str(node_id) for node_id in query.node_ids}
        requested_segment_ids = {
            str(segment_id) for segment_id in query.segment_ids
        }
        missing_node_ids = tuple(
            NodeId(node_id)
            for node_id in self._missing_in_order(
                (str(node_id) for node_id in query.node_ids),
                self._nodes,
            )
        )
        missing_segment_ids = tuple(
            SegmentId(segment_id)
            for segment_id in self._missing_in_order(
                (str(segment_id) for segment_id in query.segment_ids),
                self._segments,
            )
        )

        if not requested_node_ids and not requested_segment_ids:
            selected_node_ids = set(self._nodes)
            selected_segment_ids = set(self._segments)
        else:
            selected_node_ids = requested_node_ids & self._nodes.keys()
            selected_segment_ids = (
                requested_segment_ids & self._segments.keys()
            )
            for segment_id in selected_segment_ids:
                segment = self._segments[segment_id]
                selected_node_ids.update(
                    (
                        str(segment.source_node_id),
                        str(segment.target_node_id),
                    )
                )
            for _ in range(query.expand_hops):
                for segment_id, segment in self._segments.items():
                    endpoints = {
                        str(segment.source_node_id),
                        str(segment.target_node_id),
                    }
                    if endpoints & selected_node_ids:
                        selected_segment_ids.add(segment_id)
                        selected_node_ids.update(endpoints)

        segments = tuple(
            segment
            for segment in self._map.segments
            if str(segment.segment_id) in selected_segment_ids
            and (query.include_disabled or segment.availability.value == "enabled")
        )
        nodes = tuple(
            node
            for node in self._map.nodes
            if str(node.node_id) in selected_node_ids
        )
        return TopologyQueryResult(
            snapshot_id=snapshot.snapshot_id,
            nodes=nodes,
            segments=segments,
            missing_node_ids=missing_node_ids,
            missing_segment_ids=missing_segment_ids,
        )

    async def query_anchors(
        self,
        snapshot: MapSnapshot,
        query: AnchorQuery,
    ) -> AnchorQueryResult:
        self._require_snapshot(snapshot)
        if query.limit <= 0:
            raise MapEngineError(
                MapErrorCode.INVALID_QUERY,
                "anchor query limit must be positive",
            )

        selected_ids = {str(anchor_id) for anchor_id in query.anchor_ids}
        attached_to = {
            (entity.kind, entity.entity_id) for entity in query.attached_to
        }
        anchors = []
        for anchor in self._map.anchors:
            if selected_ids and str(anchor.anchor_id) not in selected_ids:
                continue
            if attached_to and (
                anchor.attached_to.kind,
                anchor.attached_to.entity_id,
            ) not in attached_to:
                continue
            if query.kinds and anchor.kind not in query.kinds:
                continue
            if query.purposes and not query.purposes.issubset(anchor.purposes):
                continue
            anchors.append(anchor)

        missing_anchor_ids = tuple(
            AnchorId(anchor_id)
            for anchor_id in self._missing_in_order(
                (str(anchor_id) for anchor_id in query.anchor_ids),
                self._anchors,
            )
        )
        return AnchorQueryResult(
            snapshot_id=snapshot.snapshot_id,
            anchors=tuple(anchors[: query.limit]),
            missing_anchor_ids=missing_anchor_ids,
            truncated=len(anchors) > query.limit,
        )

    async def resolve_resources(
        self,
        snapshot: MapSnapshot,
        resource_ids: tuple[ResourceId, ...],
    ) -> ResourceQueryResult:
        self._require_snapshot(snapshot)
        resources = []
        missing = []
        for resource_id in resource_ids:
            resource = self._resources.get(str(resource_id))
            if resource is None:
                missing.append(resource_id)
            else:
                resources.append(resource)
        return ResourceQueryResult(
            snapshot_id=snapshot.snapshot_id,
            resources=tuple(resources),
            missing_resource_ids=tuple(missing),
        )

    @staticmethod
    def _missing_in_order(
        requested_ids: Iterable[str],
        available: Mapping[str, object],
    ) -> tuple[str, ...]:
        missing = []
        seen = set()
        for item_id in requested_ids:
            if item_id not in available and item_id not in seen:
                missing.append(item_id)
                seen.add(item_id)
        return tuple(missing)

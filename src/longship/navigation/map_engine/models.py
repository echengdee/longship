"""Public data models for Map Engine v0.1.

The models in this file are transport-neutral.  They must not depend on ROS 2,
database clients, geometry services, or a particular map implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, NewType, TypeAlias

from longship.navigation.common import Pose3D, TimePoint


MapId = NewType("MapId", str)
MapVersion = NewType("MapVersion", str)
SnapshotId = NewType("SnapshotId", str)
PlaceId = NewType("PlaceId", str)
NodeId = NewType("NodeId", str)
SegmentId = NewType("SegmentId", str)
AnchorId = NewType("AnchorId", str)
ResourceId = NewType("ResourceId", str)

Scalar: TypeAlias = str | int | float | bool | None
Attributes: TypeAlias = Mapping[str, Scalar]


class MapCapability(str, Enum):
    SEMANTIC_PLACES = "semantic_places"
    TOPOLOGY = "topology"
    VISUAL_ANCHORS = "visual_anchors"
    METRIC_ANCHORS = "metric_anchors"
    SEGMENT_METADATA = "segment_metadata"
    RESOURCE_REFERENCES = "resource_references"


class MapEntityKind(str, Enum):
    PLACE = "place"
    NODE = "node"
    SEGMENT = "segment"


class SegmentAvailability(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"


class AnchorKind(str, Enum):
    VISUAL = "visual"
    METRIC = "metric"
    GEOMETRIC = "geometric"
    SEMANTIC = "semantic"
    CUSTOM = "custom"


class AnchorPurpose(str, Enum):
    LOCALIZATION = "localization"
    ENTRY = "entry"
    EXIT = "exit"
    TARGET = "target"
    COMPLETION = "completion"


class ResourceKind(str, Enum):
    IMAGE = "image"
    IMAGE_SEQUENCE = "image_sequence"
    METRIC_MAP = "metric_map"
    GEOMETRY = "geometry"
    MODEL_INPUT = "model_input"
    CUSTOM = "custom"


class MapErrorCode(str, Enum):
    MAP_NOT_FOUND = "map_not_found"
    VERSION_NOT_FOUND = "version_not_found"
    SNAPSHOT_NOT_FOUND = "snapshot_not_found"
    INCOMPATIBLE_SCHEMA = "incompatible_schema"
    INVALID_QUERY = "invalid_query"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    CORRUPT_MAP = "corrupt_map"


@dataclass(frozen=True, slots=True)
class MapSelector:
    """Select a published map version.

    ``version=None`` means "resolve the current published version once".  The
    returned snapshot always contains a concrete version and never follows a
    later publication automatically.
    """

    map_id: MapId
    version: MapVersion | None = None
    required_capabilities: frozenset[MapCapability] = frozenset()


@dataclass(frozen=True, slots=True)
class MapSnapshot:
    """Immutable identity and capabilities of one map version.

    This is a version token, not an in-memory copy of all map data.
    """

    snapshot_id: SnapshotId
    map_id: MapId
    version: MapVersion
    schema_version: str
    content_digest: str
    published_at: TimePoint
    map_frame: str | None
    capabilities: frozenset[MapCapability]


@dataclass(frozen=True, slots=True)
class PlaceQuery:
    text: str | None = None
    place_ids: tuple[PlaceId, ...] = ()
    kinds: tuple[str, ...] = ()
    required_tags: frozenset[str] = frozenset()
    limit: int = 20


@dataclass(frozen=True, slots=True)
class PlaceDescriptor:
    place_id: PlaceId
    canonical_name: str
    aliases: tuple[str, ...] = ()
    kind: str | None = None
    target_node_ids: tuple[NodeId, ...] = ()
    anchor_ids: tuple[AnchorId, ...] = ()
    tags: frozenset[str] = frozenset()
    attributes: Attributes = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlaceMatch:
    place: PlaceDescriptor
    rank_score: float
    matched_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlaceQueryResult:
    snapshot_id: SnapshotId
    matches: tuple[PlaceMatch, ...]
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class TopologyQuery:
    """Request the full graph or a bounded view of it.

    With no node or segment ids, the full published navigation graph is
    returned. ``expand_hops`` expands around requested nodes/segments.
    """

    node_ids: tuple[NodeId, ...] = ()
    segment_ids: tuple[SegmentId, ...] = ()
    expand_hops: int = 0
    include_disabled: bool = False


@dataclass(frozen=True, slots=True)
class TopologyNode:
    node_id: NodeId
    place_ids: tuple[PlaceId, ...] = ()
    anchor_ids: tuple[AnchorId, ...] = ()
    tags: frozenset[str] = frozenset()
    attributes: Attributes = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SegmentDescriptor:
    """One directed, traversable topological connection."""

    segment_id: SegmentId
    source_node_id: NodeId
    target_node_id: NodeId
    availability: SegmentAvailability = SegmentAvailability.ENABLED
    length_m: float | None = None
    nominal_duration_s: float | None = None
    speed_hint_mps: float | None = None
    required_capabilities: frozenset[str] = frozenset()
    anchor_ids: tuple[AnchorId, ...] = ()
    resource_ids: tuple[ResourceId, ...] = ()
    tags: frozenset[str] = frozenset()
    attributes: Attributes = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TopologyQueryResult:
    snapshot_id: SnapshotId
    nodes: tuple[TopologyNode, ...]
    segments: tuple[SegmentDescriptor, ...]
    missing_node_ids: tuple[NodeId, ...] = ()
    missing_segment_ids: tuple[SegmentId, ...] = ()


@dataclass(frozen=True, slots=True)
class MapEntityRef:
    kind: MapEntityKind
    entity_id: str


@dataclass(frozen=True, slots=True)
class AnchorQuery:
    anchor_ids: tuple[AnchorId, ...] = ()
    attached_to: tuple[MapEntityRef, ...] = ()
    kinds: frozenset[AnchorKind] = frozenset()
    purposes: frozenset[AnchorPurpose] = frozenset()
    limit: int = 200


@dataclass(frozen=True, slots=True)
class AnchorDescriptor:
    anchor_id: AnchorId
    kind: AnchorKind
    purposes: frozenset[AnchorPurpose]
    attached_to: MapEntityRef
    pose: Pose3D | None = None
    resource_ids: tuple[ResourceId, ...] = ()
    tags: frozenset[str] = frozenset()
    attributes: Attributes = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnchorQueryResult:
    snapshot_id: SnapshotId
    anchors: tuple[AnchorDescriptor, ...]
    missing_anchor_ids: tuple[AnchorId, ...] = ()
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class ResourceDescriptor:
    """Metadata and an opaque locator for a heavy map resource.

    Consumers must not parse ``locator``. They pass it to the configured
    resource/geometry adapter appropriate for ``kind``.
    """

    resource_id: ResourceId
    kind: ResourceKind
    locator: str
    media_type: str | None = None
    content_digest: str | None = None
    size_bytes: int | None = None
    attributes: Attributes = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResourceQueryResult:
    snapshot_id: SnapshotId
    resources: tuple[ResourceDescriptor, ...]
    missing_resource_ids: tuple[ResourceId, ...] = ()

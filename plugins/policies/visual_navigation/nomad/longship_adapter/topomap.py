"""Adapts a NoMaD adaptive topomap to the Longship Map Engine contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from longship.navigation.common import TimePoint
from longship.navigation.map_engine.models import (
    AnchorDescriptor,
    AnchorId,
    AnchorKind,
    AnchorPurpose,
    MapCapability,
    MapEntityKind,
    MapEntityRef,
    MapId,
    MapSnapshot,
    MapVersion,
    NodeId,
    ResourceDescriptor,
    ResourceId,
    ResourceKind,
    SegmentDescriptor,
    SegmentId,
    SnapshotId,
    TopologyNode,
)
from longship.navigation.map_engine.static import StaticMap, StaticMapEngine


_FORMAT_VERSION = 1
_SCHEMA_VERSION = "longship.nomad_topomap.v0.1"
_MANIFEST_FILENAME = "manifest.json"
_EDGES_FILENAME = "edges.json"
_SUMMARY_FILENAME = "summary.json"
_IMAGES_DIRECTORY = "images"


@dataclass(frozen=True, slots=True)
class NomadTopomapMapConfig:
    """Identity and compatibility information absent from the topomap files."""

    root: Path
    map_id: MapId
    version: MapVersion
    published_at: TimePoint
    model_artifact_id: str
    model_artifact_digest: str
    image_profile_id: str = "nomad.rgb.direct_resize_96x96.imagenet.v1"
    model_input_width: int = 96
    model_input_height: int = 96
    normalization_profile_id: str = "imagenet"
    expected_center_crop_aspect: float | None = None
    map_frame: str | None = None


def create_nomad_topomap_engine(
    config: NomadTopomapMapConfig,
) -> StaticMapEngine:
    """Loads and pins one NoMaD topomap behind a Map Engine facade."""

    return StaticMapEngine(load_nomad_topomap(config))


def load_nomad_topomap(config: NomadTopomapMapConfig) -> StaticMap:
    """Loads, validates, and converts one adaptive topomap publication."""

    _validate_config(config)
    root = config.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"NoMaD topomap directory not found: {root}")

    manifest_path = root / _MANIFEST_FILENAME
    edges_path = root / _EDGES_FILENAME
    summary_path = root / _SUMMARY_FILENAME
    images_directory = root / _IMAGES_DIRECTORY
    for path in (manifest_path, edges_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"NoMaD topomap file not found: {path}")
    if not images_directory.is_dir():
        raise FileNotFoundError(
            f"NoMaD topomap image directory not found: {images_directory}"
        )

    manifest = _require_mapping(_read_json(manifest_path), "manifest")
    edges = _require_sequence(_read_json(edges_path), "edges")
    summary = _require_mapping(_read_json(summary_path), "summary")
    _validate_header(manifest, summary, config)

    node_rows = _require_sequence(manifest.get("nodes"), "manifest.nodes")
    edge_rows = tuple(
        _require_mapping(edge, f"edges[{index}]")
        for index, edge in enumerate(edges)
    )
    _validate_graph(node_rows, edge_rows, manifest, summary)

    nodes = []
    anchors = []
    resources = []
    image_paths = []
    last_topology_node = len(node_rows) - 1
    for expected_index, value in enumerate(node_rows):
        row = _require_mapping(value, f"manifest.nodes[{expected_index}]")
        topology_index = _required_int(row, "topology_node")
        filename = _required_string(row, "filename")
        image_path = _resolve_image(images_directory, filename)
        image_paths.append(image_path)

        node_id = _node_id(topology_index)
        anchor_id = AnchorId(f"{node_id}:visual")
        resource_id = ResourceId(f"{node_id}:goal-image")
        purposes = {
            AnchorPurpose.LOCALIZATION,
            AnchorPurpose.TARGET,
        }
        tags = {"nomad", "visual-keyframe"}
        if topology_index == 0:
            purposes.add(AnchorPurpose.ENTRY)
            tags.add("route-start")
        if topology_index == last_topology_node:
            purposes.add(AnchorPurpose.COMPLETION)
            tags.add("route-end")

        common_attributes = {
            "topology_index": topology_index,
            "source_position": _required_int(row, "source_position"),
            "source_node": _required_int(row, "source_node"),
            "source_filename": _required_string(row, "source_filename"),
            "source_timestamp_s": _required_float(row, "time_s"),
            "sharpness": _optional_float(row, "sharpness"),
            "image_profile_id": config.image_profile_id,
        }
        nodes.append(
            TopologyNode(
                node_id=node_id,
                anchor_ids=(anchor_id,),
                tags=frozenset(tags),
                attributes=common_attributes,
            )
        )
        anchors.append(
            AnchorDescriptor(
                anchor_id=anchor_id,
                kind=AnchorKind.VISUAL,
                purposes=frozenset(purposes),
                attached_to=MapEntityRef(
                    kind=MapEntityKind.NODE,
                    entity_id=str(node_id),
                ),
                resource_ids=(resource_id,),
                tags=frozenset(tags),
                attributes=common_attributes,
            )
        )
        resources.append(
            ResourceDescriptor(
                resource_id=resource_id,
                kind=ResourceKind.IMAGE,
                locator=str(image_path),
                media_type=_image_media_type(image_path),
                content_digest=_file_digest(image_path),
                size_bytes=image_path.stat().st_size,
                attributes={
                    "image_profile_id": config.image_profile_id,
                    "color_space": "rgb",
                    "preprocessing_mode": _preprocessing_mode(config),
                    "center_crop_aspect": config.expected_center_crop_aspect,
                    "model_input_width": config.model_input_width,
                    "model_input_height": config.model_input_height,
                    "normalization_profile_id": (
                        config.normalization_profile_id
                    ),
                    "model_artifact_id": config.model_artifact_id,
                    "model_artifact_digest": _canonical_sha256(
                        config.model_artifact_digest
                    ),
                },
            )
        )

    selection = _require_mapping(manifest.get("selection"), "selection")
    hard_minimum = _required_float(selection, "hard_min_distance")
    hard_maximum = _required_float(selection, "hard_max_distance")
    if hard_minimum > hard_maximum:
        raise ValueError("hard distance range is inverted")
    segments = tuple(
        _convert_edge(row, hard_minimum, hard_maximum)
        for row in edge_rows
    )

    content_digest = _topomap_digest(
        config,
        (manifest_path, edges_path, summary_path, *image_paths),
    )
    snapshot = MapSnapshot(
        snapshot_id=SnapshotId(
            f"{config.map_id}:{config.version}:{content_digest[7:23]}"
        ),
        map_id=config.map_id,
        version=config.version,
        schema_version=_SCHEMA_VERSION,
        content_digest=content_digest,
        published_at=config.published_at,
        map_frame=config.map_frame,
        capabilities=frozenset(
            {
                MapCapability.TOPOLOGY,
                MapCapability.VISUAL_ANCHORS,
                MapCapability.SEGMENT_METADATA,
                MapCapability.RESOURCE_REFERENCES,
            }
        ),
    )
    return StaticMap(
        snapshot=snapshot,
        nodes=tuple(nodes),
        segments=segments,
        anchors=tuple(anchors),
        resources=tuple(resources),
    )


def _convert_edge(
    row: Mapping[str, object],
    hard_minimum: float,
    hard_maximum: float,
) -> SegmentDescriptor:
    edge_index = _required_int(row, "edge")
    source_index = _required_int(row, "source_topology_node")
    target_index = _required_int(row, "target_topology_node")
    predicted_distance = _required_float(row, "predicted_distance")
    minimum_distance = _required_float(row, "minimum_predicted_distance")
    maximum_distance = _required_float(row, "maximum_predicted_distance")
    selection_reason = _required_string(row, "selection_reason")

    if selection_reason == "terminal_short_edge":
        offline_check = "terminal"
    elif (
        predicted_distance < hard_minimum
        or predicted_distance > hard_maximum
        or minimum_distance < hard_minimum
        or maximum_distance > hard_maximum
    ):
        offline_check = "needs_review"
    else:
        offline_check = "passed"

    return SegmentDescriptor(
        segment_id=SegmentId(f"edge-{edge_index:04d}"),
        source_node_id=_node_id(source_index),
        target_node_id=_node_id(target_index),
        tags=frozenset(
            {
                "nomad",
                "forward-only",
                f"offline-model-check:{offline_check}",
                "hardware-unqualified",
            }
        ),
        attributes={
            "edge_index": edge_index,
            "source_position": _required_int(row, "source_position"),
            "target_position": _required_int(row, "target_position"),
            "source_dense_node": _required_int(row, "source_node"),
            "target_dense_node": _required_int(row, "target_node"),
            "source_timestamp_s": _required_float(row, "source_time_s"),
            "target_timestamp_s": _required_float(row, "target_time_s"),
            "offline_time_delta_s": _required_float(row, "time_delta_s"),
            "offline_predicted_distance": predicted_distance,
            "offline_minimum_predicted_distance": minimum_distance,
            "offline_maximum_predicted_distance": maximum_distance,
            "selection_reason": selection_reason,
            "candidate_count": _required_int(row, "candidate_count"),
            "offline_model_check_status": offline_check,
            "hardware_qualification_status": "unqualified",
        },
    )


def _validate_config(config: NomadTopomapMapConfig) -> None:
    text_fields = {
        "map_id": str(config.map_id),
        "version": str(config.version),
        "model_artifact_id": config.model_artifact_id,
        "image_profile_id": config.image_profile_id,
        "normalization_profile_id": config.normalization_profile_id,
    }
    for name, value in text_fields.items():
        if not value.strip():
            raise ValueError(f"{name} must not be empty")
    digest = config.model_artifact_digest.removeprefix("sha256:")
    if len(digest) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in digest
    ):
        raise ValueError("model_artifact_digest must be a SHA-256 digest")
    if config.model_input_width <= 0 or config.model_input_height <= 0:
        raise ValueError("model input dimensions must be positive")
    if (
        config.expected_center_crop_aspect is not None
        and config.expected_center_crop_aspect <= 0.0
    ):
        raise ValueError("expected_center_crop_aspect must be positive")


def _validate_header(
    manifest: Mapping[str, object],
    summary: Mapping[str, object],
    config: NomadTopomapMapConfig,
) -> None:
    if _required_int(manifest, "format_version") != _FORMAT_VERSION:
        raise ValueError("unsupported NoMaD topomap format_version")
    manifest_summary = _require_mapping(
        manifest.get("summary"), "manifest.summary"
    )
    for field in ("topology_node_count", "edge_count"):
        if _required_int(manifest_summary, field) != _required_int(
            summary, field
        ):
            raise ValueError(f"summary mismatch for {field}")

    selection = _require_mapping(manifest.get("selection"), "selection")
    center_crop_aspect = _optional_float(selection, "center_crop_aspect")
    expected = config.expected_center_crop_aspect
    if center_crop_aspect is None and expected is None:
        return
    if center_crop_aspect is None or expected is None:
        raise ValueError(
            "topomap center-crop profile does not match configured profile"
        )
    if not math.isclose(center_crop_aspect, expected, rel_tol=1e-9):
        raise ValueError(
            "topomap center-crop profile does not match configured profile"
        )


def _validate_graph(
    node_rows: Sequence[object],
    edge_rows: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
    summary: Mapping[str, object],
) -> None:
    if not node_rows:
        raise ValueError("NoMaD topomap must contain at least one node")
    expected_edge_count = max(0, len(node_rows) - 1)
    if len(edge_rows) != expected_edge_count:
        raise ValueError(
            "NoMaD sequential topomap must contain exactly node_count - 1 "
            "edges"
        )

    nodes = tuple(
        _require_mapping(value, f"manifest.nodes[{index}]")
        for index, value in enumerate(node_rows)
    )
    previous_time = None
    previous_source_position = None
    filenames = set()
    for index, row in enumerate(nodes):
        if _required_int(row, "topology_node") != index:
            raise ValueError("topology node ids must be contiguous from zero")
        filename = _required_string(row, "filename")
        if filename in filenames:
            raise ValueError(f"duplicate topology image filename: {filename}")
        filenames.add(filename)
        source_position = _required_int(row, "source_position")
        if (
            previous_source_position is not None
            and source_position <= previous_source_position
        ):
            raise ValueError("topology source positions must be increasing")
        previous_source_position = source_position
        timestamp = _required_float(row, "time_s")
        if previous_time is not None and timestamp <= previous_time:
            raise ValueError("topology node timestamps must be increasing")
        previous_time = timestamp

    for index, row in enumerate(edge_rows):
        if _required_int(row, "edge") != index:
            raise ValueError("edge ids must be contiguous from zero")
        if _required_int(row, "source_topology_node") != index:
            raise ValueError("edge source must match the sequential node")
        if _required_int(row, "target_topology_node") != index + 1:
            raise ValueError("edge target must be the next sequential node")
        if _required_float(row, "time_delta_s") <= 0.0:
            raise ValueError("edge time_delta_s must be positive")
        source = nodes[index]
        target = nodes[index + 1]
        integer_pairs = (
            ("source_position", source, "source_position"),
            ("target_position", target, "source_position"),
            ("source_node", source, "source_node"),
            ("target_node", target, "source_node"),
        )
        for edge_field, node, node_field in integer_pairs:
            if _required_int(row, edge_field) != _required_int(
                node, node_field
            ):
                raise ValueError(
                    f"edge {index} {edge_field} does not match its node"
                )
        source_time = _required_float(row, "source_time_s")
        target_time = _required_float(row, "target_time_s")
        if not math.isclose(
            source_time,
            _required_float(source, "time_s"),
            rel_tol=1e-9,
        ):
            raise ValueError(f"edge {index} source_time_s is inconsistent")
        if not math.isclose(
            target_time,
            _required_float(target, "time_s"),
            rel_tol=1e-9,
        ):
            raise ValueError(f"edge {index} target_time_s is inconsistent")
        if not math.isclose(
            target_time - source_time,
            _required_float(row, "time_delta_s"),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(f"edge {index} time_delta_s is inconsistent")

    manifest_summary = _require_mapping(
        manifest.get("summary"), "manifest.summary"
    )
    for document in (manifest_summary, summary):
        if _required_int(document, "topology_node_count") != len(node_rows):
            raise ValueError("summary topology_node_count is inconsistent")
        if _required_int(document, "edge_count") != len(edge_rows):
            raise ValueError("summary edge_count is inconsistent")


def _resolve_image(images_directory: Path, filename: str) -> Path:
    if Path(filename).name != filename:
        raise ValueError(f"image filename must not contain a path: {filename}")
    image_path = (images_directory / filename).resolve()
    if image_path.parent != images_directory.resolve():
        raise ValueError(f"image resolves outside the image directory: {filename}")
    if not image_path.is_file():
        raise FileNotFoundError(f"NoMaD topomap image not found: {image_path}")
    return image_path


def _preprocessing_mode(config: NomadTopomapMapConfig) -> str:
    if config.expected_center_crop_aspect is None:
        return "direct_resize"
    return "center_crop_then_resize"


def _node_id(topology_index: int) -> NodeId:
    return NodeId(f"node-{topology_index:04d}")


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error.msg}") from error


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_sequence(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return tuple(value)


def _required_string(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _required_int(row: Mapping[str, object], field: str) -> int:
    value = row.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def _required_float(row: Mapping[str, object], field: str) -> float:
    value = row.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _optional_float(row: Mapping[str, object], field: str) -> float | None:
    value = row.get(field)
    if value is None:
        return None
    return _required_float(row, field)


def _image_media_type(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".png":
        return "image/png"
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    return "application/octet-stream"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _topomap_digest(
    config: NomadTopomapMapConfig,
    paths: Sequence[Path],
) -> str:
    digest = hashlib.sha256()
    compatibility = {
        "model_artifact_id": config.model_artifact_id,
        "model_artifact_digest": _canonical_sha256(
            config.model_artifact_digest
        ),
        "image_profile_id": config.image_profile_id,
        "model_input_width": config.model_input_width,
        "model_input_height": config.model_input_height,
        "normalization_profile_id": config.normalization_profile_id,
        "expected_center_crop_aspect": config.expected_center_crop_aspect,
    }
    digest.update(
        json.dumps(
            compatibility,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for path in paths:
        relative_name = path.name
        if path.parent.name == _IMAGES_DIRECTORY:
            relative_name = f"{_IMAGES_DIRECTORY}/{path.name}"
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _canonical_sha256(value: str) -> str:
    return f"sha256:{value.removeprefix('sha256:').casefold()}"

"""Offline model-aware topomap construction for NoMaD."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import statistics

import torch
from torch.nn import functional as F

from nomad_runtime.image_input import ImageTensorSpec, canonicalize_image
from nomad_runtime.assets import default_checkpoint_path
from nomad_runtime.policy import NomadPolicy


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class AdaptiveTopomapConfig:
    """Selection thresholds for model-aware topomap construction."""

    minimum_gap_s: float = 0.8
    maximum_gap_s: float = 4.0
    preferred_min_distance: float = 6.0
    preferred_max_distance: float = 12.0
    hard_min_distance: float = 3.0
    hard_max_distance: float = 15.0
    target_distance: float = 9.0
    minimum_sharpness: float = 0.0
    context_jitter: int = 0
    center_crop_aspect: float | None = None

    def validate(self) -> None:
        """Raises ValueError when thresholds are inconsistent."""
        if self.minimum_gap_s <= 0.0:
            raise ValueError("minimum_gap_s must be positive")
        if self.maximum_gap_s < self.minimum_gap_s:
            raise ValueError("maximum_gap_s must not be below minimum_gap_s")
        if not (
            self.hard_min_distance
            <= self.preferred_min_distance
            <= self.target_distance
            <= self.preferred_max_distance
            <= self.hard_max_distance
        ):
            raise ValueError("distance thresholds must be monotonically nested")
        if self.minimum_sharpness < 0.0:
            raise ValueError("minimum_sharpness must be non-negative")
        if self.context_jitter < 0:
            raise ValueError("context_jitter must be non-negative")
        if (
            self.center_crop_aspect is not None
            and self.center_crop_aspect <= 0.0
        ):
            raise ValueError("center_crop_aspect must be positive")


@dataclass(frozen=True)
class FrameRecord:
    """One ordered dense input image and its source metadata."""

    position: int
    source_node: int
    filename: str
    path: Path
    timestamp_s: float
    sharpness: float | None
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class CandidateScore:
    """Aggregated NoMaD distance for one possible directed edge target."""

    position: int
    source_node: int
    timestamp_s: float
    time_delta_s: float
    distance: float
    minimum_distance: float
    maximum_distance: float


@dataclass(frozen=True)
class CandidateDecision:
    """Selected target and the reason it was preferred."""

    candidate: CandidateScore
    reason: str


def select_candidate(
    candidates: Sequence[CandidateScore],
    config: AdaptiveTopomapConfig,
) -> CandidateDecision:
    """Selects the farthest preferred candidate or the safest fallback."""
    if not candidates:
        raise ValueError("at least one candidate score is required")
    config.validate()

    preferred = [
        candidate
        for candidate in candidates
        if config.preferred_min_distance
        <= candidate.distance
        <= config.preferred_max_distance
    ]
    if preferred:
        return CandidateDecision(
            candidate=max(
                preferred,
                key=lambda candidate: (
                    candidate.timestamp_s,
                    -abs(candidate.distance - config.target_distance),
                ),
            ),
            reason="farthest_preferred",
        )

    hard_valid = [
        candidate
        for candidate in candidates
        if config.hard_min_distance
        <= candidate.distance
        <= config.hard_max_distance
    ]
    pool = hard_valid or list(candidates)
    return CandidateDecision(
        candidate=min(
            pool,
            key=lambda candidate: (
                abs(candidate.distance - config.target_distance),
                -candidate.timestamp_s,
            ),
        ),
        reason=("closest_to_target" if hard_valid else "outside_hard_range"),
    )


def _numeric_image_paths(images_dir: Path) -> list[Path]:
    paths = [
        path
        for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in _IMAGE_SUFFIXES
        and path.stem.isdigit()
    ]
    return sorted(paths, key=lambda path: int(path.stem))


def load_frame_records(
    images_dir: str | Path,
    manifest_path: str | Path | None = None,
    frame_period_s: float | None = None,
) -> list[FrameRecord]:
    """Loads numeric images and timestamps from a manifest or fixed period."""
    directory = Path(images_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory not found: {directory}")
    paths = _numeric_image_paths(directory)
    if not paths:
        raise ValueError(f"no numeric PNG/JPEG images found in {directory}")

    selected_manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else directory / "manifest.json"
    )
    metadata_by_filename: dict[str, Mapping[str, object]] = {}
    if selected_manifest.is_file():
        manifest = json.loads(selected_manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest, list):
            raise ValueError("input manifest must contain a JSON list")
        for item in manifest:
            if not isinstance(item, dict) or "filename" not in item:
                raise ValueError("every manifest row must contain filename")
            metadata_by_filename[str(item["filename"])] = item
    elif frame_period_s is None:
        raise FileNotFoundError(
            "manifest.json was not found; provide --frame-period-s"
        )

    if frame_period_s is not None and frame_period_s <= 0.0:
        raise ValueError("frame_period_s must be positive")

    records = []
    for position, path in enumerate(paths):
        metadata = metadata_by_filename.get(path.name, {})
        timestamp_value = metadata.get("time_s")
        if timestamp_value is None:
            if frame_period_s is None:
                raise ValueError(f"manifest has no time_s for {path.name}")
            timestamp = position * frame_period_s
        else:
            timestamp = float(timestamp_value)
        if not math.isfinite(timestamp):
            raise ValueError(f"non-finite timestamp for {path.name}")
        sharpness_value = metadata.get("sharpness")
        sharpness = (
            None if sharpness_value is None else float(sharpness_value)
        )
        source_node = int(metadata.get("node", path.stem))
        records.append(
            FrameRecord(
                position=position,
                source_node=source_node,
                filename=path.name,
                path=path,
                timestamp_s=timestamp,
                sharpness=sharpness,
                metadata=metadata,
            )
        )

    for previous, current in zip(records, records[1:]):
        if current.timestamp_s <= previous.timestamp_s:
            raise ValueError("input timestamps must be strictly increasing")
    return records


def _center_crop(image: object, aspect: float | None) -> object:
    if aspect is None:
        return image
    width, height = image.size
    source_aspect = width / height
    if source_aspect > aspect:
        cropped_width = max(1, round(height * aspect))
        left = (width - cropped_width) // 2
        return image.crop((left, 0, left + cropped_width, height))
    cropped_height = max(1, round(width / aspect))
    top = (height - cropped_height) // 2
    return image.crop((0, top, width, top + cropped_height))


class ImageTensorCache:
    """Lazily decodes and model-resizes source images."""

    def __init__(
        self,
        image_size: tuple[int, int],
        center_crop_aspect: float | None,
        cache_size: int = 128,
    ) -> None:
        self._image_size = image_size
        self._center_crop_aspect = center_crop_aspect
        self._load_cached = lru_cache(maxsize=cache_size)(self._load)

    def _load(self, path_string: str) -> torch.Tensor:
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "adaptive topomap construction requires Pillow; "
                "install the offline optional dependencies"
            ) from error

        with Image.open(path_string) as source_image:
            image = _center_crop(
                source_image.convert("RGB"), self._center_crop_aspect
            )
            storage = torch.ByteStorage.from_buffer(image.tobytes())
            decoded = torch.ByteTensor(storage).view(
                image.height, image.width, 3
            )
        tensor = canonicalize_image(
            decoded,
            ImageTensorSpec(layout="hwc", channel_order="rgb"),
        )
        target_width, target_height = self._image_size
        return F.interpolate(
            tensor.unsqueeze(0),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def load(self, path: Path) -> torch.Tensor:
        """Returns a cached RGB CHW float tensor at model resolution."""
        return self._load_cached(str(path))


class AdaptiveTopomapBuilder:
    """Builds a sequential directed topomap using NoMaD distance scores."""

    def __init__(
        self,
        policy: NomadPolicy,
        config: AdaptiveTopomapConfig | None = None,
    ) -> None:
        self.policy = policy
        self.config = config or AdaptiveTopomapConfig()
        self.config.validate()
        self._images = ImageTensorCache(
            policy.config.image_size,
            self.config.center_crop_aspect,
        )

    def _context_endpoints(
        self,
        source_position: int,
        first_candidate_position: int,
    ) -> list[int]:
        minimum_endpoint = self.policy.config.observation_frames - 1
        endpoints = []
        for offset in range(
            -self.config.context_jitter,
            self.config.context_jitter + 1,
        ):
            endpoint = source_position + offset
            if minimum_endpoint <= endpoint < first_candidate_position:
                endpoints.append(endpoint)
        if source_position not in endpoints:
            endpoints.append(source_position)
        return sorted(set(endpoints))

    def _score_candidates(
        self,
        records: Sequence[FrameRecord],
        source_position: int,
        candidate_positions: Sequence[int],
    ) -> list[CandidateScore]:
        goals = torch.stack(
            [
                self._images.load(records[position].path)
                for position in candidate_positions
            ]
        )
        endpoints = self._context_endpoints(
            source_position, candidate_positions[0]
        )
        distance_rows = []
        observation_frames = self.policy.config.observation_frames
        for endpoint in endpoints:
            context = torch.stack(
                [
                    self._images.load(records[position].path)
                    for position in range(
                        endpoint - observation_frames + 1,
                        endpoint + 1,
                    )
                ]
            )
            observations = context.unsqueeze(0).repeat(
                len(candidate_positions), 1, 1, 1, 1
            )
            condition = self.policy.encode_condition(observations, goals)
            distance_rows.append(self.policy.predict_distance(condition).cpu())

        distances = torch.stack(distance_rows)
        median_distances = distances.median(dim=0).values
        minimum_distances = distances.min(dim=0).values
        maximum_distances = distances.max(dim=0).values
        source_time = records[source_position].timestamp_s
        scores = []
        for index, position in enumerate(candidate_positions):
            record = records[position]
            scores.append(
                CandidateScore(
                    position=position,
                    source_node=record.source_node,
                    timestamp_s=record.timestamp_s,
                    time_delta_s=record.timestamp_s - source_time,
                    distance=float(median_distances[index].item()),
                    minimum_distance=float(minimum_distances[index].item()),
                    maximum_distance=float(maximum_distances[index].item()),
                )
            )
        return scores

    def _quality_candidates(
        self,
        records: Sequence[FrameRecord],
        positions: Sequence[int],
    ) -> tuple[list[int], bool]:
        quality_positions = [
            position
            for position in positions
            if records[position].sharpness is None
            or records[position].sharpness >= self.config.minimum_sharpness
        ]
        return (quality_positions or list(positions), not bool(quality_positions))

    def build(
        self, records: Sequence[FrameRecord]
    ) -> tuple[list[int], list[dict[str, object]]]:
        """Returns selected dense positions and directed edge diagnostics."""
        observation_frames = self.policy.config.observation_frames
        if len(records) <= observation_frames:
            raise ValueError(
                f"need more than {observation_frames} dense input frames"
            )
        current_position = observation_frames - 1
        selected_positions = [current_position]
        edges: list[dict[str, object]] = []
        final_position = len(records) - 1

        while current_position < final_position:
            source_time = records[current_position].timestamp_s
            in_time_window = [
                position
                for position in range(current_position + 1, len(records))
                if self.config.minimum_gap_s
                <= records[position].timestamp_s - source_time
                <= self.config.maximum_gap_s
            ]
            if not in_time_window:
                remaining_gap = records[final_position].timestamp_s - source_time
                if 0.0 < remaining_gap < self.config.minimum_gap_s:
                    in_time_window = [final_position]
                    terminal_short_edge = True
                else:
                    raise RuntimeError(
                        "no candidate falls within the configured time window "
                        f"after source node {records[current_position].source_node}"
                    )
            else:
                terminal_short_edge = False

            candidate_positions, used_quality_fallback = self._quality_candidates(
                records, in_time_window
            )
            scores = self._score_candidates(
                records, current_position, candidate_positions
            )
            decision = select_candidate(scores, self.config)
            quality_relaxed = False
            if (
                decision.reason == "outside_hard_range"
                and candidate_positions != in_time_window
            ):
                relaxed_scores = self._score_candidates(
                    records, current_position, in_time_window
                )
                relaxed_decision = select_candidate(
                    relaxed_scores, self.config
                )
                if relaxed_decision.reason != "outside_hard_range":
                    scores = relaxed_scores
                    decision = relaxed_decision
                    quality_relaxed = True
            if terminal_short_edge:
                reason = "terminal_short_edge"
            elif quality_relaxed:
                reason = f"{decision.reason}_quality_relaxed"
            elif used_quality_fallback:
                reason = f"{decision.reason}_quality_fallback"
            else:
                reason = decision.reason

            next_position = decision.candidate.position
            if next_position <= current_position:
                raise RuntimeError("candidate selection did not advance the route")
            edge = {
                "edge": len(edges),
                "source_topology_node": len(selected_positions) - 1,
                "target_topology_node": len(selected_positions),
                "source_position": current_position,
                "target_position": next_position,
                "source_node": records[current_position].source_node,
                "target_node": records[next_position].source_node,
                "source_time_s": records[current_position].timestamp_s,
                "target_time_s": records[next_position].timestamp_s,
                "time_delta_s": decision.candidate.time_delta_s,
                "predicted_distance": decision.candidate.distance,
                "minimum_predicted_distance": (
                    decision.candidate.minimum_distance
                ),
                "maximum_predicted_distance": (
                    decision.candidate.maximum_distance
                ),
                "selection_reason": reason,
                "candidate_count": len(scores),
                "candidates": [asdict(score) for score in scores],
            }
            edges.append(edge)
            selected_positions.append(next_position)
            current_position = next_position

        return selected_positions, edges


def _write_selected_image(
    source: Path,
    destination: Path,
    center_crop_aspect: float | None,
) -> None:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("writing topomap images requires Pillow") from error
    with Image.open(source) as source_image:
        image = _center_crop(source_image.convert("RGB"), center_crop_aspect)
        image.save(destination, format="PNG", optimize=True)


def _summary(
    source_count: int,
    selected_count: int,
    edges: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    distances = [float(edge["predicted_distance"]) for edge in edges]
    gaps = [float(edge["time_delta_s"]) for edge in edges]
    return {
        "source_frame_count": source_count,
        "topology_node_count": selected_count,
        "edge_count": len(edges),
        "compression_ratio": selected_count / source_count,
        "predicted_distance": {
            "minimum": min(distances),
            "mean": statistics.mean(distances),
            "median": statistics.median(distances),
            "maximum": max(distances),
        },
        "time_delta_s": {
            "minimum": min(gaps),
            "mean": statistics.mean(gaps),
            "median": statistics.median(gaps),
            "maximum": max(gaps),
        },
        "selection_reasons": {
            reason: sum(
                edge["selection_reason"] == reason for edge in edges
            )
            for reason in sorted(
                {str(edge["selection_reason"]) for edge in edges}
            )
        },
    }


def write_topomap(
    output_dir: str | Path,
    records: Sequence[FrameRecord],
    selected_positions: Sequence[int],
    edges: Sequence[Mapping[str, object]],
    config: AdaptiveTopomapConfig,
) -> dict[str, object]:
    """Writes official-style numeric images plus manifests and summary."""
    directory = Path(output_dir).expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"output directory is not empty: {directory}")
    images_dir = directory / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    nodes = []
    for topology_node, position in enumerate(selected_positions):
        record = records[position]
        filename = f"{topology_node:04d}.png"
        _write_selected_image(
            record.path,
            images_dir / filename,
            config.center_crop_aspect,
        )
        nodes.append(
            {
                "topology_node": topology_node,
                "filename": filename,
                "source_position": position,
                "source_node": record.source_node,
                "source_filename": record.filename,
                "time_s": record.timestamp_s,
                "sharpness": record.sharpness,
            }
        )

    summary = _summary(len(records), len(selected_positions), edges)
    manifest = {
        "format_version": 1,
        "selection": asdict(config),
        "summary": summary,
        "nodes": nodes,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (directory / "edges.json").write_text(
        json.dumps(list(edges), indent=2), encoding="utf-8"
    )
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _aspect_ratio(value: str) -> float | None:
    if value == "none":
        return None
    if ":" in value:
        width, height = value.split(":", maxsplit=1)
        return float(width) / float(height)
    return float(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a NoMaD-aware adaptive sequential topomap."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint_path(),
        help="NoMaD checkpoint (default: repository LFS asset).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--frame-period-s", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-gap-s", type=float, default=0.8)
    parser.add_argument("--maximum-gap-s", type=float, default=4.0)
    parser.add_argument("--preferred-min-distance", type=float, default=6.0)
    parser.add_argument("--preferred-max-distance", type=float, default=12.0)
    parser.add_argument("--hard-min-distance", type=float, default=3.0)
    parser.add_argument("--hard-max-distance", type=float, default=15.0)
    parser.add_argument("--target-distance", type=float, default=9.0)
    parser.add_argument("--minimum-sharpness", type=float, default=0.0)
    parser.add_argument("--context-jitter", type=int, default=0)
    parser.add_argument(
        "--center-crop-aspect",
        type=_aspect_ratio,
        default=None,
        help="Use '4:3', another ratio, or 'none' (default).",
    )
    return parser


def main() -> None:
    """Runs the adaptive topomap command-line tool."""
    args = _build_parser().parse_args()
    config = AdaptiveTopomapConfig(
        minimum_gap_s=args.minimum_gap_s,
        maximum_gap_s=args.maximum_gap_s,
        preferred_min_distance=args.preferred_min_distance,
        preferred_max_distance=args.preferred_max_distance,
        hard_min_distance=args.hard_min_distance,
        hard_max_distance=args.hard_max_distance,
        target_distance=args.target_distance,
        minimum_sharpness=args.minimum_sharpness,
        context_jitter=args.context_jitter,
        center_crop_aspect=args.center_crop_aspect,
    )
    records = load_frame_records(
        args.images,
        manifest_path=args.manifest,
        frame_period_s=args.frame_period_s,
    )
    policy = NomadPolicy.from_checkpoint(
        args.checkpoint,
        device=args.device,
        strict=True,
    )
    builder = AdaptiveTopomapBuilder(policy, config)
    selected_positions, edges = builder.build(records)
    summary = write_topomap(
        args.output,
        records,
        selected_positions,
        edges,
        config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

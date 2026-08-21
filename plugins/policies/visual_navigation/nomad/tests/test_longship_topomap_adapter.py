"""Tests for adapting NoMaD topomap artifacts to the Map Engine."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from longship_adapter import (
    NomadTopomapMapConfig,
    create_nomad_topomap_engine,
    load_nomad_topomap,
)
from longship.navigation.common import TimePoint
from longship.navigation.map_engine.interface import MapEngine
from longship.navigation.map_engine.models import (
    AnchorPurpose,
    AnchorQuery,
    MapCapability,
    MapId,
    MapSelector,
    MapVersion,
    ResourceId,
    TopologyQuery,
)


_MODEL_DIGEST = "0" * 64


class NomadTopomapAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self._write_fixture()

    def _config(self, **overrides: object) -> NomadTopomapMapConfig:
        values = {
            "root": self.root,
            "map_id": MapId("nomad-test-route"),
            "version": MapVersion("v1"),
            "published_at": TimePoint(clock_id="unix", nanoseconds=1),
            "model_artifact_id": "nomad.pth",
            "model_artifact_digest": _MODEL_DIGEST,
        }
        values.update(overrides)
        return NomadTopomapMapConfig(**values)

    def _write_fixture(self) -> None:
        images = self.root / "images"
        images.mkdir()
        for index in range(3):
            (images / f"{index:04d}.png").write_bytes(
                b"not-decoded-by-map-engine" + bytes((index,))
            )

        summary = {
            "source_frame_count": 10,
            "topology_node_count": 3,
            "edge_count": 2,
        }
        manifest = {
            "format_version": 1,
            "selection": {
                "hard_min_distance": 3.0,
                "hard_max_distance": 15.0,
                "center_crop_aspect": None,
            },
            "summary": summary,
            "nodes": [
                {
                    "topology_node": index,
                    "filename": f"{index:04d}.png",
                    "source_position": index + 3,
                    "source_node": index + 3,
                    "source_filename": f"{index + 3:04d}.png",
                    "time_s": float(index + 1),
                    "sharpness": 300.0 + index,
                }
                for index in range(3)
            ],
        }
        edges = [
            {
                "edge": 0,
                "source_topology_node": 0,
                "target_topology_node": 1,
                "source_position": 3,
                "target_position": 4,
                "source_node": 3,
                "target_node": 4,
                "source_time_s": 1.0,
                "target_time_s": 2.0,
                "time_delta_s": 1.0,
                "predicted_distance": 9.0,
                "minimum_predicted_distance": 8.0,
                "maximum_predicted_distance": 12.0,
                "selection_reason": "farthest_preferred",
                "candidate_count": 2,
                "candidates": [],
            },
            {
                "edge": 1,
                "source_topology_node": 1,
                "target_topology_node": 2,
                "source_position": 4,
                "target_position": 5,
                "source_node": 4,
                "target_node": 5,
                "source_time_s": 2.0,
                "target_time_s": 3.0,
                "time_delta_s": 1.0,
                "predicted_distance": 2.5,
                "minimum_predicted_distance": 2.0,
                "maximum_predicted_distance": 3.0,
                "selection_reason": "terminal_short_edge",
                "candidate_count": 1,
                "candidates": [],
            },
        ]
        (self.root / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (self.root / "edges.json").write_text(
            json.dumps(edges), encoding="utf-8"
        )
        (self.root / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

    async def test_exposes_topology_anchors_and_image_resources(self) -> None:
        engine = create_nomad_topomap_engine(self._config())
        self.assertIsInstance(engine, MapEngine)
        snapshot = await engine.get_snapshot(
            MapSelector(map_id=MapId("nomad-test-route"))
        )

        self.assertEqual(
            snapshot.capabilities,
            frozenset(
                {
                    MapCapability.TOPOLOGY,
                    MapCapability.VISUAL_ANCHORS,
                    MapCapability.SEGMENT_METADATA,
                    MapCapability.RESOURCE_REFERENCES,
                }
            ),
        )
        topology = await engine.query_topology(snapshot, TopologyQuery())
        self.assertEqual(len(topology.nodes), 3)
        self.assertEqual(len(topology.segments), 2)
        self.assertEqual(
            topology.segments[0].attributes["offline_model_check_status"],
            "passed",
        )
        self.assertEqual(
            topology.segments[1].attributes["offline_model_check_status"],
            "terminal",
        )
        self.assertTrue(
            all(
                segment.attributes["hardware_qualification_status"]
                == "unqualified"
                for segment in topology.segments
            )
        )

        completion = await engine.query_anchors(
            snapshot,
            AnchorQuery(
                purposes=frozenset({AnchorPurpose.COMPLETION}),
            ),
        )
        self.assertEqual(len(completion.anchors), 1)
        resource_id = completion.anchors[0].resource_ids[0]
        resources = await engine.resolve_resources(snapshot, (resource_id,))
        self.assertEqual(len(resources.resources), 1)
        resource = resources.resources[0]
        self.assertEqual(
            resource.resource_id,
            ResourceId("node-0002:goal-image"),
        )
        self.assertEqual(resource.attributes["preprocessing_mode"], "direct_resize")
        self.assertEqual(resource.attributes["model_input_width"], 96)
        self.assertTrue(Path(resource.locator).is_absolute())

    def test_snapshot_digest_pins_image_content(self) -> None:
        first = load_nomad_topomap(self._config()).snapshot.content_digest
        (self.root / "images" / "0001.png").write_bytes(b"changed")

        second = load_nomad_topomap(self._config()).snapshot.content_digest

        self.assertNotEqual(first, second)

    def test_rejects_a_preprocessing_profile_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "center-crop profile"):
            load_nomad_topomap(
                self._config(expected_center_crop_aspect=4.0 / 3.0)
            )


if __name__ == "__main__":
    unittest.main()

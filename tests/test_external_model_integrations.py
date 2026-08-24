from __future__ import annotations

import json
import unittest
from pathlib import Path

from longship.artifacts import load_model_artifact_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ExternalModelIntegrationTests(unittest.TestCase):
    def test_real_artifact_manifests_are_pinned_and_prefetch_blocked(self) -> None:
        manifests = (
            REPOSITORY_ROOT
            / "plugins/locomotion/unitree_rl_mjlab_g1_velocity"
            / "model-artifacts.experimental.json",
            REPOSITORY_ROOT
            / "plugins/locomotion/holosoma"
            / "model-artifacts.experimental.json",
        )
        loaded = tuple(load_model_artifact_manifest(path) for path in manifests)
        self.assertTrue(all(not manifest.prefetch_eligible for manifest in loaded))
        self.assertEqual(
            loaded[0].artifact("policy.onnx").sha256,
            "2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28",
        )
        self.assertEqual(
            loaded[1].artifact("fastsac_g1_29dof.onnx").size_bytes,
            895618,
        )

    def test_whole_body_plugins_declare_symmetric_conflicts(self) -> None:
        plugins = {}
        for path in (REPOSITORY_ROOT / "plugins").rglob("plugin.experimental.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            if "whole_body_motion" in data.get("actuator_scope", []):
                plugins[data["plugin_id"]] = data

        for plugin_id, plugin in plugins.items():
            expected_conflicts = set(plugins) - {plugin_id}
            self.assertEqual(set(plugin["exclusive_with"]), expected_conflicts)
            for conflict_id in plugin["exclusive_with"]:
                with self.subTest(plugin=plugin_id, conflict=conflict_id):
                    self.assertIn(conflict_id, plugins)
                    self.assertIn(
                        plugin_id,
                        plugins[conflict_id]["exclusive_with"],
                    )

    def test_no_upstream_model_or_source_tree_is_vendored(self) -> None:
        forbidden_suffixes = {".ckpt", ".onnx", ".pt", ".pth", ".safetensors"}
        roots = (REPOSITORY_ROOT / "plugins", REPOSITORY_ROOT / "src")
        copied = [
            path
            for root in roots
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in forbidden_suffixes
        ]
        self.assertEqual(copied, [])


if __name__ == "__main__":
    unittest.main()

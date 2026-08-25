"""Static checks for the Navigation Harness Skill plugin contracts."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


_PLUGIN_ROOT = (
    Path(__file__).parents[2] / "plugins" / "skills" / "navigation_harness"
)


class PluginContractTests(unittest.TestCase):
    def test_manifest_and_declared_schemas_exist(self) -> None:
        self.assertTrue((_PLUGIN_ROOT / "plugin.yaml").is_file())
        schema_paths = (
            _PLUGIN_ROOT / "schemas" / "navigate-to-request.v1.schema.json",
            _PLUGIN_ROOT / "schemas" / "navigate-to-result.v1.schema.json",
        )
        for schema_path in schema_paths:
            with self.subTest(schema_path=schema_path):
                schema = json.loads(schema_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    schema["$schema"],
                    "https://json-schema.org/draft/2020-12/schema",
                )
                self.assertEqual(schema["x-longship-status"], "draft")


if __name__ == "__main__":
    unittest.main()

"""Checks that the Harness and bundled policies share one Python baseline."""

from __future__ import annotations

from pathlib import Path
import tomllib
import unittest


_REPOSITORY_ROOT = Path(__file__).parents[2]
_NOMAD_ROOT = (
    _REPOSITORY_ROOT
    / "plugins"
    / "policies"
    / "visual_navigation"
    / "nomad"
)


def _read_project(path: Path) -> dict[str, object]:
    with path.open("rb") as project_file:
        document = tomllib.load(project_file)
    return document["project"]


class PythonVersionAlignmentTests(unittest.TestCase):
    def test_projects_require_python_3_11_or_newer(self) -> None:
        projects = (
            _read_project(_REPOSITORY_ROOT / "pyproject.toml"),
            _read_project(_NOMAD_ROOT / "pyproject.toml"),
        )

        for project in projects:
            with self.subTest(project=project["name"]):
                self.assertEqual(project["requires-python"], ">=3.11")

    def test_nomad_requires_supported_torch_baseline(self) -> None:
        project = _read_project(_NOMAD_ROOT / "pyproject.toml")

        self.assertIn("torch>=2.1", project["dependencies"])


if __name__ == "__main__":
    unittest.main()

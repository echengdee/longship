"""Dependency direction tests for the navigation harness contracts."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


_NAVIGATION_ROOT = (
    Path(__file__).parents[2] / "src" / "longship" / "navigation"
)


class DependencyDirectionTests(unittest.TestCase):
    def test_navigation_contracts_do_not_import_plugins(self) -> None:
        for source_path in _NAVIGATION_ROOT.rglob("*.py"):
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            imported_modules = self._imported_modules(tree)
            self.assertFalse(
                any(
                    module == "plugins" or module.startswith("plugins.")
                    for module in imported_modules
                ),
                msg=f"{source_path} imports a plugin",
            )

    def test_planning_does_not_import_execution_time_layers(self) -> None:
        planning_root = _NAVIGATION_ROOT / "planning_engine"
        forbidden_prefixes = (
            "longship.navigation.local_trajectory_engine",
            "longship.navigation.ports.trajectory_policy",
            "longship.navigation.runtime",
        )

        for source_path in planning_root.rglob("*.py"):
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            imported_modules = self._imported_modules(tree)
            self.assertFalse(
                any(
                    module == prefix or module.startswith(f"{prefix}.")
                    for module in imported_modules
                    for prefix in forbidden_prefixes
                ),
                msg=f"{source_path} imports an execution-time layer",
            )

    @staticmethod
    def _imported_modules(tree: ast.AST) -> tuple[str, ...]:
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        return tuple(modules)


if __name__ == "__main__":
    unittest.main()

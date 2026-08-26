from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "plugins/targets/mujoco_g1_external/doctor.py"


def _load_doctor():
    specification = importlib.util.spec_from_file_location(
        "longship_mujoco_g1_external_doctor", MODULE_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load external G1 asset doctor")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class ExternalG1AssetPluginTests(unittest.TestCase):
    def test_valid_external_bundle_is_asset_ready_but_not_dynamic_ready(self) -> None:
        module = _load_doctor()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            meshes = root / "meshes"
            meshes.mkdir()
            (meshes / "part.obj").write_text("v 0 0 0\n", encoding="utf-8")
            (root / "scene.xml").write_text(
                '<mujoco><include file="g1.xml"/></mujoco>', encoding="utf-8"
            )
            motors = "".join(
                f'<motor name="motor_{index}" joint="joint_{index}"/>'
                for index in range(29)
            )
            (root / "g1.xml").write_text(
                "<mujoco><compiler meshdir=\"meshes\"/><asset>"
                "<mesh file=\"part.obj\"/></asset><worldbody><body>"
                "<joint name=\"floating_base_joint\" type=\"free\"/>"
                f"</body></worldbody><actuator>{motors}</actuator></mujoco>",
                encoding="utf-8",
            )
            license_file = root.parent / f"{root.name}.LICENSE"
            license_file.write_text("test license record\n", encoding="utf-8")
            try:
                bundle_digest, _, _ = module.sha256_bundle(root)
                report = module.inspect_g1_asset(
                    root / "scene.xml",
                    license_file,
                    expected_bundle_sha256=bundle_digest,
                    expected_license_sha256=module.sha256_file(license_file),
                )
            finally:
                license_file.unlink()

        self.assertTrue(report["asset_ready"])
        self.assertFalse(report["dynamic_follow_ready"])
        self.assertEqual(report["motor_count"], 29)

    def test_bundle_digest_mismatch_fails_closed(self) -> None:
        module = _load_doctor()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "file.txt").write_text("asset", encoding="utf-8")
            license_file = root.parent / f"{root.name}.LICENSE"
            license_file.write_text("license", encoding="utf-8")
            try:
                with self.assertRaisesRegex(ValueError, "bundle SHA-256 mismatch"):
                    module.inspect_g1_asset(
                        root / "file.txt",
                        license_file,
                        expected_bundle_sha256="0" * 64,
                        expected_license_sha256=module.sha256_file(license_file),
                    )
            finally:
                license_file.unlink()


if __name__ == "__main__":
    unittest.main()

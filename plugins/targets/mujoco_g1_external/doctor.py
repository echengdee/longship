from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any

from longship.artifacts.external import sha256_directory
from longship.artifacts.external import sha256_file as _sha256_file


_MAX_BUNDLE_FILES = 2_048
_MAX_BUNDLE_BYTES = 256 * 1024 * 1024
_EXPECTED_G1_MOTORS = 29


def sha256_file(path: Path) -> str:
    return _sha256_file(path)


def sha256_bundle(root: Path) -> tuple[str, int, int]:
    """Hash regular files as relative-name, size, and content records."""

    return sha256_directory(
        root,
        maximum_files=_MAX_BUNDLE_FILES,
        maximum_bytes=_MAX_BUNDLE_BYTES,
    )


def inspect_g1_asset(
    scene: Path,
    license_file: Path,
    *,
    expected_bundle_sha256: str,
    expected_license_sha256: str,
) -> dict[str, Any]:
    scene_path = scene.resolve(strict=True)
    license_path = license_file.resolve(strict=True)
    bundle_root = scene_path.parent
    _require_digest(expected_bundle_sha256, "bundle")
    _require_digest(expected_license_sha256, "license")

    bundle_digest, file_count, total_bytes = sha256_bundle(bundle_root)
    license_digest = sha256_file(license_path)
    if bundle_digest != expected_bundle_sha256:
        raise ValueError("G1 asset bundle SHA-256 mismatch")
    if license_digest != expected_license_sha256:
        raise ValueError("G1 asset license SHA-256 mismatch")

    scene_root = ElementTree.parse(scene_path).getroot()
    if scene_root.tag != "mujoco":
        raise ValueError("scene root is not a MuJoCo document")
    includes = scene_root.findall("include")
    if len(includes) != 1 or not includes[0].get("file"):
        raise ValueError("G1 scene must include exactly one robot model")
    model_path = (bundle_root / str(includes[0].get("file"))).resolve(strict=True)
    if bundle_root not in model_path.parents:
        raise ValueError("included robot model escapes the asset bundle")

    robot_root = ElementTree.parse(model_path).getroot()
    if robot_root.tag != "mujoco":
        raise ValueError("included G1 model is not a MuJoCo document")
    free_joint = robot_root.find(".//joint[@name='floating_base_joint']")
    if free_joint is None or free_joint.get("type") != "free":
        raise ValueError("G1 model has no expected floating base")
    motor_count = len(robot_root.findall("./actuator/motor"))
    if motor_count != _EXPECTED_G1_MOTORS:
        raise ValueError(f"expected 29 G1 motors, found {motor_count}")
    _require_meshes(robot_root, model_path.parent)

    return {
        "schema_version": "longship.mujoco-g1-asset-doctor.v0",
        "asset_ready": True,
        "dynamic_follow_ready": False,
        "scene": str(scene_path),
        "bundle_sha256": bundle_digest,
        "license_sha256": license_digest,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "motor_count": motor_count,
        "blockers": [
            "no qualified LowCmd/LowState locomotion provider is connected",
            "the high-level Unitree LocoClient target is not a MuJoCo low-level target",
            "model-weight license, digest, command contract, and simulation "
            "evidence are required",
        ],
    }


def _require_meshes(robot_root: ElementTree.Element, model_directory: Path) -> None:
    compiler = robot_root.find("compiler")
    mesh_directory = model_directory
    if compiler is not None and compiler.get("meshdir"):
        mesh_directory = (model_directory / str(compiler.get("meshdir"))).resolve()
    for mesh in robot_root.findall("./asset/mesh"):
        filename = mesh.get("file")
        if not filename:
            raise ValueError("G1 mesh declaration has no file")
        resolved = (mesh_directory / filename).resolve(strict=True)
        if model_directory not in resolved.parents:
            raise ValueError("G1 mesh escapes the asset bundle")


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"expected {label} SHA-256 must be lowercase hexadecimal")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an external Unitree G1 MuJoCo asset without copying it"
    )
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--expected-license-sha256", required=True)
    parser.add_argument(
        "--require-dynamic-follow",
        action="store_true",
        help="fail because this asset-only seam has no qualified locomotion provider",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        report = inspect_g1_asset(
            args.scene,
            args.license,
            expected_bundle_sha256=args.expected_bundle_sha256,
            expected_license_sha256=args.expected_license_sha256,
        )
    except (OSError, ElementTree.ParseError, ValueError) as exc:
        raise SystemExit(f"BLOCKED: {exc}") from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_dynamic_follow:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

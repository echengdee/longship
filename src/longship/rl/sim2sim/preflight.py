from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import onnxruntime as ort

from longship.rl.compatibility import CompatibilityLock


@dataclass(frozen=True, slots=True)
class PreflightResult:
    backend: str
    ready: bool
    checks: tuple[str, ...]
    blockers: tuple[str, ...]


def inspect_artifact(path: Path, expected_sha256: str | None = None) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"artifact does not exist: {path}"
    size = path.stat().st_size
    if size < 1_024:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "git-lfs.github.com/spec" in text:
            return False, f"artifact is a Git LFS pointer, not model bytes: {path}"
        return False, f"artifact is unexpectedly small ({size} bytes): {path}"
    if expected_sha256 is not None:
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        digest = checksum.hexdigest()
        if digest != expected_sha256:
            return False, f"artifact SHA-256 mismatch: {path}: expected {expected_sha256}, got {digest}"
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        return False, f"ONNX artifact cannot be loaded: {path}: {exc}"
    inputs = [(value.name, value.shape) for value in session.get_inputs()]
    outputs = [(value.name, value.shape) for value in session.get_outputs()]
    return True, f"ONNX loaded: inputs={inputs}, outputs={outputs}"


def inspect_asset(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"asset does not exist: {path}"
    if path.stat().st_size < 1_024:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "git-lfs.github.com/spec" in text:
            return False, f"asset is a Git LFS pointer, not file bytes: {path}"
    return True, f"asset present: {path}"


def _configured_paths(config: dict[str, Any], key: str, fallback: str | None = None) -> tuple[str, ...]:
    values = config.get(key)
    if values is not None:
        return tuple(values)
    value = config.get(fallback) if fallback else None
    return (value,) if value else ()


def preflight_backend(root: str | Path, backend: str) -> PreflightResult:
    workspace = Path(root).resolve()
    lock = CompatibilityLock.bundled()
    try:
        config = lock.values["backends"][backend]
    except KeyError as exc:
        raise ValueError(f"unknown Sim2Sim backend {backend!r}") from exc
    checks: list[str] = []
    blockers: list[str] = []
    artifacts = _configured_paths(config, "required_artifacts", "artifact")
    digests = config.get("artifact_sha256", {})
    if not artifacts:
        blockers.append(f"{backend} has no registered model artifact")
    for artifact in artifacts:
        ok, message = inspect_artifact(workspace / artifact, digests.get(artifact))
        (checks if ok else blockers).append(message)
    for asset in _configured_paths(config, "required_assets"):
        ok, message = inspect_asset(workspace / asset)
        (checks if ok else blockers).append(message)
    capabilities = tuple(config.get("required_capabilities", ()))
    if capabilities:
        checks.append(f"host capabilities required at launch: {list(capabilities)}")
    if "nvidia_cuda_device" in capabilities:
        control_device = Path("/dev/nvidiactl")
        gpu_devices = tuple(Path("/dev").glob("nvidia[0-9]*"))
        if control_device.exists() and gpu_devices:
            checks.append(
                "NVIDIA CUDA device is mounted: "
                + ", ".join(str(path) for path in (control_device, *gpu_devices))
            )
        else:
            blockers.append(
                f"{backend} requires a CUDA device, but /dev/nvidiactl and a "
                "/dev/nvidia<N> device are not both available"
            )
    if backend == "instinctlab":
        data_root = workspace / "third_party/InstinctLab/source/instinctlab/instinctlab/tasks/parkour"
        message = f"parkour task source present: {data_root}"
        (checks if data_root.is_dir() else blockers).append(message)
    return PreflightResult(backend, not blockers, tuple(checks), tuple(blockers))


def preflight_all(root: str | Path) -> tuple[PreflightResult, ...]:
    return tuple(preflight_backend(root, name) for name in ("holosoma", "sonic", "instinctlab"))

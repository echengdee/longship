from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from longship.artifacts import (
    ArtifactApproval,
    ArtifactManifestError,
    ArtifactPolicyError,
    ArtifactReference,
    ArtifactStore,
    ArtifactVerificationError,
    ModelArtifactManifest,
    load_model_artifact_manifest,
)


class MemoryTransport:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.calls = 0

    def open(self, uri: str, *, timeout_s: float) -> io.BytesIO:
        self.calls += 1
        return io.BytesIO(self.content)


def reference(content: bytes) -> ArtifactReference:
    return ArtifactReference(
        artifact_id="policy",
        kind="policy_checkpoint",
        uri="https://models.example.org/policy.onnx",
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        media_type="application/onnx",
        required=True,
        license_id="Apache-2.0",
        gated_access=False,
    )


def manifest(
    content: bytes, *, review_status: str = "license_reviewed"
) -> ModelArtifactManifest:
    return ModelArtifactManifest(
        manifest_id="test.manifest",
        model_id="test.model",
        model_version="1",
        role="locomotion_policy",
        upstream_revision="abcdef1",
        weight_license_id="Apache-2.0",
        redistribution="redistributable",
        review_status=review_status,
        source_sha256="a" * 64,
        artifacts=(reference(content),),
    )


def approval_for(value: ModelArtifactManifest) -> ArtifactApproval:
    return ArtifactApproval(
        approval_id="trusted-registry/approval-1",
        manifest_sha256=value.source_sha256,
        artifact_ids=("policy",),
    )


def raw_manifest(content: bytes) -> dict[str, object]:
    return {
        "schema_version": "longship.model-artifact-manifest.v1",
        "manifest_id": "test.manifest",
        "model_id": "test.model",
        "model_version": "1",
        "role": "locomotion_policy",
        "upstream": {
            "project_name": "test",
            "repository_url": "https://example.org/source",
            "revision": "abcdef1",
            "code_license_id": "Apache-2.0",
            "weight_license_id": "Apache-2.0",
            "redistribution": "redistributable",
            "terms_url": "https://example.org/license",
        },
        "artifacts": [
            {
                "artifact_id": "policy",
                "kind": "policy_checkpoint",
                "uri": "https://models.example.org/policy.onnx",
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
                "media_type": "application/onnx",
                "required": True,
                "license_id": "Apache-2.0",
                "gated_access": False,
            }
        ],
        "interface": {
            "input_contracts": ["longship.policy-request.v1"],
            "output_contracts": ["longship.policy-candidate.v1"],
            "semantic_skill_ids": ["test.move"],
            "observation_profile_id": "test.obs.v1",
            "action_space_id": "test.action.v1",
        },
        "compatibility": {
            "longship_api_range": ">=0.1,<0.2",
            "target_ids": [],
            "embodiment_tags": ["test.robot"],
            "runtime_engines": ["mock"],
            "platform_architectures": ["x86_64"],
        },
        "deployment": {
            "execution_modes": ["in_process"],
            "prefetch_required": True,
            "allow_download_during_mission": False,
            "container_image": None,
            "cache_policy": {
                "key": "sha256",
                "eviction": "lru_unleased",
                "rollback_retain_count": 1,
            },
            "resources": {
                "cpu_cores": 1,
                "ram_bytes": 1,
                "disk_bytes": len(content),
                "accelerators": [],
                "vram_bytes": 0,
            },
            "warmup_timeout_ms": 1000,
        },
        "safety": {
            "output_authority": "bounded_action_chunk",
            "stale_after_ms": 20,
            "max_action_horizon_ms": 20,
            "requires_policy_guard": True,
            "requires_target_qualification": True,
            "fallback_policy": "hold_then_stop",
        },
        "provenance": {
            "created_at": "2026-08-24T00:00:00+08:00",
            "created_by": "test",
            "source_commit": "abcdef1",
            "review_status": "license_reviewed",
            "evaluation_result_refs": [],
        },
    }


class ArtifactStoreTests(unittest.TestCase):
    def test_prefetch_requires_approval_and_atomically_caches(self) -> None:
        content = b"synthetic model bytes"
        value = manifest(content)
        transport = MemoryTransport(content)
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            result = store.prefetch(
                value,
                approval_for(value),
                "policy",
                transport,
                mission_is_active=lambda: False,
            )
            self.assertEqual(result.path.read_bytes(), content)
            self.assertEqual(transport.calls, 1)
            again = store.prefetch(
                value,
                approval_for(value),
                "policy",
                transport,
                mission_is_active=lambda: False,
            )
            self.assertEqual(again.path, result.path)
            self.assertEqual(transport.calls, 1)

    def test_mission_license_and_wrong_approval_block_download(self) -> None:
        content = b"model"
        value = manifest(content)
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            with self.assertRaisesRegex(ArtifactPolicyError, "mission"):
                store.prefetch(
                    value,
                    approval_for(value),
                    "policy",
                    MemoryTransport(content),
                    mission_is_active=lambda: True,
                )
            with self.assertRaisesRegex(ArtifactPolicyError, "license"):
                draft = manifest(content, review_status="draft")
                store.prefetch(
                    draft,
                    approval_for(draft),
                    "policy",
                    MemoryTransport(content),
                    mission_is_active=lambda: False,
                )
            with self.assertRaisesRegex(ArtifactPolicyError, "approval"):
                wrong = ArtifactApproval("wrong", "b" * 64, ("policy",))
                store.prefetch(
                    value,
                    wrong,
                    "policy",
                    MemoryTransport(content),
                    mission_is_active=lambda: False,
                )

    def test_digest_mismatch_never_publishes_file(self) -> None:
        content = b"expected"
        value = manifest(content)
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            with self.assertRaisesRegex(ArtifactVerificationError, "size|digest"):
                store.prefetch(
                    value,
                    approval_for(value),
                    "policy",
                    MemoryTransport(b"tampered"),
                    mission_is_active=lambda: False,
                )
            self.assertFalse(store.path_for(reference(content)).exists())

    def test_verify_rejects_symlink(self) -> None:
        content = b"model"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(content)
            link = root / "link"
            link.symlink_to(source)
            with self.assertRaisesRegex(ArtifactVerificationError, "symlink"):
                ArtifactStore(root / "cache").verify(link, reference(content))


class ArtifactManifestTests(unittest.TestCase):
    def test_loads_complete_v1_manifest(self) -> None:
        content = b"model"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(raw_manifest(content)), encoding="utf-8")
            loaded = load_model_artifact_manifest(path)
            self.assertEqual(loaded.artifact("policy").size_bytes, len(content))
            self.assertTrue(loaded.prefetch_eligible)

    def test_rejects_non_https_duplicate_ids_and_unknown_enums(self) -> None:
        content = b"model"
        with self.assertRaises(ArtifactManifestError):
            ArtifactReference(
                artifact_id="policy",
                kind="policy_checkpoint",
                uri="file:///tmp/policy.onnx",
                sha256="0" * 64,
                size_bytes=1,
                media_type="application/onnx",
                required=True,
                license_id="Apache-2.0",
                gated_access=False,
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = raw_manifest(content)
            artifacts = duplicate["artifacts"]
            assert isinstance(artifacts, list)
            artifacts.append(dict(artifacts[0]))
            duplicate_path = root / "duplicate.json"
            duplicate_path.write_text(json.dumps(duplicate), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactManifestError, "unique"):
                load_model_artifact_manifest(duplicate_path)

            invalid = raw_manifest(content)
            upstream = invalid["upstream"]
            assert isinstance(upstream, dict)
            upstream["redistribution"] = "forbidden"
            invalid_path = root / "invalid.json"
            invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactManifestError, "redistribution"):
                load_model_artifact_manifest(invalid_path)

    def test_rejects_duplicate_json_keys(self) -> None:
        raw = '{"schema_version":"a","schema_version":"b"}'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate-key.json"
            path.write_text(raw, encoding="utf-8")
            with self.assertRaisesRegex(ArtifactManifestError, "duplicate JSON key"):
                load_model_artifact_manifest(path)


if __name__ == "__main__":
    unittest.main()

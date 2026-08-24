from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlsplit


_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,191}$")
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_ARTIFACTS = 128
_ROLES = {
    "brain",
    "dialogue",
    "asr",
    "tts",
    "perception",
    "vla_policy",
    "locomotion_policy",
    "whole_body_tracking",
    "world_model",
}
_ACTION_ROLES = {"vla_policy", "locomotion_policy", "whole_body_tracking"}
_ARTIFACT_KINDS = {
    "weights",
    "policy_checkpoint",
    "processor",
    "tokenizer",
    "normalization_statistics",
    "configuration",
    "adapter_bundle",
    "runtime_library",
    "other",
}
_REDISTRIBUTION = {"redistributable", "reference_only", "restricted", "unknown"}
_REVIEW_STATUSES = {
    "draft",
    "license_reviewed",
    "validated",
    "approved",
    "deprecated",
}
_OUTPUT_AUTHORITIES = {
    "high_level_proposal",
    "dialogue_response",
    "transcript",
    "speech_output",
    "perception_observation",
    "world_prediction",
    "bounded_action_chunk",
    "trajectory_reference",
}
_FALLBACK_POLICIES = {
    "qualified_fallback_only",
    "continue_without_role",
    "disable_role",
    "keyboard_only",
    "caption_only",
    "pause_dependent_tasks",
    "hold_then_stop",
    "immediate_safe_stop",
}


class ArtifactManifestError(ValueError):
    """Raised when an external artifact manifest is incomplete or unsafe."""


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactManifestError(f"{field} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: set[str], field: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ArtifactManifestError(
            f"{field} has invalid fields (missing={missing}, extra={extra})"
        )


def _text(value: object, field: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ArtifactManifestError(
            f"{field} must be a non-empty string of at most {maximum} characters"
        )
    return value


def _identifier(value: object, field: str) -> str:
    result = _text(value, field, maximum=192)
    if not _ID_RE.fullmatch(result):
        raise ArtifactManifestError(f"{field} is not a valid Longship identifier")
    return result


def _boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise ArtifactManifestError(f"{field} must be boolean")
    return value


def _integer(
    value: object, field: str, *, minimum: int = 0, maximum: int | None = None
) -> int:
    if type(value) is not int or value < minimum:
        raise ArtifactManifestError(f"{field} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ArtifactManifestError(f"{field} must be <= {maximum}")
    return value


def _number(value: object, field: str, *, strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactManifestError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ArtifactManifestError(f"{field} must be finite")
    if strictly_positive and result <= 0:
        raise ArtifactManifestError(f"{field} must be positive")
    if not strictly_positive and result < 0:
        raise ArtifactManifestError(f"{field} must be non-negative")
    return result


def _uri(value: object, field: str, *, https_only: bool = False) -> str:
    result = _text(value, field, maximum=4096)
    parsed = urlsplit(result)
    if not parsed.scheme or (https_only and parsed.scheme != "https"):
        raise ArtifactManifestError(f"{field} must be an absolute allowed URI")
    if parsed.scheme in {"http", "https"} and not parsed.hostname:
        raise ArtifactManifestError(f"{field} must include a hostname")
    if parsed.username or parsed.password or parsed.fragment:
        raise ArtifactManifestError(f"{field} contains forbidden URI components")
    return result


def _sequence(value: object, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ArtifactManifestError(f"{field} must be an array")
    return value


def _id_array(
    value: object, field: str, *, minimum: int = 0
) -> tuple[str, ...]:
    result = tuple(
        _identifier(item, f"{field}[{index}]")
        for index, item in enumerate(_sequence(value, field))
    )
    if len(result) < minimum or len(result) != len(set(result)):
        raise ArtifactManifestError(f"{field} has invalid length or duplicates")
    return result


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    artifact_id: str
    kind: str
    uri: str
    sha256: str
    size_bytes: int
    media_type: str
    required: bool
    license_id: str
    gated_access: bool

    def __post_init__(self) -> None:
        _identifier(self.artifact_id, "artifact_id")
        if self.kind not in _ARTIFACT_KINDS:
            raise ArtifactManifestError("artifact kind is not supported")
        _uri(self.uri, "artifact URI", https_only=True)
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ArtifactManifestError("sha256 must contain 64 lowercase hex digits")
        _integer(self.size_bytes, "size_bytes", minimum=1)
        _text(self.media_type, "media_type", maximum=128)
        _boolean(self.required, "required")
        _text(self.license_id, "license_id", maximum=128)
        _boolean(self.gated_access, "gated_access")


@dataclass(frozen=True, slots=True)
class ModelArtifactManifest:
    manifest_id: str
    model_id: str
    model_version: str
    role: str
    upstream_revision: str
    weight_license_id: str
    redistribution: str
    review_status: str
    source_sha256: str
    artifacts: tuple[ArtifactReference, ...]

    def __post_init__(self) -> None:
        _identifier(self.manifest_id, "manifest_id")
        _identifier(self.model_id, "model_id")
        _text(self.model_version, "model_version", maximum=128)
        if self.role not in _ROLES:
            raise ArtifactManifestError("manifest role is not supported")
        if len(_text(self.upstream_revision, "upstream_revision", maximum=256)) < 7:
            raise ArtifactManifestError("upstream_revision must contain 7 characters")
        _text(self.weight_license_id, "weight_license_id", maximum=128)
        if self.redistribution not in _REDISTRIBUTION:
            raise ArtifactManifestError("redistribution is not supported")
        if self.review_status not in _REVIEW_STATUSES:
            raise ArtifactManifestError("review_status is not supported")
        if not _SHA256_RE.fullmatch(self.source_sha256):
            raise ArtifactManifestError("source_sha256 must be lowercase SHA-256")
        if not isinstance(self.artifacts, tuple) or not self.artifacts:
            raise ArtifactManifestError("artifacts must be a non-empty tuple")
        if len(self.artifacts) > _MAX_ARTIFACTS:
            raise ArtifactManifestError("artifact count exceeds schema limit")
        artifact_ids = tuple(item.artifact_id for item in self.artifacts)
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ArtifactManifestError("artifact IDs must be unique")

    @property
    def prefetch_eligible(self) -> bool:
        """Return whether metadata may be submitted to a trusted approver.

        This is deliberately not download authority. ArtifactStore additionally
        requires an approval bound to this manifest's exact source digest.
        """

        reviewed = self.review_status in {"license_reviewed", "validated", "approved"}
        unknown = {"noassertion", "unknown"}
        license_known = self.weight_license_id.strip().casefold() not in unknown
        artifact_licenses_known = all(
            item.license_id.strip().casefold() not in unknown
            for item in self.artifacts
        )
        no_gated_artifacts = all(not item.gated_access for item in self.artifacts)
        return (
            reviewed
            and license_known
            and self.redistribution == "redistributable"
            and artifact_licenses_known
            and no_gated_artifacts
        )

    def artifact(self, artifact_id: str) -> ArtifactReference:
        for artifact in self.artifacts:
            if artifact.artifact_id == artifact_id:
                return artifact
        raise KeyError(artifact_id)


def _validate_complete_v1(root: Mapping[str, object]) -> None:
    _exact_keys(
        root,
        {
            "schema_version",
            "manifest_id",
            "model_id",
            "model_version",
            "role",
            "upstream",
            "artifacts",
            "interface",
            "compatibility",
            "deployment",
            "safety",
            "provenance",
        },
        "manifest",
    )
    if root["schema_version"] != "longship.model-artifact-manifest.v1":
        raise ArtifactManifestError("unsupported model artifact manifest version")

    upstream = _mapping(root["upstream"], "upstream")
    _exact_keys(
        upstream,
        {
            "project_name",
            "repository_url",
            "revision",
            "code_license_id",
            "weight_license_id",
            "redistribution",
            "terms_url",
        },
        "upstream",
    )
    _text(upstream["project_name"], "upstream.project_name")
    _uri(upstream["repository_url"], "upstream.repository_url")
    if len(_text(upstream["revision"], "upstream.revision")) < 7:
        raise ArtifactManifestError("upstream.revision must contain 7 characters")
    _text(upstream["code_license_id"], "upstream.code_license_id", maximum=128)
    _text(upstream["weight_license_id"], "upstream.weight_license_id", maximum=128)
    if upstream["redistribution"] not in _REDISTRIBUTION:
        raise ArtifactManifestError("upstream.redistribution is not supported")
    _uri(upstream["terms_url"], "upstream.terms_url")

    interface = _mapping(root["interface"], "interface")
    _exact_keys(
        interface,
        {
            "input_contracts",
            "output_contracts",
            "semantic_skill_ids",
            "observation_profile_id",
            "action_space_id",
        },
        "interface",
    )
    _id_array(interface["input_contracts"], "interface.input_contracts", minimum=1)
    _id_array(interface["output_contracts"], "interface.output_contracts", minimum=1)
    _id_array(interface["semantic_skill_ids"], "interface.semantic_skill_ids")
    _identifier(interface["observation_profile_id"], "interface.observation_profile_id")
    role = _text(root["role"], "role")
    action_space = interface["action_space_id"]
    if action_space is not None:
        _identifier(action_space, "interface.action_space_id")
    if role in _ACTION_ROLES and action_space is None:
        raise ArtifactManifestError("action-producing roles require an action space")
    if role == "brain" and action_space is not None:
        raise ArtifactManifestError("brain manifests must not declare an action space")

    compatibility = _mapping(root["compatibility"], "compatibility")
    _exact_keys(
        compatibility,
        {
            "longship_api_range",
            "target_ids",
            "embodiment_tags",
            "runtime_engines",
            "platform_architectures",
        },
        "compatibility",
    )
    _text(
        compatibility["longship_api_range"],
        "compatibility.longship_api_range",
        maximum=64,
    )
    _id_array(compatibility["target_ids"], "compatibility.target_ids")
    _id_array(compatibility["embodiment_tags"], "compatibility.embodiment_tags")
    _id_array(
        compatibility["runtime_engines"],
        "compatibility.runtime_engines",
        minimum=1,
    )
    architectures = tuple(
        _text(item, "compatibility.platform_architectures[]")
        for item in _sequence(
            compatibility["platform_architectures"],
            "compatibility.platform_architectures",
        )
    )
    if not architectures or len(architectures) != len(set(architectures)):
        raise ArtifactManifestError("platform architectures are invalid")
    if not set(architectures).issubset({"x86_64", "aarch64"}):
        raise ArtifactManifestError("platform architecture is not supported")

    deployment = _mapping(root["deployment"], "deployment")
    _exact_keys(
        deployment,
        {
            "execution_modes",
            "prefetch_required",
            "allow_download_during_mission",
            "container_image",
            "cache_policy",
            "resources",
            "warmup_timeout_ms",
        },
        "deployment",
    )
    modes = tuple(
        _text(item, "deployment.execution_modes[]")
        for item in _sequence(
            deployment["execution_modes"], "deployment.execution_modes"
        )
    )
    allowed_modes = {
        "in_process",
        "local_server",
        "edge_server",
        "remote_service",
        "offline_training",
    }
    if (
        not modes
        or len(modes) != len(set(modes))
        or not set(modes).issubset(allowed_modes)
    ):
        raise ArtifactManifestError("deployment.execution_modes is invalid")
    prefetch_required = _boolean(
        deployment["prefetch_required"], "deployment.prefetch_required"
    )
    if prefetch_required is not True:
        raise ArtifactManifestError("deployment.prefetch_required must be true")
    if _boolean(
        deployment["allow_download_during_mission"],
        "deployment.allow_download_during_mission",
    ) is not False:
        raise ArtifactManifestError("mission-time download must be false")
    if deployment["container_image"] is not None:
        container = _mapping(deployment["container_image"], "container_image")
        _exact_keys(container, {"uri", "digest"}, "container_image")
        _uri(container["uri"], "container_image.uri")
        digest = _text(container["digest"], "container_image.digest")
        if not digest.startswith("sha256:") or not _SHA256_RE.fullmatch(digest[7:]):
            raise ArtifactManifestError("container image digest is invalid")
    cache = _mapping(deployment["cache_policy"], "deployment.cache_policy")
    _exact_keys(cache, {"key", "eviction", "rollback_retain_count"}, "cache_policy")
    if cache["key"] != "sha256" or cache["eviction"] not in {
        "lru_unleased",
        "manual",
        "never",
    }:
        raise ArtifactManifestError("cache policy is invalid")
    _integer(
        cache["rollback_retain_count"],
        "rollback_retain_count",
        minimum=1,
        maximum=16,
    )
    resources = _mapping(deployment["resources"], "deployment.resources")
    _exact_keys(
        resources,
        {"cpu_cores", "ram_bytes", "disk_bytes", "accelerators", "vram_bytes"},
        "deployment.resources",
    )
    _number(resources["cpu_cores"], "resources.cpu_cores", strictly_positive=True)
    _integer(resources["ram_bytes"], "resources.ram_bytes")
    _integer(resources["disk_bytes"], "resources.disk_bytes")
    _id_array(resources["accelerators"], "resources.accelerators")
    _integer(resources["vram_bytes"], "resources.vram_bytes")
    _integer(deployment["warmup_timeout_ms"], "deployment.warmup_timeout_ms", minimum=1)

    safety = _mapping(root["safety"], "safety")
    _exact_keys(
        safety,
        {
            "output_authority",
            "stale_after_ms",
            "max_action_horizon_ms",
            "requires_policy_guard",
            "requires_target_qualification",
            "fallback_policy",
        },
        "safety",
    )
    authority = _text(safety["output_authority"], "safety.output_authority")
    if authority not in _OUTPUT_AUTHORITIES:
        raise ArtifactManifestError("safety.output_authority is invalid")
    _integer(safety["stale_after_ms"], "safety.stale_after_ms", minimum=1)
    horizon = _integer(
        safety["max_action_horizon_ms"], "safety.max_action_horizon_ms"
    )
    guard = _boolean(safety["requires_policy_guard"], "safety.requires_policy_guard")
    qualification = _boolean(
        safety["requires_target_qualification"],
        "safety.requires_target_qualification",
    )
    fallback = _text(safety["fallback_policy"], "safety.fallback_policy")
    if fallback not in _FALLBACK_POLICIES:
        raise ArtifactManifestError("safety.fallback_policy is invalid")
    if role in _ACTION_ROLES and (
        authority not in {"bounded_action_chunk", "trajectory_reference"}
        or horizon < 1
        or not guard
        or not qualification
        or fallback
        not in {"qualified_fallback_only", "hold_then_stop", "immediate_safe_stop"}
    ):
        raise ArtifactManifestError("action-producing safety policy is invalid")
    if role == "brain" and (
        authority != "high_level_proposal"
        or horizon != 0
        or guard
        or fallback
        not in {
            "qualified_fallback_only",
            "continue_without_role",
            "disable_role",
            "keyboard_only",
            "caption_only",
            "pause_dependent_tasks",
        }
    ):
        raise ArtifactManifestError("brain safety policy is invalid")

    provenance = _mapping(root["provenance"], "provenance")
    _exact_keys(
        provenance,
        {
            "created_at",
            "created_by",
            "source_commit",
            "review_status",
            "evaluation_result_refs",
        },
        "provenance",
    )
    created_at = _text(provenance["created_at"], "provenance.created_at")
    try:
        parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactManifestError("provenance.created_at is invalid") from exc
    if parsed_created_at.tzinfo is None:
        raise ArtifactManifestError("provenance.created_at must include a timezone")
    _text(provenance["created_by"], "provenance.created_by")
    source_commit = _text(
        provenance["source_commit"], "provenance.source_commit", maximum=128
    )
    if len(source_commit) < 7:
        raise ArtifactManifestError("provenance.source_commit is too short")
    if provenance["review_status"] not in _REVIEW_STATUSES:
        raise ArtifactManifestError("provenance.review_status is invalid")
    for index, value in enumerate(
        _sequence(provenance["evaluation_result_refs"], "evaluation_result_refs")
    ):
        if not isinstance(value, str):
            raise ArtifactManifestError(f"evaluation_result_refs[{index}] must be text")


def load_model_artifact_manifest(path: str | Path) -> ModelArtifactManifest:
    manifest_path = Path(path)
    try:
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ArtifactManifestError("artifact manifest must be a regular file")
        raw_bytes = manifest_path.read_bytes()
        if len(raw_bytes) > _MAX_MANIFEST_BYTES:
            raise ArtifactManifestError("artifact manifest exceeds the size limit")
        data = json.loads(
            raw_bytes.decode("utf-8"), object_pairs_hook=_unique_json_object
        )
    except ArtifactManifestError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactManifestError(
            f"cannot read artifact manifest {manifest_path}"
        ) from exc

    root = _mapping(data, "manifest")
    _validate_complete_v1(root)
    upstream = _mapping(root["upstream"], "upstream")
    provenance = _mapping(root["provenance"], "provenance")
    raw_artifacts = _sequence(root["artifacts"], "artifacts")
    if not raw_artifacts or len(raw_artifacts) > _MAX_ARTIFACTS:
        raise ArtifactManifestError("artifacts must contain 1 to 128 entries")

    artifacts: list[ArtifactReference] = []
    for index, value in enumerate(raw_artifacts):
        raw = _mapping(value, f"artifacts[{index}]")
        _exact_keys(
            raw,
            {
                "artifact_id",
                "kind",
                "uri",
                "sha256",
                "size_bytes",
                "media_type",
                "required",
                "gated_access",
                "license_id",
            },
            f"artifacts[{index}]",
        )
        artifacts.append(
            ArtifactReference(
                artifact_id=_identifier(raw["artifact_id"], "artifact_id"),
                kind=_text(raw["kind"], "kind"),
                uri=_uri(raw["uri"], "uri", https_only=True),
                sha256=_text(raw["sha256"], "sha256", maximum=64),
                size_bytes=_integer(raw["size_bytes"], "size_bytes", minimum=1),
                media_type=_text(raw["media_type"], "media_type", maximum=128),
                required=_boolean(raw["required"], "required"),
                license_id=_text(raw["license_id"], "license_id", maximum=128),
                gated_access=_boolean(raw["gated_access"], "gated_access"),
            )
        )

    return ModelArtifactManifest(
        manifest_id=_identifier(root["manifest_id"], "manifest_id"),
        model_id=_identifier(root["model_id"], "model_id"),
        model_version=_text(root["model_version"], "model_version", maximum=128),
        role=_text(root["role"], "role"),
        upstream_revision=_text(upstream["revision"], "upstream.revision"),
        weight_license_id=_text(
            upstream["weight_license_id"], "upstream.weight_license_id", maximum=128
        ),
        redistribution=_text(upstream["redistribution"], "upstream.redistribution"),
        review_status=_text(
            provenance["review_status"], "provenance.review_status"
        ),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        artifacts=tuple(artifacts),
    )

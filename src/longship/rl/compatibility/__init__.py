from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "longship.rl-compatibility.v1"


class CompatibilityLockError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CompatibilityLock:
    platform_version: str
    values: Mapping[str, Any]

    @classmethod
    def bundled(cls) -> "CompatibilityLock":
        resource = files(__package__).joinpath("longship_rl_v1.yaml")
        return cls.from_mapping(yaml.safe_load(resource.read_text(encoding="utf-8")))

    @classmethod
    def load(cls, path: str | Path) -> "CompatibilityLock":
        resolved = Path(path)
        return cls.from_mapping(yaml.safe_load(resolved.read_text(encoding="utf-8")))

    @classmethod
    def from_mapping(cls, value: Any) -> "CompatibilityLock":
        if not isinstance(value, dict):
            raise CompatibilityLockError("compatibility lock must be a YAML mapping")
        required = {"schema_version", "platform_version", "runtime", "sources", "contracts", "backends"}
        missing = required - set(value)
        unknown = set(value) - required
        if missing:
            raise CompatibilityLockError(f"compatibility lock is missing: {sorted(missing)}")
        if unknown:
            raise CompatibilityLockError(f"compatibility lock has unknown fields: {sorted(unknown)}")
        if value["schema_version"] != SCHEMA_VERSION:
            raise CompatibilityLockError(f"unsupported schema_version {value['schema_version']!r}")
        platform_version = value["platform_version"]
        if not isinstance(platform_version, str) or not platform_version:
            raise CompatibilityLockError("platform_version must be a non-empty string")
        for field in required - {"schema_version", "platform_version"}:
            if not isinstance(value[field], dict) or not value[field]:
                raise CompatibilityLockError(f"{field} must be a non-empty mapping")
        return cls(platform_version, MappingProxyType(dict(value)))


__all__ = ["CompatibilityLock", "CompatibilityLockError", "SCHEMA_VERSION"]

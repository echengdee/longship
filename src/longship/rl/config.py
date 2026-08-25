from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "longship.rl-experiment.v1"
_MAX_CONFIG_BYTES = 1_000_000
_REQUIRED_FIELDS = {"schema_version", "name", "model", "training", "environment"}
_OPTIONAL_FIELDS = {"description", "seed", "data", "tags"}


class ExperimentConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    path: Path
    name: str
    values: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        resolved = Path(path).resolve()
        if resolved.stat().st_size > _MAX_CONFIG_BYTES:
            raise ExperimentConfigError("experiment config exceeds the 1 MB limit")
        try:
            value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ExperimentConfigError(f"invalid YAML in {resolved}: {exc}") from exc
        return cls.from_mapping(value, path=resolved)

    @classmethod
    def from_mapping(
        cls, value: Any, *, path: str | Path = Path("<memory>")
    ) -> "ExperimentConfig":
        if not isinstance(value, dict):
            raise ExperimentConfigError("experiment config must be a YAML mapping")
        fields = set(value)
        missing = _REQUIRED_FIELDS - fields
        unknown = fields - _REQUIRED_FIELDS - _OPTIONAL_FIELDS
        if missing:
            raise ExperimentConfigError(f"experiment config is missing: {sorted(missing)}")
        if unknown:
            raise ExperimentConfigError(f"experiment config has unknown fields: {sorted(unknown)}")
        if value["schema_version"] != SCHEMA_VERSION:
            raise ExperimentConfigError(
                f"unsupported schema_version {value['schema_version']!r}; expected {SCHEMA_VERSION!r}"
            )
        name = value["name"]
        if not isinstance(name, str) or not name.strip():
            raise ExperimentConfigError("experiment name must be a non-empty string")
        for field in ("model", "training", "environment"):
            if not isinstance(value[field], dict) or not value[field]:
                raise ExperimentConfigError(f"{field} must be a non-empty mapping")
        model = value["model"]
        if not isinstance(model.get("type"), str) or not model["type"].strip():
            raise ExperimentConfigError("model requires a non-empty type")
        training = value["training"]
        for field in ("backend", "trainer"):
            if not isinstance(training.get(field), dict) or not training[field].get("type"):
                raise ExperimentConfigError(f"training.{field} requires a type")
        environment = value["environment"]
        for field in ("robot", "task"):
            if not isinstance(environment.get(field), str) or not environment[field].strip():
                raise ExperimentConfigError(f"environment.{field} must be a non-empty string")
        return cls(path=Path(path), name=name.strip(), values=MappingProxyType(dict(value)))

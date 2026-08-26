from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar


Factory = Callable[..., Any]
T = TypeVar("T", bound=Factory)


class RegistryError(ValueError):
    pass


class ComponentRegistry:
    """Typed registry for model parts and lifecycle adapters.

    Registrations are split by kind so an encoder cannot accidentally be used
    as a policy or deployment target merely because the names match.
    """

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Factory]] = {}

    def register(self, kind: str, name: str | None = None) -> Callable[[T], T]:
        normalized_kind = self._normalize(kind, "kind")

        def decorator(factory: T) -> T:
            component_name = self._normalize(name or factory.__name__, "name")
            entries = self._entries.setdefault(normalized_kind, {})
            if component_name in entries:
                raise RegistryError(
                    f"component {component_name!r} is already registered as {normalized_kind!r}"
                )
            entries[component_name] = factory
            return factory

        return decorator

    def create(self, kind: str, config: Mapping[str, Any], **injected: Any) -> Any:
        normalized_kind = self._normalize(kind, "kind")
        if not isinstance(config, Mapping):
            raise RegistryError(f"{normalized_kind} config must be a mapping")
        component_name = config.get("type")
        if not isinstance(component_name, str) or not component_name.strip():
            raise RegistryError(f"{normalized_kind} config requires a non-empty 'type'")
        try:
            factory = self._entries[normalized_kind][component_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._entries.get(normalized_kind, {}))) or "none"
            raise RegistryError(
                f"unknown {normalized_kind} type {component_name!r}; registered: {available}"
            ) from exc
        arguments = {key: value for key, value in config.items() if key != "type"}
        overlap = set(arguments) & set(injected)
        if overlap:
            raise RegistryError(f"injected arguments conflict with config: {sorted(overlap)}")
        return factory(**arguments, **injected)

    def names(self, kind: str) -> tuple[str, ...]:
        return tuple(sorted(self._entries.get(self._normalize(kind, "kind"), {})))

    @staticmethod
    def _normalize(value: str, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise RegistryError(f"registry {label} must be a non-empty string")
        return value.strip()


components = ComponentRegistry()

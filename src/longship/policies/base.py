from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping, Protocol


class PolicyError(RuntimeError):
    """Base error for provider-neutral policy integration failures."""


class PolicyCandidateRejected(PolicyError):
    """Raised when an untrusted policy result fails the deterministic guard."""


def _require_id(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _require_finite(value: float, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")


def _normalize_scope(values: tuple[str, ...], field: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{field} must be a non-empty tuple")
    for value in values:
        _require_id(value, field)
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must not contain duplicates")
    return values


def _freeze_payload(value: object, field: str) -> object:
    """Copy containers and reject mutable opaque payload values."""

    if value is None or isinstance(value, (str, bytes, bool, int, Enum)):
        return value
    if isinstance(value, float):
        _require_finite(value, field)
        return value
    if isinstance(value, Mapping):
        frozen: dict[object, object] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool, bytes, Enum)):
                raise TypeError(f"{field} contains a mutable or opaque mapping key")
            frozen[key] = _freeze_payload(item, f"{field}[{key!r}]")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_payload(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_payload(item, field) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        parameters = getattr(type(value), "__dataclass_params__", None)
        if parameters is None or not parameters.frozen:
            raise TypeError(f"{field} dataclass must be frozen")
        for item in fields(value):
            _assert_deeply_immutable(
                getattr(value, item.name), f"{field}.{item.name}"
            )
        return value
    raise TypeError(f"{field} contains a mutable or opaque value")


def _assert_deeply_immutable(value: object, field: str) -> None:
    if value is None or isinstance(value, (str, bytes, bool, int, Enum)):
        return
    if isinstance(value, float):
        _require_finite(value, field)
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _assert_deeply_immutable(item, f"{field}[{index}]")
        return
    if isinstance(value, frozenset):
        for item in value:
            _assert_deeply_immutable(item, field)
        return
    if isinstance(value, MappingProxyType):
        for key, item in value.items():
            _assert_deeply_immutable(key, f"{field}.key")
            _assert_deeply_immutable(item, f"{field}[{key!r}]")
        return
    if is_dataclass(value) and not isinstance(value, type):
        parameters = getattr(type(value), "__dataclass_params__", None)
        if parameters is None or not parameters.frozen:
            raise TypeError(f"{field} dataclass must be frozen")
        for item in fields(value):
            _assert_deeply_immutable(
                getattr(value, item.name), f"{field}.{item.name}"
            )
        return
    raise TypeError(f"{field} must already be deeply immutable")


@dataclass(frozen=True, slots=True)
class PolicyRequest:
    """Version-bound request sent to an action-producing policy backend.

    The payload is provider-specific, but authority and freshness fields are
    provider-neutral. A request is not permission to contact a target; it only
    authorizes one bounded inference call under an existing Runtime lease.
    """

    call_id: str
    model_binding_id: str
    lease_id: str
    lease_epoch: int
    observation_version: int
    deadline_monotonic: float
    max_action_horizon_ms: int
    resource_scope: tuple[str, ...]
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_id(self.call_id, "call_id")
        _require_id(self.model_binding_id, "model_binding_id")
        _require_id(self.lease_id, "lease_id")
        if type(self.lease_epoch) is not int or self.lease_epoch < 0:
            raise ValueError("lease_epoch must be a non-negative integer")
        if type(self.observation_version) is not int or self.observation_version < 0:
            raise ValueError("observation_version must be a non-negative integer")
        _require_finite(self.deadline_monotonic, "deadline_monotonic")
        if self.deadline_monotonic < 0:
            raise ValueError("deadline_monotonic must be non-negative")
        if (
            type(self.max_action_horizon_ms) is not int
            or self.max_action_horizon_ms <= 0
        ):
            raise ValueError("max_action_horizon_ms must be a positive integer")
        _normalize_scope(self.resource_scope, "resource_scope")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        object.__setattr__(self, "payload", _freeze_payload(self.payload, "payload"))


@dataclass(frozen=True, slots=True)
class PolicyActionFrame:
    """One numeric action vector at a relative offset in a candidate chunk."""

    offset_ms: int
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.offset_ms) is not int or self.offset_ms < 0:
            raise ValueError("offset_ms must be a non-negative integer")
        if not isinstance(self.values, tuple) or not self.values:
            raise ValueError("values must be a non-empty tuple")
        for index, value in enumerate(self.values):
            _require_finite(value, f"values[{index}]")


@dataclass(frozen=True, slots=True)
class PolicyCandidate:
    """Untrusted policy output that still requires guarding and arbitration."""

    call_id: str
    model_binding_id: str
    lease_id: str
    lease_epoch: int
    observation_version: int
    generated_at_monotonic: float
    expires_at_monotonic: float
    action_space_id: str
    resource_scope: tuple[str, ...]
    frames: tuple[PolicyActionFrame, ...]

    def __post_init__(self) -> None:
        _require_id(self.call_id, "call_id")
        _require_id(self.model_binding_id, "model_binding_id")
        _require_id(self.lease_id, "lease_id")
        if type(self.lease_epoch) is not int or self.lease_epoch < 0:
            raise ValueError("lease_epoch must be a non-negative integer")
        _require_id(self.action_space_id, "action_space_id")
        if type(self.observation_version) is not int or self.observation_version < 0:
            raise ValueError("observation_version must be a non-negative integer")
        _require_finite(self.generated_at_monotonic, "generated_at_monotonic")
        _require_finite(self.expires_at_monotonic, "expires_at_monotonic")
        if self.expires_at_monotonic <= self.generated_at_monotonic:
            raise ValueError("candidate expiry must be after generation")
        _normalize_scope(self.resource_scope, "resource_scope")
        if not isinstance(self.frames, tuple) or not self.frames:
            raise ValueError("frames must be a non-empty tuple")
        previous = -1
        for frame in self.frames:
            if not isinstance(frame, PolicyActionFrame):
                raise TypeError("frames must contain PolicyActionFrame values")
            if frame.offset_ms <= previous:
                raise ValueError("frame offsets must be strictly increasing")
            previous = frame.offset_ms


@dataclass(frozen=True, slots=True)
class PolicyGuardProfile:
    """Deterministic checks applied after provider inference."""

    action_space_id: str
    action_dimension: int
    permitted_resource_scope: tuple[str, ...]
    max_action_horizon_ms: int
    minimum_action_value: float | None = None
    maximum_action_value: float | None = None
    future_clock_tolerance_s: float = 0.005

    def __post_init__(self) -> None:
        _require_id(self.action_space_id, "action_space_id")
        if type(self.action_dimension) is not int or self.action_dimension <= 0:
            raise ValueError("action_dimension must be a positive integer")
        _normalize_scope(
            self.permitted_resource_scope, "permitted_resource_scope"
        )
        if (
            type(self.max_action_horizon_ms) is not int
            or self.max_action_horizon_ms <= 0
        ):
            raise ValueError("max_action_horizon_ms must be a positive integer")
        if self.minimum_action_value is not None:
            _require_finite(self.minimum_action_value, "minimum_action_value")
        if self.maximum_action_value is not None:
            _require_finite(self.maximum_action_value, "maximum_action_value")
        if (
            self.minimum_action_value is not None
            and self.maximum_action_value is not None
            and self.minimum_action_value > self.maximum_action_value
        ):
            raise ValueError("minimum_action_value must not exceed maximum")
        _require_finite(self.future_clock_tolerance_s, "future_clock_tolerance_s")
        if self.future_clock_tolerance_s < 0:
            raise ValueError("future_clock_tolerance_s must be non-negative")


class PolicyBackend(Protocol):
    async def infer(self, request: PolicyRequest) -> PolicyCandidate:
        ...


def guard_candidate(
    request: PolicyRequest,
    candidate: PolicyCandidate,
    profile: PolicyGuardProfile,
    *,
    now_monotonic: float,
) -> PolicyCandidate:
    """Fail closed unless a candidate is current and bound to this request."""

    _require_finite(now_monotonic, "now_monotonic")
    identities = (
        ("call_id", request.call_id, candidate.call_id),
        ("model_binding_id", request.model_binding_id, candidate.model_binding_id),
        ("lease_id", request.lease_id, candidate.lease_id),
        ("lease_epoch", request.lease_epoch, candidate.lease_epoch),
        (
            "observation_version",
            request.observation_version,
            candidate.observation_version,
        ),
    )
    for field, expected, actual in identities:
        if expected != actual:
            raise PolicyCandidateRejected(f"candidate {field} does not match request")

    if now_monotonic > request.deadline_monotonic:
        raise PolicyCandidateRejected("policy request deadline has expired")
    if candidate.generated_at_monotonic > (
        now_monotonic + profile.future_clock_tolerance_s
    ):
        raise PolicyCandidateRejected("candidate generation time is in the future")
    if now_monotonic >= candidate.expires_at_monotonic:
        raise PolicyCandidateRejected("candidate has expired")
    if candidate.action_space_id != profile.action_space_id:
        raise PolicyCandidateRejected("candidate action space is not qualified")

    request_scope = set(request.resource_scope)
    candidate_scope = set(candidate.resource_scope)
    permitted_scope = set(profile.permitted_resource_scope)
    if not candidate_scope.issubset(request_scope):
        raise PolicyCandidateRejected("candidate escalates the request resource scope")
    if not candidate_scope.issubset(permitted_scope):
        raise PolicyCandidateRejected("candidate uses an unqualified resource scope")

    allowed_horizon = min(
        request.max_action_horizon_ms, profile.max_action_horizon_ms
    )
    if candidate.frames[-1].offset_ms >= allowed_horizon:
        raise PolicyCandidateRejected("candidate action horizon exceeds the limit")
    if candidate.expires_at_monotonic > (
        candidate.generated_at_monotonic + allowed_horizon / 1000.0
    ):
        raise PolicyCandidateRejected("candidate expiry exceeds its action horizon")

    for frame in candidate.frames:
        frame_time = candidate.generated_at_monotonic + frame.offset_ms / 1000.0
        if frame_time >= candidate.expires_at_monotonic:
            raise PolicyCandidateRejected("candidate frame occurs at or after expiry")
        if len(frame.values) != profile.action_dimension:
            raise PolicyCandidateRejected("candidate action dimension is invalid")
        for value in frame.values:
            if (
                profile.minimum_action_value is not None
                and value < profile.minimum_action_value
            ):
                raise PolicyCandidateRejected("candidate action is below its bound")
            if (
                profile.maximum_action_value is not None
                and value > profile.maximum_action_value
            ):
                raise PolicyCandidateRejected("candidate action exceeds its bound")
    return candidate


class GuardedPolicyProvider:
    """Apply a deadline and deterministic guard around any injected backend."""

    def __init__(
        self,
        backend: PolicyBackend,
        profile: PolicyGuardProfile,
        *,
        lease_is_current: Callable[[PolicyRequest], bool],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(lease_is_current):
            raise TypeError("lease_is_current must be callable")
        self._backend = backend
        self._profile = profile
        self._lease_is_current = lease_is_current
        self._clock = clock

    def _require_current_lease(self, request: PolicyRequest, message: str) -> None:
        try:
            current = self._lease_is_current(request)
        except Exception as exc:
            raise PolicyCandidateRejected("policy lease validation failed") from exc
        if current is not True:
            raise PolicyCandidateRejected(message)

    async def infer(self, request: PolicyRequest) -> PolicyCandidate:
        self._require_current_lease(request, "policy lease is not current")
        remaining = request.deadline_monotonic - self._clock()
        if remaining <= 0:
            raise PolicyCandidateRejected("policy request deadline has expired")
        try:
            candidate = await asyncio.wait_for(
                self._backend.infer(request), timeout=remaining
            )
        except asyncio.TimeoutError as exc:
            raise PolicyCandidateRejected(
                "policy backend exceeded its deadline"
            ) from exc
        self._require_current_lease(
            request, "policy lease was revoked during inference"
        )
        return guard_candidate(
            request,
            candidate,
            self._profile,
            now_monotonic=self._clock(),
        )

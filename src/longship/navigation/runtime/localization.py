"""Plugin-neutral lifecycle owner for continuous localization."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
import math
from typing import Awaitable, Callable, Protocol, runtime_checkable

from longship.navigation.localization_engine.service import (
    LocalizationServiceState,
    LocalizationServiceStatus,
)


class LocalizationRuntimeState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAULTED = "faulted"


class LocalizationObservationProducerState(str, Enum):
    """Lifecycle states shared by camera and finite replay producers."""

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAULTED = "faulted"


class LocalizationObservationCompletionPolicy(str, Enum):
    """How Runtime treats natural completion of a finite producer."""

    FAULT_RUNTIME = "fault_runtime"
    ALLOW_UNTIL_STOP = "allow_until_stop"


@dataclass(frozen=True, slots=True)
class LocalizationRuntimeConfig:
    """Lifecycle bounds applied around injected runtime components."""

    component_stop_timeout_s: float = 5.0
    observation_completion_policy: (
        LocalizationObservationCompletionPolicy
    ) = LocalizationObservationCompletionPolicy.FAULT_RUNTIME

    def validate(self) -> None:
        if not math.isfinite(self.component_stop_timeout_s):
            raise ValueError("component_stop_timeout_s must be finite")
        if self.component_stop_timeout_s <= 0.0:
            raise ValueError("component_stop_timeout_s must be positive")
        if not isinstance(
            self.observation_completion_policy,
            LocalizationObservationCompletionPolicy,
        ):
            raise TypeError(
                "observation_completion_policy must be a "
                "LocalizationObservationCompletionPolicy"
            )


@dataclass(frozen=True, slots=True)
class LocalizationObservationProducerStatus:
    state: LocalizationObservationProducerState
    detail_code: str | None
    last_error: str | None


@dataclass(frozen=True, slots=True)
class LocalizationRuntimeStatus:
    state: LocalizationRuntimeState
    observation_producer: LocalizationObservationProducerStatus
    service: LocalizationServiceStatus
    detail_code: str | None = None
    last_error: str | None = None


class LocalizationRuntimeError(RuntimeError):
    """Runtime composition or lifecycle failure."""


@runtime_checkable
class LocalizationObservationProducer(Protocol):
    """Owns camera or replay submissions into an injected policy ingress."""

    def get_status(self) -> LocalizationObservationProducerStatus:
        """Returns a non-blocking lifecycle and health snapshot."""
        ...

    async def start(self) -> None:
        """Begins producing observations and returns after startup."""
        ...

    async def stop(self) -> None:
        """Stops new submissions and waits for the producer to quiesce."""
        ...

    async def wait_stopped(
        self,
        timeout_s: float | None = None,
    ) -> LocalizationObservationProducerStatus:
        """Waits for completion, an explicit stop, or a producer fault."""
        ...


@runtime_checkable
class LocalizationRuntimeResource(Protocol):
    """A resource closed after observation production and policy ticks stop."""

    async def close(self) -> None:
        """Releases the resource without accepting new policy work."""
        ...


@runtime_checkable
class LocalizationTickService(Protocol):
    """Lifecycle surface implemented by a continuous localization runner."""

    def get_status(self) -> LocalizationServiceStatus: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def wait_stopped(
        self,
        timeout_s: float | None = None,
    ) -> LocalizationServiceStatus: ...


class LocalizationRuntime:
    """Coordinates one observation producer and localization tick service.

    Components are injected so this module never imports a policy plugin,
    camera backend, tensor framework, or model executor. Shutdown order is
    fixed: stop observation submissions, stop policy ticks, then close injected
    resources in declaration order.
    """

    def __init__(
        self,
        *,
        observation_producer: LocalizationObservationProducer,
        localization_service: LocalizationTickService,
        shutdown_resources: tuple[LocalizationRuntimeResource, ...] = (),
        config: LocalizationRuntimeConfig = LocalizationRuntimeConfig(),
    ) -> None:
        config.validate()
        self._observation_producer = observation_producer
        self._localization_service = localization_service
        self._shutdown_resources = shutdown_resources
        self._config = config
        self._state = LocalizationRuntimeState.CREATED
        self._detail_code: str | None = None
        self._last_error: str | None = None
        self._stop_requested = asyncio.Event()
        self._terminated = asyncio.Event()
        self._lifecycle_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
        self._producer_monitor: asyncio.Task[None] | None = None
        self._service_monitor: asyncio.Task[None] | None = None

    def get_status(self) -> LocalizationRuntimeStatus:
        return LocalizationRuntimeStatus(
            state=self._state,
            observation_producer=self._observation_producer.get_status(),
            service=self._localization_service.get_status(),
            detail_code=self._detail_code,
            last_error=self._last_error,
        )

    async def start(self) -> None:
        """Starts observation production before continuous policy ticks."""

        async with self._lifecycle_lock:
            if self._state != LocalizationRuntimeState.CREATED:
                raise LocalizationRuntimeError(
                    "localization runtime can only start from CREATED"
                )
            self._state = LocalizationRuntimeState.STARTING
            self._detail_code = "starting_observation_producer"
            try:
                await self._observation_producer.start()
                self._detail_code = "starting_localization_service"
                await self._localization_service.start()
            except BaseException as error:
                self._state = LocalizationRuntimeState.STOPPING
                cleanup_errors = await self._cleanup_components()
                self._state = LocalizationRuntimeState.FAULTED
                self._detail_code = f"startup_failed:{type(error).__name__}"
                self._last_error = _join_errors(error, cleanup_errors)
                self._terminated.set()
                if isinstance(error, asyncio.CancelledError):
                    raise
                raise LocalizationRuntimeError(
                    "localization runtime startup failed"
                ) from error

            self._state = LocalizationRuntimeState.RUNNING
            self._detail_code = "running"
            self._producer_monitor = asyncio.create_task(
                self._monitor_observation_producer(),
                name="localization-runtime-producer-monitor",
            )
            self._service_monitor = asyncio.create_task(
                self._monitor_service(),
                name="localization-runtime-service-monitor",
            )

    async def stop(self) -> None:
        """Stops all components in dependency order and is idempotent."""

        async with self._lifecycle_lock:
            self._stop_requested.set()
            await self._shutdown(
                final_state=LocalizationRuntimeState.STOPPED,
                detail_code="stopped",
            )

    async def wait_stopped(
        self,
        timeout_s: float | None = None,
    ) -> LocalizationRuntimeStatus:
        """Waits for explicit shutdown or automatic component cleanup."""

        if timeout_s is not None:
            if not math.isfinite(timeout_s) or timeout_s < 0.0:
                raise ValueError("timeout_s must be finite and non-negative")
            await asyncio.wait_for(self._terminated.wait(), timeout=timeout_s)
        else:
            await self._terminated.wait()
        return self.get_status()

    async def _monitor_observation_producer(self) -> None:
        try:
            producer_status = (
                await self._observation_producer.wait_stopped()
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self._stop_requested.is_set():
                return
            await self._shutdown(
                final_state=LocalizationRuntimeState.FAULTED,
                detail_code=(
                    "observation_producer_monitor_failed:"
                    f"{type(error).__name__}"
                ),
                primary_error=str(error),
            )
            return

        if self._stop_requested.is_set():
            return
        if producer_status.state == (
            LocalizationObservationProducerState.COMPLETED
        ):
            if self._config.observation_completion_policy == (
                LocalizationObservationCompletionPolicy.ALLOW_UNTIL_STOP
            ):
                self._detail_code = "observation_producer_completed"
                return
            await self._shutdown(
                final_state=LocalizationRuntimeState.FAULTED,
                detail_code="observation_producer_completed_unexpectedly",
                primary_error=(
                    producer_status.last_error
                    or "continuous observation producer reached completion"
                ),
            )
            return
        if producer_status.state == (
            LocalizationObservationProducerState.FAULTED
        ):
            detail_code = producer_status.detail_code or "unknown"
            await self._shutdown(
                final_state=LocalizationRuntimeState.FAULTED,
                detail_code=(
                    f"observation_producer_faulted:{detail_code}"
                ),
                primary_error=(
                    producer_status.last_error or detail_code
                ),
            )
            return
        await self._shutdown(
            final_state=LocalizationRuntimeState.FAULTED,
            detail_code="observation_producer_stopped_unexpectedly",
            primary_error=(
                producer_status.last_error
                or "observation producer stopped without a request"
            ),
        )

    async def _monitor_service(self) -> None:
        try:
            service_status = await self._localization_service.wait_stopped()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self._stop_requested.is_set():
                return
            await self._shutdown(
                final_state=LocalizationRuntimeState.FAULTED,
                detail_code=(
                    "localization_service_monitor_failed:"
                    f"{type(error).__name__}"
                ),
                primary_error=str(error),
            )
            return
        if self._stop_requested.is_set():
            return
        if service_status.state == LocalizationServiceState.FAULTED:
            detail_code = service_status.detail_code or "unknown"
            last_error = service_status.last_error or detail_code
            await self._shutdown(
                final_state=LocalizationRuntimeState.FAULTED,
                detail_code=f"localization_service_faulted:{detail_code}",
                primary_error=last_error,
            )
            return
        await self._shutdown(
            final_state=LocalizationRuntimeState.FAULTED,
            detail_code="localization_service_stopped_unexpectedly",
            primary_error="localization service stopped without a request",
        )

    async def _shutdown(
        self,
        *,
        final_state: LocalizationRuntimeState,
        detail_code: str,
        primary_error: str | None = None,
    ) -> None:
        async with self._shutdown_lock:
            if self._state in (
                LocalizationRuntimeState.STOPPED,
                LocalizationRuntimeState.FAULTED,
            ):
                return
            self._state = LocalizationRuntimeState.STOPPING
            self._detail_code = "stopping_components"
            cleanup_errors = await self._cleanup_components()
            if cleanup_errors:
                self._state = LocalizationRuntimeState.FAULTED
                self._detail_code = "shutdown_failed"
                self._last_error = _join_errors(
                    primary_error,
                    cleanup_errors,
                )
            else:
                self._state = final_state
                self._detail_code = detail_code
                self._last_error = primary_error
            self._terminated.set()

    async def _cleanup_components(self) -> list[str]:
        operations: list[tuple[str, Callable[[], Awaitable[None]]]] = [
            ("observation_producer.stop", self._observation_producer.stop),
            ("localization_service.stop", self._localization_service.stop),
        ]
        operations.extend(
            (f"shutdown_resource[{index}].close", resource.close)
            for index, resource in enumerate(self._shutdown_resources)
        )
        errors = []
        for label, operation in operations:
            try:
                await asyncio.wait_for(
                    operation(),
                    timeout=self._config.component_stop_timeout_s,
                )
            except Exception as error:
                errors.append(f"{label}:{type(error).__name__}:{error}")
        return errors


def _join_errors(
    primary_error: BaseException | str | None,
    cleanup_errors: list[str],
) -> str:
    errors = []
    if primary_error is not None:
        errors.append(str(primary_error))
    errors.extend(cleanup_errors)
    return "; ".join(errors)

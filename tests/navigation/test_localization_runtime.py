"""Tests for plugin-neutral Localization Runtime lifecycle ownership."""

from __future__ import annotations

import asyncio
import unittest

from longship.navigation.localization_engine.service import (
    LocalizationServiceState,
    LocalizationServiceStatus,
)
from longship.navigation.runtime import (
    LocalizationObservationCompletionPolicy,
    LocalizationObservationProducerState,
    LocalizationObservationProducerStatus,
    LocalizationRuntime,
    LocalizationRuntimeConfig,
    LocalizationRuntimeError,
    LocalizationRuntimeState,
)


class _RecordingProducer:
    def __init__(
        self,
        events: list[str],
        *,
        fail_wait: bool = False,
    ) -> None:
        self._events = events
        self._fail_wait = fail_wait
        self._state = LocalizationObservationProducerState.CREATED
        self._detail_code: str | None = None
        self._last_error: str | None = None
        self._terminated = asyncio.Event()

    def get_status(self) -> LocalizationObservationProducerStatus:
        return LocalizationObservationProducerStatus(
            state=self._state,
            detail_code=self._detail_code,
            last_error=self._last_error,
        )

    async def start(self) -> None:
        self._events.append("producer.start")
        self._state = LocalizationObservationProducerState.RUNNING
        self._detail_code = "running"

    async def stop(self) -> None:
        self._events.append("producer.stop")
        if self._state != LocalizationObservationProducerState.FAULTED:
            self._state = LocalizationObservationProducerState.STOPPED
            self._detail_code = "stopped"
        self._terminated.set()

    async def wait_stopped(
        self,
        timeout_s: float | None = None,
    ) -> LocalizationObservationProducerStatus:
        if self._fail_wait:
            raise RuntimeError("producer monitor failed")
        if timeout_s is None:
            await self._terminated.wait()
        else:
            await asyncio.wait_for(self._terminated.wait(), timeout_s)
        return self.get_status()

    def complete(self) -> None:
        self._state = LocalizationObservationProducerState.COMPLETED
        self._detail_code = "source_exhausted"
        self._terminated.set()

    def fault(self) -> None:
        self._state = LocalizationObservationProducerState.FAULTED
        self._detail_code = "camera_disconnected"
        self._last_error = "camera stream ended"
        self._terminated.set()


class _RecordingService:
    def __init__(
        self,
        events: list[str],
        *,
        fail_start: bool = False,
        fail_wait: bool = False,
    ) -> None:
        self._events = events
        self._fail_start = fail_start
        self._fail_wait = fail_wait
        self._state = LocalizationServiceState.CREATED
        self._terminated = asyncio.Event()

    def get_status(self) -> LocalizationServiceStatus:
        return LocalizationServiceStatus(
            state=self._state,
            tick_period_s=0.25,
            ticks_completed=0,
            skipped_tick_slots=0,
            started_at=None,
            last_tick_at=None,
        )

    async def start(self) -> None:
        self._events.append("service.start")
        if self._fail_start:
            raise RuntimeError("scripted startup failure")
        self._state = LocalizationServiceState.RUNNING

    async def stop(self) -> None:
        self._events.append("service.stop")
        if self._state != LocalizationServiceState.FAULTED:
            self._state = LocalizationServiceState.STOPPED
        self._terminated.set()

    async def wait_stopped(
        self,
        timeout_s: float | None = None,
    ) -> LocalizationServiceStatus:
        if self._fail_wait:
            raise RuntimeError("service monitor failed")
        if timeout_s is None:
            await self._terminated.wait()
        else:
            await asyncio.wait_for(self._terminated.wait(), timeout_s)
        return self.get_status()

    def fault(self) -> None:
        self._state = LocalizationServiceState.FAULTED
        self._terminated.set()


class _RecordingResource:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        fail_close: bool = False,
    ) -> None:
        self._name = name
        self._events = events
        self._fail_close = fail_close

    async def close(self) -> None:
        self._events.append(f"{self._name}.close")
        if self._fail_close:
            raise RuntimeError(f"{self._name} close failed")


class LocalizationRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_component_lifecycle_order(self) -> None:
        events: list[str] = []
        service = _RecordingService(events)
        runtime = LocalizationRuntime(
            observation_producer=_RecordingProducer(events),
            localization_service=service,
            shutdown_resources=(
                _RecordingResource("executor", events),
                _RecordingResource("map", events),
            ),
        )

        await runtime.start()
        await runtime.stop()
        await runtime.stop()

        self.assertEqual(
            events,
            [
                "producer.start",
                "service.start",
                "producer.stop",
                "service.stop",
                "executor.close",
                "map.close",
            ],
        )
        self.assertEqual(
            runtime.get_status().state,
            LocalizationRuntimeState.STOPPED,
        )

    async def test_service_fault_stops_producer_and_resources(self) -> None:
        events: list[str] = []
        service = _RecordingService(events)
        runtime = LocalizationRuntime(
            observation_producer=_RecordingProducer(events),
            localization_service=service,
            shutdown_resources=(
                _RecordingResource("executor", events),
            ),
        )
        await runtime.start()

        service.fault()
        status = await runtime.wait_stopped(timeout_s=0.5)

        self.assertEqual(status.state, LocalizationRuntimeState.FAULTED)
        self.assertEqual(
            status.detail_code,
            "localization_service_faulted:unknown",
        )
        self.assertEqual(
            events,
            [
                "producer.start",
                "service.start",
                "producer.stop",
                "service.stop",
                "executor.close",
            ],
        )

    async def test_producer_fault_stops_service_and_resources(self) -> None:
        events: list[str] = []
        producer = _RecordingProducer(events)
        service = _RecordingService(events)
        runtime = LocalizationRuntime(
            observation_producer=producer,
            localization_service=service,
            shutdown_resources=(
                _RecordingResource("executor", events),
            ),
        )
        await runtime.start()

        producer.fault()
        status = await runtime.wait_stopped(timeout_s=0.5)

        self.assertEqual(status.state, LocalizationRuntimeState.FAULTED)
        self.assertEqual(
            status.detail_code,
            "observation_producer_faulted:camera_disconnected",
        )
        self.assertEqual(
            status.observation_producer.state,
            LocalizationObservationProducerState.FAULTED,
        )
        self.assertEqual(
            events,
            [
                "producer.start",
                "service.start",
                "producer.stop",
                "service.stop",
                "executor.close",
            ],
        )

    async def test_unexpected_producer_completion_faults_runtime(
        self,
    ) -> None:
        events: list[str] = []
        producer = _RecordingProducer(events)
        runtime = LocalizationRuntime(
            observation_producer=producer,
            localization_service=_RecordingService(events),
        )
        await runtime.start()

        producer.complete()
        status = await runtime.wait_stopped(timeout_s=0.5)

        self.assertEqual(status.state, LocalizationRuntimeState.FAULTED)
        self.assertEqual(
            status.detail_code,
            "observation_producer_completed_unexpectedly",
        )

    async def test_finite_producer_completion_can_wait_for_explicit_stop(
        self,
    ) -> None:
        events: list[str] = []
        producer = _RecordingProducer(events)
        runtime = LocalizationRuntime(
            observation_producer=producer,
            localization_service=_RecordingService(events),
            config=LocalizationRuntimeConfig(
                observation_completion_policy=(
                    LocalizationObservationCompletionPolicy.ALLOW_UNTIL_STOP
                )
            ),
        )
        await runtime.start()

        producer.complete()
        for _ in range(3):
            await asyncio.sleep(0)

        status = runtime.get_status()
        self.assertEqual(status.state, LocalizationRuntimeState.RUNNING)
        self.assertEqual(
            status.detail_code,
            "observation_producer_completed",
        )

        await runtime.stop()
        self.assertEqual(
            runtime.get_status().state,
            LocalizationRuntimeState.STOPPED,
        )

    async def test_producer_monitor_failure_faults_runtime(self) -> None:
        events: list[str] = []
        runtime = LocalizationRuntime(
            observation_producer=_RecordingProducer(
                events,
                fail_wait=True,
            ),
            localization_service=_RecordingService(events),
        )
        await runtime.start()

        status = await runtime.wait_stopped(timeout_s=0.5)

        self.assertEqual(status.state, LocalizationRuntimeState.FAULTED)
        self.assertEqual(
            status.detail_code,
            "observation_producer_monitor_failed:RuntimeError",
        )

    async def test_service_monitor_failure_faults_runtime(self) -> None:
        events: list[str] = []
        runtime = LocalizationRuntime(
            observation_producer=_RecordingProducer(events),
            localization_service=_RecordingService(
                events,
                fail_wait=True,
            ),
        )
        await runtime.start()

        status = await runtime.wait_stopped(timeout_s=0.5)

        self.assertEqual(status.state, LocalizationRuntimeState.FAULTED)
        self.assertEqual(
            status.detail_code,
            "localization_service_monitor_failed:RuntimeError",
        )

    async def test_startup_failure_cleans_up_all_components(self) -> None:
        events: list[str] = []
        runtime = LocalizationRuntime(
            observation_producer=_RecordingProducer(events),
            localization_service=_RecordingService(
                events,
                fail_start=True,
            ),
            shutdown_resources=(
                _RecordingResource("executor", events),
            ),
        )

        with self.assertRaisesRegex(
            LocalizationRuntimeError,
            "startup failed",
        ):
            await runtime.start()

        self.assertEqual(
            runtime.get_status().state,
            LocalizationRuntimeState.FAULTED,
        )
        self.assertEqual(
            events,
            [
                "producer.start",
                "service.start",
                "producer.stop",
                "service.stop",
                "executor.close",
            ],
        )

    async def test_cleanup_failure_does_not_skip_later_resources(self) -> None:
        events: list[str] = []
        runtime = LocalizationRuntime(
            observation_producer=_RecordingProducer(events),
            localization_service=_RecordingService(events),
            shutdown_resources=(
                _RecordingResource(
                    "failing-resource",
                    events,
                    fail_close=True,
                ),
                _RecordingResource("last-resource", events),
            ),
        )
        await runtime.start()

        await runtime.stop()

        status = runtime.get_status()
        self.assertEqual(status.state, LocalizationRuntimeState.FAULTED)
        self.assertEqual(status.detail_code, "shutdown_failed")
        self.assertIn("failing-resource close failed", status.last_error or "")
        self.assertEqual(events[-1], "last-resource.close")


if __name__ == "__main__":
    unittest.main()

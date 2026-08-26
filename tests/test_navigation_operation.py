"""Tests for the optional live-trajectory navigation operation boundary."""

from __future__ import annotations

import asyncio
import unittest

from longship.navigation import (
    NavigationAuthority,
    NavigationRequest,
    NavigationResult,
    NavigationStopRequest,
    StopResult,
)
from longship.navigation.operation import StreamBackedNavigationPort


class _TrajectoryStream:
    def get_latest(self) -> object:
        return object()

    async def wait_for_update(self, request: object) -> object:
        del request
        return object()


class _Session:
    def __init__(
        self,
        result: NavigationResult,
        completion: asyncio.Event | None = None,
    ) -> None:
        self.trajectory_stream = _TrajectoryStream()
        self._result = result
        self._completion = completion
        self.paused = 0
        self.resumed = 0
        self.stop_requests: list[NavigationStopRequest] = []

    async def wait_result(self) -> NavigationResult:
        if self._completion is not None:
            await self._completion.wait()
        return self._result

    async def pause(self, authority: NavigationAuthority) -> None:
        authority.ensure_active()
        self.paused += 1

    async def resume(self, authority: NavigationAuthority) -> None:
        authority.ensure_active()
        self.resumed += 1

    async def stop(self, request: NavigationStopRequest) -> StopResult:
        self.stop_requests.append(request)
        return StopResult(
            request_id=request.request_id,
            revoked_through_epoch=request.revoke_through_epoch,
            requested=True,
            verified_stopped=True,
            evidence="test.stop",
        )


class _SessionFactory:
    def __init__(
        self,
        result: NavigationResult,
        completion: asyncio.Event | None = None,
    ) -> None:
        self.session = _Session(result, completion)
        self.started: list[tuple[NavigationRequest, NavigationAuthority]] = []

    async def start_session(
        self,
        request: NavigationRequest,
        authority: NavigationAuthority,
    ) -> _Session:
        self.started.append((request, authority))
        return self.session


def _request() -> NavigationRequest:
    return NavigationRequest(
        request_id="request-1",
        authority_epoch=7,
        map_id="map",
        map_version="version",
        route_id="route",
        waypoint_id="waypoint",
    )


def _result(request: NavigationRequest) -> NavigationResult:
    return NavigationResult(
        arrived=True,
        request_id=request.request_id,
        authority_epoch=request.authority_epoch,
        map_id=request.map_id,
        map_version=request.map_version,
        route_id=request.route_id,
        waypoint_id=request.waypoint_id,
        evidence="test.arrived",
    )


class StreamBackedNavigationPortTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_exposes_the_session_trajectory_stream(self) -> None:
        request = _request()
        authority = NavigationAuthority(request.authority_epoch)
        factory = _SessionFactory(_result(request))
        navigation = StreamBackedNavigationPort(factory)

        operation = await navigation.start_navigation(request, authority)

        self.assertEqual(operation.request_id, request.request_id)
        self.assertIs(operation.trajectory_stream, factory.session.trajectory_stream)
        self.assertEqual(factory.started, [(request, authority)])
        self.assertEqual(await operation.wait_result(), _result(request))

    async def test_pause_resume_and_stop_use_the_active_session(self) -> None:
        request = _request()
        authority = NavigationAuthority(request.authority_epoch)
        factory = _SessionFactory(_result(request))
        navigation = StreamBackedNavigationPort(factory)
        await navigation.start_navigation(request, authority)

        await navigation.pause(authority)
        await navigation.resume(authority)
        stop = await navigation.stop(
            NavigationStopRequest(
                request_id=request.request_id,
                reason="operator request",
                revoke_through_epoch=authority.epoch,
            )
        )

        self.assertEqual(factory.session.paused, 1)
        self.assertEqual(factory.session.resumed, 1)
        self.assertEqual(len(factory.session.stop_requests), 1)
        self.assertTrue(stop.verified_stopped)

    async def test_navigate_to_remains_a_terminal_result_convenience(self) -> None:
        request = _request()
        authority = NavigationAuthority(request.authority_epoch)
        factory = _SessionFactory(_result(request))
        navigation = StreamBackedNavigationPort(factory)

        result = await navigation.navigate_to(request, authority)

        self.assertEqual(result, _result(request))
        with self.assertRaisesRegex(ValueError, "not active"):
            await navigation.stop(
                NavigationStopRequest(
                    request_id=request.request_id,
                    reason="too late",
                    revoke_through_epoch=authority.epoch,
                )
            )

    async def test_terminal_session_is_released_without_waiting_for_result(
        self,
    ) -> None:
        request = _request()
        authority = NavigationAuthority(request.authority_epoch)
        factory = _SessionFactory(_result(request))
        navigation = StreamBackedNavigationPort(factory)
        await navigation.start_navigation(request, authority)

        await asyncio.sleep(0)

        with self.assertRaisesRegex(ValueError, "not active"):
            await navigation.stop(
                NavigationStopRequest(
                    request_id=request.request_id,
                    reason="already completed",
                    revoke_through_epoch=authority.epoch,
                )
            )

    async def test_cancelling_a_waiter_does_not_cancel_the_session(self) -> None:
        request = _request()
        authority = NavigationAuthority(request.authority_epoch)
        completion = asyncio.Event()
        factory = _SessionFactory(_result(request), completion)
        navigation = StreamBackedNavigationPort(factory)
        operation = await navigation.start_navigation(request, authority)
        waiter = asyncio.create_task(operation.wait_result())

        await asyncio.sleep(0)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter

        await navigation.pause(authority)
        self.assertEqual(factory.session.paused, 1)
        completion.set()
        await asyncio.sleep(0)

    async def test_start_rejects_an_authority_that_already_owns_an_operation(
        self,
    ) -> None:
        request = _request()
        authority = NavigationAuthority(request.authority_epoch)
        factory = _SessionFactory(_result(request))
        navigation = StreamBackedNavigationPort(factory)
        await navigation.start_navigation(request, authority)

        second_request = NavigationRequest(
            request_id="request-2",
            authority_epoch=authority.epoch,
            map_id=request.map_id,
            map_version=request.map_version,
            route_id=request.route_id,
            waypoint_id=request.waypoint_id,
        )
        with self.assertRaisesRegex(ValueError, "already owns"):
            await navigation.start_navigation(second_request, authority)


if __name__ == "__main__":
    unittest.main()

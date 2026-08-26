from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable

from longship.contracts.skills.follow_person import FollowScene


class HttpFollowSceneSource:
    """Read the versioned scene from one local RGB-D owner over HTTP."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 0.1,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        maximum_response_bytes: int = 1_000_000,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("follow scene URL must be an http URL with a host")
        has_credentials = parsed.username is not None or parsed.password is not None
        has_suffix = bool(parsed.query) or bool(parsed.fragment)
        if has_credentials or has_suffix:
            raise ValueError(
                "follow scene base URL must not contain credentials or query"
            )
        if parsed.hostname.lower() not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError(
                "follow scene provider must be loopback so monotonic timestamps "
                "share a host"
            )
        if not 0.01 <= timeout_s <= 1.0:
            raise ValueError("scene HTTP timeout must be between 0.01 and 1 second")
        if not 1_024 <= maximum_response_bytes <= 4_000_000:
            raise ValueError("maximum response size is outside supported bounds")
        self._url = base_url.rstrip("/") + "/v1/follow-scene"
        self._origin = (parsed.hostname.lower(), parsed.port or 80)
        self._timeout_s = timeout_s
        self._clock_ns = clock_ns
        self._maximum_response_bytes = maximum_response_bytes
        self._failure_sequence = 0

    def read(self) -> FollowScene:
        try:
            request = urllib.request.Request(
                self._url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "longship-follow/0",
                },
            )
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                if response.status != 200:
                    raise ValueError(f"scene endpoint returned HTTP {response.status}")
                final = urllib.parse.urlparse(response.geturl())
                if (final.hostname or "").lower() != self._origin[0] or (
                    final.port or 80
                ) != self._origin[1]:
                    raise ValueError("scene endpoint redirected to another origin")
                body = response.read(self._maximum_response_bytes + 1)
            if len(body) > self._maximum_response_bytes:
                raise ValueError("scene response exceeds the configured size limit")
            value = json.loads(body)
            if not isinstance(value, dict):
                raise ValueError("scene response must be a JSON object")
            scene = FollowScene.from_mapping(
                value, received_monotonic_ns=self._clock_ns()
            )
            self._failure_sequence = max(self._failure_sequence, scene.sequence)
            return scene
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            ValueError,
        ) as exc:
            now = self._clock_ns()
            self._failure_sequence += 1
            return FollowScene(
                sequence=self._failure_sequence,
                captured_monotonic_ns=now,
                received_monotonic_ns=now,
                healthy=False,
                calibration_id="unavailable",
                calibration_valid=False,
                detector_ready=False,
                floor_valid=False,
                tracks=(),
                obstacles=(),
                raw_forward_clearance_m=None,
                detail=f"scene request failed: {type(exc).__name__}",
            )

    def require_ready(self) -> FollowScene:
        scene = self.read()
        flags = (
            scene.healthy,
            scene.calibration_valid,
            scene.detector_ready,
            scene.floor_valid,
            scene.raw_forward_clearance_m is not None,
        )
        if not all(flags):
            raise RuntimeError(f"RGB-D scene preflight failed: {scene.detail}")
        return scene

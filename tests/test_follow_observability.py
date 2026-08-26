from __future__ import annotations

import json
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from longship.observability.follow_person import (
    BufferedEventSink,
    FollowDashboard,
    JsonlEventSink,
)


EVENT = {
    "schema_version": "longship.follow-runtime-event.v1",
    "snapshot": {
        "revision": 1,
        "state": "following",
        "detail": "synthetic event",
    },
    "scene": {"sequence": 3, "tracks": [], "obstacles": []},
}

SYSTEM_EVENT = {
    "schema_version": "longship.follow-system-event.v0",
    "event_type": "brain.decision.accepted",
    "active_skill_call_id": "follow-skill-call-1",
    "task_graph": {
        "graph_id": "follow-task-graph-1",
        "state": "running",
        "current_operation_id": "navigation.follow_person.pause",
    },
}


class FollowObservabilityTests(unittest.TestCase):
    def test_dashboard_is_read_only_and_exposes_latest_event(self) -> None:
        dashboard = FollowDashboard(port=0)
        try:
            dashboard.start()
        except PermissionError:
            self.skipTest("local socket creation is disabled by the test sandbox")
        try:
            dashboard.publish(EVENT)
            dashboard.publish(SYSTEM_EVENT)
            dashboard.publish_camera_frame(
                7,
                b"\xff\xd8test-frame\xff\xd9",
                source="synthetic-g1-camera",
            )
            with urllib.request.urlopen(
                f"http://127.0.0.1:{dashboard.port}/api/snapshot", timeout=1.0
            ) as response:
                value = json.loads(response.read())
            with urllib.request.urlopen(
                f"http://127.0.0.1:{dashboard.port}/camera.jpg", timeout=1.0
            ) as response:
                camera = response.read()
            request = urllib.request.Request(
                f"http://127.0.0.1:{dashboard.port}/api/snapshot",
                data=b"{}",
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request, timeout=1.0)
        finally:
            dashboard.close()

        self.assertEqual(value["event"]["snapshot"]["state"], "following")
        self.assertEqual(
            value["system_event"]["event_type"], "brain.decision.accepted"
        )
        self.assertEqual(
            value["system_event"]["task_graph"]["current_operation_id"],
            "navigation.follow_person.pause",
        )
        self.assertEqual(value["camera"]["sequence"], 7)
        self.assertEqual(value["last_scene"]["sequence"], 3)
        self.assertEqual(camera, b"\xff\xd8test-frame\xff\xd9")
        self.assertEqual(raised.exception.code, 405)

    def test_dashboard_camera_url_is_loopback_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            FollowDashboard(camera_preview_url="http://camera.example/preview.jpg")

    def test_buffered_journal_delivers_without_blocking_publisher(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with JsonlEventSink(path) as journal:
                with BufferedEventSink(journal, maximum_pending_events=8) as buffered:
                    started = time.monotonic()
                    buffered.publish(EVENT)
                    publish_elapsed = time.monotonic() - started

            values = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertLess(publish_elapsed, 0.05)
        self.assertEqual(values, [EVENT])


if __name__ == "__main__":
    unittest.main()

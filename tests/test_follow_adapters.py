from __future__ import annotations

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from longship.perception.follow_http import HttpFollowSceneSource


class FollowAdapterTests(unittest.TestCase):
    def test_http_source_reads_atomic_scene_contract(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                now = time.monotonic_ns()
                body = json.dumps(
                    {
                        "schema_version": "longship.follow-scene.v1",
                        "sequence": 7,
                        "captured_monotonic_ns": now,
                        "healthy": True,
                        "calibration_id": "synthetic-http-calibration",
                        "calibration_valid": True,
                        "detector_ready": True,
                        "floor_valid": True,
                        "tracks": [
                            {
                                "track_id": "synthetic-person",
                                "forward_m": 2.0,
                                "left_m": 0.1,
                                "confidence": 0.9,
                            }
                        ],
                        "obstacles": [],
                        "raw_forward_clearance_m": 4.0,
                        "detail": "synthetic HTTP scene",
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except PermissionError:
            self.skipTest("local socket creation is disabled by the test sandbox")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            source = HttpFollowSceneSource(
                f"http://127.0.0.1:{server.server_address[1]}"
            )
            scene = source.require_ready()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1.0)

        self.assertEqual(scene.sequence, 7)
        self.assertEqual(scene.tracks[0].track_id, "synthetic-person")

    def test_http_failure_becomes_fail_closed_scene_not_exception(self) -> None:
        source = HttpFollowSceneSource("http://127.0.0.1:1", timeout_s=0.01)

        scene = source.read()

        self.assertFalse(scene.healthy)
        self.assertIsNone(scene.raw_forward_clearance_m)

    def test_http_source_rejects_cross_host_clock_domain(self) -> None:
        with self.assertRaises(ValueError):
            HttpFollowSceneSource("http://192.0.2.10:8780")


if __name__ == "__main__":
    unittest.main()

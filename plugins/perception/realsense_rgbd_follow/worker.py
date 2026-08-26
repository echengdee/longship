#!/usr/bin/env python3
"""Own one RealSense camera and publish a synchronized FollowScene.

This experimental provider contains no actuator path. It uses OpenCV's bundled
HOG people detector as a dependency-light baseline and intentionally refuses to
start without an explicitly confirmed camera-to-base transform.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
import urllib.parse
from collections import defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from longship.contracts.skills.follow_person import (
    FollowScene,
    ObstaclePoint,
    PersonTrack,
)
from longship.perception.rgbd import BoundingBox, RigidTransform, ShortTrackAssigner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish Longship RGB-D FollowScene V1"
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--detection-every", type=int, default=3)
    parser.add_argument("--sample-step-px", type=int, default=8)
    parser.add_argument("--maximum-depth-m", type=float, default=6.0)
    parser.add_argument("--corridor-half-width-m", type=float, default=0.45)
    parser.add_argument("--obstacle-min-height-m", type=float, default=0.06)
    parser.add_argument("--obstacle-max-height-m", type=float, default=1.6)
    parser.add_argument("--occupancy-cell-m", type=float, default=0.15)
    parser.add_argument("--minimum-cell-points", type=int, default=3)
    parser.add_argument("--minimum-floor-points", type=int, default=20)
    return parser


class PublishedState:
    def __init__(self, calibration_id: str) -> None:
        now = time.monotonic_ns()
        self._scene = FollowScene(
            sequence=0,
            captured_monotonic_ns=now,
            received_monotonic_ns=now,
            healthy=False,
            calibration_id=calibration_id,
            calibration_valid=True,
            detector_ready=False,
            floor_valid=False,
            tracks=(),
            obstacles=(),
            raw_forward_clearance_m=None,
            detail=f"camera starting with calibration {calibration_id}",
        )
        self._preview = b""
        self._lock = threading.Lock()

    def publish(self, scene: FollowScene, preview: bytes) -> None:
        with self._lock:
            self._scene = scene
            self._preview = preview

    def snapshot(self) -> tuple[FollowScene, bytes]:
        with self._lock:
            return self._scene, self._preview


class DiagnosticServer:
    def __init__(self, state: PublishedState, host: str, port: int) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                parsed = urllib.parse.urlparse(self.path)
                route = parsed.path
                scene, preview = owner.state.snapshot()
                if route == "/v1/follow-scene":
                    self._json(scene.to_dict())
                    return
                if route == "/health":
                    age_s = scene.age_s(time.monotonic_ns())
                    self._json(
                        {
                            "healthy": scene.healthy,
                            "calibration_valid": scene.calibration_valid,
                            "detector_ready": scene.detector_ready,
                            "floor_valid": scene.floor_valid,
                            "sequence": scene.sequence,
                            "camera_age_s": age_s,
                            "track_count": len(scene.tracks),
                            "detail": scene.detail,
                        }
                    )
                    return
                if route == "/preview.jpg" and preview:
                    requested = urllib.parse.parse_qs(parsed.query).get("seq")
                    if requested and requested[-1] != str(scene.sequence):
                        self._send(
                            HTTPStatus.CONFLICT,
                            "text/plain",
                            b"preview sequence changed; fetch the scene again\n",
                        )
                        return
                    self._send(HTTPStatus.OK, "image/jpeg", preview)
                    return
                if route == "/":
                    self._send(
                        HTTPStatus.OK,
                        "text/html; charset=utf-8",
                        _DIAGNOSTIC_HTML.encode("utf-8"),
                    )
                    return
                self._send(HTTPStatus.NOT_FOUND, "text/plain", b"not found\n")

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                self._send(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    "text/plain",
                    b"perception diagnostics are read-only\n",
                )

            def log_message(self, format: str, *args: object) -> None:
                del format, args

            def _json(self, value: object) -> None:
                body = json.dumps(
                    value, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                self._send(HTTPStatus.OK, "application/json", body)

            def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

        self.state = state
        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="longship-rgbd-diagnostics",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)


class RealSenseFollowWorker:
    def __init__(
        self,
        args: argparse.Namespace,
        transform: RigidTransform,
        calibration_id: str,
    ) -> None:
        self.args = args
        self.transform = transform
        self.calibration_id = calibration_id
        self.sequence = 0
        self.tracker = ShortTrackAssigner()
        self._last_boxes: tuple[BoundingBox, ...] = ()

    def run(self, state: PublishedState) -> None:
        try:
            import cv2
            import numpy as np
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "worker requires pyrealsense2, OpenCV, and NumPy in one environment"
            ) from exc

        pipeline = rs.pipeline()
        configuration = rs.config()
        configuration.enable_stream(
            rs.stream.depth,
            self.args.width,
            self.args.height,
            rs.format.z16,
            self.args.fps,
        )
        configuration.enable_stream(
            rs.stream.color,
            self.args.width,
            self.args.height,
            rs.format.bgr8,
            self.args.fps,
        )
        pipeline.start(configuration)
        align = rs.align(rs.stream.color)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        try:
            while True:
                try:
                    frames = align.process(pipeline.wait_for_frames(timeout_ms=1_000))
                    depth = frames.get_depth_frame()
                    color = frames.get_color_frame()
                    if not depth or not color:
                        raise RuntimeError("aligned depth or color frame is missing")
                    image = np.asanyarray(color.get_data())
                    intrinsics = depth.profile.as_video_stream_profile().intrinsics
                    self.sequence += 1
                    if (
                        self.sequence % self.args.detection_every == 1
                        or not self._last_boxes
                    ):
                        self._last_boxes = self._detect_people(hog, image)
                    tracked = self.tracker.update(self._last_boxes)
                    tracks = self._person_tracks(rs, depth, intrinsics, tracked)
                    obstacles, raw_clearance, floor_valid = self._depth_geometry(
                        rs, depth, intrinsics
                    )
                    preview = image.copy()
                    for tracked_box in tracked:
                        box = tracked_box.box
                        cv2.rectangle(
                            preview,
                            (box.left_px, box.top_px),
                            (box.right_px, box.bottom_px),
                            (92, 235, 164),
                            2,
                        )
                        cv2.putText(
                            preview,
                            tracked_box.track_id,
                            (box.left_px, max(18, box.top_px - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (92, 235, 164),
                            1,
                            cv2.LINE_AA,
                        )
                    cv2.putText(
                        preview,
                        f"seq={self.sequence} "
                        f"floor={'ok' if floor_valid else 'invalid'} "
                        f"clearance={raw_clearance:.2f}m",
                        (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (250, 210, 100),
                        2,
                        cv2.LINE_AA,
                    )
                    encoded, jpeg = cv2.imencode(
                        ".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 78]
                    )
                    now = time.monotonic_ns()
                    state.publish(
                        FollowScene(
                            sequence=self.sequence,
                            captured_monotonic_ns=now,
                            received_monotonic_ns=now,
                            healthy=floor_valid,
                            calibration_id=self.calibration_id,
                            calibration_valid=True,
                            detector_ready=True,
                            floor_valid=floor_valid,
                            tracks=tracks,
                            obstacles=obstacles,
                            raw_forward_clearance_m=raw_clearance,
                            detail=(
                                "RGB-D frame synchronized"
                                if floor_valid
                                else "insufficient transformed floor support"
                            ),
                        ),
                        jpeg.tobytes() if encoded else b"",
                    )
                except Exception as exc:
                    self.sequence += 1
                    now = time.monotonic_ns()
                    state.publish(
                        FollowScene(
                            sequence=self.sequence,
                            captured_monotonic_ns=now,
                            received_monotonic_ns=now,
                            healthy=False,
                            calibration_id=self.calibration_id,
                            calibration_valid=True,
                            detector_ready=True,
                            floor_valid=False,
                            tracks=(),
                            obstacles=(),
                            raw_forward_clearance_m=None,
                            detail=f"camera frame failed: {type(exc).__name__}",
                        ),
                        b"",
                    )
                    time.sleep(0.05)
        finally:
            pipeline.stop()

    @staticmethod
    def _detect_people(hog: Any, image: Any) -> tuple[BoundingBox, ...]:
        rectangles, weights = hog.detectMultiScale(
            image,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        boxes = []
        for rectangle, weight in zip(rectangles, weights):
            left, top, width, height = (int(item) for item in rectangle)
            raw_weight = float(weight)
            confidence = 1.0 / (1.0 + math.exp(-raw_weight))
            if confidence >= 0.55:
                boxes.append(BoundingBox(left, top, width, height, confidence))
        return tuple(boxes)

    def _person_tracks(
        self, rs: Any, depth: Any, intrinsics: Any, tracked: tuple[Any, ...]
    ) -> tuple[PersonTrack, ...]:
        output = []
        for item in tracked:
            box = item.box
            centre_u = min(self.args.width - 1, box.left_px + box.width_px // 2)
            centre_v = min(
                self.args.height - 1, box.top_px + round(box.height_px * 0.62)
            )
            distances = []
            for offset_u in (-18, -9, 0, 9, 18):
                for offset_v in (-12, 0, 12):
                    u = max(0, min(self.args.width - 1, centre_u + offset_u))
                    v = max(0, min(self.args.height - 1, centre_v + offset_v))
                    distance = depth.get_distance(u, v)
                    if 0.2 <= distance <= self.args.maximum_depth_m:
                        distances.append(distance)
            if len(distances) < 5:
                continue
            distance = statistics.median(distances)
            optical = rs.rs2_deproject_pixel_to_point(
                intrinsics, [centre_u, centre_v], distance
            )
            forward, left, _ = self.transform.apply(tuple(float(v) for v in optical))
            if forward <= 0.0:
                continue
            output.append(
                PersonTrack(item.track_id, forward, left, box.confidence)
            )
        return tuple(output)

    def _depth_geometry(
        self, rs: Any, depth: Any, intrinsics: Any
    ) -> tuple[tuple[ObstaclePoint, ...], float, bool]:
        bins: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
        raw_candidates = []
        floor_points = 0
        cell_size = self.args.occupancy_cell_m
        for v in range(0, self.args.height, self.args.sample_step_px):
            for u in range(0, self.args.width, self.args.sample_step_px):
                distance = depth.get_distance(u, v)
                if not 0.2 <= distance <= self.args.maximum_depth_m:
                    continue
                optical = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], distance)
                forward, left, height = self.transform.apply(
                    tuple(float(item) for item in optical)
                )
                if forward <= 0.0 or forward > self.args.maximum_depth_m:
                    continue
                if abs(height) <= 0.08:
                    floor_points += 1
                if not (
                    self.args.obstacle_min_height_m
                    <= height
                    <= self.args.obstacle_max_height_m
                ):
                    continue
                if abs(left) <= self.args.corridor_half_width_m:
                    raw_candidates.append(forward)
                key = (round(forward / cell_size), round(left / cell_size))
                bins[key].append((forward, left))
        obstacles = []
        for points in bins.values():
            if len(points) < self.args.minimum_cell_points:
                continue
            obstacles.append(
                ObstaclePoint(
                    forward_m=sum(point[0] for point in points) / len(points),
                    left_m=sum(point[1] for point in points) / len(points),
                    radius_m=cell_size * 0.71,
                )
            )
        obstacles.sort(key=lambda item: (item.forward_m, item.left_m))
        raw_clearance = min(raw_candidates, default=self.args.maximum_depth_m)
        return (
            tuple(obstacles[:10_000]),
            raw_clearance,
            floor_points >= self.args.minimum_floor_points,
        )


def _validate_args(args: argparse.Namespace) -> None:
    if not isinstance(args.host, str) or not args.host:
        raise ValueError("diagnostic host is required")
    if not 1 <= args.port <= 65_535:
        raise ValueError("port is invalid")
    if args.width < 320 or args.height < 240 or not 5 <= args.fps <= 30:
        raise ValueError("camera dimensions or rate are outside supported bounds")
    if not 1 <= args.detection_every <= 30 or not 2 <= args.sample_step_px <= 32:
        raise ValueError("detector interval or depth sample step is invalid")
    if not 1.0 <= args.maximum_depth_m <= 10.0:
        raise ValueError("maximum depth must be between 1 and 10 metres")
    if not 0.1 <= args.corridor_half_width_m <= 2.0:
        raise ValueError("corridor half-width is invalid")
    if not (
        0.0 <= args.obstacle_min_height_m < args.obstacle_max_height_m <= 3.0
    ):
        raise ValueError("obstacle height band is invalid")
    if not 0.05 <= args.occupancy_cell_m <= 0.3:
        raise ValueError("occupancy cell size is invalid")
    if not 1 <= args.minimum_cell_points <= 100:
        raise ValueError("minimum cell support is invalid")
    if not 1 <= args.minimum_floor_points <= 10_000:
        raise ValueError("minimum floor support is invalid")


def main() -> None:
    args = _parser().parse_args()
    try:
        _validate_args(args)
        if args.calibration.stat().st_size > 64_000:
            raise ValueError("calibration file exceeds 64 KB")
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
        calibration_id, transform = RigidTransform.from_calibration(calibration)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"BLOCKED: {exc}") from exc
    state = PublishedState(calibration_id)
    server = DiagnosticServer(state, args.host, args.port)
    server.start()
    print(f"FollowScene: http://{args.host}:{server.port}/v1/follow-scene")
    print(f"Read-only camera diagnostics: http://{args.host}:{server.port}/")
    try:
        RealSenseFollowWorker(args, transform, calibration_id).run(state)
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        raise SystemExit(f"BLOCKED: {exc}") from exc
    finally:
        server.close()


_DIAGNOSTIC_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Longship RGB-D</title><style>
body{font:15px system-ui;background:#0a111a;color:#e7eef7;margin:0;padding:20px}main{max-width:980px;margin:auto}
img,pre{width:100%;box-sizing:border-box;border:1px solid #2b3c50;border-radius:12px;background:#111d2b}
pre{padding:14px;white-space:pre-wrap;color:#9dd9bc}h1{font-size:20px}.note{color:#aab9ca}
</style></head><body><main><h1>Longship RGB-D FollowScene</h1>
<p class="note">Read-only camera owner. This page has no command route.</p><img id="preview" alt="camera preview"><pre id="state">waiting</pre>
<script>async function poll(){try{const r=await fetch('/v1/follow-scene',{cache:'no-store'}),s=await r.json();
document.getElementById('state').textContent=JSON.stringify(s,null,2);document.getElementById('preview').src='/preview.jpg?seq='+s.sequence}catch(e){}
setTimeout(poll,250)}poll()</script></main></body></html>"""


if __name__ == "__main__":
    main()

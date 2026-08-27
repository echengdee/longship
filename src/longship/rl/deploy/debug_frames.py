from __future__ import annotations

import json
import time
from typing import Any

import numpy as np


class DebugFramePublisher:
    """Best-effort JPEG debug frames; inference never waits for the UI."""

    def __init__(self, endpoint: str | None, channel: str, fps: float = 10.0) -> None:
        self.channel = channel
        self.period = 1.0 / fps
        self.next_publish_at = 0.0
        self.socket: Any | None = None
        if endpoint:
            import zmq

            self.zmq = zmq
            self.socket = zmq.Context.instance().socket(zmq.PUSH)
            self.socket.setsockopt(zmq.SNDHWM, 1)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.connect(endpoint)

    def publish_depth(self, depth: np.ndarray, *, normalized: bool) -> None:
        if self.socket is None:
            return
        now = time.monotonic()
        if now < self.next_publish_at:
            return
        self.next_publish_at = now + self.period
        image = np.asarray(depth, dtype=np.float32)
        finite = np.isfinite(image)
        if normalized:
            scaled = np.clip(image, 0.0, 1.0)
        else:
            scaled = np.clip((image - 0.2) / (2.5 - 0.2), 0.0, 1.0)
        gray = np.asarray(np.rint((1.0 - scaled) * 255.0), dtype=np.uint8)
        gray[~finite] = 0
        import cv2

        color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        color[~finite] = 0
        if image.shape[0] < 100:
            color = cv2.resize(color, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        ok, encoded = cv2.imencode(".jpg", color, (cv2.IMWRITE_JPEG_QUALITY, 85))
        if not ok:
            return
        valid_values = image[finite]
        metadata = {
            "timestamp": time.time(),
            "shape": list(image.shape),
            "min": float(valid_values.min()) if valid_values.size else None,
            "max": float(valid_values.max()) if valid_values.size else None,
            "normalized": normalized,
        }
        try:
            self.socket.send_multipart(
                (
                    self.channel.encode(),
                    json.dumps(metadata, separators=(",", ":")).encode(),
                    encoded.tobytes(),
                ),
                flags=self.zmq.NOBLOCK,
            )
        except self.zmq.Again:
            pass

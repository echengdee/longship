from __future__ import annotations

import time

import numpy as np

from longship.rl.deploy.debug_frames import DebugFramePublisher
from longship.rl.deploy.web_monitor import FrameStore
from longship.rl.sim2sim.hiking_pipeline import HikingOnnxPolicy


def test_hiking_visualized_model_frame_is_exact_preprocessing_output() -> None:
    raw = np.linspace(0.1, 3.0, 270 * 480, dtype=np.float32).reshape(270, 480)

    model_input = HikingOnnxPolicy.preprocess_depth(raw)

    assert model_input.shape == (18, 32)
    assert model_input.dtype == np.float32
    assert 0.0 <= float(model_input.min()) <= float(model_input.max()) <= 1.0


def test_disabled_debug_publisher_is_a_noop_without_image_dependencies() -> None:
    publisher = DebugFramePublisher(None, "model_depth")

    publisher.publish_depth(np.zeros((18, 32), dtype=np.float32), normalized=True)


def test_monitor_status_reports_frame_metadata() -> None:
    store = FrameStore()
    timestamp = time.time()

    store.update("camera_depth", {"timestamp": timestamp, "shape": [270, 480]}, b"jpeg")

    assert store.status()["camera_depth"]["shape"] == [270, 480]
    assert store.wait("camera_depth", 0, timeout=0.01).jpeg == b"jpeg"  # type: ignore[union-attr]

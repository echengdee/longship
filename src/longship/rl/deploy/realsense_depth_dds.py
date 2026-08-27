#!/usr/bin/env python3
"""Publish a physical RealSense depth stream with the Sim2Sim DDS contract."""

from __future__ import annotations

import argparse
import time

import numpy as np

from longship.rl.sim2sim.dds import DEPTH_TOPIC
from longship.rl.sim2sim.simulator import encode_depth
from longship.rl.deploy.debug_frames import DebugFramePublisher


def run(args: argparse.Namespace) -> None:
    import cv2
    import pyrealsense2 as rs
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(
        rs.stream.depth,
        args.raw_width,
        args.raw_height,
        rs.format.z16,
        args.fps,
    )
    profile = pipeline.start(config)
    device = profile.get_device()
    name = str(device.get_info(rs.camera_info.name))
    serial = str(device.get_info(rs.camera_info.serial_number))
    if args.expected_model.lower() not in name.lower():
        pipeline.stop()
        raise RuntimeError(
            f"expected RealSense model containing {args.expected_model!r}, got {name!r}"
        )
    depth_scale = float(device.first_depth_sensor().get_depth_scale())
    if not 0.0 < depth_scale < 0.1:
        pipeline.stop()
        raise RuntimeError(f"invalid RealSense depth scale {depth_scale}")

    ChannelFactoryInitialize(args.domain_id, args.interface)
    publisher = ChannelPublisher(DEPTH_TOPIC, PointCloud2_)
    publisher.Init()
    started = time.monotonic()
    published = 0
    debug_frames = DebugFramePublisher(
        args.debug_frame_endpoint, "camera_depth", args.debug_frame_fps
    )
    try:
        for _ in range(args.warmup_frames):
            pipeline.wait_for_frames(args.timeout_ms)
        print(
            f"REALSENSE DDS READY: model={name} serial={serial} "
            f"raw={args.raw_width}x{args.raw_height}@{args.fps} "
            f"output={args.output_width}x{args.output_height} topic={DEPTH_TOPIC}",
            flush=True,
        )
        while args.duration == 0 or time.monotonic() - started < args.duration:
            frames = pipeline.wait_for_frames(args.timeout_ms)
            frame = frames.get_depth_frame()
            if not frame:
                raise RuntimeError("RealSense returned a frame set without depth")
            depth_m = np.asanyarray(frame.get_data(), dtype=np.uint16).astype(np.float32)
            depth_m *= depth_scale
            if depth_m.shape != (args.output_height, args.output_width):
                depth_m = cv2.resize(
                    depth_m,
                    (args.output_width, args.output_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            debug_frames.publish_depth(depth_m, normalized=False)
            publisher.Write(encode_depth(depth_m, time.monotonic() - started))
            published += 1
    finally:
        pipeline.stop()
        print(f"REALSENSE DDS DONE: frames={published}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--expected-model", default="D435I")
    parser.add_argument("--raw-width", type=int, default=848)
    parser.add_argument("--raw-height", type=int, default=480)
    parser.add_argument("--output-width", type=int, default=480)
    parser.add_argument("--output-height", type=int, default=270)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--timeout-ms", type=int, default=1000)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--debug-frame-endpoint")
    parser.add_argument("--debug-frame-fps", type=float, default=10.0)
    args = parser.parse_args()
    dimensions = (args.raw_width, args.raw_height, args.output_width, args.output_height)
    if not all(1 <= value <= 4096 for value in dimensions):
        parser.error("camera dimensions must be between 1 and 4096")
    if not 1 <= args.fps <= 90 or args.warmup_frames < 0 or args.timeout_ms <= 0:
        parser.error("fps, warmup frames, or timeout is invalid")
    if args.duration < 0 or not args.serial.strip() or args.debug_frame_fps <= 0:
        parser.error("duration must be non-negative, serial explicit, and debug fps positive")
    return args


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

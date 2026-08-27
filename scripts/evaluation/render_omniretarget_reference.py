#!/usr/bin/env python3
"""Render a reference trajectory kinematically, without policy or PD control."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--ffmpeg", type=Path, help="FFmpeg executable (uses PATH by default)")
    args = parser.parse_args()

    ref = np.load(args.reference, allow_pickle=True)
    fps = int(np.asarray(ref["fps"]).reshape(-1)[0])
    model = mujoco.MjModel.from_xml_path(str(args.model_xml.resolve()))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)
    if model.nq != ref["joint_pos"].shape[1] + 7:
        raise ValueError(f"Model nq={model.nq}, expected {ref['joint_pos'].shape[1] + 7}")

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.2
    camera.azimuth = 145
    camera.elevation = -13
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = str(args.ffmpeg) if args.ffmpeg else shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found; pass --ffmpeg /path/to/ffmpeg")
    writer = subprocess.Popen(
        [
            ffmpeg, "-y", "-loglevel", "warning",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{args.width}x{args.height}", "-framerate", str(fps),
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", "18", "-pix_fmt", "yuv420p", str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for frame in range(len(ref["joint_pos"])):
            data.qpos[: ref["joint_pos"].shape[1]] = ref["joint_pos"][frame]
            data.qpos[-7:-4] = ref["object_pos_w"][frame]
            data.qpos[-4:] = ref["object_quat_w"][frame]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.65]
            renderer.update_scene(data, camera=camera)
            assert writer.stdin is not None
            writer.stdin.write(np.ascontiguousarray(renderer.render()).tobytes())
    finally:
        if writer.stdin is not None:
            writer.stdin.close()
        return_code = writer.wait()
        renderer.close()
    if return_code:
        raise RuntimeError(f"FFmpeg exited with status {return_code}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()

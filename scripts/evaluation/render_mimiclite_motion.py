#!/usr/bin/env python3
"""Render an any4hdmi G1 qpos trajectory to MP4 with MuJoCo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio_ffmpeg
import mujoco
import numpy as np


def _build_model(mjcf: Path, *, width: int, height: int) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(mjcf))
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    collision = {
        "contype": 1,
        "conaffinity": 1,
        "condim": 3,
        "friction": (0.9, 0.005, 0.0001),
    }
    spec.worldbody.add_geom(
        name="render_ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0.0, 0.0, 0.05),
        rgba=(0.16, 0.18, 0.20, 1.0),
        **collision,
    )
    x_min, x_max = -0.57366969, 0.56492993
    y_min, y_max = -1.16311446, 0.24821548
    floor_height = 0.71
    spec.worldbody.add_geom(
        name="render_box71",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, floor_height * 0.5),
        size=((x_max - x_min) * 0.5, (y_max - y_min) * 0.5, floor_height * 0.5),
        rgba=(0.25, 0.42, 0.56, 1.0),
        **collision,
    )
    roof_bottom = floor_height + 1.16
    spec.worldbody.add_geom(
        name="render_clearance_roof",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, roof_bottom + 0.02),
        size=((x_max - x_min) * 0.5, (y_max - y_min) * 0.5, 0.02),
        rgba=(0.85, 0.88, 0.92, 0.18),
        contype=0,
        conaffinity=0,
    )
    spec.worldbody.add_light(
        pos=(1.5, -2.0, 4.0),
        dir=(-0.25, 0.25, -1.0),
        diffuse=(1.0, 1.0, 1.0),
        specular=(0.25, 0.25, 0.25),
    )
    spec.worldbody.add_light(
        pos=(-2.0, 1.5, 2.5),
        dir=(0.4, -0.2, -1.0),
        diffuse=(0.55, 0.60, 0.68),
        specular=(0.1, 0.1, 0.1),
    )
    return spec.compile()


def render(
    motion: Path,
    mjcf: Path,
    output: Path,
    *,
    width: int,
    height: int,
) -> None:
    with np.load(motion, allow_pickle=False) as payload:
        qpos = np.asarray(payload["qpos"], dtype=np.float64)
        fps = float(np.asarray(payload["fps"]).item())
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"Expected qpos [T, 36], got {qpos.shape}")

    model = _build_model(mjcf, width=width, height=height)
    if model.nq != qpos.shape[1]:
        raise ValueError(f"MJCF nq={model.nq} does not match motion width {qpos.shape[1]}")
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (-0.08, -0.47, 0.86)
    camera.distance = 3.25
    camera.azimuth = 132.0
    camera.elevation = -18.0

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output),
        (width, height),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=2,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        for frame, pose in enumerate(qpos):
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            writer.send(renderer.render())
            if frame % 100 == 0 or frame + 1 == len(qpos):
                print(f"rendered {frame + 1}/{len(qpos)}", flush=True)
    finally:
        renderer.close()
        writer.close()
    print(f"wrote {output} ({len(qpos)} frames at {fps:g} fps)")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion",
        type=Path,
        default=root / "third_party/mimiclite-assets/g1-climb-turn-sit-71cm/motions/climb_turn_floor_sit_71cm.npz",
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=root / "third_party/mimiclite-assets/g1_xmls/g1-mode_13_15.xml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs/mimiclite/climb_turn_floor_sit_71cm.mp4",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()
    render(args.motion, args.mjcf, args.output, width=args.width, height=args.height)


if __name__ == "__main__":
    main()

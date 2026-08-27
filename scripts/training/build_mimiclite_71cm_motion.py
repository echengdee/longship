#!/usr/bin/env python3
"""Build a MimicLite/any4hdmi climb-turn-floor-sit reference motion.

The source OmniRetarget trajectory uses ``[quat_wxyz, root_xyz, joints]``.
MimicLite's any4hdmi datasets use MuJoCo qpos ordering
``[root_xyz, quat_wxyz, joints]``.  The real climb and turn are preserved;
only the short settle and floor-sit tail are synthesized.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


QPOS_NAMES = [
    "root_tx", "root_ty", "root_tz", "root_qw", "root_qx", "root_qy", "root_qz",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

SOURCE_PLATFORM_HEIGHT_M = 0.7044943820224719


def _smoothstep5(t: np.ndarray) -> np.ndarray:
    return t**3 * (10.0 - 15.0 * t + 6.0 * t**2)


def _normalize_quaternions(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True).clip(min=1e-12)
    for index in range(1, len(quat)):
        if np.dot(quat[index - 1], quat[index]) < 0.0:
            quat[index] *= -1.0
    return quat


def _slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q0[None, :] + t[:, None] * (q1 - q0)[None, :]
        return _normalize_quaternions(result)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return (
        np.sin((1.0 - t) * theta)[:, None] / sin_theta * q0[None, :]
        + np.sin(t * theta)[:, None] / sin_theta * q1[None, :]
    )


def _resample(qpos: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    duration = (len(qpos) - 1) / source_fps
    target_count = int(round(duration * target_fps)) + 1
    source_t = np.arange(len(qpos), dtype=np.float64) / source_fps
    target_t = np.arange(target_count, dtype=np.float64) / target_fps
    target_t[-1] = duration
    output = np.empty((target_count, qpos.shape[1]), dtype=np.float64)
    for column in list(range(3)) + list(range(7, qpos.shape[1])):
        output[:, column] = np.interp(target_t, source_t, qpos[:, column])
    quats = _normalize_quaternions(qpos[:, 3:7])
    left = np.minimum(np.searchsorted(source_t, target_t, side="right") - 1, len(source_t) - 2)
    left = np.maximum(left, 0)
    alpha = (target_t - source_t[left]) / (source_t[left + 1] - source_t[left])
    for index in range(target_count):
        output[index, 3:7] = _slerp(quats[left[index]], quats[left[index] + 1], alpha[index:index + 1])[0]
    return output


def _yaw_from_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _seated_joint_target(source: np.ndarray) -> np.ndarray:
    target = source.copy()
    # Symmetric bent-leg floor sit.  The ankle target remains inside the G1
    # joint limits and leaves the feet in front of the pelvis rather than
    # folding both knees under the body (which would be a kneel).
    target[[7, 13]] = -2.40
    target[[8, 14]] = 0.0
    target[[9, 15]] = 0.0
    target[[10, 16]] = 1.97
    target[[11, 17]] = 0.40
    target[[12, 18]] = 0.0
    target[19:22] = 0.0
    # Arms forward and slightly out, elbows bent; they are not used as
    # required contacts so the reference stays compatible with mode-15 G1.
    target[22:29] = (0.35, 0.18, 0.0, 0.75, 0.0, 0.0, 0.0)
    target[29:36] = (0.35, -0.18, 0.0, 0.75, 0.0, 0.0, 0.0)
    return target


def build_motion(
    source_path: Path,
    *,
    climb_end_frame: int,
    cargo_floor_height_m: float,
    target_fps: float,
    settle_s: float,
    sit_s: float,
    hold_s: float,
    move_inside_m: float,
    turn_deg: float,
) -> tuple[np.ndarray, dict[str, int]]:
    payload = np.load(source_path, allow_pickle=False)
    source = np.asarray(payload["qpos"], dtype=np.float64)
    source_fps = float(payload["fps"])
    if source.shape[1] != 36:
        raise ValueError(f"Expected 36D OmniRetarget qpos, got {source.shape}")
    if not 1 <= climb_end_frame < len(source):
        raise ValueError(f"climb_end_frame must be in [1, {len(source) - 1}]")

    # Include the end frame and convert quaternion-first OmniRetarget ordering.
    omni = source[: climb_end_frame + 1]
    climb = np.concatenate((omni[:, 4:7], omni[:, 0:4], omni[:, 7:36]), axis=1)
    climb[:, 2] += cargo_floor_height_m - SOURCE_PLATFORM_HEIGHT_M
    climb[:, 3:7] = _normalize_quaternions(climb[:, 3:7])
    climb = _resample(climb, source_fps, target_fps)

    settle_count = max(1, int(round(settle_s * target_fps)))
    settle = np.repeat(climb[-1][None, :], settle_count, axis=0)

    sit_count = max(2, int(round(sit_s * target_fps)))
    alpha = _smoothstep5(np.linspace(0.0, 1.0, sit_count, dtype=np.float64))
    sit_start = settle[-1]
    sit_target = _seated_joint_target(sit_start)
    # Pelvis sphere bottom is root_z - 0.15 m in the released G1 MJCF.
    # Add 2 cm clearance; the final foot capsule bottoms are then just above
    # the cargo floor and can settle into contact during training.
    sit_target[2] = cargo_floor_height_m + 0.17
    yaw = _yaw_from_wxyz(sit_start[3:7])
    forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    sit_target[:2] = sit_start[:2] + move_inside_m * forward
    target_yaw = yaw + math.radians(turn_deg)
    sit_target[3:7] = np.asarray(
        (math.cos(target_yaw / 2.0), 0.0, 0.0, math.sin(target_yaw / 2.0))
    )

    sit = sit_start[None, :] + alpha[:, None] * (sit_target - sit_start)[None, :]
    sit[:, 3:7] = _slerp(sit_start[3:7], sit_target[3:7], alpha)

    hold_count = max(1, int(round(hold_s * target_fps)))
    hold = np.repeat(sit[-1][None, :], hold_count, axis=0)
    qpos = np.concatenate((climb, settle[1:], sit[1:], hold), axis=0)
    qpos[:, 3:7] = _normalize_quaternions(qpos[:, 3:7])
    phases = {
        "climb_end": len(climb) - 1,
        "settle_end": len(climb) + len(settle) - 2,
        "turn_sit_end": len(climb) + len(settle) + len(sit) - 3,
        "motion_end": len(qpos) - 1,
    }
    return qpos.astype(np.float32), phases


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "third_party/OmniRetarget_Dataset/robot-terrain/climb_12_z_scale_1.0.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "third_party/mimiclite-assets/g1-climb-turn-sit-71cm",
    )
    # Frame 240 is fully on the platform but remains below the 1.87 m cargo
    # roof. The later source turn stands upright and cannot fit the opening.
    parser.add_argument("--climb-end-frame", type=int, default=240)
    parser.add_argument("--cargo-floor-height-m", type=float, default=0.71)
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--settle-s", type=float, default=0.25)
    parser.add_argument("--sit-s", type=float, default=3.0)
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument("--move-inside-m", type=float, default=0.25)
    parser.add_argument("--turn-deg", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    qpos, phases = build_motion(
        args.source,
        climb_end_frame=args.climb_end_frame,
        cargo_floor_height_m=args.cargo_floor_height_m,
        target_fps=args.target_fps,
        settle_s=args.settle_s,
        sit_s=args.sit_s,
        hold_s=args.hold_s,
        move_inside_m=args.move_inside_m,
        turn_deg=args.turn_deg,
    )
    motions_dir = args.output / "motions"
    motions_dir.mkdir(parents=True, exist_ok=True)
    motion_path = motions_dir / "climb_turn_floor_sit_71cm.npz"
    np.savez_compressed(
        motion_path,
        qpos=qpos,
        fps=np.asarray(args.target_fps, dtype=np.float32),
        qpos_layout=np.asarray("mujoco_xyz_wxyz_joints"),
    )
    manifest = {
        "format_version": 2,
        "dataset_name": "g1_climb_turn_floor_sit_71cm_v0",
        "mjcf": "../g1_xmls/g1-mode_13_15.xml",
        "motions_subdir": "motions",
        "timestep": 1.0 / args.target_fps,
        "qpos_dim": len(QPOS_NAMES),
        "qpos_names": QPOS_NAMES,
        "num_motions": 1,
        "source": {
            "dataset": "omniretarget/OmniRetarget_Dataset",
            "trajectory": str(args.source),
            "source_platform_height_m": SOURCE_PLATFORM_HEIGHT_M,
            "cargo_floor_height_m": args.cargo_floor_height_m,
            "climb_end_frame_inclusive": args.climb_end_frame,
            "synthetic_tail": "quintic low-clearance turn and floor-sit with symmetric bent legs",
            "move_inside_m": args.move_inside_m,
            "turn_deg": args.turn_deg,
            "phase_end_frames": phases,
        },
        "total_hours": len(qpos) / args.target_fps / 3600.0,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {motion_path}")
    print(f"frames={len(qpos)} fps={args.target_fps:g} duration_s={len(qpos) / args.target_fps:.3f}")
    print(f"phase_end_frames={phases}")
    print(f"root_z_range_m=[{qpos[:, 2].min():.6f}, {qpos[:, 2].max():.6f}]")
    print(f"final_root_xyz_m={qpos[-1, :3].tolist()}")


if __name__ == "__main__":
    main()

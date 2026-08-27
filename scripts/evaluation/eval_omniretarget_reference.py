#!/usr/bin/env python3
"""Evaluate an OmniRetarget/HoloSoma whole-body reference without a policy.

The checks deliberately separate reference quality from policy tracking quality:
finite values and continuity, MuJoCo joint limits, foot/ground geometry,
object/ground geometry, hand/object proximity, and consistency with the public
OmniRetarget source sequence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(x)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    dot = np.clip(np.abs(np.sum(q0 * q1, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def slerp_wxyz(q: np.ndarray, t_old: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    xyzw = q[:, [1, 2, 3, 0]]
    result = Slerp(t_old, Rotation.from_quat(xyzw))(t_new).as_quat()
    return result[:, [3, 0, 1, 2]]


def interp_rows(x: np.ndarray, t_old: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t_new, t_old, x[:, i]) for i in range(x.shape[1])], axis=1)


def load_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open(encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    return np.asarray(vertices, dtype=np.float64)


def joint_limit_report(xml_path: Path, names: list[str], q: np.ndarray) -> dict:
    root = ET.parse(xml_path).getroot()
    ranges = {}
    for elem in root.iter("joint"):
        if elem.get("name") and elem.get("range"):
            ranges[elem.get("name")] = [float(v) for v in elem.get("range").split()]
    missing = [name for name in names if name not in ranges]
    lower = np.array([ranges[name][0] for name in names if name in ranges])
    upper = np.array([ranges[name][1] for name in names if name in ranges])
    q_known = q[:, [i for i, name in enumerate(names) if name in ranges]]
    below = np.maximum(lower - q_known, 0.0)
    above = np.maximum(q_known - upper, 0.0)
    violation = np.maximum(below, above)
    margin = np.minimum(q_known - lower, upper - q_known)
    where = np.unravel_index(np.argmax(violation), violation.shape)
    known_names = [name for name in names if name in ranges]
    near_limit = {
        name: int(np.count_nonzero(margin[:, i] < 1e-3))
        for i, name in enumerate(known_names)
        if np.any(margin[:, i] < 1e-3)
    }
    return {
        "missing_joint_names": missing,
        "violating_samples": int(np.count_nonzero(violation > 1e-6)),
        "violating_frames": int(np.count_nonzero(np.any(violation > 1e-6, axis=1))),
        "max_violation_rad": float(violation[where]),
        "max_violation_frame": int(where[0]),
        "max_violation_joint": known_names[where[1]],
        "minimum_limit_margin_rad": float(np.min(margin)),
        "frames_within_1e-3_rad_by_joint": near_limit,
    }


def box_geometry(vertices: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotations = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
    world_vertices = np.empty((len(pos), len(vertices), 3), dtype=np.float64)
    for i, rotation in enumerate(rotations):
        world_vertices[i] = rotation.apply(vertices) + pos[i]
    return world_vertices, np.min(world_vertices[..., 2], axis=1)


def aabb_surface_distance(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outside = np.maximum(np.maximum(lo - points, points - hi), 0.0)
    outside_dist = np.linalg.norm(outside, axis=1)
    inside = np.all((points >= lo) & (points <= hi), axis=1)
    to_face = np.min(np.concatenate([points - lo, hi - points], axis=1), axis=1)
    signed = np.where(inside, -to_face, outside_dist)
    return outside_dist, signed, inside


def hand_box_report(
    data: dict,
    vertices: np.ndarray,
    lifted: np.ndarray,
    hand_mesh_paths: dict[str, Path | None],
    box_mesh_path: Path,
) -> dict:
    names = [str(v) for v in data["body_names"]]
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    box_rot = Rotation.from_quat(data["object_quat_w"][:, [1, 2, 3, 0]])
    result = {"note": "Distances use the box mesh AABB in box-local coordinates; negative signed distance means the hand body origin lies inside that AABB."}
    for side in ("left", "right"):
        idx = names.index(f"{side}_rubber_hand_link")
        local = box_rot.inv().apply(data["body_pos_w"][:, idx] - data["object_pos_w"])
        distance, signed, inside = aabb_surface_distance(local, lo, hi)
        subset = lifted if np.any(lifted) else np.ones(len(local), dtype=bool)
        result[side] = {
            "all_frames_surface_distance_m": stats(distance),
            "lifted_surface_distance_m": stats(distance[subset]),
            "inside_aabb_fraction_all": float(np.mean(inside)),
            "inside_aabb_fraction_lifted": float(np.mean(inside[subset])),
            "deepest_inside_aabb_m": float(max(0.0, -np.min(signed[subset]))),
        }
    if any(hand_mesh_paths.values()):
        import trimesh

        box_mesh = trimesh.load(box_mesh_path, force="mesh", process=False)
        result["mesh_surface_note"] = (
            "Actual mesh nearest-surface distance is reliable for contact proximity. "
            "The box mesh is not watertight, so it is not used to classify penetration."
        )
        lifted_ids = np.flatnonzero(lifted)
        for side, mesh_path in hand_mesh_paths.items():
            if mesh_path is None:
                continue
            hand_mesh = trimesh.load(mesh_path, force="mesh", process=False)
            hand_vertices = np.asarray(hand_mesh.vertices, dtype=np.float64)
            stride = max(1, len(hand_vertices) // 50)
            hand_vertices = hand_vertices[::stride]
            idx = names.index(f"{side}_rubber_hand_link")
            hand_rot = Rotation.from_quat(data["body_quat_w"][:, idx][:, [1, 2, 3, 0]])
            local_batches = []
            for frame in lifted_ids:
                world = hand_rot[frame].apply(hand_vertices) + data["body_pos_w"][frame, idx]
                local = box_rot[frame].inv().apply(world - data["object_pos_w"][frame])
                local_batches.append(local)
            local_points = np.concatenate(local_batches, axis=0)
            _, distances, _ = trimesh.proximity.closest_point(box_mesh, local_points)
            nearest_array = distances.reshape(len(lifted_ids), len(hand_vertices)).min(axis=1)
            result[side]["lifted_actual_mesh_nearest_surface_m"] = stats(nearest_array)
            result[side]["lifted_contact_fraction_le_10mm"] = float(np.mean(nearest_array <= 0.01))
    return result


def source_report(source_path: Path, ref: dict, fps: float) -> dict:
    source = np.load(source_path, allow_pickle=True)
    q = np.asarray(source["qpos"], dtype=np.float64)
    source_fps = float(np.asarray(source.get("fps", [30])).reshape(-1)[0])
    t_old = np.arange(len(q)) / source_fps
    t_new = np.arange(len(ref["joint_pos"])) / fps
    t_new = np.minimum(t_new, t_old[-1])
    src_root_q = slerp_wxyz(q[:, 0:4], t_old, t_new)
    src_root_p = interp_rows(q[:, 4:7], t_old, t_new)
    src_joints = interp_rows(q[:, 7:36], t_old, t_new)
    src_obj_q = slerp_wxyz(q[:, 36:40], t_old, t_new)
    src_obj_p = interp_rows(q[:, 40:43], t_old, t_new)
    return {
        "source_fps": source_fps,
        "source_frames": int(len(q)),
        "robot_root_position_error_m": stats(np.linalg.norm(ref["joint_pos"][:, :3] - src_root_p, axis=1)),
        "robot_root_orientation_error_deg": stats(quat_angle_deg(ref["joint_pos"][:, 3:7], src_root_q)),
        "robot_joint_error_rad": stats(np.abs(ref["joint_pos"][:, 7:36] - src_joints)),
        "object_position_error_m": stats(np.linalg.norm(ref["object_pos_w"] - src_obj_p, axis=1)),
        "object_orientation_error_deg": stats(quat_angle_deg(ref["object_quat_w"], src_obj_q)),
        "interpretation": "Object agreement with simultaneous robot disagreement indicates that this reference is not reproducible from the supplied public source by interpolation alone.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--model-xml", type=Path, required=True)
    parser.add_argument("--box-mesh", type=Path, required=True)
    parser.add_argument("--left-hand-mesh", type=Path)
    parser.add_argument("--right-hand-mesh", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    loaded = np.load(args.reference, allow_pickle=True)
    data = {key: loaded[key] for key in loaded.files}
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    frames = len(data["joint_pos"])
    dt = 1.0 / fps
    names = [str(v) for v in data["body_names"]]
    vertices = load_obj_vertices(args.box_mesh)
    _, box_bottom = box_geometry(vertices, data["object_pos_w"], data["object_quat_w"])
    lifted = box_bottom > 0.05

    left_ids = [i for i, name in enumerate(names) if name.startswith("left_ankle_roll_sphere_")]
    right_ids = [i for i, name in enumerate(names) if name.startswith("right_ankle_roll_sphere_")]
    left_z = np.min(data["body_pos_w"][:, left_ids, 2] - 0.005, axis=1)
    right_z = np.min(data["body_pos_w"][:, right_ids, 2] - 0.005, axis=1)
    root_step = np.linalg.norm(np.diff(data["joint_pos"][:, :3], axis=0), axis=1)
    object_step = np.linalg.norm(np.diff(data["object_pos_w"], axis=0), axis=1)
    root_rot_step = quat_angle_deg(data["joint_pos"][:-1, 3:7], data["joint_pos"][1:, 3:7])
    object_rot_step = quat_angle_deg(data["object_quat_w"][:-1], data["object_quat_w"][1:])
    arrays = {key: value for key, value in data.items() if isinstance(value, np.ndarray) and value.dtype.kind in "fiu"}

    report = {
        "reference": str(args.reference.resolve()),
        "frames": frames,
        "fps": fps,
        "duration_s": frames / fps,
        "integrity": {
            "all_numeric_values_finite": bool(all(np.all(np.isfinite(v)) for v in arrays.values())),
            "root_quaternion_max_norm_error": float(np.max(np.abs(np.linalg.norm(data["joint_pos"][:, 3:7], axis=1) - 1))),
            "body_quaternion_max_norm_error": float(np.max(np.abs(np.linalg.norm(data["body_quat_w"], axis=2) - 1))),
            "object_quaternion_max_norm_error": float(np.max(np.abs(np.linalg.norm(data["object_quat_w"], axis=1) - 1))),
        },
        "continuity": {
            "root_speed_mps": stats(root_step / dt),
            "object_speed_mps": stats(object_step / dt),
            "root_rotation_step_deg": stats(root_rot_step),
            "object_rotation_step_deg": stats(object_rot_step),
            "joint_speed_radps": stats(np.abs(data["joint_vel"][:, 6:])),
        },
        "joint_limits": joint_limit_report(args.model_xml, [str(v) for v in data["joint_names"]], data["joint_pos"][:, 7:]),
        "feet_ground": {
            "note": "Clearance is the lowest 5 mm foot marker sphere surface relative to z=0.",
            "left_clearance_m": stats(left_z),
            "right_clearance_m": stats(right_z),
            "left_contact_fraction_le_10mm": float(np.mean(left_z <= 0.01)),
            "right_contact_fraction_le_10mm": float(np.mean(right_z <= 0.01)),
            "both_feet_airborne_fraction_gt_20mm": float(np.mean((left_z > 0.02) & (right_z > 0.02))),
            "ground_penetration_frames_below_minus_10mm": int(np.count_nonzero((left_z < -0.01) | (right_z < -0.01))),
        },
        "object_ground": {
            "bottom_height_m": stats(box_bottom),
            "ground_penetration_frames_below_minus_10mm": int(np.count_nonzero(box_bottom < -0.01)),
            "lifted_threshold_m": 0.05,
            "lifted_frames": int(np.count_nonzero(lifted)),
            "first_lifted_frame": int(np.flatnonzero(lifted)[0]) if np.any(lifted) else None,
            "last_lifted_frame": int(np.flatnonzero(lifted)[-1]) if np.any(lifted) else None,
        },
        "hands_vs_box": hand_box_report(
            data,
            vertices,
            lifted,
            {"left": args.left_hand_mesh, "right": args.right_hand_mesh},
            args.box_mesh,
        ),
    }
    if args.source:
        report["public_source_consistency"] = source_report(args.source, data, fps)

    issues = []
    if not report["integrity"]["all_numeric_values_finite"]:
        issues.append("non-finite values")
    if report["joint_limits"]["violating_frames"]:
        issues.append("joint-limit violations")
    if report["feet_ground"]["ground_penetration_frames_below_minus_10mm"]:
        issues.append("foot/ground penetration")
    if report["object_ground"]["ground_penetration_frames_below_minus_10mm"]:
        issues.append("box/ground penetration")
    if args.source and report["public_source_consistency"]["robot_root_position_error_m"]["max"] > 0.01:
        issues.append("robot trajectory does not reproduce from public source")
    report["summary"] = {
        "reference_is_numerically_valid": not any(v in issues for v in ("non-finite values", "joint-limit violations")),
        "reference_provenance_is_reproducible": "robot trajectory does not reproduce from public source" not in issues,
        "issues": issues,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(args.output.resolve())


if __name__ == "__main__":
    main()

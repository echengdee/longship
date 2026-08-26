"""This is a tool that convert robot base pose to another frame on given robot chain"""

import argparse
import functools
import multiprocessing as mp
import numpy as np
import os
import pickle as pkl
import torch
import tqdm

import pytorch_kinematics as pk


def load_GMR_src_file(src_file):
    """Load from GMR source file"""
    with open(src_file, "rb") as f:
        motion_data = pkl.load(f)
    joint_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    joint_pos = motion_data["dof_pos"]  # (N, 29)
    base_pos_w = motion_data["root_pos"]  # (N, 3)
    base_quat_w_ = motion_data["root_rot"]  # (N, 4), xyzw order
    base_quat_w = base_quat_w_[..., [3, 0, 1, 2]]  # convert to wxyz order
    framerate = motion_data["fps"]
    return {
        "joint_names": joint_names,
        "joint_pos": joint_pos,
        "base_pos_w": base_pos_w,
        "base_quat_w": base_quat_w,
        "framerate": framerate,
    }


def convert_file(
    src_tgt_pairs,
    urdf: str,
    src_frame_name: str,
    tgt_frame_name: str,
    joints_to_revert: list = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
):
    src_file, tgt_file = src_tgt_pairs

    src_npz = load_GMR_src_file(src_file)

    with open(urdf) as f:
        robot_chain = pk.build_chain_from_urdf(f.read())

    joint_pos = torch.as_tensor(src_npz["joint_pos"], dtype=torch.float32)
    joint_names = src_npz["joint_names"]
    src_base_pos_w = src_npz["base_pos_w"]
    src_base_quat_w = src_npz["base_quat_w"]
    src_base_transform = pk.Transform3d(
        rot=src_base_quat_w,  # wxyz order
        pos=src_base_pos_w,  # xyz order
    )

    # reverse joint names to match the new robot urdf if needed
    for joint_name in joints_to_revert:
        joint_idx_src = joint_names.index(joint_name)
        joint_pos[:, joint_idx_src] *= -1.0

    joint_order_src_to_pk = torch.zeros(len(joint_names), dtype=torch.long)
    for joint_i, joint_name in enumerate(robot_chain.get_joint_parameter_names()):
        joint_idx_src = joint_names.index(joint_name)
        joint_order_src_to_pk[joint_i] = joint_idx_src
    joint_pos_pk = joint_pos[:, joint_order_src_to_pk]

    src_tgt_frame_indices = robot_chain.get_frame_indices(src_frame_name, tgt_frame_name)
    frame_poses = robot_chain.forward_kinematics(joint_pos_pk, src_tgt_frame_indices)
    src_frame_poses = frame_poses[src_frame_name]  # pk.Transform3d
    tgt_frame_poses = frame_poses[tgt_frame_name]  # pk.Transform3d

    tgt_in_src_frame_poses = src_frame_poses.inverse().compose(tgt_frame_poses)  # pk.Transform3d
    tgt_base_transform = src_base_transform.compose(tgt_in_src_frame_poses)
    tgt_base_matrix = tgt_base_transform.get_matrix()  # (N, 4, 4)
    tgt_base_quat_w = pk.matrix_to_quaternion(tgt_base_matrix[:, :3, :3])
    tgt_base_pos_w = tgt_base_matrix[:, :3, 3]

    # pack the file and store
    np.savez(
        tgt_file,
        framerate=src_npz["framerate"],
        joint_names=joint_names,
        joint_pos=joint_pos.numpy(),
        base_pos_w=tgt_base_pos_w,
        base_quat_w=tgt_base_quat_w,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="Input file or folder")
    parser.add_argument("--tgt", type=str, help="Target file or folder. Structure preserved if folder")
    parser.add_argument(
        "--urdf", type=str, help="Robot urdf to convert the base, does not need to match the source base or target base"
    )
    parser.add_argument("--src_frame", type=str, default="pelvis")
    parser.add_argument("--tgt_frame", type=str, default="torso_link")
    parser.add_argument("--num_cpus", default=10)

    args = parser.parse_args()

    # walk through the source folder and make folders in target folder if needed
    src_tgt_pairs = []
    if os.path.isfile(args.src):
        src_tgt_pairs.append((args.src, args.tgt))
    else:
        if not os.path.exists(args.tgt):
            os.makedirs(args.tgt, exist_ok=True)
        for root, _, filenames in os.walk(args.src):
            target_dirpath = os.path.join(args.tgt, os.path.relpath(root, args.src))
            os.makedirs(target_dirpath, exist_ok=True)
            for filename in filenames:
                if not filename.endswith(".pkl"):
                    continue
                src_tgt_pairs.append(
                    (
                        os.path.join(root, filename),
                        os.path.join(target_dirpath, filename.replace(".pkl", "_retargeted.npz")),
                    )
                )

    with mp.Pool(args.num_cpus) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap_unordered(
                    functools.partial(
                        convert_file,
                        urdf=args.urdf,
                        src_frame_name=args.src_frame,
                        tgt_frame_name=args.tgt_frame,
                    ),
                    src_tgt_pairs,
                ),
                total=len(src_tgt_pairs),
            )
        )


if __name__ == "__main__":
    main()

import functools
import multiprocessing as mp
import numpy as np
import os
import torch
import tqdm
import yaml

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(dict(headless=True))
simulation_app = app_launcher.app

import pytorch_kinematics as pk

import isaaclab.utils.math as math_utils

from instinctlab.assets.unitree_g1 import G1_29DOF_TORSOBASE_CFG
from instinctlab.utils.humanoid_ik import HumanoidSmplRotationalIK

LINK_OF_INTERESTS = [
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "pelvis",
    "torso_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "left_knee_link",
    "right_knee_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

KNEE_JOINT_NAMES = [
    "left_knee_joint",
    "right_knee_joint",
]

FOOT_LINK_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

TORSO_LINK_NAME = "torso_link"

POTENTIAL_GIMBAL_LOCK_JOINTS = [
    ("left_hip_roll_joint", np.pi / 2 + 0.2793),
    ("right_hip_roll_joint", -np.pi / 2 - 0.2793),
    ("left_shoulder_roll_joint", np.pi / 2),
    ("right_shoulder_roll_joint", -np.pi / 2),
]


@torch.no_grad()
def determine_motion_validity(filepath, args) -> tuple[str, bool]:
    """Determines if the motion is valid based on the file path and the arguments."""

    with open(args.urdf_path, mode="rb") as f:
        urdf_str = f.read()
    robot_chain = pk.build_chain_from_urdf(urdf_str)
    joint_limits_low, joint_limits_high = robot_chain.get_joint_limits()  # (num_joints,)
    joint_limits_low = torch.as_tensor(joint_limits_low)
    joint_limits_high = torch.as_tensor(joint_limits_high)
    joint_names = [j.name for j in robot_chain.get_joints()]
    knee_joint_idxs = [joint_names.index(name) for name in KNEE_JOINT_NAMES]

    try:
        if filepath.endswith("poses.npz"):
            motion = np.load(os.path.join(args.datadir, filepath))
            framerate = motion["mocap_framerate"]
            poses = torch.from_numpy(motion["poses"]).to(torch.float32)
            poses = poses[:, : 24 * 3].reshape(-1, 24, 3)
            root_trans = torch.from_numpy(motion["trans"]).to(torch.float32)
            traj_length_s = poses.shape[0] / framerate
        elif filepath.endswith("retargetted.npz"):
            motion = np.load(os.path.join(args.datadir, filepath), allow_pickle=True)
            framerate = motion["framerate"]
            retargetted_joint_names = (
                motion["joint_names"] if isinstance(motion["joint_names"], list) else motion["joint_names"].tolist()
            )
            root_pos = torch.from_numpy(motion["base_pos_w"]).to(torch.float32)
            root_quat = torch.from_numpy(motion["base_quat_w"]).to(torch.float32)
            robot_joint_pos = torch.from_numpy(motion["joint_pos"]).to(torch.float32)
            traj_length_s = root_pos.shape[0] / framerate
            # re-order joint positions to match the order in robot chain
            chain_joint_names = [j.name for j in robot_chain.get_joints()]
            robot_joint_pos = robot_joint_pos[:, [retargetted_joint_names.index(name) for name in chain_joint_names]]
    except Exception as e:
        print("Error in loading", filepath, ":", e)
        return filepath, False

    vel_smooth_frames = int(args.vel_smooth_window * framerate)

    # check the trajectory time
    if traj_length_s < args.min_traj_time or traj_length_s < vel_smooth_frames / framerate:
        print("Trajectory time too short:", filepath)
        return filepath, False

    if filepath.endswith("poses.npz"):
        # Initializing the IK function and retargetting
        try:
            retargetting_func = HumanoidSmplRotationalIK(
                robot_chain=robot_chain,
                smpl_root_in_robot_link_name="pelvis",
                translation_scaling=0.75,
                translation_height_offset=0.0,
            )
        except Exception as e:
            print(f"Error in initializing IK {filepath}:", e)
            return filepath, False
        try:
            if filepath.endswith("poses.npz"):
                robot_joint_pos, robot_root_poses = retargetting_func(poses, root_trans)
            root_pos = robot_root_poses[:, :3]  # (batch, 3)
            root_quat = robot_root_poses[:, 3:]  # (batch, 4)
        except Exception as e:
            print(f"Error in retargetting {filepath}:", e)

    # From here, we assume root_pos, root_quat, and robot_joint_pos are available
    root_pos, root_quat, robot_joint_pos = (
        root_pos.to(torch.float32),
        root_quat.to(torch.float32),
        robot_joint_pos.to(torch.float32),
    )
    root_yaw = math_utils.euler_xyz_from_quat(root_quat)[2]  # (batch,)

    with torch.no_grad():
        link_poses_b = robot_chain.forward_kinematics(robot_joint_pos)
    link_poses_b = torch.stack(
        [link_poses_b[name].get_matrix().reshape(-1, 4, 4) for name in LINK_OF_INTERESTS],
        dim=1,
    )  # (batch, num_links, 4, 4)
    link_pos_b = link_poses_b[:, :, :3, 3]  # (batch, num_links, 3)
    link_quat_b = math_utils.quat_from_matrix(link_poses_b[:, :, :3, :3])
    link_pos_w, link_quat_w = math_utils.combine_frame_transforms(
        root_pos.unsqueeze(1).expand(-1, link_pos_b.shape[1], -1),
        root_quat.unsqueeze(1).expand(-1, link_pos_b.shape[1], -1),
        link_pos_b,
        link_quat_b,
    )

    # compute the maximum angular velocity
    if args.max_angvel is not None:
        try:
            if args.angvel_frame == "world":
                link_quat = link_quat_w
            elif args.angvel_frame == "base":
                link_quat = link_quat_b
            link_axisang = math_utils.axis_angle_from_quat(link_quat)  # (batch, num_links, 3)
            link_angvel = (
                link_axisang[vel_smooth_frames:] - link_axisang[:-vel_smooth_frames]
            ) / args.vel_smooth_window  # (batch-1, num_links, 3)
            max_angvel = torch.max(torch.norm(link_angvel, dim=-1)).item()
            if max_angvel > args.max_angvel:
                max_angvel_idx = torch.argmax(torch.norm(link_angvel, dim=-1)).tolist()
                max_angvel_frame_idx = max_angvel_idx // link_angvel.shape[1]
                max_angvel_link_idx = max_angvel_idx % link_angvel.shape[1]
                print(
                    "Exceeding maximum angular velocity:",
                    filepath,
                    "framerate:",
                    framerate,
                    "reaches:",
                    max_angvel,
                    "at:",
                    max_angvel_frame_idx,
                    max_angvel_link_idx,
                    "preframe:",
                    link_axisang[max_angvel_frame_idx, max_angvel_link_idx],
                    "postframe:",
                    link_axisang[max_angvel_frame_idx + 1, max_angvel_link_idx],
                )
                return filepath, False
        except Exception as e:
            print(f"Error in computing angular velocity {filepath}:", e)
            return filepath, False

    link_airborne = (link_pos_w[:, :, 2] > args.airborne_height_threshold).all(dim=-1)  # (batch,)
    if args.airborne_height_threshold is not None and args.max_airborne_time is not None:
        # compute the maximum connected airborne time
        airborne_time = 0
        max_airborne_time = 0
        for i in range(link_airborne.shape[0]):
            if link_airborne[i]:
                airborne_time += 1 / framerate
                max_airborne_time = max(max_airborne_time, airborne_time)
            else:
                airborne_time = 0
            if max_airborne_time > args.max_airborne_time:
                print("Exceeding maximum airborne time:", filepath)
                return filepath, False

    if args.min_base_height is not None:
        # check if the base height is above the minimum height
        base_height = root_pos[:, 2]
        if (base_height < args.min_base_height).any():
            print(
                "Base height below minimum:",
                filepath,
                f"min_base_height: {args.min_base_height:.3f}",
                f"base_height: {base_height.min().item():.3f}",
            )
            return filepath, False

    if args.max_base_height is not None:
        # check if the base height is below the maximum height
        base_height = root_pos[:, 2]
        if (base_height > args.max_base_height).any():
            print(
                "Base height above maximum:",
                filepath,
                f"max_base_height: {args.max_base_height:.3f}",
                f"base_height: {base_height.max().item():.3f}",
            )
            return filepath, False

    if args.max_linvel is not None:
        # compute the maximum linear velocity
        root_vel = (root_pos[vel_smooth_frames:] - root_pos[:-vel_smooth_frames]) / args.vel_smooth_window
        max_linvel = torch.max(torch.norm(root_vel, dim=-1)).item()
        if max_linvel > args.max_linvel:
            print(
                "Exceeding maximum linear velocity:",
                filepath,
                f"linvel: {max_linvel:.3f}",
            )
            return filepath, False

    if args.sit_leg_bend_threshold is not None and args.sit_foot_center_offset_max is not None:
        # check if the leg is bent
        knee_bend = robot_joint_pos[:, knee_joint_idxs].abs() > args.sit_leg_bend_threshold
        knee_bend = knee_bend.all(dim=-1)  # (batch,)
        # check if the foot center is too far from the base link
        foot_link_idxs = [LINK_OF_INTERESTS.index(name) for name in FOOT_LINK_NAMES]
        foot_center_w = link_pos_w[:, foot_link_idxs, :].mean(dim=1)  # (batch, 3)
        foot_center_offset = foot_center_w - root_pos
        foot_center_forward_offset = foot_center_offset[:, 0] * torch.cos(root_yaw) + foot_center_offset[
            :, 1
        ] * torch.sin(root_yaw)
        foot_away = torch.abs(foot_center_forward_offset) > args.sit_foot_center_offset_max
        # check if the entire body has more than 3 support links
        num_support_links = (link_pos_w[:, :, 2] < args.airborne_height_threshold).sum(dim=-1)  # (batch,)
        not_crawling = num_support_links == 2  # (batch,)
        # summarize the conditions
        possible_sit = knee_bend & foot_away & not_crawling  # (batch,)
        if possible_sit.any():
            sit_idx = torch.where(possible_sit)[0]
            print(
                "Possible sit detected:",
                filepath,
                f"at {sit_idx}",
                f"foot_center_offset min: {min([x for x in foot_center_forward_offset[sit_idx].cpu().numpy()])}",
                "num_support_links min:"
                f" {min([np.format_float_positional(x, precision=2) for x in num_support_links[sit_idx].cpu().numpy()])}",
            )
            return filepath, False

    if args.max_rollpitch is not None:
        roll, pitch, _ = math_utils.euler_xyz_from_quat(root_quat)
        roll = math_utils.wrap_to_pi(roll)
        pitch = math_utils.wrap_to_pi(pitch)
        max_roll = torch.max(torch.abs(roll)).item()
        max_pitch = torch.max(torch.abs(pitch)).item()
        if max_roll > args.max_rollpitch or max_pitch > args.max_rollpitch:
            # check if the roll/pitch is exceeding the threshold
            roll_idx = torch.argmax(torch.abs(roll)).tolist()
            pitch_idx = torch.argmax(torch.abs(pitch)).tolist()
            print(
                "Exceeding maximum roll/pitch:",
                filepath,
                f"roll: {max_roll:.3f} at {roll_idx}",
                f"pitch: {max_pitch:.3f} at {pitch_idx}",
            )
            return filepath, False

    if args.gimbal_lock_threshold is not None and args.gimbal_lock_time:
        # check if the potential gimbal lock joints are both over the velocity threshold
        for joint_name, gimbal_lock_pos in POTENTIAL_GIMBAL_LOCK_JOINTS:
            joint_idx = joint_names.index(joint_name)
            joint_pos = robot_joint_pos[:, joint_idx]
            in_gimbal_lock = (joint_pos - gimbal_lock_pos).abs() < args.gimbal_lock_threshold
            # return false if the gimbal lock lasts for a certain time
            if in_gimbal_lock.any():
                # find the connected frames that are in gimbal lock
                gimbal_lock_time = 0
                for i in range(in_gimbal_lock.shape[0]):
                    if in_gimbal_lock[i]:
                        gimbal_lock_time += 1 / framerate
                    else:
                        gimbal_lock_time = 0
                    if gimbal_lock_time > args.gimbal_lock_time:
                        print(
                            "Exceeding maximum gimbal lock time:",
                            filepath,
                            f"joint: {joint_name}",
                            f"gimbal_lock_pos: {gimbal_lock_pos:.3f}",
                            f"gimbal_lock_time: {gimbal_lock_time:.3f}",
                        )
                        return filepath, False

    if args.joint_limit_time is not None:
        out_of_limit_masks = torch.logical_or(
            robot_joint_pos > joint_limits_high.unsqueeze(0),
            robot_joint_pos < joint_limits_low.unsqueeze(0),
        )
        out_of_limit_count = out_of_limit_masks.sum(-1)  # (batch,)
        out_of_limit_frame_count = (out_of_limit_count > 2).sum()
        if out_of_limit_frame_count > args.joint_limit_time * framerate:
            print(f"Exceeding joint limit time (count only more than 3 joints): filepath")
            return filepath, False

    if args.max_joint_vel is not None:
        joint_vel = (
            robot_joint_pos[vel_smooth_frames:] - robot_joint_pos[:-vel_smooth_frames]
        ) / args.vel_smooth_window
        if (torch.abs(joint_vel) > args.max_joint_vel).any():
            print(f"Exceeding maximum joint vel at {filepath}")
            return filepath, False

    return filepath, True


def main(args):
    # list all the files
    args.datadir = os.path.abspath(args.datadir)
    if args.startfromfiles is None:
        all_files = []
        for root, _, files in os.walk(args.datadir, followlinks=True):
            for file in files:
                if file.endswith("poses.npz") or file.endswith("retargetted.npz"):
                    # store the relative path (relative to datadir)
                    all_files.append(os.path.join(os.path.relpath(root, args.datadir), file))
    else:
        yaml_file = yaml.load(open(args.startfromfiles), Loader=yaml.FullLoader)
        all_files = yaml_file["selected_files"]

    # filter the files
    print(f"Total {len(all_files)} files to filter from {args.datadir}")
    if args.mp:
        print("Filtering the files using multiprocessing...")
        with mp.Pool() as pool:
            files_validity = list(
                tqdm.tqdm(
                    pool.imap_unordered(
                        functools.partial(
                            determine_motion_validity,
                            args=args,
                        ),
                        all_files,
                    ),
                    total=len(all_files),
                )
            )
            save_filtered_files(files_validity, args)
    elif args.pack:
        print("Filtering the files using pack-to-gpu method...")
        max_file_size = 10 * 1024 * 1024  # 10MB
        while len(all_files) > 0:
            file_size_sum = 0
            pack_files = []
            while len(all_files) > 0:
                file = all_files.pop(0)
                pack_files.append(file)
                file_size_sum += os.path.getsize(os.path.join(args.datadir, file))
                if file_size_sum > max_file_size:
                    break
            # filter the pack_files
            files_validity = []
            for file in tqdm.tqdm(pack_files, desc="Filtering"):
                files_validity.append(
                    determine_motion_validity(
                        file,
                        args=args,
                    )
                )
            save_filtered_files(files_validity, args)
    else:
        print("Filtering the files iteratively...")
        files_validity = []
        for file in tqdm.tqdm(all_files, desc="Filtering"):
            files_validity.append(
                determine_motion_validity(
                    file,
                    args=args,
                )
            )
        save_filtered_files(files_validity, args)


def save_filtered_files(files_validity, args):
    # store the list
    print("Storing the list of selected files...")
    selected_files = [file for file, valid in files_validity if valid]
    selected_files = sorted(selected_files)
    filtering_configs = {
        "min_traj_time": args.min_traj_time,
        "vel_smooth_window": args.vel_smooth_window,
        "angvel_frame": args.angvel_frame,
        "max_angvel": args.max_angvel,
        "max_airborne_time": args.max_airborne_time,
        "airborne_height_threshold": args.airborne_height_threshold,
        "max_rollpitch": args.max_rollpitch,
        "max_linvel": args.max_linvel,
        "sit_leg_bend_threshold": args.sit_leg_bend_threshold,
        "sit_foot_center_offset_max": args.sit_foot_center_offset_max,
        "sit_torso_pitch_threshold": args.sit_torso_pitch_threshold,
        "min_base_height": args.min_base_height,
        "max_base_height": args.max_base_height,
        "gimbal_lock_threshold": args.gimbal_lock_threshold,
        "gimbal_lock_time": args.gimbal_lock_time,
        "joint_limit_time": args.joint_limit_time,
        "max_joint_vel": args.max_joint_vel,
        "from_files": args.startfromfiles,
    }
    output = {
        "selected_files": selected_files,
        "filtering_configs": filtering_configs,
        "urdf_path": args.urdf_path,
        "datadir": args.datadir,
    }
    yaml.safe_dump(
        output,
        open(args.output, "w"),
        default_flow_style=False,
    )

    print(f"AMASS filtered done, {len(selected_files)} files stored in {args.output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AMASS filtering script")
    parser.add_argument("--datadir", help="path to the directory containing the AMASS dataset", required=True)
    parser.add_argument("--urdf_path", help="path to the URDF file", default=G1_29DOF_CFG.spawn.asset_path)
    parser.add_argument("--startfromfiles", help="the subset list to filter from", type=str, default=None)
    parser.add_argument("--output", help="output file", default="output.yaml")
    parser.add_argument("--mp", action="store_true", help="use multiprocessing")
    parser.add_argument("--pack", action="store_true", help="use pack-to-gpu method to accelerate the filtering")
    parser.add_argument("--min_traj_time", type=float, help="minimum trajectory time allowed", default=1.0)
    parser.add_argument(
        "--vel_smooth_window", type=float, help="window size for velocity smoothing [second]", default=0.2
    )
    parser.add_argument("--angvel_frame", choices=["world", "base"], default="base")
    parser.add_argument("--max_angvel", type=float, help="maximum angular velocity allowed", default=None)
    parser.add_argument("--max_airborne_time", type=float, help="maximum airborne time allowed", default=0.5)
    parser.add_argument(
        "--airborne_height_threshold",
        type=float,
        help="the height threshold to determine if the character is airborne",
        default=0.2,
    )
    parser.add_argument("--max_rollpitch", type=float, help="maximum roll/pitch angle allowed", default=None)
    parser.add_argument("--max_linvel", type=float, help="maximum linear velocity allowed", default=3.0)
    parser.add_argument(
        "--sit_leg_bend_threshold",
        type=float,
        help="the threshold to determine if the leg is bent (in radians)",
        default=0.8,
    )
    parser.add_argument(
        "--sit_foot_center_offset_max",
        type=float,
        help="the maximum offset of the foot center from the base link",
        default=0.2,
    )
    parser.add_argument(
        "--sit_torso_pitch_threshold",
        type=float,
        help="the maximum pitch of the torso (in radians) when sitting",
        default=0.5,
    )
    parser.add_argument(
        "--min_base_height",
        type=float,
        help="If set, the base height must be above this value to be considered valid",
        default=None,
    )
    parser.add_argument(
        "--max_base_height",
        type=float,
        help="If set, the base height must be below this value to be considered valid",
        default=1.0,
    )
    parser.add_argument(
        "--gimbal_lock_threshold",
        type=float,
        help="The threshold to determine if the joints are in gimbal lock position (in radian)",
        default=None,  # 0.2,
    )
    parser.add_argument(
        "--gimbal_lock_time",
        type=float,
        help="The time duration to consider as gimbal lock (in seconds)",
        default=0.5,
    )
    parser.add_argument(
        "--joint_limit_time",
        type=float,
        help=(
            "The time (s) maximum allows if more-than-3 joints are reaching joint limits. It possibly due to the"
            " retargeting failure."
        ),
        default=0.5,
    )
    parser.add_argument(
        "--max_joint_vel",
        type=float,
        help="The maximum joint velocity, whenever it is reached",
        default=30.0,
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print("Error:", e)
        app_launcher.close()
        raise e

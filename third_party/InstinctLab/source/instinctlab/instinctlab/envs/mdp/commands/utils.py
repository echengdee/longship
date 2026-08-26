""" Some utilities to compute the statistics of the robot w.r.t the shadowing command. """

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

import instinctlab.utils.math as instinct_math_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject

    from instinctlab.envs.mdp import (
        JointPosRefCommand,
        JointVelRefCommand,
        LinkPosRefCommand,
        LinkRotRefCommand,
        PositionRefCommand,
        RotationRefCommand,
        ShadowingCommandBase,
    )


def get_joint_pos_diff_to_cmd(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_term: JointPosRefCommand,
) -> torch.Tensor:
    """Get the joint position difference based on the command.
    Args:
        env: the environment
        asset_cfg: the configuration of the asset
        command_name: the name of the command to be used, e.g., "joint_pos_ref_command"
    """
    # extract useful elements
    asset: Articulation = env.scene[asset_cfg.name]

    # obtain the joint position of the robot
    joint_pos = asset.data.joint_pos
    # obtain the reference joint position from the command
    ref_joint_pos = command_term.command[command_term.ALL_INDICES, command_term.aiming_frame_idx]

    joint_diff = joint_pos - ref_joint_pos  # (batch_size, num_joints)

    return joint_diff  # (batch_size, num_joints)


def get_joint_vel_diff_to_cmd(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_term: JointVelRefCommand,
) -> torch.Tensor:
    """Get the joint velocity difference based on the command.
    Args:
        env: the environment
        asset_cfg: the configuration of the asset
        command_name: the name of the command to be used, e.g., "joint_vel_ref_command"
    """
    # extract useful elements
    asset: Articulation = env.scene[asset_cfg.name]

    # obtain the joint velocity of the robot
    joint_vel = asset.data.joint_vel
    # obtain the reference joint velocity from the command
    ref_joint_vel = command_term.command[command_term.ALL_INDICES, command_term.aiming_frame_idx]

    joint_diff = joint_vel - ref_joint_vel  # (batch_size, num_joints)

    return joint_diff  # (batch_size, num_joints)


def get_link_pos_diff_to_cmd(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_term: LinkPosRefCommand,
) -> torch.Tensor:
    """Get the link position difference based on the command.
    ## NOTE
        Currently, we consider the LinkPosRefCommand only specify local link positions.
    Args:
        env: the environment
        asset_cfg: the configuration of the asset
        command_name: the name of the command to be used, e.g., "link_pos_ref_command"
    """
    # extract useful elements
    asset: RigidObject = env.scene[asset_cfg.name]
    links = asset.find_bodies(command_term.link_of_interests, preserve_order=True)
    link_indices = links[0]

    # obtain the link position of the robot
    link_pos_w = asset.data.body_pos_w[:, link_indices]  # (batch_size, num_links, 3)
    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    root_pos_w_inv, root_quat_w_inv = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w)
    link_pos = math_utils.transform_points(
        link_pos_w,
        root_pos_w_inv,
        root_quat_w_inv,
    )
    # obtain the reference link position from the command
    ref_link_pos = command_term.command[command_term.ALL_INDICES, command_term.aiming_frame_idx]

    link_diff = link_pos - ref_link_pos  # (batch_size, num_links, 3)

    return link_diff  # (batch_size, num_links, 3)


def get_link_rot_diff_mag_to_cmd(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_term: LinkRotRefCommand,
) -> torch.Tensor:
    """Get the link rotation difference magnitude based on the command.
    Args:
        env: the environment
        asset_cfg: the configuration of the asset
        command_name: the name of the command to be used, e.g., "link_rot_ref_command"
    """
    # extract useful elements
    asset: RigidObject = env.scene[asset_cfg.name]
    links = asset.find_bodies(command_term.link_of_interests, preserve_order=True)
    link_indices = links[0]

    # obtain the link rotation of the robot
    link_rot = asset.data.body_quat_w  # (batch_size, num_all_links, 4)
    link_rot = link_rot[:, link_indices, :]  # (batch_size, num_links, 4)
    # obtain the reference link rotation from the command
    ref_link_rot = command_term.command[command_term.ALL_INDICES, command_term.aiming_frame_idx]
    if command_term.cfg.rotation_mode == "euler":
        # convert the reference link rotation from euler angles to quaternion
        ref_link_rot = math_utils.quat_from_euler_xyz(
            ref_link_rot[..., 0],  # roll
            ref_link_rot[..., 1],  # pitch
            ref_link_rot[..., 2],  # yaw
        )
    elif command_term.cfg.rotation_mode == "axis_angle":
        # convert the reference link rotation from axis-angle to quaternion
        ref_link_rot = math_utils.quat_from_angle_axis(
            torch.norm(ref_link_rot, dim=-1),  # angle
            ref_link_rot,
        )
    elif command_term.cfg.rotation_mode == "tannorm":
        ref_link_rot = instinct_math_utils.tan_norm_to_quat(ref_link_rot)
    elif command_term.cfg.rotation_mode == "quaternion":
        pass
    else:
        raise ValueError(f"Unsupported rotation mode: {command_term.cfg.rotation_mode}")

    # compute the difference in quaternion representation
    link_diff = math_utils.quat_error_magnitude(link_rot, ref_link_rot)  # (batch_size, num_links)

    return link_diff  # (batch_size, num_links)


def matching_command_timing(
    buffer: torch.Tensor,
    shadowing_command: ShadowingCommandBase,
    check_at_keyframe_threshold: float,
) -> torch.Tensor:
    """Multiple the buffer by keyframe timing and (maybe) frame interval."""
    if check_at_keyframe_threshold >= 0:
        # return nonzero reward when the time_to_aiming_frame is less than the threshold
        buffer = buffer * (shadowing_command.time_to_aiming_frame <= check_at_keyframe_threshold).float()

    # maskout the reward when the reference if all frames are done/invalid
    buffer = buffer * (shadowing_command.aiming_frame_idx >= 0)

    return buffer

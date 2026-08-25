from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

import instinctlab.envs.mdp.commands.utils as command_utils

if TYPE_CHECKING:
    from instinctlab.envs.mdp import (
        JointPosRefCommand,
        JointVelRefCommand,
        LinkPosRefCommand,
        LinkRotRefCommand,
        PositionRefCommand,
        RotationRefCommand,
    )


def track_joint_pos_shadowing_cmd_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std=0.5,
    masked: bool = True,
    combine_method: Literal["prod", "sum"] = "prod",
):
    """Reward based on the joint position difference from the command.
    Args:
        env: the environment
        asset_cfg: the configuration of the asset
        command_name: the name of the command to be used, e.g., "joint_pos_ref_command"
        masked: whether to use the reference data's joint position mask in computing distance
    """
    # extract useful elements
    command_term: JointPosRefCommand = env.command_manager.get_term(command_name)
    if command_term is None:
        raise ValueError(f"Command '{command_name}' not found in the environment.")

    # obtain the joint position difference w.r.t the command
    joint_diff = command_utils.get_joint_pos_diff_to_cmd(
        env,
        asset_cfg=asset_cfg,
        command_term=command_term,
    )  # (batch_size, num_joints)
    # compute the reward based on the joint position difference
    joint_diff = torch.abs(joint_diff)  # (batch_size, num_joints)

    rewards = torch.exp(-torch.square(joint_diff) / (std * std))  # (batch_size, num_joints)
    rewards = (
        rewards * command_term.mask[command_term.ALL_INDICES, command_term.aiming_frame_idx] if masked else rewards
    )  # (batch_size, num_joints)

    if combine_method == "sum":
        rewards = torch.sum(rewards, dim=-1)  # (batch_size,)
    elif combine_method == "prod":
        rewards = torch.prod(rewards, dim=-1)

    return rewards


def track_joint_vel_shadowing_cmd_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std=0.5,
    masked: bool = True,
    combine_method: Literal["prod", "sum"] = "prod",
):
    """Reward based on the joint velocity difference from the command.
    Args:
        env: the environment
        asset_cfg: the configuration of the asset
        command_name: the name of the command to be used, e.g., "joint_vel_ref_command"
        masked: whether to use the reference data's joint velocity mask in computing distance
    """
    # extract useful elements
    command_term: JointVelRefCommand = env.command_manager.get_term(command_name)
    if command_term is None:
        raise ValueError(f"Command '{command_name}' not found in the environment.")

    # obtain the joint velocity difference w.r.t the command
    joint_diff = command_utils.get_joint_vel_diff_to_cmd(
        env,
        asset_cfg=asset_cfg,
        command_term=command_term,
    )  # (batch_size, num_joints)
    # compute the reward based on the joint velocity difference
    joint_diff = torch.abs(joint_diff)  # (batch_size, num_joints)

    rewards = torch.exp(-torch.square(joint_diff) / (std * std))  # (batch_size, num_joints)
    rewards = (
        rewards * command_term.mask[command_term.ALL_INDICES, command_term.aiming_frame_idx] if masked else rewards
    )  # (batch_size, num_joints)

    if combine_method == "sum":
        rewards = torch.sum(rewards, dim=-1)  # (batch_size,)
    elif combine_method == "prod":
        rewards = torch.prod(rewards, dim=-1)

    return rewards


def track_link_pos_shadowing_cmd_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std=0.3,
    masked: bool = True,
    combine_method: Literal["prod", "sum"] = "prod",
):
    """Reward based on the link position difference from the command.
    Args:
        env: the environment
        asset_cfg: the configuration of the asset
        command_name: the name of the command to be used, e.g., "link_pos_ref_command"
        masked: whether to use the reference data's link position mask in computing distance
    """
    # extract useful elements
    command_term: LinkPosRefCommand = env.command_manager.get_term(command_name)
    if command_term is None:
        raise ValueError(f"Command '{command_name}' not found in the environment.")

    # obtain the link position difference w.r.t the command
    link_diff = command_utils.get_link_pos_diff_to_cmd(
        env,
        asset_cfg=asset_cfg,
        command_term=command_term,
    )  # (batch_size, num_links, 3)
    # compute the reward based on the link position difference
    link_diff = torch.abs(link_diff)  # (batch_size, num_links, 3)

    rewards = torch.exp(-torch.sum(torch.square(link_diff), dim=-1) / (std * std))  # (batch_size, num_links)
    rewards = (
        rewards * command_term.mask[command_term.ALL_INDICES, command_term.aiming_frame_idx] if masked else rewards
    )  # (batch_size, num_links)

    if combine_method == "sum":
        rewards = torch.sum(rewards, dim=-1)  # (batch_size,)
    elif combine_method == "prod":
        rewards = torch.prod(rewards, dim=-1)

    return rewards


def track_link_rot_shadowing_cmd_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std=0.3,
    masked: bool = True,
    combine_method: Literal["prod", "sum"] = "prod",
):
    """Reward based on the link rotation difference from the command.
    Args:
        env: the environment
        asset_cfg: the configuration of the asset
        command_name: the name of the command to be used, e.g., "link_rot_ref_command"
        masked: whether to use the reference data's link rotation mask in computing distance
    """
    # extract useful elements
    command_term: LinkRotRefCommand = env.command_manager.get_term(command_name)
    if command_term is None:
        raise ValueError(f"Command '{command_name}' not found in the environment.")

    # obtain the link rotation difference w.r.t the command
    link_diff = command_utils.get_link_rot_diff_mag_to_cmd(
        env,
        asset_cfg=asset_cfg,
        command_term=command_term,
    )  # (batch_size, num_links)

    rewards = torch.exp(-torch.square(link_diff) / (std * std))  # (batch_size, num_links)
    rewards = (
        rewards * command_term.mask[command_term.ALL_INDICES, command_term.aiming_frame_idx] if masked else rewards
    )  # (batch_size, num_links)

    if combine_method == "sum":
        rewards = torch.sum(rewards, dim=-1)  # (batch_size,)
    elif combine_method == "prod":
        rewards = torch.prod(rewards, dim=-1)

    return rewards

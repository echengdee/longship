from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import base_ang_vel, joint_pos_rel, joint_vel, last_action, projected_gravity
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def expanded_base_ang_vel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    num_frames: int = 1,
) -> torch.Tensor:
    """The angular velocity of the base, but expanded in a new dimension (to meet the need of sequential models).
    Returns:
        (num_envs, num_frames, 3)
    """
    ang_vel = base_ang_vel(env, asset_cfg)
    ang_vel = ang_vel.unsqueeze(1).expand(-1, num_frames, -1)
    return ang_vel


def expanded_projected_gravity(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    num_frames: int = 1,
) -> torch.Tensor:
    """The projected gravity, but expanded in a new dimension (to meet the need of sequential models).
    Returns:
        (num_envs, num_frames, 3)
    """
    gravity = projected_gravity(env, asset_cfg)
    gravity = gravity.unsqueeze(1).expand(-1, num_frames, -1)
    return gravity


def expanded_joint_pos_rel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    num_frames: int = 1,
) -> torch.Tensor:
    """The relative joint positions, but expanded in a new dimension (to meet the need of sequential models).
    Returns:
        (num_envs, num_frames, num_joints)
    """
    joint_pos_ = joint_pos_rel(env, asset_cfg)
    joint_pos_ = joint_pos_.unsqueeze(1).expand(-1, num_frames, -1)
    return joint_pos_


def expanded_joint_vel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    num_frames: int = 1,
) -> torch.Tensor:
    """The joint velocities, but expanded in a new dimension (to meet the need of sequential models).
    Returns:
        (num_envs, num_frames, num_joints)
    """
    vel = joint_vel(env, asset_cfg)
    vel = vel.unsqueeze(1).expand(-1, num_frames, -1)
    return vel


def expanded_last_action(
    env: ManagerBasedEnv,
    action_name: str | None = None,
    num_frames: int = 1,
) -> torch.Tensor:
    """The last action taken, but expanded in a new dimension (to meet the need of sequential models).
    Returns:
        (num_envs, num_frames, num_actions)
    """
    action = last_action(env, action_name)
    action = action.unsqueeze(1).expand(-1, num_frames, -1)
    return action

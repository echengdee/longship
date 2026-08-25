from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Literal

import pytorch_kinematics as pk

import isaaclab.utils.math as math_utils

import instinctlab.utils.math as intinct_math

from .retarget_smpl_to_joint import retarget_smpl_to_g1_29dof_joints

SMPL_PARENTS_PELVIS_B = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]
SMPL_ROOT_PELVIS = 0

SMPL_PARENTS_TORSO_B = [
    3,
    0,
    0,
    6,
    1,
    2,
    -1,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]
SMPL_ROOT_TORSO = 6


class HumanoidSmplRotationalIK:
    def __init__(
        self,
        robot_chain: str | pk.Chain,
        joint_names: list[str] | None = None,
        smpl_root_in_robot_link_name: str = "pelvis",
        retargetting_func_core: callable = retarget_smpl_to_g1_29dof_joints,
        hip_waist_mixing: float = 0.9,
        translation_scaling=0.75,
        translation_height_offset=0.02,
        device="cpu",
    ):
        """A wrapper to compute inverse kinematics for humanoid robots, callable. Returning the joint angles (in the
        input `joint_names` order and link poses in the order of `link_mapping`) to reach the target poses in SMPL.
        ## Args:
        - robot_chain_path: The kinematic chain of the humanoid robot.
        - link_mapping: The mapping between the link names of the robot and the indices of the smpl poses.
        - joint_names: The joint names of the robot. The returned joint angles will be in this order.
        - starting_joint_pose: The starting joint pose of the robot. If None, the zero joint positions will be used.
        -
        """
        self.device = torch.device(device)
        self.robot_chain = (
            pk.build_chain_from_urdf(open(robot_chain, mode="rb").read()).to(device=self.device)
            if isinstance(robot_chain, str)
            else robot_chain
        )
        self.joint_names: list[str] = (
            joint_names if joint_names is not None else self.robot_chain.get_joint_parameter_names()
        )
        self.smpl_root_in_robot_link_name = smpl_root_in_robot_link_name
        self.retargetting_func_core = retargetting_func_core  # a temporary manual retargetting function
        self.hip_waist_mixing = hip_waist_mixing  # a parameter for temporary manual retargetting function
        self.translation_scaling = translation_scaling  # a parameter for scaling the translation
        self.translation_height_offset = translation_height_offset  # a parameter for offsetting the translation

        # build a serial chain from the robot root to the smpl_root_in_robot_link
        self._root_translation_chain = pk.SerialChain(
            self.robot_chain,
            end_frame_name=smpl_root_in_robot_link_name,
        ).to(device=self.device)

        # initialize joint translation from retargetted to smpl_transformation_chain joints
        _retargetted_joints_to_root_transformation_joints_ids = []
        for j_name in self._root_translation_chain.get_joint_parameter_names():
            _retargetted_joints_to_root_transformation_joints_ids.append(
                self.retargetting_func_core.joint_names.index(j_name)
            )
        self._retargetted_joints_to_root_transformation_joints_ids = torch.as_tensor(
            _retargetted_joints_to_root_transformation_joints_ids,
            dtype=torch.long,
        ).to(self.device)

        # initialize joint translation from retargetted to output joints
        _retargetted_joints_to_output_joints_ids = []
        for j_name in self.joint_names:
            _retargetted_joints_to_output_joints_ids.append(self.retargetting_func_core.joint_names.index(j_name))
        self._retargetted_joints_to_output_joints_ids = torch.as_tensor(
            _retargetted_joints_to_output_joints_ids,
            dtype=torch.long,
        ).to(self.device)

        # build axis swap quaterion for smpl coordinate system
        swap_rot_axis_angle = torch.ones((1, 3), device=self.device) * np.pi * -2 / 3 / np.sqrt(3)  # shape [1, 3]
        self.swap_rot_quat = math_utils.quat_from_angle_axis(
            torch.norm(swap_rot_axis_angle, dim=-1), swap_rot_axis_angle
        )  # shape [1, 4]

    def __call__(self, smpl_poses: torch.Tensor, smpl_root_trans: torch.Tensor, **kwargs):
        """smpl_poses -> retargetted_joint_poses, smpl_root_quat -> output_joint_poses, robot_root_quat
        ## Args:
        - smpl_poses: The target poses in SMPL format, shape [N, 24, 3]. In SMPL format (+z forward, +y up).
        - root_trans: The root translation of the SMPL poses, shape [N, 3]. In SMPL format (x-y as the ground plane).
        ## Returns:
        - joint_positions: The joint positions of the robot, shape [N, len(joint_names)], if joint_names provided.
            Otherwise, all joints in the order of robot_chain.
        - robot_root_poses: The root poses of the robot_chain, shape [N, 7].
            (x, y, z, qw, qx, qy, qz), (+x forward, +z up).
        """

        retargetted_joints, smpl_root_quat = self.retargetting_func_core(
            smpl_poses.contiguous(),
            hip_waist_mixing=self.hip_waist_mixing,
        )  # shape [N, 29], [N, 4]

        # update smpl_root_quat in isaacsim coordinate system
        smpl_root_quat = math_utils.quat_mul(
            smpl_root_quat,
            self.swap_rot_quat.expand(retargetted_joints.shape[0], -1),
        ).to(torch.float)

        # compute the robot root pose
        _root_transformation_chain_joints = retargetted_joints[
            :, self._retargetted_joints_to_root_transformation_joints_ids
        ]
        smpl_in_robot_mat = (
            self._root_translation_chain.forward_kinematics(
                _root_transformation_chain_joints,
                end_only=True,
            )
            .get_matrix()
            .reshape(-1, 4, 4)
        )
        smpl_in_robot_pos_b = smpl_in_robot_mat[:, :3, 3]
        smpl_in_robot_quat_b = math_utils.quat_from_matrix(smpl_in_robot_mat[:, :3, :3])
        robot_root_trans, robot_root_quat = math_utils.combine_frame_transforms(
            smpl_root_trans,
            smpl_root_quat,
            *math_utils.subtract_frame_transforms(
                smpl_in_robot_pos_b,
                smpl_in_robot_quat_b,
            ),
        )
        root_poses = torch.cat([robot_root_trans, robot_root_quat], dim=-1)
        root_poses[:, :3] *= self.translation_scaling
        root_poses[:, 2] += self.translation_height_offset

        # translate retargetted_joints to output joints
        joint_positions = retargetted_joints[:, self._retargetted_joints_to_output_joints_ids]

        return joint_positions, root_poses

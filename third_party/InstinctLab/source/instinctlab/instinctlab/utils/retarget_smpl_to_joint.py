""" A module of various retargetting implementations
Each of them retargets smpl poses to robot's base pose and joint positions
"""

import numpy as np
import torch

import isaaclab.utils.math as math_utils

import instinctlab.utils.math as instinct_math

"""
refer to https://khanhha.github.io/posts/SMPL-model-introduction/ for SMPL joint definition.
refer to https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py for SMPL joint names.
"""


def retarget_smpl_to_h1_joints(
    smpl_poses: torch.Tensor,
    mixing_hip_waist: bool = True,  # when torso has no pitch/roll motion, mixing hip and waist joint angles
):
    """Retargetting SMPL motion to H1 motion, can select joint position or link orientation.
    ------
    Args:
    - smpl_poses: np.ndarray, shape (N, 24, 3), each vector is a joint axis-angle w.r.t its parent joint.
    - return_link_orientations: bool, whether to return link orientation in base link
    """
    raise NotImplementedError("This function is not checked with IsaacSim joint order yet.")
    smpl_quats = math_utils.quat_from_angle_axis(
        torch.norm(smpl_poses.view(-1, 3), dim=-1),
        smpl_poses.view(-1, 3),
    ).view(-1, 24, 4)

    z_plus_dot43 = math_utils.quat_from_euler_xyz(
        torch.zeros_like(smpl_poses[:, 0, 0]),
        torch.zeros_like(smpl_poses[:, 0, 0]),
        torch.ones_like(smpl_poses[:, 0, 0]) * 0.43,
    )
    z_plus_half_pi = math_utils.quat_from_euler_xyz(
        torch.zeros_like(smpl_poses[:, 0, 0]),
        torch.zeros_like(smpl_poses[:, 0, 0]),
        torch.ones_like(smpl_poses[:, 0, 0]) * np.pi / 2,
    )

    # get rotation sequence on waist joint
    pelvis_to_torso_q = math_utils.quat_conjugate(
        math_utils.quat_mul(
            smpl_quats[:, 3],
            smpl_quats[:, 6],
        )
    )  # (N, 4)
    hip_pitch_offset, torso_joint, _ = math_utils.euler_xyz_from_quat(pelvis_to_torso_q)  # (N,)
    if not mixing_hip_waist:
        hip_pitch_offset[:] = 0

    # get rotation sequence on left hip joints
    left_hip_q = smpl_quats[:, 1]
    left_hip_mat = math_utils.matrix_from_quat(left_hip_q)
    left_hip_x, left_hip_z, left_hip_y = instinct_math.rotmat_to_euler_xzy(left_hip_mat)
    left_hip_yaw = left_hip_y
    left_hip_roll = left_hip_z
    left_hip_pitch = left_hip_x + hip_pitch_offset
    left_knee_joint = math_utils.euler_xyz_from_quat(smpl_quats[:, 4])[0]
    left_ankle_joint = math_utils.euler_xyz_from_quat(smpl_quats[:, 7])[0]

    # get rotation sequence on right hip joints
    right_hip_q = smpl_quats[:, 2]
    right_hip_mat = math_utils.matrix_from_quat(right_hip_q)
    right_hip_x, right_hip_z, right_hip_y = instinct_math.rotmat_to_euler_xzy(right_hip_mat)
    right_hip_yaw = right_hip_y
    right_hip_roll = right_hip_z
    right_hip_pitch = right_hip_x + hip_pitch_offset
    right_knee_joint = math_utils.euler_xyz_from_quat(smpl_quats[:, 5])[0]
    right_ankle_joint = math_utils.euler_xyz_from_quat(smpl_quats[:, 8])[0]

    # get rotation sequence on left arm joints
    r_13_16 = math_utils.quat_mul(smpl_quats[:, 13], smpl_quats[:, 16])
    left_shoulder_q = math_utils.quat_mul(
        math_utils.quat_conjugate(z_plus_dot43),
        math_utils.quat_mul(
            r_13_16,
            z_plus_half_pi,
        ),
    )
    left_shoulder_mat = math_utils.matrix_from_quat(left_shoulder_q)
    # get the rotation parameters in y(q0)x(q1)z(q2) sequence
    left_shoulder_y, left_shoulder_z, left_shoulder_x = instinct_math.rotmat_to_euler_yzx(left_shoulder_mat)
    left_shoulder_pitch = left_shoulder_x
    left_shoulder_roll = left_shoulder_z + 0.43633
    left_shoulder_yaw = left_shoulder_y
    left_elbow_joint = math_utils.euler_xyz_from_quat(smpl_quats[:, 18])[1] + np.pi / 2

    # get rotation sequence on right arm joints
    r_14_17 = math_utils.quat_mul(smpl_quats[:, 14], smpl_quats[:, 17])
    right_shoulder_q = math_utils.quat_mul(
        z_plus_dot43,
        math_utils.quat_mul(
            r_14_17,
            math_utils.quat_conjugate(z_plus_half_pi),
        ),
    )
    right_shoulder_mat = math_utils.matrix_from_quat(right_shoulder_q)
    # get the rotation parameters in y(q0)x(q1)z(q2) sequence
    right_shoulder_y, right_shoulder_z, right_shoulder_x = instinct_math.rotmat_to_euler_yzx(right_shoulder_mat)
    right_shoulder_pitch = right_shoulder_x
    right_shoulder_roll = right_shoulder_z - 0.43633
    right_shoulder_yaw = right_shoulder_y
    right_elbow_joint = -math_utils.euler_xyz_from_quat(smpl_quats[:, 19])[1] + np.pi / 2

    # remapping joint angles to (-pi, pi)
    pelvis_to_torso_yaw = math_utils.wrap_to_pi(torso_joint)
    left_hip_yaw = math_utils.wrap_to_pi(left_hip_yaw)
    left_hip_roll = math_utils.wrap_to_pi(left_hip_roll)
    left_hip_pitch = math_utils.wrap_to_pi(left_hip_pitch)
    left_knee_joint = math_utils.wrap_to_pi(left_knee_joint)
    left_ankle_joint = math_utils.wrap_to_pi(left_ankle_joint)
    right_hip_yaw = math_utils.wrap_to_pi(right_hip_yaw)
    right_hip_roll = math_utils.wrap_to_pi(right_hip_roll)
    right_hip_pitch = math_utils.wrap_to_pi(right_hip_pitch)
    right_knee_joint = math_utils.wrap_to_pi(right_knee_joint)
    right_ankle_joint = math_utils.wrap_to_pi(right_ankle_joint)
    left_shoulder_pitch = math_utils.wrap_to_pi(left_shoulder_pitch)
    left_shoulder_roll = math_utils.wrap_to_pi(left_shoulder_roll)
    left_shoulder_yaw = math_utils.wrap_to_pi(left_shoulder_yaw)
    left_elbow_joint = math_utils.wrap_to_pi(left_elbow_joint)
    right_shoulder_pitch = math_utils.wrap_to_pi(right_shoulder_pitch)
    right_shoulder_roll = math_utils.wrap_to_pi(right_shoulder_roll)
    right_shoulder_yaw = math_utils.wrap_to_pi(right_shoulder_yaw)
    right_elbow_joint = math_utils.wrap_to_pi(right_elbow_joint)

    # assemble joints angle sequence
    robot_joints = torch.zeros((smpl_poses.shape[0], 19))
    robot_joints[:, 8] = pelvis_to_torso_yaw
    robot_joints[:, 9] = left_hip_yaw
    robot_joints[:, 10] = left_hip_roll
    robot_joints[:, 11] = left_hip_pitch
    robot_joints[:, 12] = left_knee_joint
    robot_joints[:, 13] = left_ankle_joint
    robot_joints[:, 14] = right_hip_yaw
    robot_joints[:, 15] = right_hip_roll
    robot_joints[:, 16] = right_hip_pitch
    robot_joints[:, 17] = right_knee_joint
    robot_joints[:, 18] = right_ankle_joint
    robot_joints[:, 0] = left_shoulder_pitch
    robot_joints[:, 1] = left_shoulder_roll
    robot_joints[:, 2] = left_shoulder_yaw
    robot_joints[:, 3] = left_elbow_joint
    robot_joints[:, 4] = right_shoulder_pitch
    robot_joints[:, 5] = right_shoulder_roll
    robot_joints[:, 6] = right_shoulder_yaw
    robot_joints[:, 7] = right_elbow_joint

    return robot_joints


def retarget_smpl_to_g1_joints(
    smpl_poses: torch.Tensor,
    hip_waist_mixing: float = 1.0,  # when torso has no pitch/roll motion, mixing hip and waist joint angles
) -> tuple[torch.Tensor, torch.Tensor]:
    smpl_quats = math_utils.quat_from_angle_axis(
        torch.norm(smpl_poses.view(-1, 3), dim=-1),
        smpl_poses.view(-1, 3),
    ).view(-1, 24, 4)

    z_plus_dot27 = math_utils.quat_from_euler_xyz(
        torch.zeros_like(smpl_poses[:, 0, 0]),
        torch.zeros_like(smpl_poses[:, 0, 0]),
        torch.ones_like(smpl_poses[:, 0, 0]) * 0.27925,
    )
    z_plus_half_pi = math_utils.quat_from_euler_xyz(
        torch.zeros_like(smpl_poses[:, 0, 0]),
        torch.zeros_like(smpl_poses[:, 0, 0]),
        torch.ones_like(smpl_poses[:, 0, 0]) * np.pi / 2,
    )

    # get rotation sequence on waist joint
    pelvis_to_torso_q = math_utils.quat_conjugate(
        math_utils.quat_mul(
            smpl_quats[:, 3],
            math_utils.quat_mul(
                smpl_quats[:, 6],
                smpl_quats[:, 9],
            ),
        )
    )  # (N, 4)
    hip_pitch_offset, torso_joint, _ = math_utils.euler_xyz_from_quat(pelvis_to_torso_q)  # (N,)
    hip_pitch_offset = math_utils.wrap_to_pi(hip_pitch_offset)
    hip_pitch_offset[:] *= hip_waist_mixing

    # depending on the hip_waist_mixing, pelvis rotation will be updated
    smpl0_pelvis_q = smpl_quats[:, 0]
    pelvis_quat = math_utils.quat_mul(
        smpl0_pelvis_q,
        math_utils.quat_from_euler_xyz(
            -torch.ones_like(smpl_poses[:, 0, 0]) * hip_pitch_offset,
            torch.zeros_like(smpl_poses[:, 0, 0]),
            torch.zeros_like(smpl_poses[:, 0, 0]),
        ),
    )

    # get rotation sequence on left hip joints
    left_hip_q = smpl_quats[:, 1]
    left_hip_mat = math_utils.matrix_from_quat(left_hip_q)
    left_hip_y, left_hip_z, left_hip_x = instinct_math.rotmat_to_euler_yzx(left_hip_mat)
    left_hip_pitch_joint = left_hip_x + hip_pitch_offset
    left_hip_roll_joint = left_hip_z
    left_hip_yaw_joint = left_hip_y
    left_knee_joint = math_utils.euler_xyz_from_quat(smpl_quats[:, 4])[0]
    _, left_ankle_roll_joint, left_ankle_pitch_joint = instinct_math.rotmat_to_euler_yzx(
        math_utils.matrix_from_quat(smpl_quats[:, 7])
    )

    # get rotation sequence on right hip joints
    right_hip_q = smpl_quats[:, 2]
    right_hip_mat = math_utils.matrix_from_quat(right_hip_q)
    right_hip_y, right_hip_z, right_hip_x = instinct_math.rotmat_to_euler_yzx(right_hip_mat)
    right_hip_pitch_joint = right_hip_x + hip_pitch_offset
    right_hip_roll_joint = right_hip_z
    right_hip_yaw_joint = right_hip_y
    right_knee_joint = math_utils.euler_xyz_from_quat(smpl_quats[:, 5])[0]
    _, right_ankle_roll_joint, right_ankle_pitch_joint = instinct_math.rotmat_to_euler_yzx(
        math_utils.matrix_from_quat(smpl_quats[:, 8])
    )

    # get rotation sequence on left arm joints
    r_13_16 = math_utils.quat_mul(smpl_quats[:, 13], smpl_quats[:, 16])
    left_shoulder_q = math_utils.quat_mul(
        math_utils.quat_conjugate(z_plus_dot27),
        math_utils.quat_mul(
            r_13_16,
            z_plus_half_pi,
        ),
    )
    left_shoulder_mat = math_utils.matrix_from_quat(left_shoulder_q)
    # get the rotation parameters in y(q0)x(q1)z(q2) sequence
    left_shoulder_y, left_shoulder_z, left_shoulder_x = instinct_math.rotmat_to_euler_yzx(left_shoulder_mat)
    left_shoulder_pitch = left_shoulder_x
    left_shoulder_roll = left_shoulder_z + 0.27925
    left_shoulder_yaw = left_shoulder_y
    left_elbow_pitch_joint = math_utils.euler_xyz_from_quat(smpl_quats[:, 18])[1] + np.pi / 2
    left_elbow_roll_joint = math_utils.euler_xyz_from_quat(smpl_quats[:, 20])[0]

    # get rotation sequence on right arm joints
    r_14_17 = math_utils.quat_mul(smpl_quats[:, 14], smpl_quats[:, 17])
    right_shoulder_q = math_utils.quat_mul(
        z_plus_dot27,
        math_utils.quat_mul(
            r_14_17,
            math_utils.quat_conjugate(z_plus_half_pi),
        ),
    )
    right_shoulder_mat = math_utils.matrix_from_quat(right_shoulder_q)
    # get the rotation parameters in y(q0)x(q1)z(q2) sequence
    right_shoulder_y, right_shoulder_z, right_shoulder_x = instinct_math.rotmat_to_euler_yzx(right_shoulder_mat)
    right_shoulder_pitch = right_shoulder_x
    right_shoulder_roll = right_shoulder_z - 0.27925
    right_shoulder_yaw = right_shoulder_y
    right_elbow_pitch_joint = -math_utils.euler_xyz_from_quat(smpl_quats[:, 19])[1] + np.pi / 2
    right_elbow_roll_joint = -math_utils.euler_xyz_from_quat(smpl_quats[:, 21])[0]

    # remapping joint angles to (-pi, pi)
    torso = math_utils.wrap_to_pi(torso_joint)
    left_hip_pitch = math_utils.wrap_to_pi(left_hip_pitch_joint)
    left_hip_roll = math_utils.wrap_to_pi(left_hip_roll_joint)
    left_hip_yaw = math_utils.wrap_to_pi(left_hip_yaw_joint)
    left_knee = math_utils.wrap_to_pi(left_knee_joint)
    left_ankle_pitch = math_utils.wrap_to_pi(left_ankle_pitch_joint)
    left_ankle_roll = math_utils.wrap_to_pi(left_ankle_roll_joint)
    right_hip_pitch = math_utils.wrap_to_pi(right_hip_pitch_joint)
    right_hip_roll = math_utils.wrap_to_pi(right_hip_roll_joint)
    right_hip_yaw = math_utils.wrap_to_pi(right_hip_yaw_joint)
    right_knee = math_utils.wrap_to_pi(right_knee_joint)
    right_ankle_pitch = math_utils.wrap_to_pi(right_ankle_pitch_joint)
    right_ankle_roll = math_utils.wrap_to_pi(right_ankle_roll_joint)
    left_shoulder_pitch = math_utils.wrap_to_pi(left_shoulder_pitch)
    left_shoulder_roll = math_utils.wrap_to_pi(left_shoulder_roll)
    left_shoulder_yaw = math_utils.wrap_to_pi(left_shoulder_yaw)
    left_elbow_pitch = math_utils.wrap_to_pi(left_elbow_pitch_joint)
    left_elbow_roll = math_utils.wrap_to_pi(left_elbow_roll_joint)
    right_shoulder_pitch = math_utils.wrap_to_pi(right_shoulder_pitch)
    right_shoulder_roll = math_utils.wrap_to_pi(right_shoulder_roll)
    right_shoulder_yaw = math_utils.wrap_to_pi(right_shoulder_yaw)
    right_elbow_pitch = math_utils.wrap_to_pi(right_elbow_pitch_joint)
    right_elbow_roll = math_utils.wrap_to_pi(right_elbow_roll_joint)

    # assemble joints angle sequence
    robot_joints = torch.zeros((smpl_poses.shape[0], 23), device=smpl_poses.device)
    robot_joints[:, 0] = left_shoulder_pitch
    robot_joints[:, 1] = right_shoulder_pitch
    robot_joints[:, 2] = left_shoulder_roll
    robot_joints[:, 3] = right_shoulder_roll
    robot_joints[:, 4] = left_shoulder_yaw
    robot_joints[:, 5] = right_shoulder_yaw
    robot_joints[:, 6] = torso
    robot_joints[:, 7] = left_elbow_pitch
    robot_joints[:, 8] = right_elbow_pitch
    robot_joints[:, 9] = left_hip_pitch
    robot_joints[:, 10] = right_hip_pitch
    robot_joints[:, 11] = left_elbow_roll
    robot_joints[:, 12] = right_elbow_roll
    robot_joints[:, 13] = left_hip_roll
    robot_joints[:, 14] = right_hip_roll
    robot_joints[:, 15] = left_hip_yaw
    robot_joints[:, 16] = right_hip_yaw
    robot_joints[:, 17] = left_knee
    robot_joints[:, 18] = right_knee
    robot_joints[:, 19] = left_ankle_pitch
    robot_joints[:, 20] = right_ankle_pitch
    robot_joints[:, 21] = left_ankle_roll
    robot_joints[:, 22] = right_ankle_roll

    return robot_joints, pelvis_quat


retarget_smpl_to_g1_joints.joint_names = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "torso_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_elbow_roll_joint",
    "right_elbow_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]


def retarget_smpl_to_g1_29dof_joints(
    smpl_poses: torch.Tensor,
    hip_waist_mixing: float = 0.5,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Similar to `retarget_smpl_to_g1_joints`, but with 29 DoF.

    Args:
        hip_waist_mixing: float, considering original g1 robot does not has waist pitch/roll motion,
            this parameter is used to mix the waist pitch motion into hip pitch motion.
    """
    robot_joints_23dof, pelvis_quat = retarget_smpl_to_g1_joints(smpl_poses, hip_waist_mixing=hip_waist_mixing)
    smpl_quats = math_utils.quat_from_angle_axis(
        torch.norm(smpl_poses.view(-1, 3), dim=-1),
        smpl_poses.view(-1, 3),
    ).view(-1, 24, 4)

    # add 6 DoF torso base joints
    robot_joints_29dof = torch.zeros((robot_joints_23dof.shape[0], 29), device=smpl_poses.device)

    # prepare missing data compared with g1_23dof
    waist_pitch, waist_yaw, waist_roll = math_utils.euler_xyz_from_quat(
        math_utils.quat_mul(
            smpl_quats[:, 3],
            math_utils.quat_mul(
                smpl_quats[:, 6],
                smpl_quats[:, 9],
            ),
        )
    )
    # left wrist
    _, left_wrist_pitch, left_wrist_yaw = math_utils.euler_xyz_from_quat(smpl_quats[:, 22])
    # right wrist
    _, right_wrist_pitch, right_wrist_yaw = math_utils.euler_xyz_from_quat(smpl_quats[:, 23])

    # assign values to each corresponding joints
    # shoulder_pitch
    robot_joints_29dof[:, :2] = robot_joints_23dof[:, :2]
    # waist pitch
    robot_joints_29dof[:, 2] = -math_utils.wrap_to_pi(waist_pitch) * (1 - hip_waist_mixing)
    # shoulder_roll
    robot_joints_29dof[:, 3:5] = robot_joints_23dof[:, 2:4]
    # waist roll
    robot_joints_29dof[:, 5] = -math_utils.wrap_to_pi(waist_roll)
    # shoulder_yaw
    robot_joints_29dof[:, 6:8] = robot_joints_23dof[:, 4:6]
    # waist yaw
    robot_joints_29dof[:, 8] = -math_utils.wrap_to_pi(waist_yaw)
    # elbow
    robot_joints_29dof[:, 9:11] = robot_joints_23dof[:, 7:9]
    # hip_pitch
    robot_joints_29dof[:, 11:13] = robot_joints_23dof[:, 9:11]
    # wrist_roll
    robot_joints_29dof[:, 13:15] = robot_joints_23dof[:, 11:13]
    # hip_roll
    robot_joints_29dof[:, 15:17] = robot_joints_23dof[:, 13:15]
    # wrist_pitch
    robot_joints_29dof[:, 17] = math_utils.wrap_to_pi(left_wrist_pitch)
    robot_joints_29dof[:, 18] = -math_utils.wrap_to_pi(right_wrist_pitch)
    # hip_yaw
    robot_joints_29dof[:, 19:21] = robot_joints_23dof[:, 15:17]
    # wrist_yaw
    robot_joints_29dof[:, 21] = -math_utils.wrap_to_pi(left_wrist_yaw)
    robot_joints_29dof[:, 22] = math_utils.wrap_to_pi(right_wrist_yaw)
    # knee
    robot_joints_29dof[:, 23:25] = robot_joints_23dof[:, 17:19]
    # ankle_pitch
    robot_joints_29dof[:, 25:27] = robot_joints_23dof[:, 19:21]
    # ankle_roll
    robot_joints_29dof[:, 27:29] = robot_joints_23dof[:, 21:23]

    return robot_joints_29dof, pelvis_quat


retarget_smpl_to_g1_29dof_joints.joint_names = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "waist_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

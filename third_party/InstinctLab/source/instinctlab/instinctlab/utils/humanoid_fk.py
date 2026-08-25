from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING, Sequence, Tuple

import pytorch_kinematics as pk

import isaaclab.utils.math as math_utils


def export_forward_kinematics_as_onnx(
    pk_chain: pk.Chain,
    key_link_names: Sequence[str],
    sim_joint_names: Sequence[str],
    file_dir: str,
):
    """Export the process of forward kinematics from pytorch_kinematics as an ONNXProgram.
    This ONNXProgram will should treat the robot as a parameterized model and only the joint positions
    as inputs. The output will be the positions of the key links.

    ## ONNX program structure
    - Input: joint_pos (batch_size, num_joints) in simulation joint order
    - Output:
        - link_pos (batch_size, num_links, 3) in the order of key_link_names
        - link_tan (batch_size, num_links, 3) in the order of key_link_names
        - link_norm (batch_size, num_links, 3) in the order of key_link_names

        (Where tan/norm are the tangent and normal vectors of the link orientations,
        a.k.a the 0-th and 2-nd column vectors of the link rotation matrix)
    """

    # get the mapping from simluation joint order to pk_chain joint order
    pk_joint_names = pk_chain.get_joint_parameter_names()
    # joint_pos_pk = joint_pos_sim[_order]
    sim_to_pk_order = torch.tensor([sim_joint_names.index(joint) for joint in pk_joint_names]).to(
        device=pk_chain.device
    )
    key_link_indices = pk_chain.get_frame_indices(*key_link_names)

    # create a dummy input tensor
    joint_pos = torch.zeros(1, len(pk_joint_names), device=pk_chain.device)

    # create the torch module
    class ForwardKinematics(torch.nn.Module):
        def __init__(
            self,
            pk_chain: pk.Chain,
            key_link_names: Sequence[str],
            key_link_indices: torch.Tensor,
        ):
            super().__init__()
            self.pk_chain = pk_chain
            self.key_link_names = key_link_names
            self.key_link_indices = key_link_indices

        def forward(self, joint_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            joint_pos_pk = joint_pos[:, sim_to_pk_order]
            all_link_poses = self.pk_chain.forward_kinematics(joint_pos_pk, frame_indices=self.key_link_indices)
            links_pos = torch.stack(
                [
                    all_link_poses[link_name].get_matrix().reshape(1, 4, 4)[:, :3, 3]
                    for link_name in self.key_link_names
                ],
                dim=1,
            )  # [1, num_links, 3]
            links_rotmat = torch.stack(
                [
                    all_link_poses[link_name].get_matrix().reshape(1, 4, 4)[:, :3, :3]
                    for link_name in self.key_link_names
                ],
                dim=1,
            )  # [1, num_links, 3, 3]
            links_quat = math_utils.quat_from_matrix(links_rotmat)  # [1, num_links, 4], (w, x, y, z) order
            return links_pos, links_quat

    forward_kinematics = ForwardKinematics(pk_chain, key_link_names, key_link_indices)

    # export the program
    torch.onnx.export(
        forward_kinematics,
        joint_pos,
        os.path.join(file_dir, "forward_kinematics.onnx"),
        input_names=["joint_pos"],
        output_names=["links_pos", "links_quat"],
        dynamic_axes={
            "joint_pos": {0: "batch"},
            "links_pos": {0: "batch"},
            "links_quat": {0: "batch"},
        },
        opset_version=11,
    )

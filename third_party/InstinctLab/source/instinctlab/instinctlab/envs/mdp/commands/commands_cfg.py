from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, List, Literal

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .shadowing_command import (
    BaseHeightRefCommand,
    BaseLinVelRefCommand,
    HeadingErrorRefCommand,
    HeadingRefCommand,
    JointPosErrRefCommand,
    JointPosRefCommand,
    JointVelRefCommand,
    LinkPosErrRefCommand,
    LinkPosRefCommand,
    LinkRefCommand,
    LinkRotErrRefCommand,
    LinkRotRefCommand,
    PoseRefCommand,
    PositionRefCommand,
    ProjectedGravityRefCommand,
    RotationRefCommand,
    ShadowingCommandBase,
    TimeToTargetCommand,
)


@configclass
class ShadowingCommandBaseCfg(CommandTermCfg):
    """Command space for the MDP."""

    class_type: type = ShadowingCommandBase

    motion_reference: SceneEntityCfg = SceneEntityCfg("motion_reference")
    """ The configuration for accessing motion reference """

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    """ The configuration for accessing the robot asset and its states. """

    resampling_time_range: List[float] = [1e4, 1e5]

    current_state_command: bool = False
    """ Whether to include the current state in the command space. If True,
    the command sequence will be 1-frame longer than the motion reference.
    """


@configclass
class PoseRefCommandCfg(ShadowingCommandBaseCfg):
    """Command for base pose reference, w.r.t current robot/reference position."""

    class_type: type = PoseRefCommand

    anchor_frame: Literal["reference", "robot"] = "robot"
    """To generate local pose reference, the reference can be represented in either the robot's local frame or the
    reference motion's local frame. The reference motion's local frame is read from the motion reference.
    """

    rotation_mode: Literal["quaterion", "axis_angle", "euler", "tannorm"] = "axis_angle"
    """ n_dims:
        - if rotation_mode is "quaternion", data_dim = 3 + 4
        - if rotation_mode is "axis_angle", data_dim = 3 + 3
        - if rotation_mode is "euler", data_dim = 3 + 3
        - if rotation_mode is "tannorm", data_dim = 3 + 6, where 6 consists of tangent and normal vectors
    """

    realtime_mode: int = 0
    """ Since this reference is represented in robot's local-frame, the reference can be updated
    in real-time or not. However, the rotation can be updated in real-time using onboard IMU.
    Then, there are 3 options:
    - 0: No real-time update
    - 1: Real-time update in both position and rotation
    - -1: Real-time update in rotation only
    NOTE: Using 0, 1, -1 for backward compatibility, because int(True) == 1 and int(False) == 0
    """

    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/PoseRefCommand",
        markers={
            "pose": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.15, 0.15, 0.15),
            ),
        },
    )


@configclass
class PositionRefCommandCfg(ShadowingCommandBaseCfg):
    """Command for base position reference, w.r.t current robot/reference position."""

    class_type: type = PositionRefCommand

    anchor_frame: Literal["reference", "robot", "ref_rot_robot_pos", "ref_pos_robot_rot"] = "robot"
    """To generate local position reference, the reference can be represented in either the robot's local frame or the
    reference motion's local frame. The reference motion's local frame is read from the motion reference.
    """

    realtime_mode: bool = False
    """ Since this reference is represented in robot's local-frame, the reference can be updated
    in real-time or not.
    """

    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/PositionRefCommand",
        markers={
            "point": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
        },
    )


@configclass
class RotationRefCommandCfg(ShadowingCommandBaseCfg):
    class_type: type = RotationRefCommand

    rotation_mode: Literal["quaterion", "axis_angle", "euler", "tannorm"] = "axis_angle"
    """ n_dims:
        - if rotation_mode is "quaternion", data_dim = 4
        - if rotation_mode is "axis_angle", data_dim = 3
        - if rotation_mode is "euler", data_dim = 3
        - if rotation_mode is "tannorm", data_dim = 6, where 6 consists of tangent and normal vectors
    """

    in_base_frame: bool = True
    """ Whether to represent the rotation in the robot base frame or or in the world frame.
    """

    realtime_mode: bool = False
    """ If this reference is represented in robot's local-frame (in_base_frame == True), the reference
    can be updated in real-time or not.
    """

    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/RotationRefCommand",
        markers={
            "pose": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.15, 0.15, 0.15),
            ),
        },
    )


@configclass
class ProjectedGravityRefCommandCfg(ShadowingCommandBaseCfg):
    class_type: type = ProjectedGravityRefCommand

    visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/ProjectedGravityRefCommand",
    )


@configclass
class HeadingRefCommandCfg(ShadowingCommandBaseCfg):
    class_type: type = HeadingRefCommand

    visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/HeadingRefCommand",
    )


@configclass
class HeadingErrorRefCommandCfg(ShadowingCommandBaseCfg):
    """Command for heading error reference, w.r.t current robot/reference position."""

    class_type: type = HeadingErrorRefCommand

    visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/HeadingRefCommand",
    )

    realtime_mode: bool = True
    """ Since this reference is represented in robot's local-frame, the reference can be updated
    in real-time or not.
    """


@configclass
class BaseHeightRefCommandCfg(ShadowingCommandBaseCfg):
    class_type: type = BaseHeightRefCommand

    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/BaseHeightRefCommand",
        markers={
            "patch": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/disk.usd",
                scale=(0.4, 0.4, 0.01),
                visual_material=sim_utils.GlassMdlCfg(
                    glass_color=(0.0, 1.0, 0.0),
                    glass_ior=1.3,
                ),
            ),
        },
    )


@configclass
class BaseLinVelRefCommandCfg(ShadowingCommandBaseCfg):
    class_type: type = BaseLinVelRefCommand
    """Command for base linear velocity reference, w.r.t current robot/reference position."""


@configclass
class JointPosRefCommandCfg(ShadowingCommandBaseCfg):
    class_type: type = JointPosRefCommand
    """To get the robot's default joint positions."""


@configclass
class JointPosErrRefCommandCfg(ShadowingCommandBaseCfg):
    class_type: type = JointPosErrRefCommand

    realtime_mode: bool = False
    """ Since this reference is represented in robot's local-frame, the reference can be updated
    in real-time or not.
    """


@configclass
class JointVelRefCommandCfg(ShadowingCommandBaseCfg):
    class_type: type = JointVelRefCommand


@configclass
class LinkRefCommandCfg(ShadowingCommandBaseCfg):
    class_type: type = LinkRefCommand

    rotation_mode: Literal["quaterion", "axis_angle", "euler", "tannorm"] = "axis_angle"
    """ NOTE: link of interests are specified in MotionReferenceManagerCfg
    n_dims = (
      n_links,
      data_dim,
    )
      - n_links: the number of links to track
      - data_dim: the number of dimensions for each link
        - if rotation_mode is "quaternion", data_dim = 3 + 4
        - if rotation_mode is "axis_angle", data_dim = 3 + 3
        - if rotation_mode is "euler", data_dim = 3 + 3
        - if rotation_mode is "tannorm", data_dim = 3 + 6, where 6 consists of tangent and normal vectors
    """

    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/LinkRefCommand",
        markers={
            "point": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )


@configclass
class LinkPosRefCommandCfg(LinkRefCommandCfg):
    class_type: type = LinkPosRefCommand

    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/LinkPosRefCommand",
        markers={
            "point": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )


@configclass
class LinkRotRefCommandCfg(LinkRefCommandCfg):
    class_type: type = LinkRotRefCommand

    visualizer_cfg: VisualizationMarkersCfg = None


@configclass
class LinkPosErrRefCommandCfg(LinkRefCommandCfg):
    class_type: type = LinkPosErrRefCommand

    realtime_mode: bool = False
    """ Since this reference is represented in robot's local-frame, the reference can be updated
    in real-time or not. (But is always represented in robot's local-frame)
    """

    # in_base_frame: bool = True
    # """ Whether to compute the error in the robot base frame or in the world frame.
    # If True, the reference and state will be link_pos_b
    # If False, the reference and state will be link_pos_w
    # """
    # TODO: Currently, not implemented.

    visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/LinkPosErrRefCommand",
    )


@configclass
class LinkRotErrRefCommandCfg(LinkRefCommandCfg):
    class_type: type = LinkRotErrRefCommand

    realtime_mode: bool = False
    """ Since this reference is represented in robot's local-frame, the reference can be updated
    in real-time or not. (But is always represented in robot's local-frame)
    """

    in_base_frame: bool = True
    """ Whether to compute the error in the robot base frame or in the world frame.
    If True, the reference and state will be link_rot_b
    If False, the reference and state will be link_rot_w
    """

    visualizer_cfg: VisualizationMarkersCfg = None


@configclass
class TimeToTargetCommandCfg(ShadowingCommandBaseCfg):
    class_type: type = TimeToTargetCommand

    realtime_mode: bool = False
    """ In real-time mode, the time-to-target will always be the time currently to the target frame.
    In non-real-time mode, the time-to-target will be the time to the target frame at the time of motion reference refresh.
    """

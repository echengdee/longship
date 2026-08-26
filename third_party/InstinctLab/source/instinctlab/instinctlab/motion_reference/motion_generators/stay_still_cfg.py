from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from collections.abc import Callable

import torch

from instinctlab.motion_reference.motion_reference_cfg import MotionBufferCfg

from .stay_still import StayStillMotion


@configclass
class StayStillMotionCfg(MotionBufferCfg):
    """Configuration for the stay still motion, which generates motion that asks the robot to stay still."""

    class_type: type = StayStillMotion

    pseudo_num_trajectories: int = 1
    """ The number of trajectories in the motion buffer, which effects the ratio of assigning robot to this buffer. """

    mark_base_rest_pose_time: float | tuple[float, float] = 0.1
    """ The time to record the base rest pose. Tuple for a range of time in case of in-accuracy.
    """

    resting_pose_spawn_height_offset: float = 0.8
    """ The height of the robot when it is in the resting pose in case of penetrating the ground. """

    buffer_device: Literal["cpu", "output_device"] = "cpu"
    """ the device for the motion buffer. """

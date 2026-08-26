from isaaclab.utils import configclass

from ..grouped_ray_caster import GroupedRayCasterCameraCfg
from .noisy_camera_cfg import NoisyCameraCfgMixin
from .noisy_grouped_raycaster_camera import NoisyGroupedRayCasterCamera


@configclass
class NoisyGroupedRayCasterCameraCfg(NoisyCameraCfgMixin, GroupedRayCasterCameraCfg):
    """
    Configuration class for the NoisyGroupedRayCasterCamera sensor and manages image transforms and their parameters.
    """

    class_type: type = NoisyGroupedRayCasterCamera

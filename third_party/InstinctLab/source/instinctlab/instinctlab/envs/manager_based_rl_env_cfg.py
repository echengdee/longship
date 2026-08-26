from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from instinctlab.envs.ui import InstinctLabRLEnvWindow


@configclass
class InstinctLabRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for a reinforcement learning environment with the manager-based workflow."""

    # ui settings
    ui_window_class_type: type | None = InstinctLabRLEnvWindow
    """Inherit from :class:`isaaclab.envs.ui.manager_based_rl_env_window.ManagerBasedRLEnvWindow` class."""

    # monitor settings
    monitors: object | None = None
    """Monitor Settings

    Please refer to the :class:`instinctlab.monitors.MonitorManager` class for more details.
    """

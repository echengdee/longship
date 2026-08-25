from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.ui import ManagerBasedRLEnvWindow

if TYPE_CHECKING:
    from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv


class InstinctLabRLEnvWindow(ManagerBasedRLEnvWindow):
    """Window manager for the RL environment.

    On top of the isaaclab manager-based RL environment window, this class adds more controls for InstinctLab-specific.
    This includes visualization of the command manager.
    """

    def __init__(self, env: ManagerBasedRLEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)

        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._visualize_manager(title="Monitors", class_name="monitor_manager")

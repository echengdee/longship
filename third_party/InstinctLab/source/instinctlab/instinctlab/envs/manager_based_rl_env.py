import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.ui.widgets import ManagerLiveVisualizer

from instinctlab.managers import DummyRewardCfg, MultiRewardCfg, MultiRewardManager
from instinctlab.monitors import MonitorManager


class InstinctRlEnv(ManagerBasedRLEnv):
    """This class adds additional logging mechanism on sensors to get more
    comprehensive running statistics.
    """

    def load_managers(self):

        # check and routing the reward manager to the multi reward manager
        if isinstance(self.cfg.rewards, MultiRewardCfg):
            reward_group_cfg = self.cfg.rewards
            self.cfg.rewards = DummyRewardCfg()
        super().load_managers()
        # replace the parent class's reward manager
        if "reward_group_cfg" in locals():
            self.cfg.rewards = reward_group_cfg
            self.reward_manager = MultiRewardManager(self.cfg.rewards, self)
            print("[INFO] Multi-Reward Manager: ", self.reward_manager)

        self.monitor_manager = MonitorManager(self.cfg.monitors, self)
        print("[INFO] Monitor Manager: ", self.monitor_manager)

    def setup_manager_visualizers(self):
        super().setup_manager_visualizers()
        self.manager_visualizers["monitor_manager"] = ManagerLiveVisualizer(manager=self.monitor_manager)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        return_ = super().step(action)
        monitor_infos = self.monitor_manager.update(dt=self.step_dt)
        self.extras["step"] = self.extras.get("step", {})
        self.extras["step"].update(monitor_infos)
        return return_

    def _reset_idx(self, env_ids: Sequence[int]):
        monitor_infos = self.monitor_manager.reset(env_ids, is_episode=True)
        return_ = super()._reset_idx(env_ids)
        self.extras["log"] = self.extras.get("log", {})
        self.extras["log"].update(monitor_infos)
        return return_

    """
    Properties.
    """

    @property
    def num_rewards(self) -> int:
        return getattr(self.reward_manager, "num_rewards", 1)

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp import JointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ActionTerm, ActionTermCfg

    from . import action_cfg


class ActionOverridenMixin:
    """Override some action dimensions with the provided values constantly."""

    def __init__(self: ActionTerm, cfg: ActionTermCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)  # type: ignore
        self._override_action_ids = self._env.scene[cfg.asset_cfg.name].find_joints(cfg.asset_cfg.joint_names)[0]
        self._override_value = cfg.override_value

    def process_actions(self: ActionTerm, action: torch.Tensor):
        _raw_actions = action
        action = _raw_actions.clone()
        action[:, self._override_action_ids] = self._override_value
        super().process_actions(action)
        self._raw_actions[:] = _raw_actions


class ActionOverridenJointPositionAction(ActionOverridenMixin, JointPositionAction):
    """Delayed joint position action term that overrides some action dimensions with the provided values constantly."""

    cfg: action_cfg.ActionOverridenJointPositionActionCfg

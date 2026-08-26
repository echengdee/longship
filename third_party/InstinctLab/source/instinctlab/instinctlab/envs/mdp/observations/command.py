from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedEnv

if TYPE_CHECKING:
    from instinctlab.envs.mdp import ShadowingCommandBase


def command_mask(
    env: ManagerBasedEnv,
    command_name: str,
):
    """
    Args:
        command_name: the name of the command in the env.
    """
    command: ShadowingCommandBase = env.command_manager.get_term(command_name)
    return command.mask.to(torch.float32)

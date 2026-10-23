from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def goal_future_frames(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(N, K*68) future reference-frame features (plan D5)."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return command.goal

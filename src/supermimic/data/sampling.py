"""Preserve within-group adaptive sampling while setting replay-group mass."""
import torch


def mix_replay_mass(p: torch.Tensor, mask: torch.Tensor, fraction: float) -> torch.Tensor:
    replay = p * mask
    fresh = p * ~mask
    return fraction * replay / replay.sum() + (1.0 - fraction) * fresh / fresh.sum()

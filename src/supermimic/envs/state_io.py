"""Read / write the full simulator state (qpos | qvel) as one 71-d tensor."""

from __future__ import annotations

from supermimic.robots.dimensions import NJ, NQ, NV, STATE_DIM, FEAT_DIM, DELTA_DIM, XI_DIM, OBS_DIM, GOAL_FEAT_DIM, PROFILE, ROBOT

import torch

STATE_DIM = NQ + NV
NQ = NQ
NV = NV


class StateIO:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        # States are stored relative to each env's origin so that they are directly comparable
        # with the (origin-free) motion library and translation-invariant across envs.
        self.origins = env.scene.env_origins.clone()  # (N, 3)
        assert self.sim.data.qpos.shape[-1] == NQ and self.sim.data.qvel.shape[-1] == NV, (
            self.sim.data.qpos.shape,
            self.sim.data.qvel.shape,
        )

    def read(self) -> torch.Tensor:
        """(N, 71) copy of [qpos | qvel]."""
        # Index with [:] to get plain tensors out of mjlab's TorchArray proxies (torch.cat on the
        # proxies recurses inside their __torch_function__).
        s = torch.cat([self.sim.data.qpos[:], self.sim.data.qvel[:]], dim=-1).clone()
        s[:, 0:3] -= self.origins
        return s

    @torch.no_grad()
    def write(self, env_ids: torch.Tensor, state: torch.Tensor, forward: bool = True) -> None:
        """Overwrite qpos/qvel of ``env_ids`` and refresh derived quantities."""
        state = state.to(self.sim.data.qpos.dtype).clone()
        state[:, 0:3] += self.origins[env_ids]
        self.sim.data.qpos[env_ids] = state[:, :NQ]
        self.sim.data.qvel[env_ids] = state[:, NQ:]
        if hasattr(self.sim.data, "qacc_warmstart"):
            self.sim.data.qacc_warmstart[env_ids] = 0.0
        if forward:
            self.sim.forward()

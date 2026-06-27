"""
Twin Q-critics for TD3.  Each takes the concatenated (state, action) vector.

State is pre-encoded via Actor.encode_state() before being passed here,
so Critic only needs a flat MLP.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class TwinCritic(nn.Module):
    """Two independent Q-networks; returns both Q-values for TD3."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        K = cfg["n_quote_levels"]
        priv_dim = 1 + 4 * K + 7
        state_dim = cfg["tcsa_out_dim"] + cfg["n_horizons"] + priv_dim
        action_dim = 2 + 2 * K
        hidden = cfg["fc_hidden"]

        self.q1 = _QNet(state_dim, action_dim, hidden)
        self.q2 = _QNet(state_dim, action_dim, hidden)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action), self.q2(state, action)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

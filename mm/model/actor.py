"""
IMM Actor network: (s_m, s_s, s_p) → action (m*, δ*, φ_bid, φ_ask).

Action encoding (paper §3.2):
  m*     — desired quoted mid-price offset from p_ref (ticks), tanh-scaled
  δ*     — half-spread (ticks), softplus to ensure ≥ 0
  φ_bid  — K-dim volume ratios for bid levels, softmax → scaled by V
  φ_ask  — K-dim volume ratios for ask levels, softmax → scaled by V

Total action dim = 2 + 2K.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mm.model.tcsa import TCSA
from mm.model.sl import SLPredictor


class Actor(nn.Module):

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        K = cfg["n_quote_levels"]
        self.action_dim = 2 + 2 * K

        self.tcsa = TCSA(
            f_dim=cfg["f_dim"],
            seq_len=cfg["tcsa_seq_len"],
            hidden=cfg["tcsa_channels"],
            n_layers=cfg["tcsa_layers"],
            out_dim=cfg["tcsa_out_dim"],
        )
        self.sl = SLPredictor(
            in_dim=cfg["f_dim"],
            hidden=cfg["sl_hidden"],
            n_horizons=cfg["n_horizons"],
        )

        # Private state dim: z(1) + q_bid(K) + q_ask(K) + v_bid(K) + v_ask(K)
        #   + time_ep(1) + time_day(1) + pnl_r(1) + pnl_u(1) + fill_bid(1)
        #   + fill_ask(1) + staleness(1)
        priv_dim = 1 + 4 * K + 7

        state_dim = cfg["tcsa_out_dim"] + cfg["n_horizons"] + priv_dim
        hidden = cfg["fc_hidden"]

        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head_mid    = nn.Linear(hidden, 1)    # m*
        self.head_spread = nn.Linear(hidden, 1)    # δ*
        self.head_phi_bid = nn.Linear(hidden, K)   # φ_bid logits
        self.head_phi_ask = nn.Linear(hidden, K)   # φ_ask logits

        self._max_offset = float(cfg["max_offset_ticks"])
        self._max_spread = float(cfg["max_spread_ticks"])
        self._K = K

    def encode_state(
        self,
        x: torch.Tensor,       # [B, F, L] or [F, L] market feature window
        s_p: torch.Tensor,     # [B, priv_dim] private state
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if s_p.dim() == 1:
            s_p = s_p.unsqueeze(0)

        s_m = self.tcsa(x)                              # [B, tcsa_out_dim]
        x_flat = x[:, :, -1]                           # last tick features [B, F]
        s_s = self.sl.signal(x_flat)                   # [B, n_horizons]
        return torch.cat([s_m, s_s, s_p], dim=-1)      # [B, state_dim]

    def forward(
        self,
        x: torch.Tensor,
        s_p: torch.Tensor,
    ) -> torch.Tensor:
        state = self.encode_state(x, s_p)
        h = self.fc(state)

        m_star = torch.tanh(self.head_mid(h)) * self._max_offset    # [B, 1]
        delta  = F.softplus(self.head_spread(h))                     # [B, 1]
        delta  = torch.clamp(delta, 0.0, self._max_spread)
        phi_bid = F.softmax(self.head_phi_bid(h), dim=-1)            # [B, K]
        phi_ask = F.softmax(self.head_phi_ask(h), dim=-1)            # [B, K]

        return torch.cat([m_star, delta, phi_bid, phi_ask], dim=-1)  # [B, 2+2K]

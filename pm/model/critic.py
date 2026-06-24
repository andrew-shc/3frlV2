"""Critic network: Q(s, a) → scalar Q-value."""
from __future__ import annotations

import torch
import torch.nn as nn

from pm.model.extractor import FeatureExtractor


class Critic(nn.Module):

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.feat = FeatureExtractor(
            n_assets=cfg["n_assets"],
            m_days=cfg["m_days"],
            n_indicators=cfg["n_indicators"],
            conv_filters=cfg["conv_filters"],
            tucker_ranks=cfg["tucker_ranks"],
        )
        self.fc = nn.Sequential(
            nn.Linear(self.feat.output_dim + cfg["n_actions"], cfg["fc_hidden"]),
            nn.ReLU(),
            nn.Linear(cfg["fc_hidden"], 1),
        )

    def forward(self, v_t: torch.Tensor, cor_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([self.feat(v_t, cor_t), action], dim=-1))

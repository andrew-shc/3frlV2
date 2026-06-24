"""
Full feature extraction pipeline from raw observations to flat embedding.

  V_t [batch, 4, n, m]  +  Cor_t [batch, 4, n, n]
    → fused   [batch, 4, n, m, n]   (outer-product broadcast)
    → Conv3D  [batch, 32, n, m-2, n]
    → Tucker  [batch, R1, R2, R3, R4]
    → flatten [batch, R1·R2·R3·R4]
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pm.model.tucker import TuckerCompressor


class FeatureExtractor(nn.Module):

    def __init__(
        self,
        n_assets: int,
        m_days: int,
        n_indicators: int = 4,
        conv_filters: int = 32,
        tucker_ranks: list[int] | None = None,
    ) -> None:
        super().__init__()
        tucker_ranks = tucker_ranks or [8, 8, 8, 8]
        m_out = m_days - 2  # Conv3D kernel (1,3,1) with no padding reduces m by 2

        self.n = n_assets
        self.m = m_days
        self.k = n_indicators

        self.conv = nn.Conv3d(
            in_channels=n_indicators,
            out_channels=conv_filters,
            kernel_size=(1, 3, 1),
            padding=0,
        )
        self.act = nn.ReLU()
        self.tucker = TuckerCompressor(
            input_shape=(conv_filters, n_assets, m_out, n_assets),
            ranks=tucker_ranks,
        )
        self.output_dim: int = tucker_ranks[0] * tucker_ranks[1] * tucker_ranks[2] * tucker_ranks[3]

    def trace_shapes(self, v_t: torch.Tensor, cor_t: torch.Tensor) -> dict[str, list[int]]:
        shapes: dict[str, list[int]] = {}
        shapes["V_t"]   = list(v_t.shape)
        shapes["Cor_t"] = list(cor_t.shape)
        fused = v_t.unsqueeze(-1) * cor_t.unsqueeze(-2)
        shapes["fused F_t (outer product)"] = list(fused.shape)
        x = self.act(self.conv(fused))
        shapes["after Conv3D"] = list(x.shape)
        x = self.tucker(x)
        shapes["after Tucker"] = list(x.shape)
        shapes["flatten (FC input)"] = list(x.flatten(start_dim=1).shape)
        return shapes

    def forward(self, v_t: torch.Tensor, cor_t: torch.Tensor) -> torch.Tensor:
        fused = v_t.unsqueeze(-1) * cor_t.unsqueeze(-2)  # [batch, 4, n, m, n]
        x = self.act(self.conv(fused))                    # [batch, 32, n, m-2, n]
        x = self.tucker(x)                                # [batch, R1, R2, R3, R4]
        return x.flatten(start_dim=1)

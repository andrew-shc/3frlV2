"""
Temporal Convolution + Spatial Attention (TCSA) encoder.

Input:  x ∈ R[B, F, L]  — F features over L ticks
Output: s_m ∈ R[B, D]   — market state representation

Architecture (following IMM paper §4.1):
  1. TCN: stack of dilated causal conv1d blocks → H_hat ∈ R[B, F, L]
  2. Spatial attention over feature axis:
       S_hat = V · sigmoid(W1 · (W2^T + b))   — F×F attention matrix
       S = row-softmax(S_hat)
       H = S ⊗ H_hat + x                       — ResNet residual
  3. FC readout: s_m = sigmoid(W4 · ReLU(H · W3 + b3) + b4)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalConvBlock(nn.Module):
    """Single dilated causal convolution residual block."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self._pad = pad
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        out = self.conv1(x)[:, :, :x.size(2)]   # trim right for causality
        out = self.drop(F.relu(self.norm1(out.transpose(1, 2)).transpose(1, 2)))
        out = self.conv2(out)[:, :, :x.size(2)]
        out = self.drop(F.relu(self.norm2(out.transpose(1, 2)).transpose(1, 2)))
        return out + x


class TCN(nn.Module):
    """Stack of dilated causal conv blocks with exponentially increasing dilation."""

    def __init__(self, in_channels: int, hidden: int, n_layers: int) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_channels, hidden, 1)
        self.blocks = nn.ModuleList([
            _CausalConvBlock(hidden, dilation=2 ** i)
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, L]
        h = self.proj(x)
        for block in self.blocks:
            h = block(h)
        return h   # [B, hidden, L]


class SpatialAttention(nn.Module):
    """Feature-axis self-attention (spatial attention in the paper)."""

    def __init__(self, n_features: int, seq_len: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(seq_len, seq_len, bias=False)
        self.W2 = nn.Linear(seq_len, seq_len, bias=False)
        self.V  = nn.Linear(n_features, n_features, bias=True)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: [B, F, L]
        # S_hat[i,j] = V( sigmoid( W1(H_i) · W2(H_j)^T + b ) )
        S_hat = self.V(torch.sigmoid(
            self.W1(H).unsqueeze(2) * self.W2(H).unsqueeze(1)  # [B, F, F, L] → reduce L
        ).mean(-1))   # [B, F, F]
        S = F.softmax(S_hat, dim=-1)   # row-normalise
        # H_out = S ⊗ H_hat + skip
        return torch.bmm(S, H)         # [B, F, L]


class TCSA(nn.Module):
    """Full TCSA encoder: x → s_m."""

    def __init__(
        self,
        f_dim: int,
        seq_len: int,
        hidden: int,
        n_layers: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.tcn  = TCN(f_dim, hidden, n_layers)
        self.attn = SpatialAttention(hidden, seq_len)
        # FC readout on last time step
        self.fc = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, L]
        H_hat = self.tcn(x)              # [B, hidden, L]
        H = self.attn(H_hat) + H_hat     # ResNet skip
        h = H[:, :, -1]                  # take last time step [B, hidden]
        return self.fc(h)                # [B, out_dim]

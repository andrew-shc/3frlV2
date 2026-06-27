"""
Signal Label (SL) predictor — supervised pre-training module.

Predicts multi-horizon price direction for horizons h ∈ {20, 120, 240, 600} ticks.
Label y^h ∈ {-1, 0, +1}:
  +1  if (p_{t+h} - p_t) / p_t >  ε_h
  -1  if (p_{t+h} - p_t) / p_t < -ε_h
   0  otherwise
where ε_h = mean one-sided spread over period h.

Architecture: shallow MLP on the flattened current x (or s_m).
During RL training the parameters are frozen.

Public API
----------
  SLPredictor(cfg)              — module
  make_sl_labels(ob, horizons)  — build label array for supervised training
  train_sl(model, X, labels)    — simple supervised training loop
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 3          # {down, neutral, up} → labels {0,1,2} internally
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
HORIZONS = [20, 120, 240, 600]


class SLPredictor(nn.Module):
    """MLP: flat features → logits [B, n_horizons, N_CLASSES]."""

    def __init__(self, in_dim: int, hidden: int, n_horizons: int = 4) -> None:
        super().__init__()
        self.n_horizons = n_horizons
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden, N_CLASSES) for _ in range(n_horizons)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        h = self.encoder(x)
        logits = torch.stack([head(h) for head in self.heads], dim=1)  # [B, H, 3]
        return logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class indices [B, H] ∈ {0,1,2}."""
        return self.forward(x).argmax(dim=-1)

    @torch.no_grad()
    def signal(self, x: torch.Tensor) -> torch.Tensor:
        """Return labels mapped to {-1, 0, +1} as float [B, H]."""
        cls = self.predict(x).float()
        return cls - 1.0   # 0→-1, 1→0, 2→+1


# --------------------------------------------------------------------------
# Label construction helpers
# --------------------------------------------------------------------------

def make_sl_labels(
    mid: np.ndarray,
    spread: np.ndarray,
    horizons: list[int] = HORIZONS,
    tick_size: float = 0.01,
) -> np.ndarray:
    """
    Build integer labels {0,1,2} for each horizon.

    Args:
        mid:       mid-price array [T]
        spread:    bid-ask spread array [T] (unused, kept for API compat)
        horizons:  list of tick horizons
        tick_size: minimum price increment (threshold = 1 tick)

    Returns: int8 array [T, len(horizons)], padded with 1 (neutral) at tail.
    """
    T = len(mid)
    labels = np.ones((T, len(horizons)), dtype=np.int8)  # default: neutral

    for hi, h in enumerate(horizons):
        # Absolute price move — DO NOT use relative return (ret / mid):
        # eps is in dollars, so compare to dollar delta directly.
        delta = mid[h:] - mid[:-h]          # shape [T-h], absolute $-move
        eps = tick_size                      # 1-tick threshold (direction signal)
        up   = delta >  eps
        down = delta < -eps
        labels[:T - h, hi] = np.where(up, 2, np.where(down, 0, 1))

    return labels


# --------------------------------------------------------------------------
# Supervised training
# --------------------------------------------------------------------------

def train_sl(
    model: SLPredictor,
    X: np.ndarray,          # [T, in_dim]
    labels: np.ndarray,     # [T, n_horizons] int
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 512,
    device: str = "cpu",
) -> list[float]:
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    T = len(X)
    losses = []

    for epoch in range(epochs):
        idx = np.random.permutation(T)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, T, batch_size):
            batch_idx = idx[start:start + batch_size]
            xb = torch.as_tensor(X[batch_idx], dtype=torch.float32).to(device)
            yb = torch.as_tensor(labels[batch_idx], dtype=torch.long).to(device)

            logits = model(xb)   # [B, H, 3]
            # Cross-entropy across all horizons jointly
            loss = sum(
                F.cross_entropy(logits[:, h, :], yb[:, h])
                for h in range(model.n_horizons)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.detach().item()
            n_batches += 1

        losses.append(epoch_loss / max(n_batches, 1))

    model.eval()
    return losses

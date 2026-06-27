"""
D4PG-QR actor / critic networks and replay buffer.

Critic:
  Outputs N quantile values z_1 ≤ … ≤ z_N (sorted only during actor update).
  Trained with the asymmetric quantile-Huber loss (QR-DQN style).

Actor:
  Outputs α ∈ [0, 1] via sigmoid.
  Loss is derived from critic quantiles depending on the objective:
    mean_std  →  -(mean(z) − λ·std(z))
    var       →  z_{ceil(N·α)}                   (worst-α quantile)
    cvar      →  mean(z[:ceil(N·α)])              (expected shortfall)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


# ── Actor ─────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    def __init__(self, obs_dim: int = 5, hidden: int = 256):
        super().__init__()
        self.net = _mlp(obs_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns α ∈ [0, 1], shape (B, 1)."""
        return torch.sigmoid(self.net(obs))


# ── QR-Critic ─────────────────────────────────────────────────────────────────

class QRCritic(nn.Module):
    def __init__(self, obs_dim: int = 5, act_dim: int = 1,
                 n_quantiles: int = 32, hidden: int = 256):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.net = _mlp(obs_dim + act_dim, hidden, n_quantiles)
        # Fixed mid-quantile levels τ_i = (2i−1)/(2N)
        taus = (2 * torch.arange(1, n_quantiles + 1) - 1) / (2 * n_quantiles)
        self.register_buffer("taus", taus)

    def forward(self, obs: torch.Tensor,
                act: torch.Tensor) -> torch.Tensor:
        """Returns quantiles z, shape (B, N)."""
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


# ── Quantile-Huber (QR) loss ─────────────────────────────────────────────────

def qr_huber_loss(pred: torch.Tensor, target: torch.Tensor,
                  taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """
    Args:
        pred:   (B, N)  —  predicted quantiles from online critic
        target: (B, M)  —  target samples (Bellman-updated quantiles)
        taus:   (N,)    —  quantile levels for pred
        kappa:  Huber threshold
    Returns scalar loss.
    """
    # (B, N, 1) vs (B, 1, M)  →  (B, N, M)
    u = target.unsqueeze(1) - pred.unsqueeze(2)
    huber = torch.where(
        u.abs() <= kappa,
        0.5 * u**2,
        kappa * (u.abs() - 0.5 * kappa),
    )
    tau = taus.view(1, -1, 1)          # (1, N, 1)
    loss = (tau - (u < 0).float()).abs() * huber
    return loss.mean()


# ── Actor objective ───────────────────────────────────────────────────────────

def actor_loss(quantiles: torch.Tensor, taus: torch.Tensor,
               objective: str, risk_lambda: float, risk_alpha: float
               ) -> torch.Tensor:
    """
    Derive actor loss from critic quantiles (B, N).
    We MINIMISE the loss, so all objectives negate the "good" signal.
    """
    N = quantiles.shape[1]

    if objective == "mean_std":
        mu  = quantiles.mean(dim=1)
        std = quantiles.std(dim=1, unbiased=False)
        return -(mu - risk_lambda * std).mean()

    # Sort ascending so tail is the left end
    z_sorted, _ = torch.sort(quantiles, dim=1)   # (B, N)
    k = max(1, int(np.ceil(N * risk_alpha)))      # number of tail quantiles

    if objective == "var":
        return -z_sorted[:, k - 1].mean()

    if objective == "cvar":
        return -z_sorted[:, :k].mean()

    raise ValueError(f"Unknown objective: {objective!r}")


# ── Ornstein-Uhlenbeck noise ──────────────────────────────────────────────────

class OUNoise:
    def __init__(self, size: int, theta: float = 0.15, sigma: float = 0.2,
                 mu: float = 0.0):
        self.mu    = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(*self.state.shape)
        self.state += dx
        return self.state.copy()


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int,
                 capacity: int, device: torch.device):
        self.capacity = capacity
        self.device   = device
        self.ptr      = 0
        self.size     = 0

        self.obs     = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act     = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew     = np.zeros((capacity, 1),       dtype=np.float32)
        self.next_obs= np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done    = np.zeros((capacity, 1),       dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done):
        i = self.ptr
        self.obs     [i] = obs
        self.act     [i] = act
        self.rew     [i] = rew
        self.next_obs[i] = next_obs
        self.done    [i] = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs     [idx], device=self.device),
            torch.as_tensor(self.act     [idx], device=self.device),
            torch.as_tensor(self.rew     [idx], device=self.device),
            torch.as_tensor(self.next_obs[idx], device=self.device),
            torch.as_tensor(self.done    [idx], device=self.device),
        )

    def __len__(self):
        return self.size

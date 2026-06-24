"""
Portfolio gymnasium environment.

Observation:  flat concatenation of V_t and Cor_t
  V_t   [4, n, m]  — TA indicators per asset over last m bars
  Cor_t [4, n, n]  — rolling Pearson correlation matrix per indicator

Action:       portfolio weights [n+1] (n assets + 1 cash), sums to 1

Reward:       portfolio step-return minus transaction cost
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PortfolioDataset(gym.Env):
    """
    Args:
        indicators: dict mapping indicator name → (T, n) ndarray
                    ordered as [close, ma, rsi, macd]
        m_days:     bars in the V_t lookback window
        cor_window: bars used to compute rolling Cor_t
        cost_bps:   one-way transaction cost in basis points
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        indicators: dict[str, np.ndarray],
        m_days: int = 28,
        cor_window: int = 252,
        cost_bps: float = 5.0,
    ) -> None:
        super().__init__()

        keys = ["close", "ma", "rsi", "macd"]
        self.data = np.stack([indicators[k] for k in keys], axis=0)  # [4, T, n]
        self.T, self.n = self.data.shape[1], self.data.shape[2]
        self.k = 4
        self.m = m_days
        self.cor_win = cor_window
        self.cost_bps = cost_bps / 10_000.0
        self.n_actions = self.n + 1

        obs_dim = self.k * self.n * self.m + self.k * self.n * self.n
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_actions,), dtype=np.float32
        )
        self._t: int = 0
        self._weights: np.ndarray = np.zeros(self.n, dtype=np.float32)

    def _obs(self) -> np.ndarray:
        t = self._t
        v_t = self.data[:, t - self.m : t, :].transpose(0, 2, 1)  # [4, n, m]

        start = max(0, t - self.cor_win)
        cor_t = np.zeros((self.k, self.n, self.n), dtype=np.float32)
        for ki in range(self.k):
            window = self.data[ki, start:t, :]
            if window.shape[0] > 1:
                C = np.corrcoef(window.T)
                np.nan_to_num(C, nan=0.0, copy=False)
                np.fill_diagonal(C, 1.0)
                np.clip(C, -1.0, 1.0, out=C)
                cor_t[ki] = C
            else:
                np.fill_diagonal(cor_t[ki], 1.0)

        return np.concatenate([
            v_t.astype(np.float32).ravel(),
            cor_t.ravel(),
        ])

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._t = self.m
        self._weights = np.zeros(self.n, dtype=np.float32)
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, 0.0, None)
        total = action.sum()
        action = action / total if total >= 1e-8 else np.ones(self.n_actions) / self.n_actions
        asset_w = action[: self.n]

        turnover = np.abs(asset_w - self._weights).sum()
        cost = turnover * self.cost_bps

        prev_close = self.data[0, self._t - 1, :]
        curr_close = self.data[0, self._t, :]
        valid = prev_close != 0
        step_ret = np.zeros(self.n, dtype=np.float32)
        step_ret[valid] = curr_close[valid] / prev_close[valid] - 1.0

        reward = float((asset_w * step_ret).sum()) - cost
        self._weights = asset_w.copy()
        self._t += 1

        terminated = self._t >= self.T
        obs = self._obs() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        return obs, reward, terminated, False, {"step_return": reward}

    def obs_shapes(self) -> tuple[tuple, tuple]:
        """Return (v_t_shape, cor_t_shape) for use in network construction."""
        return (self.k, self.n, self.m), (self.k, self.n, self.n)

    @staticmethod
    def split_obs(obs: "np.ndarray | torch.Tensor", k: int, n: int, m: int):
        """Split flat obs back into (v_t, cor_t) tensors."""
        import torch
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        v_size = k * n * m
        v_t = obs[..., :v_size].reshape(-1, k, n, m)
        cor_t = obs[..., v_size:].reshape(-1, k, n, n)
        return v_t, cor_t


"""Greedy rollout and benchmark construction for backtest evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from pm.model.actor import Actor
from pm.model.env import PortfolioDataset


def run_inference(
    cfg: dict,
    ckpt_path: str,
    test_indicators: dict,
    test_dates: pd.DatetimeIndex,
    name: str = "model",
) -> tuple[pd.Series, np.ndarray]:
    """
    Greedy rollout (no exploration noise) on a test split.

    Returns:
        returns_series : pd.Series   portfolio step-returns, DatetimeIndex
        weights_array  : np.ndarray  shape (steps, n_assets)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = PortfolioDataset(
        indicators=test_indicators,
        m_days=cfg["m_days"],
        cor_window=cfg["cor_window"],
        cost_bps=cfg["cost_bps"],
    )
    k, n, m = env.k, env.n, env.m

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor = Actor(ckpt.get("cfg", cfg)).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    obs, _ = env.reset()
    step_returns: list[float]      = []
    step_dates:   list             = []
    weights_list: list[np.ndarray] = []

    while True:
        v_t, cor_t = PortfolioDataset.split_obs(obs, k, n, m)
        with torch.no_grad():
            action = actor(v_t.to(device), cor_t.to(device)).cpu().numpy().squeeze(0)

        obs, reward, terminated, truncated, _ = env.step(action)
        step_dates.append(test_dates[min(env._t - 1, len(test_dates) - 1)])
        step_returns.append(reward)
        weights_list.append(action[:n])
        if terminated or truncated:
            break

    return pd.Series(step_returns, index=step_dates, name=name), np.array(weights_list)


def build_benchmarks(
    test_indicators: dict,
    test_dates: pd.DatetimeIndex,
    model_index: pd.DatetimeIndex,
) -> dict[str, pd.Series]:
    """
    Equal-weight and buy-and-hold (first asset alphabetically)
    aligned to the model inference index.
    """
    close = test_indicators["close"]
    price_returns = np.diff(close, axis=0) / (close[:-1] + 1e-10)

    idx_map = {d: i for i, d in enumerate(test_dates)}
    pr_indices = [max(0, idx_map.get(d, 0) - 1) for d in model_index]

    return {
        "equal-weight": pd.Series(
            [float(price_returns[i].mean()) for i in pr_indices],
            index=model_index, name="equal-weight",
        ),
        "buy-hold-1st": pd.Series(
            [float(price_returns[i, 0]) for i in pr_indices],
            index=model_index, name="buy-hold-1st",
        ),
    }

"""
Market-making evaluation metrics.

Primary metrics (following IMM paper Table 1):
  EPnL   — total realized + unrealized PnL per episode
  MAP    — Mean Absolute Position (mean |inventory| over episode)
  PnLMAP — EPnL / MAP  (risk-adjusted return; higher is better)

Additional metrics:
  ASR    — Adverse Selection Ratio: fraction of filled ticks where mid-price
            subsequently moved against the fill direction
  fill_rate   — total filled volume / total desired volume
  avg_spread  — mean quoted half-spread (ticks) per episode
  inv_std     — std of inventory (measures how well agent manages position)
  sharpe_ep   — Sharpe of episodic PnL across multiple episodes
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import wandb


def episode_metrics(record: dict) -> dict[str, float]:
    """
    Compute per-episode metrics from a trajectory record dict.

    Expected keys in record:
      pnl_history   : list[float]  — cumulative PnL at each tick
      inv_history   : list[int]    — inventory at each tick
      fill_bid_hist : list[int]    — bid fills per tick
      fill_ask_hist : list[int]    — ask fills per tick
      mid_history   : list[float]  — mid-price at each tick
      desired_vol   : int          — total desired volume per side per tick
    """
    pnl = np.array(record["pnl_history"], dtype=np.float64)
    inv = np.array(record["inv_history"], dtype=np.float64)
    fill_bid = np.array(record.get("fill_bid_hist", []), dtype=np.float64)
    fill_ask = np.array(record.get("fill_ask_hist", []), dtype=np.float64)
    mid = np.array(record.get("mid_history", []), dtype=np.float64)
    desired = float(record.get("desired_vol", 1))

    epnl = float(pnl[-1] - pnl[0]) if len(pnl) > 0 else 0.0
    map_ = float(np.mean(np.abs(inv))) if len(inv) > 0 else 1.0
    pnl_map = epnl / (map_ + 1e-8)

    total_fills = float(fill_bid.sum() + fill_ask.sum())
    total_ticks = max(len(fill_bid), 1)
    total_desired = desired * total_ticks * 2
    fill_rate = total_fills / (total_desired + 1e-8)

    # Adverse selection ratio: fill at tick t is adversarial if mid
    # moves against us in the next tick
    asr = _adverse_selection_ratio(fill_bid, fill_ask, mid)

    return {
        "epnl":       epnl,
        "map":        map_,
        "pnl_map":    pnl_map,
        "asr":        asr,
        "fill_rate":  fill_rate,
        "inv_std":    float(np.std(inv)) if len(inv) > 0 else 0.0,
        "ep_len":     len(pnl),
    }


def _adverse_selection_ratio(
    fill_bid: np.ndarray,
    fill_ask: np.ndarray,
    mid: np.ndarray,
) -> float:
    if len(mid) < 2:
        return float("nan")
    mid_delta = np.diff(mid)      # [T-1]
    n_fills = len(fill_bid) - 1
    if n_fills <= 0:
        return float("nan")

    adverse = 0
    total = 0
    for t in range(min(n_fills, len(mid_delta))):
        if fill_bid[t] > 0:      # we bought — adverse if mid fell
            total += 1
            if mid_delta[t] < 0:
                adverse += 1
        if fill_ask[t] > 0:      # we sold — adverse if mid rose
            total += 1
            if mid_delta[t] > 0:
                adverse += 1

    return adverse / (total + 1e-8)


def aggregate_metrics(episode_records: list[dict]) -> dict[str, float]:
    """Aggregate per-episode dicts into mean ± std summary."""
    keys = ["epnl", "map", "pnl_map", "asr", "fill_rate", "inv_std"]
    per_ep = [episode_metrics(r) for r in episode_records]
    out: dict[str, float] = {}
    epnl_vals = [m["epnl"] for m in per_ep]
    out["sharpe_ep"] = (
        float(np.mean(epnl_vals) / (np.std(epnl_vals) + 1e-8))
        if len(epnl_vals) > 1 else 0.0
    )
    for k in keys:
        vals = [m[k] for m in per_ep if np.isfinite(m.get(k, np.nan))]
        out[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[f"{k}_std"]  = float(np.std(vals))  if vals else float("nan")
    return out


def log_metrics_table(
    results: dict[str, list[dict]],
    wandb_run,
    prefix: str = "mm/backtest",
) -> pd.DataFrame:
    """
    results: {strategy_name: [episode_record, ...]}
    Logs a W&B summary table and returns a DataFrame.
    """
    rows = []
    for name, records in results.items():
        m = aggregate_metrics(records)
        m["name"] = name
        rows.append(m)
        print(
            f"  {name:20s}  EPnL={m['epnl_mean']:+.2f}±{m['epnl_std']:.2f}"
            f"  MAP={m['map_mean']:.2f}  PnLMAP={m['pnl_map_mean']:+.4f}"
            f"  ASR={m['asr_mean']:.3f}  fill={m['fill_rate_mean']:.3f}"
            f"  sharpe={m['sharpe_ep']:+.3f}"
        )

    df = pd.DataFrame(rows).set_index("name")
    wandb_run.log({
        f"{prefix}/metrics": wandb.Table(dataframe=df.reset_index())
    })
    return df

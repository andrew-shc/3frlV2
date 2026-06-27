"""
Matplotlib plots for MM backtest — logged to W&B as images.

  plot_pnl_curves()      — cumulative PnL per episode across strategies
  plot_inventory()       — inventory trajectory for a sample episode
  plot_spread_hist()     — histogram of quoted spreads (if recorded)
  plot_fill_rates()      — fill rate bar chart per strategy
  plot_metrics_bars()    — key metric bars (EPnL, MAP, PnLMAP, ASR)
  plot_pnl_histogram()   — distribution of episodic PnLs
  run_all_plots()        — convenience wrapper
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

STRATEGY_COLORS = {
    "imm":         "#1f77b4",
    "ltiic":       "#ff7f0e",
    "foic":        "#2ca02c",
    "liic":        "#9467bd",
    "equal-weight":"#8c564b",
}

def _color(name: str) -> str:
    for k, v in STRATEGY_COLORS.items():
        if k in name.lower():
            return v
    return "#7f7f7f"


def plot_pnl_curves(
    all_records: dict[str, list[dict]],
    wandb_run,
    max_episodes: int = 5,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, records in all_records.items():
        for ep_i, rec in enumerate(records[:max_episodes]):
            pnl = np.array(rec["pnl_history"])
            label = name if ep_i == 0 else None
            ax.plot(pnl, color=_color(name), alpha=0.6, linewidth=1.0, label=label)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_title("Cumulative PnL — sample episodes")
    ax.set_xlabel("Tick")
    ax.set_ylabel("PnL ($)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    wandb_run.log({"mm/backtest/pnl_curves": wandb.Image(fig)})
    plt.close(fig)


def plot_inventory(
    all_records: dict[str, list[dict]],
    wandb_run,
    episode_idx: int = 0,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, records in all_records.items():
        if episode_idx >= len(records):
            continue
        inv = np.array(records[episode_idx]["inv_history"])
        ax.plot(inv, color=_color(name), linewidth=1.2, label=name)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_title(f"Inventory trajectory — episode {episode_idx}")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Net inventory (units)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    wandb_run.log({"mm/backtest/inventory": wandb.Image(fig)})
    plt.close(fig)


def plot_fill_rates(metrics_df: pd.DataFrame, wandb_run) -> None:
    if "fill_rate_mean" not in metrics_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = metrics_df["fill_rate_mean"]
    errs = metrics_df.get("fill_rate_std", pd.Series(0, index=vals.index))
    ax.bar(vals.index, vals.values,
           yerr=errs.values, capsize=4,
           color=[_color(n) for n in vals.index])
    ax.set_title("Fill Rate (filled / desired volume)")
    ax.set_ylabel("Fill rate")
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    wandb_run.log({"mm/backtest/fill_rates": wandb.Image(fig)})
    plt.close(fig)


def plot_metrics_bars(metrics_df: pd.DataFrame, wandb_run) -> None:
    fields = [
        ("epnl_mean",    "EPnL (mean $)"),
        ("map_mean",     "MAP (mean |inv|)"),
        ("pnl_map_mean", "PnL/MAP"),
        ("asr_mean",     "Adverse Selection Ratio"),
        ("sharpe_ep",    "Sharpe (episodic)"),
    ]
    available = [(c, t) for c, t in fields if c in metrics_df.columns]
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (col, title) in zip(axes, available):
        vals = metrics_df[col]
        ax.bar(vals.index, vals.values,
               color=[_color(nm) for nm in vals.index])
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Backtest Metrics — Strategy Comparison", fontsize=10)
    fig.tight_layout()
    wandb_run.log({"mm/backtest/metrics_bars": wandb.Image(fig)})
    plt.close(fig)


def plot_pnl_histogram(
    all_records: dict[str, list[dict]],
    wandb_run,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, records in all_records.items():
        epnls = [r["pnl_history"][-1] - r["pnl_history"][0]
                 for r in records if len(r["pnl_history"]) > 0]
        if epnls:
            ax.hist(epnls, bins=20, alpha=0.5, label=name, color=_color(name))
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title("Episodic PnL distribution")
    ax.set_xlabel("Episode PnL ($)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    wandb_run.log({"mm/backtest/pnl_histogram": wandb.Image(fig)})
    plt.close(fig)


def run_all_plots(
    all_records: dict[str, list[dict]],
    metrics_df: pd.DataFrame,
    wandb_run,
) -> None:
    plot_pnl_curves(all_records, wandb_run)
    plot_inventory(all_records, wandb_run)
    plot_fill_rates(metrics_df, wandb_run)
    plot_metrics_bars(metrics_df, wandb_run)
    plot_pnl_histogram(all_records, wandb_run)

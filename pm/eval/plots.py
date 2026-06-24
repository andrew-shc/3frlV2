"""Cumulative-return, drawdown, rolling-Sharpe, and metrics-bar plots for backtest."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from pm.eval.metrics import BARS_PER_YEAR

COLORS = {
    "trial-008":    "#1f77b4",
    "trial-011":    "#ff7f0e",
    "trial-018":    "#2ca02c",
    "equal-weight": "#9467bd",
    "buy-hold-1st": "#8c564b",
}


def cum_returns(rets: pd.Series) -> pd.Series:
    return (1 + rets).cumprod() - 1


def rolling_sharpe(rets: pd.Series, window: int = 390) -> pd.Series:
    """Rolling annualized Sharpe over `window` bars (≈ 5 trading days)."""
    mu  = rets.rolling(window).mean()
    sig = rets.rolling(window).std()
    return (mu / (sig + 1e-10)) * np.sqrt(BARS_PER_YEAR)


def drawdown_series(rets: pd.Series) -> pd.Series:
    peak = (1 + rets).cumprod().cummax()
    return ((1 + rets).cumprod() - peak) / (peak + 1e-10)


def _line_style(name: str) -> dict:
    return dict(
        color=COLORS.get(name),
        linewidth=1.5 if "trial" in name else 1.0,
        linestyle="-" if "trial" in name else "--",
    )


def plot_cumulative_returns(all_returns: dict[str, pd.Series], wandb_run) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, rets in all_returns.items():
        ax.plot(cum_returns(rets).index, cum_returns(rets).values * 100,
                label=name, **_line_style(name))
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_title("Cumulative Return — Test Set (5-min bars)")
    ax.set_ylabel("Return (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    wandb_run.log({"backtest/cumulative_returns": wandb.Image(fig)})
    plt.close(fig)


def plot_drawdown(all_returns: dict[str, pd.Series], wandb_run) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, rets in all_returns.items():
        dd = drawdown_series(rets) * 100
        ax.fill_between(dd.index, dd.values, 0, alpha=0.35, label=name, color=COLORS.get(name))
        ax.plot(dd.index, dd.values, color=COLORS.get(name), linewidth=0.8)
    ax.set_title("Drawdown — Test Set")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    wandb_run.log({"backtest/drawdown": wandb.Image(fig)})
    plt.close(fig)


def plot_rolling_sharpe(all_returns: dict[str, pd.Series], wandb_run) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, rets in all_returns.items():
        if "trial" not in name:
            continue
        rs = rolling_sharpe(rets, window=390)
        ax.plot(rs.index, rs.values, label=name, color=COLORS.get(name), linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_title("Rolling Sharpe (390-bar ≈ 5 days)")
    ax.set_ylabel("Sharpe")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    wandb_run.log({"backtest/rolling_sharpe": wandb.Image(fig)})
    plt.close(fig)


def plot_metrics_bars(metrics_df: pd.DataFrame, wandb_run) -> None:
    trial_metrics = metrics_df[metrics_df.index.str.startswith("trial")]
    if trial_metrics.empty:
        return

    fields = ["sharpe", "sortino", "calmar", "max_drawdown", "total_return", "win_rate"]
    titles = ["Sharpe", "Sortino", "Calmar", "Max Drawdown", "Total Return (%)", "Win Rate"]

    fig, axes = plt.subplots(1, len(fields), figsize=(18, 4))
    for ax, field, title in zip(axes, fields, titles):
        vals = trial_metrics[field].copy()
        if field == "total_return":
            vals = vals * 100
        elif field == "win_rate":
            vals = vals * 100
        ax.bar(vals.index, vals.values, color=[COLORS.get(n, "gray") for n in vals.index])
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", rotation=20, labelsize=7)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Backtest Metrics — Top-3 Configs", fontsize=10)
    fig.tight_layout()
    wandb_run.log({"backtest/metrics_bars": wandb.Image(fig)})
    plt.close(fig)


def plot_equity_curve(all_returns: dict[str, pd.Series], wandb_run) -> None:
    try:
        combined = pd.concat(
            [cum_returns(r) for r in all_returns.values()],
            axis=1, keys=list(all_returns.keys()),
        ).ffill().fillna(0)
        fig, ax = plt.subplots(figsize=(12, 5))
        (combined * 100).plot(ax=ax, linewidth=1.5)
        ax.set_title("Equity Curve (cumulative %, test set)")
        ax.set_ylabel("Return (%)")
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        wandb_run.log({"backtest/equity_curve": wandb.Image(fig)})
        plt.close(fig)
    except Exception as exc:
        print(f"  equity curve failed: {exc}")


def run_all_plots(
    all_returns: dict[str, pd.Series],
    metrics_df: pd.DataFrame,
    wandb_run,
) -> None:
    plot_cumulative_returns(all_returns, wandb_run)
    plot_drawdown(all_returns, wandb_run)
    plot_rolling_sharpe(all_returns, wandb_run)
    plot_metrics_bars(metrics_df, wandb_run)
    plot_equity_curve(all_returns, wandb_run)

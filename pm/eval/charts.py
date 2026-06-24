"""
Matplotlib visualization functions for TA features and correlation analysis.

All functions return a plt.Figure; callers are responsible for closing it.
"""
from __future__ import annotations

from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pm.data.ta import MA_PERIOD, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL

VARIABLES = ["close", "ma", "rsi", "macd"]
SAMPLE_SYM = "NVDA"

# 30-stock subsample for correlation heatmaps (representative cross-sector)
CORR_SUBSAMPLE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "JPM", "V",    "UNH",
    "XOM",  "LLY",  "JNJ",  "WMT",  "MA",    "CVX",  "HD",   "MRK", "ABBV", "PG",
    "COST", "BAC",  "CRM",  "NFLX", "AMD",   "ORCL", "ACN",  "TMO", "ADBE", "INTC",
]

_HI_COLORS = {"close": "#1f77b4", "ma": "#ff7f0e", "rsi": "#2ca02c", "macd": "#d62728"}


def _day_ticks(idx: pd.DatetimeIndex) -> tuple[list, list]:
    """Return (tick_timestamps, tick_labels) with one tick at each day open."""
    ticks, labels, prev = [], [], None
    for ts in idx:
        d = ts.date()
        if d != prev:
            ticks.append(ts)
            labels.append(ts.strftime("%b %-d"))
            prev = d
    return ticks, labels


def nvda_chart(features: dict[str, pd.DataFrame], symbol: str = SAMPLE_SYM) -> plt.Figure:
    """4-panel figure: Close+MA overlay, MA deviation, RSI, and MACD line."""
    idx   = features["close"].index
    close = features["close"][symbol].values
    ma    = features["ma"][symbol].values
    rsi   = features["rsi"][symbol].values
    macd  = features["macd"][symbol].values

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"{symbol} — TA features (5-min bars)", fontsize=13, fontweight="bold")

    axes[0].plot(idx, close, label="Close", color="#1f77b4", linewidth=1.2)
    axes[0].plot(idx, ma, label=f"MA({MA_PERIOD})", color="#ff7f0e", linewidth=1.2, linestyle="--")
    axes[0].set_ylabel("Price ($)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(idx, ma - close, color="#ff7f0e", linewidth=1.0)
    axes[1].axhline(0, color="grey", linewidth=0.7, linestyle="--")
    axes[1].set_ylabel(f"MA({MA_PERIOD}) − Close")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(idx, rsi, color="#2ca02c", linewidth=1.2)
    axes[2].axhline(70, color="red",   linewidth=0.7, linestyle="--", alpha=0.6)
    axes[2].axhline(30, color="green", linewidth=0.7, linestyle="--", alpha=0.6)
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel(f"RSI({RSI_PERIOD})")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(idx, macd, color="#d62728", linewidth=1.2)
    axes[3].axhline(0, color="grey", linewidth=0.7, linestyle="--")
    axes[3].set_ylabel(f"MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})")
    axes[3].grid(True, alpha=0.3)

    tick_step = max(1, len(idx) // 10)
    tick_pos = list(range(0, len(idx), tick_step))
    axes[3].set_xticks([idx[i] for i in tick_pos])
    axes[3].set_xticklabels([str(idx[i])[11:16] for i in tick_pos], rotation=30, fontsize=7)
    axes[3].set_xlabel("Time (HH:MM)")

    fig.tight_layout()
    return fig


def ti_lines_chart(features: dict[str, pd.DataFrame], n_days: int = 10) -> plt.Figure:
    """
    4-panel figure: one panel per indicator, all stocks as thin gray lines,
    NVDA highlighted. Close/MA/MACD are z-scored per stock; RSI is raw.
    """
    ylabels = {
        "close": "Close (z-scored)",
        "ma":    f"MA({MA_PERIOD}) (z-scored)",
        "rsi":   f"RSI({RSI_PERIOD})",
        "macd":  f"MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL}) (z-scored)",
    }

    idx = features["close"].index
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(
        f"S&P 500 — TA indicators, last {n_days} trading days (5-min bars)",
        fontsize=12, fontweight="bold",
    )

    for ax, v in zip(axes, VARIABLES):
        vals = features[v].values.copy().astype(float)
        if v != "rsi":
            mu = np.nanmean(vals, axis=0, keepdims=True)
            sd = np.nanstd(vals,  axis=0, keepdims=True)
            sd[sd < 1e-8] = 1.0
            vals = (vals - mu) / sd

        ax.plot(idx, vals, color="gray", alpha=0.05, linewidth=0.5, rasterized=True)

        if SAMPLE_SYM in features[v].columns:
            j = features[v].columns.get_loc(SAMPLE_SYM)
            ax.plot(idx, vals[:, j], color=_HI_COLORS[v], linewidth=1.8, label=SAMPLE_SYM, zorder=5)

        if v == "rsi":
            ax.axhline(70, color="red",   linewidth=0.8, linestyle="--", alpha=0.5)
            ax.axhline(30, color="green", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_ylim(0, 100)

        ax.set_ylabel(ylabels[v], fontsize=8)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)

    ticks, labels = _day_ticks(idx)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(labels, rotation=35, fontsize=9)
    axes[-1].set_xlabel("Trading day")
    fig.tight_layout()
    return fig


def nvda_daily_boxplot(features: dict[str, pd.DataFrame], last_dates: list) -> plt.Figure:
    """
    4-panel chart: one panel per indicator showing NVDA's intraday distribution
    per trading day as box plots (~78 five-minute bars per box).
    """
    ylabels = {
        "close": "Close ($)",
        "ma":    f"MA({MA_PERIOD}) ($)",
        "rsi":   f"RSI({RSI_PERIOD})",
        "macd":  "MACD",
    }
    n_days = len(last_dates)
    positions = list(range(1, n_days + 1))
    idx = features["close"].index

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"{SAMPLE_SYM} — intraday distribution per trading day (last {n_days} days, 5-min bars)",
        fontsize=12, fontweight="bold",
    )

    for ax, v in zip(axes, VARIABLES):
        series = features[v][SAMPLE_SYM]
        boxes = []
        for d in last_dates:
            ts_s = pd.Timestamp(d)
            mask = (idx >= ts_s) & (idx < ts_s + pd.Timedelta("1D"))
            vals = series[mask].dropna().values
            boxes.append(vals if len(vals) > 1 else np.array([np.nan]))

        ax.boxplot(
            boxes, positions=positions, widths=0.55, patch_artist=True,
            boxprops    =dict(facecolor=_HI_COLORS[v], alpha=0.35),
            medianprops =dict(color=_HI_COLORS[v],     linewidth=2.0),
            whiskerprops=dict(color=_HI_COLORS[v],     alpha=0.6),
            capprops    =dict(color=_HI_COLORS[v],     alpha=0.6),
            flierprops  =dict(marker=".", color=_HI_COLORS[v], alpha=0.3, markersize=3),
        )
        if v == "rsi":
            ax.axhline(70, color="red",   lw=0.8, ls="--", alpha=0.5)
            ax.axhline(30, color="green", lw=0.8, ls="--", alpha=0.5)
        ax.set_ylabel(ylabels[v], fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(
        [pd.Timestamp(d).strftime("%b %-d") for d in last_dates], rotation=30, fontsize=9
    )
    axes[-1].set_xlabel("Trading day")
    fig.tight_layout()
    return fig


def corr_heatmap_fig(
    features: dict[str, pd.DataFrame],
    subsample: list[str],
    day_date: date,
) -> plt.Figure:
    """2×2 heatmap grid for one trading day: one panel per indicator."""
    day_ts  = pd.Timestamp(day_date)
    day_end = day_ts + pd.Timedelta("1D")
    mask    = (features["close"].index >= day_ts) & (features["close"].index < day_end)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        f"Correlation matrices — {day_date}  (n={len(subsample)} stocks)",
        fontsize=12, fontweight="bold",
    )

    for ax, v in zip(axes.flat, VARIABLES):
        sub = features[v].loc[mask, subsample]
        sub = sub.loc[:, sub.notna().any()]
        if sub.shape[0] < 5 or sub.shape[1] < 2:
            ax.set_title(v.upper(), fontsize=10)
            ax.axis("off")
            continue

        corr  = sub.corr(method="pearson")
        ticks = range(len(corr))
        im    = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(corr.columns.tolist(), rotation=90, fontsize=5)
        ax.set_yticklabels(corr.index.tolist(),   fontsize=5)
        ax.set_title(v.upper(), fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig

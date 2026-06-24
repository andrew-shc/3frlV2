"""
W&B logging for rolling TA correlation analysis.

Two modes:
  run_hist(n_days)   — historical parquet: spaghetti/heatmap/boxplot charts
  run(window)        — today's parquet: per-bar rolling correlation

Usage:
    python -m pm.eval.ta_analysis               # last 10 trading days
    python -m pm.eval.ta_analysis --days 5
    python -m pm.eval.ta_analysis --today --window 20
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv

from pm.data.ta import compute_ta_df, MA_PERIOD, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL
from pm.eval.charts import VARIABLES, SAMPLE_SYM, CORR_SUBSAMPLE
from pm.eval.charts import nvda_chart, ti_lines_chart, nvda_daily_boxplot, corr_heatmap_fig
from pm.eval.correlation import rolling_mean_corr

load_dotenv()

HIST_PARQUET = "dataset/hist/sp500_5min_5yr.parquet"
PROJECT = os.getenv("WANDB_PROJECT", "3frlV2")


def _log_rolling_steps(roll_corr: dict, features: dict) -> None:
    all_idx = sorted(set().union(*[s.index for s in roll_corr.values()]))
    nvda = {v: features[v][SAMPLE_SYM] for v in VARIABLES}
    for i, ts in enumerate(all_idx):
        row: dict = {"bar": i}
        for v in VARIABLES:
            val = roll_corr[v].get(ts, np.nan)
            if np.isfinite(val):
                row[f"roll_corr/{v}"] = val
            nv = nvda[v].get(ts, np.nan)
            if np.isfinite(nv):
                row[f"{SAMPLE_SYM}/{v}"] = nv
        wandb.log(row, step=i)


def _log_corr_summary(roll_corr: dict) -> None:
    rows = []
    for v in VARIABLES:
        s = roll_corr[v].dropna()
        if len(s):
            rows.append([v, round(float(s.mean()), 4), round(float(s.std()), 4),
                         round(float(s.min()), 4), round(float(s.max()), 4)])
    wandb.log({"tables/corr_summary": wandb.Table(
        columns=["variable", "mean", "std", "min", "max"], data=rows,
    )})


def run(window: int = 20) -> str:
    """Today-mode: log rolling cross-asset correlation + NVDA chart."""
    stocks = pd.read_parquet("dataset/sp500_5min_today.parquet")
    close_wide = stocks.pivot(index="date", columns="symbol", values="close").sort_index()
    close_wide.index.name = "timestamp"

    if SAMPLE_SYM not in close_wide.columns:
        raise ValueError(f"{SAMPLE_SYM} not found in dataset")

    print(f"Data: {close_wide.shape[0]} bars × {close_wide.shape[1]} symbols")
    features  = compute_ta_df(close_wide)
    roll_corr = {v: rolling_mean_corr(features[v], window=window) for v in VARIABLES}

    run_obj = wandb.init(
        project=PROJECT, name=f"ta-corr-w{window}", tags=["ta", "correlation", "sp500"],
        config={"sample_symbol": SAMPLE_SYM, "ma_period": MA_PERIOD, "rsi_period": RSI_PERIOD,
                "macd_fast": MACD_FAST, "macd_slow": MACD_SLOW, "macd_signal": MACD_SIGNAL,
                "roll_window": window, "n_symbols": close_wide.shape[1], "n_bars": close_wide.shape[0]},
    )

    _log_rolling_steps(roll_corr, features)
    _log_corr_summary(roll_corr)

    nvda_df = pd.DataFrame({
        "timestamp": features["close"].index.astype(str),
        "close":  features["close"][SAMPLE_SYM].values,
        "ma_28":  features["ma"][SAMPLE_SYM].values,
        "rsi_14": features["rsi"][SAMPLE_SYM].values,
        "macd":   features["macd"][SAMPLE_SYM].values,
    })
    wandb.log({f"tables/{SAMPLE_SYM}_ta": wandb.Table(dataframe=nvda_df)})

    fig = nvda_chart(features, symbol=SAMPLE_SYM)
    wandb.log({f"charts/{SAMPLE_SYM}_ta_chart": wandb.Image(fig)})
    plt.close(fig)

    run_obj.finish()
    return run_obj.url


def run_hist(n_days: int = 10, roll_window: int = 20) -> str:
    """Historical mode: spaghetti lines, box plots, heatmaps, rolling correlation."""
    print(f"Loading last {n_days} trading days from {HIST_PARQUET} ...")
    df = pd.read_parquet(HIST_PARQUET)
    df["date"] = pd.to_datetime(df["date"])

    all_dates  = sorted(df["date"].dt.date.unique())
    last_dates = all_dates[-n_days:]
    df = df[df["date"].dt.date.isin(set(last_dates))]

    close_wide = df.pivot(index="date", columns="symbol", values="close").sort_index()
    close_wide.index.name = "timestamp"
    n_bars, n_syms = close_wide.shape
    print(f"Data: {n_bars} bars x {n_syms} symbols  ({last_dates[0]} -> {last_dates[-1]})")

    if SAMPLE_SYM not in close_wide.columns:
        raise ValueError(f"{SAMPLE_SYM} not found in dataset")

    print("Computing TA features ...")
    features  = compute_ta_df(close_wide)
    valid_sub = [s for s in CORR_SUBSAMPLE if s in close_wide.columns]
    print(f"Correlation subsample: {len(valid_sub)}/{len(CORR_SUBSAMPLE)} stocks")

    print(f"Computing rolling mean pairwise correlation (window={roll_window}) ...")
    roll_corr = {v: rolling_mean_corr(features[v], window=roll_window) for v in VARIABLES}

    run_obj = wandb.init(
        project=PROJECT, name=f"portfolio-vis-{n_days}d",
        tags=["ta", "correlation", "historical", "heatmap", "spaghetti"],
        config={"ma_period": MA_PERIOD, "rsi_period": RSI_PERIOD,
                "macd_fast": MACD_FAST, "macd_slow": MACD_SLOW, "macd_signal": MACD_SIGNAL,
                "n_days": n_days, "date_from": str(last_dates[0]), "date_to": str(last_dates[-1]),
                "n_symbols": n_syms, "n_bars": n_bars, "sample_symbol": SAMPLE_SYM,
                "roll_window": roll_window, "corr_subsample_n": len(valid_sub)},
    )

    print("Generating TI spaghetti chart ...")
    ti_fig = ti_lines_chart(features, n_days=n_days)
    wandb.log({"charts/ti_lines_all_stocks": wandb.Image(ti_fig)})
    plt.close(ti_fig)

    print(f"Generating {SAMPLE_SYM} daily box plots ...")
    bp_fig = nvda_daily_boxplot(features, last_dates)
    wandb.log({"charts/nvda_daily_boxplot": wandb.Image(bp_fig)})
    plt.close(bp_fig)

    print(f"Generating {SAMPLE_SYM} 4-panel TA chart ...")
    ta_fig = nvda_chart(features, symbol=SAMPLE_SYM)
    wandb.log({"charts/nvda_ta_chart": wandb.Image(ta_fig)})
    plt.close(ta_fig)

    print(f"Generating {len(last_dates)} daily correlation heatmaps ...")
    hm_dict: dict = {}
    for d in last_dates:
        fig = corr_heatmap_fig(features, valid_sub, d)
        hm_dict[f"charts/corr_heatmap/{d}"] = wandb.Image(fig)
        plt.close(fig)
        print(f"  {d} OK")
    wandb.log(hm_dict)

    print("Logging rolling correlation time series ...")
    _log_rolling_steps(roll_corr, features)
    _log_corr_summary(roll_corr)

    run_obj.finish()
    return run_obj.url


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",   type=int, default=10)
    parser.add_argument("--today",  action="store_true")
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    url = run(window=args.window) if args.today else run_hist(n_days=args.days)
    print(f"\nW&B run: {url}")

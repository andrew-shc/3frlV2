"""
Log S&P 500 5-minute bar data to Weights & Biases.

Creates (or reuses) the project named by WANDB_PROJECT in .env.

Usage:
    python -m pm.eval.market_log           # uses cached parquet from data/
    python -m pm.eval.market_log --fetch   # re-polls FMP before logging
"""
from __future__ import annotations

import argparse
import os
from datetime import date

import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
PROJECT  = os.getenv("WANDB_PROJECT", "3frlV2")


def _load_or_fetch(fetch: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    stocks_path = f"{DATA_DIR}/sp500_5min_today.parquet"
    index_path  = f"{DATA_DIR}/gspc_5min_today.parquet"

    if fetch or not os.path.exists(stocks_path):
        from pm.data.fmp_client import poll_sp500_5min
        stocks, index = poll_sp500_5min()
        stocks.to_parquet(stocks_path)
        index.to_parquet(index_path)
    else:
        stocks = pd.read_parquet(stocks_path)
        index  = pd.read_parquet(index_path)

    return stocks, index


def log_to_wandb(fetch: bool = False) -> str:
    stocks_df, index_df = _load_or_fetch(fetch)
    today = date.today().isoformat()

    run = wandb.init(
        project=PROJECT,
        name=f"sp500-5min-{today}",
        tags=["market-data", "5min", "sp500"],
        config={"date": today, "interval": "5min", "universe": "S&P 500"},
    )

    # 1. ^GSPC index — step-logged line chart
    if not index_df.empty:
        index_df = index_df.sort_values("date").reset_index(drop=True)
        for i, row in index_df.iterrows():
            wandb.log({
                "index/close":  row["close"],
                "index/open":   row["open"],
                "index/high":   row["high"],
                "index/low":    row["low"],
                "index/volume": row["volume"],
                "bar":          i,
                "timestamp":    str(row["date"]),
            }, step=i)

    # 2. Full OHLCV table — browsable in Artifacts tab
    stocks_df["date"] = stocks_df["date"].astype(str)
    wandb.log({"sp500_5min_bars": wandb.Table(dataframe=stocks_df)})

    # 3. Per-stock close-price pivot for custom chart panels
    pivot = (
        stocks_df.pivot(index="date", columns="symbol", values="close")
        .reset_index()
        .rename(columns={"date": "timestamp"})
    )
    wandb.log({"sp500_close_pivot": wandb.Table(dataframe=pivot)})

    # 4. Summary stats per symbol
    summary = (
        stocks_df.groupby("symbol")
        .agg(bars=("close", "count"), open=("open", "first"), close=("close", "last"),
             high=("high", "max"), low=("low", "min"), total_volume=("volume", "sum"))
        .assign(day_return=lambda d: (d["close"] / d["open"] - 1) * 100)
        .reset_index()
        .sort_values("day_return", ascending=False)
    )
    wandb.log({"sp500_summary": wandb.Table(dataframe=summary)})

    wandb.summary["top_mover"]      = summary.iloc[0]["symbol"]
    wandb.summary["top_return_%"]   = round(summary.iloc[0]["day_return"], 2)
    wandb.summary["bot_mover"]      = summary.iloc[-1]["symbol"]
    wandb.summary["bot_return_%"]   = round(summary.iloc[-1]["day_return"], 2)
    wandb.summary["symbols_logged"] = summary.shape[0]
    wandb.summary["total_bars"]     = len(stocks_df)

    run.finish()
    return run.url


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="Re-poll FMP before logging")
    args = parser.parse_args()

    url = log_to_wandb(fetch=args.fetch)
    print(f"\nW&B run: {url}")

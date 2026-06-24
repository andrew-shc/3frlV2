"""
Backtest pipeline — trains top-3 Optuna configs, runs greedy inference,
reports quantstats metrics, and logs vectorbt plots to W&B.

Split: last TEST_YEARS of data = held-out test set; everything before = train.

Usage:
    python -m pm.eval.backtest
    python -m pm.eval.backtest --train-steps 50000 --skip-train
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import wandb
from dotenv import load_dotenv

from pm.data.loader import load_indicators
from pm.model.config import DEFAULT_CFG
from pm.model.ddpg import train
from pm.eval.inference import build_benchmarks, run_inference
from pm.eval.metrics import run_quantstats
from pm.eval.plots import run_all_plots

load_dotenv()

PARQUET    = "dataset/hist/sp500_5min_5yr.parquet"
PROJECT    = os.getenv("WANDB_PROJECT", "3frlV2")
GROUP_NAME = "tucker-ddpg-backtest"
TEST_YEARS = 1        # hold out last N years as test set
CKPT_DIR   = "checkpoints"

TOP3 = [
    {
        "name": "trial-008",
        "tucker_rank": 8,  "actor_lr": 8.781e-4,   "critic_lr": 8.412e-4,
        "fc_hidden": 256,  "batch_size": 128,       "tau": 7.146e-3,
        "exploration_noise": 0.0629, "gamma": 0.9637, "cor_window": 20,
    },
    {
        "name": "trial-011",
        "tucker_rank": 4,  "actor_lr": 9.724e-4,   "critic_lr": 5.154e-5,
        "fc_hidden": 256,  "batch_size": 32,        "tau": 8.640e-3,
        "exploration_noise": 0.0531, "gamma": 0.9760, "cor_window": 20,
    },
    {
        "name": "trial-018",
        "tucker_rank": 4,  "actor_lr": 2.837e-4,   "critic_lr": 4.257e-4,
        "fc_hidden": 256,  "batch_size": 32,        "tau": 1.561e-2,
        "exploration_noise": 0.0777, "gamma": 0.9770, "cor_window": 20,
    },
]


def _build_cfg(params: dict, train_steps: int) -> dict:
    rank = params["tucker_rank"]
    return {
        **DEFAULT_CFG,
        "tucker_ranks":      [rank] * 4,
        "actor_lr":          params["actor_lr"],
        "critic_lr":         params["critic_lr"],
        "fc_hidden":         params["fc_hidden"],
        "batch_size":        params["batch_size"],
        "tau":               params["tau"],
        "exploration_noise": params["exploration_noise"],
        "gamma":             params["gamma"],
        "cor_window":        params["cor_window"],
        "total_steps":       train_steps,
        "n_actions":         DEFAULT_CFG["n_assets"] + 1,
    }


def _get_or_train(
    params: dict, train_indicators: dict, train_steps: int, ckpt_dir: str,
    force: bool = False,
) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{params['name']}.pt")
    if os.path.exists(ckpt_path) and not force:
        print(f"  Checkpoint exists, skipping: {ckpt_path}")
        return ckpt_path
    cfg = _build_cfg(params, train_steps)
    train(cfg, PARQUET, run_name=f"backtest-{params['name']}",
          run_group=GROUP_NAME, indicators=train_indicators, ckpt_path=ckpt_path)
    return ckpt_path


def _calendar_split(
    indicators: dict, dates: pd.DatetimeIndex, test_years: int = TEST_YEARS
) -> tuple[dict, dict, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split by calendar: last `test_years` years = test, rest = train."""
    cutoff = dates[-1] - pd.DateOffset(years=test_years)
    T_train = int((dates < cutoff).sum())
    return (
        {k: v[:T_train] for k, v in indicators.items()},
        {k: v[T_train:] for k, v in indicators.items()},
        dates[:T_train],
        dates[T_train:],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps",   type=int, default=30_000)
    parser.add_argument("--test-years",    type=int, default=TEST_YEARS,
                        help="Calendar years held out as test set")
    parser.add_argument("--skip-train",    action="store_true")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Retrain even if checkpoint already exists")
    parser.add_argument("--configs-path",  type=str, default=None,
                        help="JSON file from tune.py (tune_top3.json) to override TOP3")
    args = parser.parse_args()

    configs = TOP3
    if args.configs_path:
        with open(args.configs_path) as f:
            configs = json.load(f)
        print(f"Loaded {len(configs)} configs from {args.configs_path}")

    print(f"\n{'='*60}")
    print(f"  Tucker-DDPG Backtest  ({len(configs)} configs)")
    print(f"  test_years={args.test_years}  train_steps={args.train_steps}")
    print(f"{'='*60}\n")

    indicators, dates, _ = load_indicators(PARQUET, DEFAULT_CFG["n_assets"])
    train_indicators, test_indicators, train_dates, test_dates = _calendar_split(
        indicators, dates, args.test_years
    )
    T       = len(dates)
    T_train = len(train_dates)
    T_test  = len(test_dates)

    print(f"  Total bars: {T}  |  Train: {T_train} ({T_train/T:.0%})"
          f"  |  Test: {T_test} ({T_test/T:.0%})")
    print(f"  Train period: {train_dates[0].date()} → {train_dates[-1].date()}")
    print(f"  Test period:  {test_dates[0].date()} → {test_dates[-1].date()}\n")

    # ── Train ────────────────────────────────────────────────────────────────
    ckpt_paths: dict[str, str] = {}
    for params in configs:
        name = params["name"]
        print(f"[{name}] Training ({args.train_steps} steps)...")
        if args.skip_train:
            ckpt = os.path.join(CKPT_DIR, f"{name}.pt")
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
            ckpt_paths[name] = ckpt
        else:
            ckpt_paths[name] = _get_or_train(
                params, train_indicators, args.train_steps, CKPT_DIR,
                force=args.force_retrain,
            )

    # ── Inference ────────────────────────────────────────────────────────────
    print("\nRunning greedy inference on test set...")
    model_returns: dict = {}
    for params in configs:
        name = params["name"]
        cfg  = _build_cfg(params, train_steps=1)
        print(f"  {name}...")
        ret_series, _ = run_inference(cfg, ckpt_paths[name], test_indicators, test_dates, name)
        model_returns[name] = ret_series
        print(f"    steps={len(ret_series)}  mean_return={ret_series.mean():.6f}")

    # ── Benchmarks ───────────────────────────────────────────────────────────
    ref_index  = next(iter(model_returns.values())).index
    all_returns = {**model_returns, **build_benchmarks(test_indicators, test_dates, ref_index)}

    # ── W&B summary run ──────────────────────────────────────────────────────
    summary_run = wandb.init(
        project=PROJECT, name=f"{GROUP_NAME}-summary",
        group=GROUP_NAME, tags=["backtest", "summary"],
        config={"test_years": args.test_years, "train_steps": args.train_steps,
                "test_start": str(test_dates[0].date()), "test_end": str(test_dates[-1].date()),
                "T_train": T_train, "T_test": T_test,
                "n_assets": DEFAULT_CFG["n_assets"], "configs": [p["name"] for p in configs]},
    )

    print("\nRunning quantstats...")
    metrics_df = run_quantstats(all_returns, report_dir="reports", wandb_run=summary_run)

    print("\nRunning plots...")
    run_all_plots(all_returns, metrics_df, wandb_run=summary_run)

    summary_run.finish()
    print(f"\nDone.  W&B group: {GROUP_NAME}")


if __name__ == "__main__":
    main()

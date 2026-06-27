"""
IMM backtest pipeline — runs inference, benchmarks, metrics, plots, W&B.

Usage
-----
  python -m mm.eval.backtest
  python -m mm.eval.backtest --ticker MSFT --ckpt checkpoints/imm_MSFT.pt
  python -m mm.eval.backtest --ticker MSFT --train   # train first, then backtest
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import wandb
from dotenv import load_dotenv

from mm.data.loader import load_lobster
from mm.data.features import build_feature_matrix
from mm.model.config import DEFAULT_CFG
from mm.eval.inference import run_inference, ltiic_rollout, foic_rollout
from mm.eval.metrics import log_metrics_table
from mm.eval.plots import run_all_plots

load_dotenv()

DATASET_DIR  = "dataset/mm"
CKPT_DIR     = "checkpoints"
REPORT_DIR   = "reports"
PROJECT      = os.getenv("WANDB_PROJECT", "3frlV2")
GROUP_NAME   = "mm-imm-backtest"


def _find_lobster_files(ticker: str, levels: int = 10):
    import glob
    msgs = sorted(glob.glob(f"{DATASET_DIR}/{ticker}_*_message_{levels}.csv"))
    obs  = sorted(glob.glob(f"{DATASET_DIR}/{ticker}_*_orderbook_{levels}.csv"))
    if not msgs or not obs:
        raise FileNotFoundError(f"No LOBSTER files for {ticker} depth={levels}")
    return msgs[0], obs[0]


def run_backtest(
    cfg: dict,
    ticker: str,
    ckpt_path: str,
    n_episodes: int = 20,
    seed: int = 42,
) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  IMM Backtest — {ticker}")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  episodes: {n_episodes}")
    print(f"{'='*60}\n")

    # ── Load cfg + train/test split from checkpoint ─────────────
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt_data.get("cfg", cfg)

    # ── Data ────────────────────────────────────────────────────
    print("Loading data...")
    msg_path, ob_path = _find_lobster_files(ticker, cfg["lob_levels"])
    msg, ob = load_lobster(msg_path, ob_path, n_levels=cfg["lob_levels"])
    X = build_feature_matrix(msg, ob, tick_size=cfg["tick_size"])
    print(f"  {len(msg)} ticks  |  feature matrix: {X.shape}")

    # Recover the exact split used during training
    train_t = ckpt_data.get("train_t", int(len(msg) * cfg.get("train_split", 0.8)))
    test_t  = len(msg) - train_t
    print(f"  Train/test split: [{0}, {train_t}) train  |  [{train_t}, {len(msg)}) test  "
          f"({test_t} test ticks)")

    # ── Inference (held-out test split only) ────────────────────
    print("\nRunning IMM rollout (test split)...")
    imm_records = run_inference(cfg, ckpt_path, X, msg, ob,
                                n_episodes=n_episodes, seed=seed,
                                t_min=train_t)

    print("Running LTIIC benchmark (test split)...")
    ltiic_records = ltiic_rollout(cfg, X, msg, ob,
                                  n_episodes=n_episodes, seed=seed,
                                  t_min=train_t)

    print("Running FOIC benchmark (test split)...")
    foic_records = foic_rollout(cfg, X, msg, ob,
                                n_episodes=n_episodes, seed=seed,
                                t_min=train_t)

    all_records = {
        "imm":   imm_records,
        "ltiic": ltiic_records,
        "foic":  foic_records,
    }

    # ── W&B summary run ──────────────────────────────────────────
    summary_run = wandb.init(
        project=PROJECT,
        name=f"{GROUP_NAME}-{ticker}-{int(time.time())}",
        group=GROUP_NAME,
        tags=["mm", "backtest", "summary", ticker],
        config={
            **cfg,
            "ticker":     ticker,
            "n_episodes": n_episodes,
            "ckpt_path":  ckpt_path,
            "train_t":    train_t,
            "test_t":     test_t,
        },
    )
    summary_run.log({
        "mm/data/total_ticks": len(msg),
        "mm/data/train_ticks": train_t,
        "mm/data/test_ticks":  test_t,
    })

    print("\nComputing metrics...")
    metrics_df = log_metrics_table(all_records, summary_run, prefix="mm/backtest")

    print("\nGenerating plots...")
    run_all_plots(all_records, metrics_df, summary_run)

    summary_run.finish()
    print(f"\nDone.  W&B group: {GROUP_NAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",     default="MSFT")
    parser.add_argument("--ckpt",       default=None)
    parser.add_argument("--episodes",   type=int, default=20)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--train",      action="store_true",
                        help="Run training first, then backtest")
    parser.add_argument("--total-steps",type=int, default=DEFAULT_CFG["total_steps"])
    args = parser.parse_args()

    ckpt = args.ckpt or os.path.join(CKPT_DIR, f"imm_{args.ticker}.pt")

    if args.train or not os.path.exists(ckpt):
        from mm.model.td3 import train
        cfg = {**DEFAULT_CFG, "total_steps": args.total_steps, "seed": args.seed}
        ckpt = train(cfg, ticker=args.ticker)
    else:
        cfg = DEFAULT_CFG

    run_backtest(cfg, ticker=args.ticker, ckpt_path=ckpt,
                 n_episodes=args.episodes, seed=args.seed)

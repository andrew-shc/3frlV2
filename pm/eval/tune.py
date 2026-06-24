"""
Optuna hyperparameter search for Tucker-DDPG.

Each trial runs a short training run and reports full metrics to W&B
(one W&B run per trial, grouped under "optuna-<study_name>").
After all trials, a summary run logs optimization history,
parameter importances, and the best configuration.

Objective: mean per-step portfolio return over the last half of
post-warmup training steps (higher = better).

Search space
------------
  tucker_rank       : 4 | 8 | 16
  actor_lr          : log-uniform [1e-5, 1e-3]
  critic_lr         : log-uniform [1e-5, 1e-3]
  fc_hidden         : 128 | 256 | 512
  batch_size        : 32 | 64 | 128
  tau               : log-uniform [0.001, 0.05]
  exploration_noise : uniform [0.05, 0.3]
  gamma             : uniform [0.95, 0.999]
  cor_window        : 20 | 50 | 100

Usage:
    python -m pm.eval.tune
    python -m pm.eval.tune --n-trials 30 --trial-steps 5000
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import wandb
from dotenv import load_dotenv
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)

import pandas as pd

from pm.data.loader import load_indicators
from pm.model.config import DEFAULT_CFG
from pm.model.ddpg import train

load_dotenv()

PARQUET    = "dataset/hist/sp500_5min_5yr.parquet"
PROJECT    = os.getenv("WANDB_PROJECT", "3frlV2")
STUDY_NAME = "tucker-ddpg-search"
TEST_YEARS = 1        # mirror backtest.py — tune only on train portion


def objective(
    trial: optuna.Trial, trial_steps: int, train_indicators: dict
) -> float:
    rank = trial.suggest_categorical("tucker_rank", [4, 8, 16])
    cfg = {
        **DEFAULT_CFG,
        "tucker_ranks":      [rank] * 4,
        "actor_lr":          trial.suggest_float("actor_lr",  1e-5, 1e-3, log=True),
        "critic_lr":         trial.suggest_float("critic_lr", 1e-5, 1e-3, log=True),
        "fc_hidden":         trial.suggest_categorical("fc_hidden",  [128, 256, 512]),
        "batch_size":        trial.suggest_categorical("batch_size", [32, 64, 128]),
        "tau":               trial.suggest_float("tau", 1e-3, 0.05, log=True),
        "exploration_noise": trial.suggest_float("exploration_noise", 0.05, 0.3),
        "gamma":             trial.suggest_float("gamma", 0.95, 0.999),
        "cor_window":        trial.suggest_categorical("cor_window", [20, 50, 100]),
        "total_steps":       trial_steps,
        "n_actions":         DEFAULT_CFG["n_assets"] + 1,
    }
    return train(
        cfg, PARQUET,
        run_name=f"trial-{trial.number:03d}",
        run_group=STUDY_NAME,
        indicators=train_indicators,
    )


TOP3_JSON = "checkpoints/tune_top3.json"


def save_top3(study: optuna.Study, out_path: str = TOP3_JSON, n: int = 3) -> None:
    """Save top-N completed trials as JSON for backtest.py to consume."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top = sorted(completed, key=lambda t: t.value, reverse=True)[:n]
    records = [
        {"name": f"trial-{t.number:03d}", "value": t.value, **t.params}
        for t in top
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Top-{n} configs saved → {out_path}")
    for r in records:
        print(f"  {r['name']}  value={r['value']:.6f}  {r}")


def log_study_summary(study: optuna.Study) -> None:
    """Log study-level visualisations and best params to a single W&B run."""
    run = wandb.init(
        project=PROJECT,
        name=f"{STUDY_NAME}-summary",
        group=STUDY_NAME,
        tags=["optuna", "summary"],
        config={"n_trials": len(study.trials), "study_name": STUDY_NAME},
    )

    best = study.best_trial
    print(f"\nBest trial #{best.number}  value={best.value:.6f}")
    print("  Params:", best.params)
    wandb.log({
        "best/value":        best.value,
        "best/trial_number": best.number,
        **{f"best/{k}": v for k, v in best.params.items()},
    })

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df = df[df["state"] == "COMPLETE"].drop(columns=["state"])
    wandb.log({"tables/all_trials": wandb.Table(dataframe=df)})

    figs: dict[str, plt.Figure] = {}
    for name, fn in [
        ("plots/optimization_history", plot_optimization_history),
        ("plots/param_importances",    plot_param_importances),
        ("plots/parallel_coordinate",  plot_parallel_coordinate),
    ]:
        try:
            ax = fn(study)
            figs[name] = ax.get_figure()
        except Exception:
            pass

    wandb.log({k: wandb.Image(v) for k, v in figs.items()})
    for fig in figs.values():
        plt.close(fig)

    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials",    type=int, default=20)
    parser.add_argument("--trial-steps", type=int, default=5_000)
    parser.add_argument("--n-jobs",      type=int, default=1)
    parser.add_argument("--test-years",  type=int, default=TEST_YEARS,
                        help="Calendar years held out for backtest (not used in tuning)")
    args = parser.parse_args()

    # Load once; pass train slice to every trial (no test leakage)
    print("Loading indicators...")
    indicators, dates, _ = load_indicators(PARQUET, DEFAULT_CFG["n_assets"])
    cutoff = dates[-1] - pd.DateOffset(years=args.test_years)
    T_train = int((dates < cutoff).sum())
    train_indicators = {k: v[:T_train] for k, v in indicators.items()}
    print(f"  Train: {dates[0].date()} → {dates[T_train-1].date()} ({T_train} bars)")
    print(f"  Test held-out: {dates[T_train].date()} → {dates[-1].date()} (not used in tuning)\n")

    print(f"Starting Optuna search: {args.n_trials} trials × {args.trial_steps} steps")
    print(f"W&B project: {PROJECT}  group: {STUDY_NAME}\n")

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective(trial, args.trial_steps, train_indicators),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    print(f"\nSearch complete. Best value: {study.best_value:.6f}")
    save_top3(study)
    log_study_summary(study)


if __name__ == "__main__":
    main()

"""
Comprehensive quantstats metrics for 5-minute bar returns.

BARS_PER_YEAR sets the annualization factor: 78 bars/day × 252 days = 19,656.
All period-sensitive metrics (Sharpe, Sortino, volatility …) are annualized
using this factor so that results are comparable across different timeframes.

The HTML tearsheet (quantstats.reports.html) is the primary deliverable and
already includes every metric quantstats computes.  The W&B table produced by
run_quantstats() is a structured version of that same full metric set, logged
programmatically so it can be queried and compared across runs.
"""
from __future__ import annotations

import os

import pandas as pd
import quantstats as qs
import wandb

BARS_PER_YEAR = 78 * 252   # ~78 five-minute bars per US trading day


def _safe(fn, *args, **kwargs) -> float:
    """Call a quantstats function; return nan on any error."""
    try:
        v = fn(*args, **kwargs)
        return float(v.iloc[0]) if isinstance(v, pd.Series) else float(v)
    except Exception:
        return float("nan")


def compute_metrics(returns: pd.Series, periods: int = BARS_PER_YEAR) -> dict[str, float]:
    """
    Full quantstats metric set, annualized for 5-min bars.

    Covers: risk-adjusted returns, drawdown, trade stats, tail risk, and
    consecutive win/loss streaks.
    """
    r = returns.dropna()
    p = periods

    return {
        # ── Returns ──────────────────────────────────────────────────────────
        "total_return":      float((1 + r).prod() - 1),
        "avg_return":        float(r.mean()),
        "best_bar":          _safe(qs.stats.best,  r),
        "worst_bar":         _safe(qs.stats.worst, r),
        "avg_win":           _safe(qs.stats.avg_win,  r),
        "avg_loss":          _safe(qs.stats.avg_loss, r),

        # ── Risk-adjusted ────────────────────────────────────────────────────
        "sharpe":            _safe(qs.stats.sharpe,   r, annualize=True, periods=p),
        "sortino":           _safe(qs.stats.sortino,  r, annualize=True, periods=p),
        "calmar":            _safe(qs.stats.calmar,   r),
        "omega":             _safe(qs.stats.omega,    r),

        # ── Volatility & tail ────────────────────────────────────────────────
        "volatility":        _safe(qs.stats.volatility, r, annualize=True, periods=p),
        "skew":              _safe(qs.stats.skew,        r),
        "kurtosis":          _safe(qs.stats.kurtosis,    r),
        "tail_ratio":        _safe(qs.stats.tail_ratio,  r),
        "var_95":            _safe(qs.stats.value_at_risk, r),
        "cvar_95":           _safe(qs.stats.cvar,          r),

        # ── Drawdown ─────────────────────────────────────────────────────────
        "max_drawdown":      _safe(qs.stats.max_drawdown,    r),
        "recovery_factor":   _safe(qs.stats.recovery_factor, r),

        # ── Trade quality ────────────────────────────────────────────────────
        "win_rate":          _safe(qs.stats.win_rate,      r),
        "payoff_ratio":      _safe(qs.stats.payoff_ratio,  r),
        "profit_factor":     _safe(qs.stats.profit_factor, r),
        "exposure":          _safe(qs.stats.exposure,      r),
        "consecutive_wins":  _safe(qs.stats.consecutive_wins,   r),
        "consecutive_losses":_safe(qs.stats.consecutive_losses, r),
    }


def run_quantstats(
    all_returns: dict[str, pd.Series],
    report_dir: str,
    wandb_run,
) -> pd.DataFrame:
    """
    For each strategy:
      1. Generate an HTML tearsheet saved to report_dir and uploaded as W&B artifact.
      2. Compute the full metric set and collect into a DataFrame.
    Logs the metrics DataFrame as a W&B table and returns it.
    """
    os.makedirs(report_dir, exist_ok=True)
    rows = []

    for name, rets in all_returns.items():
        html_path = os.path.join(report_dir, f"tearsheet_{name}.html")
        try:
            qs.reports.html(
                rets,
                output=html_path,
                title=f"Tucker-DDPG  {name}",
                periods_per_year=BARS_PER_YEAR,
                download_filename=html_path,
            )
            artifact = wandb.Artifact(f"tearsheet-{name}", type="report")
            artifact.add_file(html_path)
            wandb_run.log_artifact(artifact)
        except Exception as exc:
            print(f"  tearsheet failed for {name}: {exc}")

        m = compute_metrics(rets)
        m["name"] = name
        rows.append(m)
        print(
            f"  {name:20s}  sharpe={m['sharpe']:+.3f}  sortino={m['sortino']:+.3f}"
            f"  calmar={m['calmar']:+.3f}  maxDD={m['max_drawdown']:.3f}"
            f"  total={m['total_return']:+.4f}  win={m['win_rate']:.2%}"
        )

    metrics_df = pd.DataFrame(rows).set_index("name")
    wandb_run.log({"backtest/metrics": wandb.Table(dataframe=metrics_df.reset_index())})
    return metrics_df

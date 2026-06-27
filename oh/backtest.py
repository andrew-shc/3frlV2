"""
Backtesting / evaluation for the D4PG-QR hedging agent.

Metrics reported per eval run:
  mean_return, std_return, sharpe, VaR_5, CVaR_5, max_drawdown
  + P&L distribution plot

Usage:
  python -m gv.backtest --ckpt checkpoints/gv/<run>.pt [--market heston] [--n_episodes 1000]
"""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass, asdict

import numpy as np
import torch
import matplotlib.pyplot as plt

from .config import HedgeConfig
from .env.hedging_env import HedgingEnv
from .agent.networks import Actor, QRCritic


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class EvalMetrics:
    mean_return:  float
    std_return:   float
    sharpe:       float
    var_05:       float   # 5th-pct loss (negative = good)
    cvar_05:      float   # expected loss beyond VaR_5
    max_drawdown: float
    n_episodes:   int

    def __str__(self):
        return (
            f"  mean={self.mean_return:+.4f}  std={self.std_return:.4f}  "
            f"sharpe={self.sharpe:+.3f}\n"
            f"  VaR5={self.var_05:+.4f}  CVaR5={self.cvar_05:+.4f}  "
            f"MDD={self.max_drawdown:.4f}  N={self.n_episodes}"
        )


def _max_drawdown(returns: np.ndarray) -> float:
    """Average per-episode max drawdown (each element is an episode total)."""
    # returns here are episode totals; to get intra-episode MDD we
    # treat each as a one-step episode — meaningful only for step-level data.
    # Here we report the mean magnitude of negative returns as a proxy.
    neg = returns[returns < 0]
    return float(-neg.mean()) if len(neg) > 0 else 0.0


def compute_metrics(returns: np.ndarray) -> EvalMetrics:
    r = np.array(returns)
    mu  = float(r.mean())
    std = float(r.std())
    sharpe = mu / (std + 1e-8) * np.sqrt(252 / 30)   # annualised (30-day episodes)
    var05  = float(np.percentile(r, 5))
    cvar05 = float(r[r <= var05].mean()) if (r <= var05).any() else var05
    mdd    = _max_drawdown(r)
    return EvalMetrics(mu, std, sharpe, var05, cvar05, mdd, len(r))


# ── Baselines ─────────────────────────────────────────────────────────────────

def run_baseline_no_hedge(cfg: HedgeConfig, n_episodes: int,
                          seed: int = 0) -> np.ndarray:
    """Always alpha=0 (no gamma/vega hedge, just delta-neutral)."""
    env = HedgingEnv(cfg, seed=seed)
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            obs, r, done, _, _ = env.step(np.array([0.0]))
            ep_ret += r
        returns.append(ep_ret)
    return np.array(returns)


def run_full_hedge(cfg: HedgeConfig, n_episodes: int,
                   seed: int = 0) -> np.ndarray:
    """Always alpha=1 (maximum hedging every day)."""
    env = HedgingEnv(cfg, seed=seed)
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            obs, r, done, _, _ = env.step(np.array([1.0]))
            ep_ret += r
        returns.append(ep_ret)
    return np.array(returns)


# ── Agent evaluation ──────────────────────────────────────────────────────────

def evaluate_agent(actor: Actor, cfg: HedgeConfig,
                   n_episodes: int = 1000, seed: int = 100) -> np.ndarray:
    """Deterministic rollout of a trained actor."""
    device = next(actor.parameters()).device
    actor.eval()
    env = HedgingEnv(cfg, seed=seed)
    returns = []

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_ret = 0.0
            done = False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                        device=device).unsqueeze(0)
                alpha = actor(obs_t).cpu().numpy()[0]
                obs, r, done, _, _ = env.step(alpha)
                ep_ret += r
            returns.append(ep_ret)

    actor.train()
    return np.array(returns)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_returns(results: dict[str, np.ndarray],
                 title: str = "Episode P&L distribution",
                 save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, rets in results.items():
        ax.hist(rets, bins=60, alpha=0.5, density=True, label=label)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Normalised episode return (/ S₀)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_quantile_fan(actor: Actor, critic: QRCritic,
                      cfg: HedgeConfig, n_episodes: int = 200,
                      save_path: str | None = None):
    """Plot the critic's quantile fan over episode steps for a few rollouts."""
    device = next(actor.parameters()).device
    actor.eval(); critic.eval()
    env = HedgingEnv(cfg, seed=999)

    all_q_means = []
    all_q_lower = []
    all_q_upper = []

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            q_means, q_lowers, q_uppers = [], [], []
            done = False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                        device=device).unsqueeze(0)
                alpha = actor(obs_t)
                z = critic(obs_t, alpha)[0].cpu().numpy()
                q_means.append(z.mean())
                q_lowers.append(np.percentile(z, 5))
                q_uppers.append(np.percentile(z, 95))
                obs, _, done, _, _ = env.step(alpha.cpu().numpy()[0])
            all_q_means.append(q_means)
            all_q_lower.append(q_lowers)
            all_q_upper.append(q_uppers)

    T = cfg.episode_len
    mean_arr  = np.mean(all_q_means,  axis=0)
    lower_arr = np.mean(all_q_lower,  axis=0)
    upper_arr = np.mean(all_q_upper,  axis=0)
    steps = np.arange(T)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, mean_arr,  label="Q-mean", color="steelblue")
    ax.fill_between(steps, lower_arr, upper_arr, alpha=0.3,
                    color="steelblue", label="Q 5–95%")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Day within episode")
    ax.set_ylabel("Critic quantile estimate (/ S₀)")
    ax.set_title("Return distribution fan over episode")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Robustness: train on GBM, test on Heston/VG ──────────────────────────────

def robustness_eval(actor: Actor, base_cfg: HedgeConfig,
                    n_episodes: int = 1000) -> dict[str, EvalMetrics]:
    results = {}
    for market in ["gbm", "heston", "vg"]:
        import copy
        cfg_test = copy.copy(base_cfg)
        cfg_test.market = market
        rets = evaluate_agent(actor, cfg_test, n_episodes, seed=42)
        results[market] = compute_metrics(rets)
        print(f"\n── {market.upper()} ──")
        print(results[market])
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       type=str, required=True)
    p.add_argument("--market",     type=str, default=None,
                   help="Override eval market (gbm|heston|vg). Default: from ckpt.")
    p.add_argument("--n_episodes", type=int, default=1000)
    p.add_argument("--robustness", action="store_true",
                   help="Also test on Heston and VG.")
    p.add_argument("--plot",       action="store_true")
    p.add_argument("--save_dir",   type=str, default="reports/gv")
    args = p.parse_args()

    # ── Load checkpoint ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.ckpt, map_location=device)
    cfg    = HedgeConfig(**ckpt["cfg"])
    if args.market:
        cfg.market = args.market

    actor  = Actor(5, cfg.hidden_dim).to(device)
    critic = QRCritic(5, 1, cfg.n_quantiles, cfg.hidden_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    print(f"Loaded: {args.ckpt}")

    # ── Agent eval ────────────────────────────────────────────────────────────
    rets_agent = evaluate_agent(actor, cfg, args.n_episodes)
    m_agent = compute_metrics(rets_agent)
    print(f"\n── AGENT ({cfg.market}) ──\n{m_agent}")

    # ── Baselines ─────────────────────────────────────────────────────────────
    rets_no   = run_baseline_no_hedge(cfg, args.n_episodes)
    rets_full = run_full_hedge(cfg, args.n_episodes)
    m_no   = compute_metrics(rets_no)
    m_full = compute_metrics(rets_full)
    print(f"\n── NO-HEDGE baseline ──\n{m_no}")
    print(f"\n── FULL-HEDGE baseline ──\n{m_full}")

    if args.robustness:
        robustness_eval(actor, cfg, args.n_episodes)

    if args.plot:
        save_dir = pathlib.Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_stem = pathlib.Path(args.ckpt).stem

        plot_returns(
            {"Agent": rets_agent, "No-hedge": rets_no, "Full-hedge": rets_full},
            title=f"P&L distribution | {cfg.objective} | κ={cfg.tc_ratio}",
            save_path=str(save_dir / f"{ckpt_stem}_returns.png"),
        )
        plot_quantile_fan(
            actor, critic, cfg,
            save_path=str(save_dir / f"{ckpt_stem}_qfan.png"),
        )


if __name__ == "__main__":
    main()

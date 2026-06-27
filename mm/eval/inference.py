"""
Greedy rollout and benchmark construction for MM backtest evaluation.

run_inference()  — deterministic rollout of a trained IMM actor
ltiic_rollout()  — LTIIC rule-based benchmark
foic_rollout()   — Fixed-Offset with Inventory Constraint benchmark
"""
from __future__ import annotations

import numpy as np
import torch

from mm.data.features import build_feature_matrix
from mm.model.actor import Actor
from mm.model.env import MarketMakingEnv
from mm.model.trainer import ltiic_action


# ------------------------------------------------------------------
# Trajectory collector helper
# ------------------------------------------------------------------

def _collect_episode(env: MarketMakingEnv, action_fn) -> dict:
    """Run one full episode using action_fn(obs) → action."""
    obs, _ = env.reset()
    pnl_hist, inv_hist, fill_bid_hist, fill_ask_hist, mid_hist = [], [], [], [], []
    cum_pnl = 0.0

    while True:
        action = action_fn(obs)
        obs, reward, done, _, info = env.step(action)
        cum_pnl += reward
        pnl_hist.append(cum_pnl)
        inv_hist.append(info["inventory"])
        fill_bid_hist.append(info.get("fills_bid", 0))
        fill_ask_hist.append(info.get("fills_ask", 0))
        mid_hist.append(info.get("mid", 0.0))
        if done:
            break

    return {
        "pnl_history":   pnl_hist,
        "inv_history":   inv_hist,
        "fill_bid_hist": fill_bid_hist,
        "fill_ask_hist": fill_ask_hist,
        "mid_history":   mid_hist,
        "desired_vol":   env.cfg["total_volume"],
    }


# ------------------------------------------------------------------
# IMM greedy rollout
# ------------------------------------------------------------------

def run_inference(
    cfg: dict,
    ckpt_path: str,
    feature_matrix: np.ndarray,
    msg,
    ob,
    n_episodes: int = 10,
    seed: int = 42,
    t_min: int | None = None,
    t_max: int | None = None,
) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor = Actor(ckpt.get("cfg", cfg)).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    f, l = cfg["f_dim"], cfg["tcsa_seq_len"]
    priv = 1 + 4 * cfg["n_quote_levels"] + 7

    @torch.no_grad()
    def act(obs: np.ndarray) -> np.ndarray:
        x, s_p = MarketMakingEnv.split_obs(obs, f, l, priv)
        return actor(x.to(device), s_p.to(device)).cpu().numpy().squeeze(0)

    rng = np.random.default_rng(seed)
    records = []
    for ep in range(n_episodes):
        env = MarketMakingEnv(feature_matrix, msg, ob, cfg, t_min=t_min, t_max=t_max)
        env.reset(seed=int(rng.integers(0, 2**31)))
        records.append(_collect_episode(env, act))

    return records


# ------------------------------------------------------------------
# Rule-based benchmarks
# ------------------------------------------------------------------

def ltiic_rollout(
    cfg: dict,
    feature_matrix: np.ndarray,
    msg,
    ob,
    n_episodes: int = 10,
    seed: int = 42,
    a: float = 0.0,      # 0 = quote at best bid/ask touch
    b: float = 0.5,
    c: float = 1.0,
    d: float = 5.0,
    t_min: int | None = None,
    t_max: int | None = None,
) -> list[dict]:
    """LTIIC expert strategy rollout (benchmark)."""
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n_episodes):
        env = MarketMakingEnv(feature_matrix, msg, ob, cfg, t_min=t_min, t_max=t_max)
        env.reset(seed=int(rng.integers(0, 2**31)))

        def act(obs: np.ndarray) -> np.ndarray:
            info    = env.exchange.state
            tick    = cfg["tick_size"]
            ob_row  = env.ob.iloc[env._t]
            mid     = float(ob_row.get("mid", 0.0))
            best_bid = float(ob_row.get("bid_p1", mid - tick))
            best_ask = float(ob_row.get("ask_p1", mid + tick))
            if not np.isfinite(best_bid): best_bid = mid - tick
            if not np.isfinite(best_ask): best_ask = mid + tick
            y_hat = 0.0
            return ltiic_action(
                z=info.inventory, y_hat=y_hat,
                best_bid=best_bid, best_ask=best_ask, cfg=cfg,
                a=a, b=b, c=c, d=d,
            )

        records.append(_collect_episode(env, act))
    return records


def foic_rollout(
    cfg: dict,
    feature_matrix: np.ndarray,
    msg,
    ob,
    n_episodes: int = 10,
    seed: int = 42,
    fixed_offset: float = 1.0,
    inv_limit: float = 5.0,
    t_min: int | None = None,
    t_max: int | None = None,
) -> list[dict]:
    """Fixed Offset with Inventory Constraint (FOIC) benchmark."""
    rng = np.random.default_rng(seed)
    records = []
    K = cfg["n_quote_levels"]
    max_offset = cfg["max_offset_ticks"]
    max_spread = cfg["max_spread_ticks"]

    for _ in range(n_episodes):
        env = MarketMakingEnv(feature_matrix, msg, ob, cfg, t_min=t_min, t_max=t_max)
        env.reset(seed=int(rng.integers(0, 2**31)))

        def act(obs: np.ndarray) -> np.ndarray:
            z = env.exchange.state.inventory
            spread_norm = np.clip(fixed_offset * 2 / (max_spread + 1e-8), 0, 1)
            phi = np.ones(K, dtype=np.float32) / K
            if abs(z) > inv_limit:
                # Post only on the side that reduces inventory
                phi_bid = phi if z < 0 else np.zeros(K, dtype=np.float32)
                phi_ask = phi if z > 0 else np.zeros(K, dtype=np.float32)
            else:
                phi_bid = phi_ask = phi
            return np.concatenate([[0.0], [spread_norm], phi_bid, phi_ask]).astype(np.float32)

        records.append(_collect_episode(env, act))
    return records

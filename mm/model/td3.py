"""
IMM training orchestration — TD3 + Behaviour Cloning.

Pipeline
--------
  1. Load LOBSTER data + compute feature matrix
  2. Pre-train SL signal predictor (or load checkpoint)
  3. Collect LTIIC expert demonstrations
  4. Main TD3+BC loop (actor/twin-critic, BC loss decays over time)
  5. Log to W&B; save checkpoint

Usage
-----
  python -m mm.model.td3
  python -m mm.model.td3 --ticker MSFT --total-steps 50000
"""
from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.optim as optim
import wandb
from dotenv import load_dotenv
from stable_baselines3.common.buffers import ReplayBuffer

from mm.data.loader import load_lobster
from mm.data.features import build_feature_matrix
from mm.model.config import DEFAULT_CFG
from mm.model.actor import Actor
from mm.model.critic import TwinCritic
from mm.model.env import MarketMakingEnv
from mm.model.sl import SLPredictor, make_sl_labels, train_sl
from mm.model.trainer import (
    PreparedBatch, prepare_batch, select_action,
    update_critic, update_actor, soft_update, ltiic_action,
)

load_dotenv()

DATASET_DIR = "dataset/mm"
CKPT_DIR    = "checkpoints"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _init_wandb(cfg: dict, run_name: str | None, ticker: str):
    return wandb.init(
        project=os.getenv("WANDB_PROJECT", "3frlV2"),
        name=run_name or f"mm-imm-{ticker}-{int(time.time())}",
        group="mm-imm-train",
        tags=["mm", "imm", "td3", "market-making", ticker],
        config=cfg,
        save_code=True,
    )


def _find_lobster_files(ticker: str, levels: int = 10):
    """Return (msg_path, ob_path) for the requested ticker and level depth."""
    import glob
    msgs = glob.glob(f"{DATASET_DIR}/{ticker}_*_message_{levels}.csv")
    obs  = glob.glob(f"{DATASET_DIR}/{ticker}_*_orderbook_{levels}.csv")
    if not msgs or not obs:
        raise FileNotFoundError(
            f"No LOBSTER files found for {ticker} depth={levels} in {DATASET_DIR}"
        )
    return sorted(msgs)[0], sorted(obs)[0]


def _pretrain_sl(
    cfg: dict,
    feature_matrix: np.ndarray,
    ob,
    sl_ckpt: str,
    device: torch.device,
) -> SLPredictor:
    sl = SLPredictor(
        in_dim=cfg["f_dim"],
        hidden=cfg["sl_hidden"],
        n_horizons=cfg["n_horizons"],
    )
    if os.path.exists(sl_ckpt):
        sl.load_state_dict(torch.load(sl_ckpt, map_location=device, weights_only=True))
        sl.eval()
        print(f"  SL loaded from {sl_ckpt}")
        return sl.to(device)

    print("  Pre-training SL signal predictor...")
    mid    = ob["mid"].values.astype(np.float32)
    spread = ob["spread"].ffill().fillna(0).values.astype(np.float32)
    labels = make_sl_labels(mid, spread, tick_size=cfg["tick_size"])
    losses = train_sl(sl, feature_matrix, labels, epochs=20, lr=cfg["sl_lr"], device=str(device))
    print(f"  SL training done. Final loss: {losses[-1]:.4f}")
    torch.save(sl.state_dict(), sl_ckpt)
    return sl.to(device)


def _collect_expert_demos(
    cfg: dict,
    feature_matrix: np.ndarray,
    msg,
    ob,
    n_steps: int,
    device: torch.device,
    t_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out LTIIC and collect (obs, expert_action) pairs."""
    env = MarketMakingEnv(feature_matrix, msg, ob, cfg, t_max=t_max)
    obs_buf  = np.zeros((n_steps, env.observation_space.shape[0]), dtype=np.float32)
    act_buf  = np.zeros((n_steps, env.action_space.shape[0]), dtype=np.float32)
    obs, _   = env.reset(seed=cfg["seed"])
    collected = 0

    while collected < n_steps:
        state   = env.exchange.state
        ob_row  = ob.iloc[env._t]
        tick    = cfg["tick_size"]
        mid     = float(ob["mid"].iloc[env._t])
        best_bid = float(ob_row.get("bid_p1", mid - tick))
        best_ask = float(ob_row.get("ask_p1", mid + tick))
        if not np.isfinite(best_bid): best_bid = mid - tick
        if not np.isfinite(best_ask): best_ask = mid + tick
        # Simple signal proxy: 0 (neutral) when SL unavailable here
        action = ltiic_action(
            z=state.inventory, y_hat=0.0,
            best_bid=best_bid, best_ask=best_ask, cfg=cfg,
        )
        obs_buf[collected] = obs
        act_buf[collected] = action
        obs, _, done, _, _ = env.step(action)
        collected += 1
        if done:
            obs, _ = env.reset()

    return obs_buf, act_buf


def _log_step(step: int, c_loss: float, a_loss: float | None, bc_coef: float) -> None:
    log = {"mm/loss/critic": c_loss, "mm/train/bc_coef": bc_coef, "global_step": step}
    if a_loss is not None:
        log["mm/loss/actor"] = a_loss
    wandb.log(log, step=step)


# ------------------------------------------------------------------
# Main training function
# ------------------------------------------------------------------

def train(
    cfg: dict,
    ticker: str = "MSFT",
    run_name: str | None = None,
    ckpt_path: str | None = None,
) -> str:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    if ckpt_path is None:
        ckpt_path = os.path.join(CKPT_DIR, f"imm_{ticker}.pt")
    sl_ckpt = os.path.join(CKPT_DIR, f"sl_{ticker}.pt")

    run = _init_wandb(cfg, run_name, ticker)

    # ── Data ──────────────────────────────────────────────────────
    print(f"\nLoading LOBSTER data: {ticker}...")
    msg_path, ob_path = _find_lobster_files(ticker, cfg["lob_levels"])
    msg, ob = load_lobster(msg_path, ob_path, n_levels=cfg["lob_levels"])
    print(f"  {len(msg)} ticks loaded")

    print("  Computing feature matrix...")
    X = build_feature_matrix(msg, ob, tick_size=cfg["tick_size"])
    print(f"  Feature matrix: {X.shape}")

    # ── Train / test split (temporal) ─────────────────────────────
    train_t = int(len(ob) * cfg.get("train_split", 0.8))
    print(f"  Train split: ticks [0, {train_t}) | Test: [{train_t}, {len(ob)})")
    wandb.log({
        "mm/data/total_ticks": len(ob),
        "mm/data/train_ticks": train_t,
        "mm/data/test_ticks":  len(ob) - train_t,
    })

    # ── SL pre-training ───────────────────────────────────────────
    print("\nSL pre-training...")
    sl_model = _pretrain_sl(cfg, X, ob, sl_ckpt, device)
    wandb.log({"mm/sl/pretrained": True})

    # ── Expert demos for BC ───────────────────────────────────────
    n_expert = max(cfg["learning_starts"], 2_000)
    print(f"\nCollecting {n_expert} LTIIC expert steps...")
    expert_obs, expert_acts = _collect_expert_demos(cfg, X, msg, ob, n_expert, device,
                                                    t_max=train_t)

    # ── Networks ──────────────────────────────────────────────────
    env = MarketMakingEnv(X, msg, ob, cfg, t_max=train_t)

    # Inject pre-trained SL weights into actor
    actor  = Actor(cfg).to(device)
    actor.sl.load_state_dict(sl_model.state_dict())
    for p in actor.sl.parameters():
        p.requires_grad_(False)   # freeze SL during RL

    critic = TwinCritic(cfg).to(device)
    actor_target  = copy.deepcopy(actor)
    critic_target = copy.deepcopy(critic)
    for net in (actor_target, critic_target):
        for p in net.parameters():
            p.requires_grad_(False)

    actor_opt  = optim.Adam(
        [p for p in actor.parameters() if p.requires_grad], lr=cfg["actor_lr"]
    )
    critic_opt = optim.Adam(critic.parameters(), lr=cfg["critic_lr"])

    rb = ReplayBuffer(
        buffer_size=cfg["buffer_size"],
        observation_space=env.observation_space,
        action_space=env.action_space,
        device="cpu",
        handle_timeout_termination=False,
    )

    # ── Training loop ─────────────────────────────────────────────
    obs, _    = env.reset(seed=cfg["seed"])
    ep_return = 0.0
    ep_len    = 0
    ep_count  = 0
    ep_fills  = 0
    bc_coef   = cfg["bc_coef"]
    policy_updates = 0

    print(f"\nTraining for {cfg['total_steps']} steps...")
    for step in range(1, cfg["total_steps"] + 1):
        action = select_action(
            actor, obs, cfg, device,
            exploration_noise=cfg["exploration_noise"],
            is_exploring=(step < cfg["learning_starts"]),
            action_space=env.action_space,
        )
        next_obs, reward, done, _, info = env.step(action)
        ep_return += reward
        ep_len    += 1
        ep_fills  += info.get("fills_bid", 0) + info.get("fills_ask", 0)

        rb.add(obs, next_obs, action, reward, done, [info])
        obs = next_obs

        if done:
            ep_count += 1
            V = cfg["total_volume"]
            fill_rate = ep_fills / (2 * V * ep_len + 1e-8)
            wandb.log({
                "mm/episode/return":    ep_return,
                "mm/episode/length":    ep_len,
                "mm/episode/count":     ep_count,
                "mm/episode/inventory": info.get("inventory", 0),
                "mm/episode/fill_rate": fill_rate,
                "global_step":          step,
            }, step=step)
            obs, _ = env.reset()
            ep_return = 0.0
            ep_len    = 0
            ep_fills  = 0

        if step < cfg["learning_starts"]:
            continue

        # ── Sample batch + expert actions for BC ─────────────────
        raw_batch = rb.sample(cfg["batch_size"])

        expert_idx    = np.random.randint(0, n_expert, size=cfg["batch_size"])
        expert_acts_b = torch.as_tensor(
            expert_acts[expert_idx], dtype=torch.float32
        )

        batch = prepare_batch(raw_batch, cfg, device, expert_acts=expert_acts_b)

        c_loss = update_critic(
            critic, critic_opt, critic_target, actor_target, batch, cfg
        )
        a_loss = None
        if step % cfg["policy_freq"] == 0:
            a_loss = update_actor(actor, actor_opt, critic, batch, cfg, bc_coef)
            soft_update(actor,  actor_target,  cfg["tau"])
            soft_update(critic, critic_target, cfg["tau"])
            bc_coef = max(bc_coef * cfg["bc_decay"], cfg.get("min_bc_coef", 0.05))
            policy_updates += 1

        if step % cfg["log_interval"] == 0:
            _log_step(step, c_loss, a_loss, bc_coef)

    # ── Save ──────────────────────────────────────────────────────
    torch.save({
        "actor":   actor.state_dict(),
        "critic":  critic.state_dict(),
        "cfg":     cfg,
        "train_t": train_t,
    }, ckpt_path)
    wandb.save(ckpt_path)
    print(f"\nCheckpoint → {ckpt_path}")

    run.finish()
    return ckpt_path


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",        default="MSFT")
    parser.add_argument("--total-steps",   type=int,   default=DEFAULT_CFG["total_steps"])
    parser.add_argument("--batch-size",    type=int,   default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--actor-lr",      type=float, default=DEFAULT_CFG["actor_lr"])
    parser.add_argument("--critic-lr",     type=float, default=DEFAULT_CFG["critic_lr"])
    parser.add_argument("--bc-coef",       type=float, default=DEFAULT_CFG["bc_coef"])
    parser.add_argument("--seed",          type=int,   default=DEFAULT_CFG["seed"])
    parser.add_argument("--run-name",      default=None)
    args = parser.parse_args()

    cfg = {
        **DEFAULT_CFG,
        "total_steps": args.total_steps,
        "batch_size":  args.batch_size,
        "actor_lr":    args.actor_lr,
        "critic_lr":   args.critic_lr,
        "bc_coef":     args.bc_coef,
        "seed":        args.seed,
    }
    train(cfg, ticker=args.ticker, run_name=args.run_name)

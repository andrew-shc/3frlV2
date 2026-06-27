"""
D4PG-QR training (CleanRL style, single file).

Usage:
  python -m gv.train [overrides]

Key overrides (passed as --key=value):
  --objective=cvar|var|mean_std
  --tc_ratio=0.01
  --market=gbm|heston|vg
  --hedge_maturity_days=30
  --total_timesteps=1000000
  --seed=42
"""
from __future__ import annotations

import argparse
import copy
import time
import random

import numpy as np
import torch
import torch.optim as optim
import wandb

from .config import HedgeConfig
from .env.hedging_env import HedgingEnv
from .agent.networks import (
    Actor, QRCritic, OUNoise, ReplayBuffer,
    qr_huber_loss, actor_loss,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> HedgeConfig:
    p = argparse.ArgumentParser()
    cfg = HedgeConfig()
    for field, val in cfg.__dict__.items():
        t = type(val)
        if t == bool:
            p.add_argument(f"--{field}", type=lambda x: x.lower() == "true",
                           default=val)
        else:
            p.add_argument(f"--{field}", type=t, default=val)
    ns = p.parse_args()
    return HedgeConfig(**vars(ns))


# ── Main ──────────────────────────────────────────────────────────────────────

def train(cfg: HedgeConfig | None = None):
    if cfg is None:
        cfg = parse_args()

    # ── Seeding ──────────────────────────────────────────────────────────────
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Objective: {cfg.objective} | Market: {cfg.market}")

    # ── W&B ──────────────────────────────────────────────────────────────────
    run = wandb.init(
        project=cfg.wandb_project,
        name=f"{cfg.run_name}_{cfg.objective}_k{cfg.tc_ratio}_T{cfg.hedge_maturity_days}d",
        config=cfg.__dict__,
        save_code=True,
        mode=cfg.wandb_mode,
    )

    # ── Env ───────────────────────────────────────────────────────────────────
    env = HedgingEnv(cfg, seed=cfg.seed)

    obs_dim = env.observation_space.shape[0]   # 5
    act_dim = env.action_space.shape[0]         # 1

    # ── Networks ──────────────────────────────────────────────────────────────
    actor         = Actor(obs_dim, cfg.hidden_dim).to(device)
    critic        = QRCritic(obs_dim, act_dim, cfg.n_quantiles, cfg.hidden_dim).to(device)
    actor_target  = copy.deepcopy(actor)
    critic_target = copy.deepcopy(critic)
    for p in (*actor_target.parameters(), *critic_target.parameters()):
        p.requires_grad_(False)

    actor_opt  = optim.Adam(actor.parameters(),  lr=cfg.actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    buf   = ReplayBuffer(obs_dim, act_dim, cfg.buffer_size, device)
    noise = OUNoise(act_dim, theta=cfg.ou_theta, sigma=cfg.ou_sigma)

    # ── Training loop ─────────────────────────────────────────────────────────
    obs, _ = env.reset()
    noise.reset()

    ep_returns: list[float] = []
    ep_ret = 0.0
    ep_len = 0
    start  = time.time()

    for global_step in range(1, cfg.total_timesteps + 1):

        # ── Action ───────────────────────────────────────────────────────────
        if global_step < cfg.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t  = torch.as_tensor(obs, dtype=torch.float32,
                                         device=device).unsqueeze(0)
                action = actor(obs_t).cpu().numpy()[0]
            action = np.clip(action + noise.sample(), 0.0, 1.0)

        # ── Step ─────────────────────────────────────────────────────────────
        next_obs, reward, done, _, info = env.step(action)
        ep_ret += reward
        ep_len += 1

        # gymnasium stores terminal obs in info["final_observation"] if truncated
        real_next_obs = next_obs if not done else obs   # terminal is dummy zeros
        buf.add(obs, action, reward, real_next_obs, float(done))
        obs = next_obs

        if done:
            ep_returns.append(ep_ret)
            obs, _ = env.reset()
            noise.reset()
            ep_ret = 0.0
            ep_len = 0

        # ── Update ───────────────────────────────────────────────────────────
        if (global_step >= cfg.learning_starts
                and global_step % cfg.train_freq == 0
                and len(buf) >= cfg.batch_size):

            s, a, r, s2, d = buf.sample(cfg.batch_size)

            # ── Critic update ─────────────────────────────────────────────
            with torch.no_grad():
                a2  = actor_target(s2)
                z2  = critic_target(s2, a2)                    # (B, N)
                target = r + cfg.gamma * (1.0 - d) * z2        # (B, N)

            z_pred = critic(s, a)                               # (B, N)
            c_loss = qr_huber_loss(z_pred, target, critic.taus)

            critic_opt.zero_grad()
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()

            # ── Actor update ──────────────────────────────────────────────
            a_new = actor(s)
            z_new = critic(s, a_new)                           # (B, N)
            a_loss = actor_loss(z_new, critic.taus,
                                cfg.objective, cfg.risk_lambda, cfg.risk_alpha)

            actor_opt.zero_grad()
            a_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_opt.step()

            # ── Soft target update ────────────────────────────────────────
            τ = cfg.tau_soft
            for p, tp in zip(actor.parameters(), actor_target.parameters()):
                tp.data.copy_(τ * p.data + (1 - τ) * tp.data)
            for p, tp in zip(critic.parameters(), critic_target.parameters()):
                tp.data.copy_(τ * p.data + (1 - τ) * tp.data)

        # ── Logging ──────────────────────────────────────────────────────────
        if global_step % 5_000 == 0 and ep_returns:
            recent = ep_returns[-50:]
            elapsed = time.time() - start
            sps = global_step / elapsed
            log = {
                "train/ep_return_mean": np.mean(recent),
                "train/ep_return_std":  np.std(recent),
                "train/ep_return_min":  np.min(recent),
                "train/ep_return_p05":  np.percentile(recent, 5),
                "train/SPS": sps,
                "global_step": global_step,
            }
            wandb.log(log, step=global_step)
            print(f"step={global_step:>8d} | "
                  f"ret={np.mean(recent):+.4f} ± {np.std(recent):.4f} | "
                  f"SPS={sps:.0f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    import os, pathlib
    ckpt_dir = pathlib.Path("checkpoints/oh")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{run.name}.pt"
    torch.save({
        "actor":        actor.state_dict(),
        "critic":       critic.state_dict(),
        "actor_target": actor_target.state_dict(),
        "critic_target":critic_target.state_dict(),
        "cfg":          cfg.__dict__,
    }, ckpt_path)
    print(f"Saved checkpoint → {ckpt_path}")
    wandb.save(str(ckpt_path))
    run.finish()
    return actor, critic


if __name__ == "__main__":
    train()

"""
Tucker-DDPG training orchestration.

Usage:
    python -m pm.model.ddpg
    python -m pm.model.ddpg --total-steps 20000 --batch-size 64
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

from pm.data.loader import load_indicators
from pm.model.actor import Actor
from pm.model.config import DEFAULT_CFG
from pm.model.critic import Critic
from pm.model.env import PortfolioDataset
from pm.model.trainer import prepare_batch, select_action, soft_update, update_actor, update_critic

load_dotenv()


def _init_wandb(cfg: dict, run_name: str | None, run_group: str | None):
    return wandb.init(
        project=os.getenv("WANDB_PROJECT", "3frlV2"),
        name=run_name or f"tucker-ddpg-{int(time.time())}",
        group=run_group,
        tags=["ddpg", "tucker", "portfolio"],
        config=cfg,
        save_code=True,
    )


def _build_env_and_nets(cfg: dict, indicators: dict, device: torch.device):
    env = PortfolioDataset(
        indicators=indicators,
        m_days=cfg["m_days"],
        cor_window=cfg["cor_window"],
        cost_bps=cfg["cost_bps"],
    )
    actor  = Actor(cfg).to(device)
    critic = Critic(cfg).to(device)
    return env, actor, critic, copy.deepcopy(actor), copy.deepcopy(critic)


def _log_step(global_step: int, critic_loss: float, actor_loss: float | None, cfg: dict) -> None:
    log: dict = {"loss/critic": critic_loss, "global_step": global_step}
    if actor_loss is not None:
        log["loss/actor"] = actor_loss
    wandb.log(log, step=global_step)


def train(
    cfg: dict,
    parquet_path: str,
    run_name: str | None = None,
    run_group: str | None = None,
    indicators: dict | None = None,
    ckpt_path: str = "tucker_ddpg_final.pt",
) -> float:
    """Run training; returns mean per-step reward over last half of post-warmup steps."""
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run = _init_wandb(cfg, run_name, run_group)

    if indicators is None:
        indicators, _, _ = load_indicators(parquet_path, cfg["n_assets"])

    actual_n = next(iter(indicators.values())).shape[1]
    T_bars   = next(iter(indicators.values())).shape[0]
    if actual_n != cfg["n_assets"]:
        cfg = {**cfg, "n_assets": actual_n, "n_actions": actual_n + 1}
        wandb.config.update({"n_assets": actual_n, "n_actions": actual_n + 1})

    env, actor, critic, actor_target, critic_target = _build_env_and_nets(cfg, indicators, device)
    k, n, m = env.k, env.n, env.m
    print(f"Env: T={env.T} bars, m={m}, steps/episode={env.T - env.m}")

    actor_opt  = optim.Adam(actor.parameters(),  lr=cfg["actor_lr"])
    critic_opt = optim.Adam(critic.parameters(), lr=cfg["critic_lr"])

    from pm.eval.shapes import log_shapes
    log_shapes(actor, critic, cfg, device, T=T_bars)

    rb = ReplayBuffer(
        buffer_size=cfg["buffer_size"],
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        handle_timeout_termination=False,
    )

    obs, _    = env.reset(seed=cfg["seed"])
    ep_return = 0.0
    ep_len    = 0
    ep_count  = 0
    step_rewards: list[float] = []

    for global_step in range(1, cfg["total_steps"] + 1):
        action = select_action(
            actor, env, obs, k, n, m, device,
            cfg["exploration_noise"],
            is_exploring=(global_step < cfg["learning_starts"]),
        )
        next_obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        ep_len    += 1

        rb.add(obs, next_obs, action, reward, terminated, [info])
        obs = next_obs

        if terminated or truncated:
            ep_count += 1
            wandb.log({"episode/return": ep_return, "episode/length": ep_len,
                       "episode/count": ep_count, "global_step": global_step}, step=global_step)
            obs, _ = env.reset()
            ep_return = 0.0
            ep_len    = 0

        if global_step < cfg["learning_starts"]:
            continue

        step_rewards.append(reward)
        batch = prepare_batch(rb.sample(cfg["batch_size"]), k, n, m, device)
        critic_loss = update_critic(critic, critic_opt, critic_target, actor_target, batch, cfg)

        actor_loss = None
        if global_step % cfg["policy_freq"] == 0:
            actor_loss = update_actor(actor, actor_opt, critic, batch)
            soft_update(actor,  actor_target,  cfg["tau"])
            soft_update(critic, critic_target, cfg["tau"])

        if global_step % cfg["log_interval"] == 0:
            _log_step(global_step, critic_loss, actor_loss, cfg)

    half = max(1, len(step_rewards) // 2)
    mean_reward = float(np.mean(step_rewards[-half:])) if step_rewards else float("-inf")
    wandb.log({"final/mean_step_reward": mean_reward})
    print(f"Mean step reward (last half): {mean_reward:.6f}")

    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "cfg": cfg}, ckpt_path)
    wandb.save(ckpt_path)
    print(f"Checkpoint → {ckpt_path}")

    run.finish()
    return mean_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",         default="dataset/hist/sp500_5min_5yr.parquet")
    parser.add_argument("--total-steps",     type=int,   default=DEFAULT_CFG["total_steps"])
    parser.add_argument("--batch-size",      type=int,   default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--n-assets",        type=int,   default=DEFAULT_CFG["n_assets"])
    parser.add_argument("--m-days",          type=int,   default=DEFAULT_CFG["m_days"])
    parser.add_argument("--cor-window",      type=int,   default=DEFAULT_CFG["cor_window"])
    parser.add_argument("--tucker-ranks",    type=int,   nargs=4,
                        default=DEFAULT_CFG["tucker_ranks"], metavar=("R1", "R2", "R3", "R4"))
    parser.add_argument("--fc-hidden",       type=int,   default=DEFAULT_CFG["fc_hidden"])
    parser.add_argument("--actor-lr",        type=float, default=DEFAULT_CFG["actor_lr"])
    parser.add_argument("--critic-lr",       type=float, default=DEFAULT_CFG["critic_lr"])
    parser.add_argument("--seed",            type=int,   default=DEFAULT_CFG["seed"])
    args = parser.parse_args()

    cfg = {
        **DEFAULT_CFG,
        "total_steps":  args.total_steps,
        "batch_size":   args.batch_size,
        "n_assets":     args.n_assets,
        "m_days":       args.m_days,
        "cor_window":   args.cor_window,
        "tucker_ranks": args.tucker_ranks,
        "n_actions":    args.n_assets + 1,
        "fc_hidden":    args.fc_hidden,
        "actor_lr":     args.actor_lr,
        "critic_lr":    args.critic_lr,
        "seed":         args.seed,
    }
    train(cfg, parquet_path=args.parquet)

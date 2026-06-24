"""
Modular DDPG training-step functions.

There is no off-the-shelf RL library that fits this setup cleanly:
  - SB3 has DDPG but couples it to its own feature extraction
  - TorchRL is modular but requires significant wrapper work for Tucker obs
We keep it explicit here so every step is readable and testable individually.

Public API
----------
  prepare_batch(batch, k, n, m, device)     → PreparedBatch
  select_action(actor, env, obs, ...)        → np.ndarray
  update_critic(critic, ..., batch, cfg)     → float (loss)
  update_actor(actor, ..., batch)            → float (loss)
  soft_update(source, target, tau)           → None
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from pm.model.env import PortfolioDataset


@dataclass
class PreparedBatch:
    bv:      torch.Tensor   # obs V_t       [B, k, n, m]
    bc:      torch.Tensor   # obs Cor_t     [B, k, n, n]
    bvn:     torch.Tensor   # next V_t
    bcn:     torch.Tensor   # next Cor_t
    actions: torch.Tensor   # [B, n_actions]
    rewards: torch.Tensor   # [B, 1]
    dones:   torch.Tensor   # [B, 1]


def prepare_batch(batch, k: int, n: int, m: int, device: torch.device) -> PreparedBatch:
    bv,  bc  = PortfolioDataset.split_obs(batch.observations,      k, n, m)
    bvn, bcn = PortfolioDataset.split_obs(batch.next_observations, k, n, m)
    return PreparedBatch(
        bv=bv.to(device),   bc=bc.to(device),
        bvn=bvn.to(device), bcn=bcn.to(device),
        actions=batch.actions.to(device),
        rewards=batch.rewards.to(device),
        dones=batch.dones.to(device),
    )


def select_action(
    actor,
    env: PortfolioDataset,
    obs: np.ndarray,
    k: int,
    n: int,
    m: int,
    device: torch.device,
    exploration_noise: float,
    is_exploring: bool,
) -> np.ndarray:
    if is_exploring:
        return env.action_space.sample()
    with torch.no_grad():
        v_t, cor_t = PortfolioDataset.split_obs(obs, k, n, m)
        action = actor(v_t.to(device), cor_t.to(device)).cpu().numpy().squeeze(0)
    noise = np.random.normal(0, exploration_noise, size=action.shape)
    action = np.clip(action + noise, 0.0, 1.0)
    action /= action.sum() + 1e-8
    return action


def update_critic(
    critic,
    critic_opt: torch.optim.Optimizer,
    critic_target,
    actor_target,
    batch: PreparedBatch,
    cfg: dict,
) -> float:
    with torch.no_grad():
        next_a   = actor_target(batch.bvn, batch.bcn)
        q_target = batch.rewards + cfg["gamma"] * (1.0 - batch.dones) * critic_target(batch.bvn, batch.bcn, next_a)

    loss = F.mse_loss(critic(batch.bv, batch.bc, batch.actions), q_target)
    critic_opt.zero_grad()
    loss.backward()
    critic_opt.step()
    return loss.item()


def update_actor(
    actor,
    actor_opt: torch.optim.Optimizer,
    critic,
    batch: PreparedBatch,
) -> float:
    loss = -critic(batch.bv, batch.bc, actor(batch.bv, batch.bc)).mean()
    actor_opt.zero_grad()
    loss.backward()
    actor_opt.step()
    return loss.item()


def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    for sp, tp in zip(source.parameters(), target.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

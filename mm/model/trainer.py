"""
TD3 + Behaviour Cloning training for IMM.

TD3 (Fujimoto et al. 2018) with an additive BC loss (paper §4.2):
  π* = argmax_π E[Q(s, π(s))] + λ · (-L_BC)
  L_BC = MSE(π(s), a_expert)
  λ decays with each policy update step.

Public API
----------
  PreparedBatch                                dataclass
  prepare_batch(batch, cfg, device)            → PreparedBatch
  select_action(actor, obs, cfg, device, ...)  → np.ndarray
  update_critic(critic, ..., batch, cfg)       → float
  update_actor(actor, ..., batch, cfg)         → float
  soft_update(source, target, tau)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from mm.model.env import MarketMakingEnv


@dataclass
class PreparedBatch:
    x:       torch.Tensor   # [B, F, L]
    s_p:     torch.Tensor   # [B, priv_dim]
    xn:      torch.Tensor   # next x
    s_pn:    torch.Tensor   # next s_p
    actions: torch.Tensor   # [B, action_dim]
    rewards: torch.Tensor   # [B, 1]
    dones:   torch.Tensor   # [B, 1]
    # Expert actions (LTIIC), may be None when not using BC
    expert_actions: torch.Tensor | None = None


def prepare_batch(
    batch,
    cfg: dict,
    device: torch.device,
    expert_acts: torch.Tensor | None = None,
) -> PreparedBatch:
    f, l = cfg["f_dim"], cfg["tcsa_seq_len"]
    priv = 1 + 4 * cfg["n_quote_levels"] + 7

    x,   s_p  = MarketMakingEnv.split_obs(batch.observations,      f, l, priv)
    xn,  s_pn = MarketMakingEnv.split_obs(batch.next_observations, f, l, priv)

    expert = None
    if expert_acts is not None:
        expert = expert_acts.to(device) if isinstance(expert_acts, torch.Tensor) \
                 else torch.as_tensor(expert_acts, dtype=torch.float32).to(device)

    return PreparedBatch(
        x=x.to(device),          s_p=s_p.to(device),
        xn=xn.to(device),        s_pn=s_pn.to(device),
        actions=torch.as_tensor(batch.actions,  dtype=torch.float32).to(device),
        rewards=torch.as_tensor(batch.rewards,  dtype=torch.float32).to(device),
        dones=torch.as_tensor(batch.dones,    dtype=torch.float32).to(device),
        expert_actions=expert,
    )


@torch.no_grad()
def select_action(
    actor,
    obs: np.ndarray,
    cfg: dict,
    device: torch.device,
    exploration_noise: float = 0.0,
    is_exploring: bool = False,
    action_space=None,
) -> np.ndarray:
    if is_exploring and action_space is not None:
        return action_space.sample()

    f, l = cfg["f_dim"], cfg["tcsa_seq_len"]
    priv = 1 + 4 * cfg["n_quote_levels"] + 7
    x, s_p = MarketMakingEnv.split_obs(obs, f, l, priv)
    action = actor(x.to(device), s_p.to(device)).cpu().numpy().squeeze(0)

    if exploration_noise > 0:
        noise = np.random.normal(0, exploration_noise, size=action.shape)
        action = np.clip(action + noise, -1.0, 1.0)
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
        # Target policy smoothing
        noise = torch.randn_like(batch.actions) * cfg["target_noise"]
        noise = noise.clamp(-cfg["noise_clip"], cfg["noise_clip"])
        next_a = (actor_target(batch.xn, batch.s_pn) + noise).clamp(-1.0, 1.0)

        # Encode next state
        next_state = actor_target.encode_state(batch.xn, batch.s_pn)
        q1_t, q2_t = critic_target(next_state, next_a)
        q_target = batch.rewards + cfg["gamma"] * (1.0 - batch.dones) * torch.min(q1_t, q2_t)

    state = actor_target.encode_state(batch.x, batch.s_p)
    q1, q2 = critic(state, batch.actions)
    loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
    critic_opt.zero_grad()
    loss.backward()
    critic_opt.step()
    return loss.item()


def update_actor(
    actor,
    actor_opt: torch.optim.Optimizer,
    critic,
    batch: PreparedBatch,
    cfg: dict,
    bc_coef: float,
) -> float:
    state = actor.encode_state(batch.x, batch.s_p)
    pred_a = actor(batch.x, batch.s_p)

    q_loss = -critic.q_min(state, pred_a).mean()

    if bc_coef > 0.0 and batch.expert_actions is not None:
        bc_loss = F.mse_loss(pred_a, batch.expert_actions)
        loss = q_loss + bc_coef * bc_loss
    else:
        loss = q_loss

    actor_opt.zero_grad()
    loss.backward()
    actor_opt.step()
    return loss.item()


def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    for sp, tp in zip(source.parameters(), target.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# ------------------------------------------------------------------
# LTIIC expert strategy (paper §4.2)
# ------------------------------------------------------------------

def ltiic_action(
    z: float,            # current inventory
    y_hat: float,        # short-term signal (e.g. y^20)
    best_bid: float,     # current LOB best bid price
    best_ask: float,     # current LOB best ask price
    cfg: dict,
    a: float = 0.0,      # extra ticks OUTSIDE best bid/ask (0 = quote at best)
    b: float = 0.5,      # inventory skew coefficient
    c: float = 1.0,      # signal adjustment
    d: float = 5.0,      # inventory limit for one-sided posting
) -> np.ndarray:
    """
    Return normalised action vector matching the actor's output format.
    Used to generate expert demonstrations for BC.

    Quotes are anchored to best_bid/ask (not mid), so a=0 quotes at the
    touch and gets filled as soon as the queue ahead is consumed.

    Inventory skew: negative b*z shifts both quotes down when long (z>0),
    making the ask more competitive (more sales) and the bid less aggressive
    (fewer purchases) — Avellaneda-Stoikov reservation-price direction.
    """
    tick = cfg["tick_size"]
    K = cfg["n_quote_levels"]
    max_offset = cfg["max_offset_ticks"]
    max_spread = cfg["max_spread_ticks"]

    mid = (best_bid + best_ask) / 2.0

    # Reservation-price skew: negative when long → shifts both quotes down
    skew = -b * z * tick + c * y_hat * tick
    bid_q = best_bid - a * tick + skew
    ask_q = best_ask + a * tick + skew

    # Clamp: never quote inside the spread (passive fills need LOB-level matching)
    bid_q = min(bid_q, best_bid)
    ask_q = max(ask_q, best_ask)

    # Encode to action format
    actual_mid = (bid_q + ask_q) / 2.0
    mid_star = np.clip((actual_mid - mid) / (max_offset * tick + 1e-8), -1.0, 1.0)

    # delta_outside = average ticks outside best bid/ask
    ask_outside = max(0.0, ask_q - best_ask) / tick
    bid_outside = max(0.0, best_bid - bid_q) / tick
    delta_outside_ticks = (ask_outside + bid_outside) / 2.0
    spread_norm = np.clip(delta_outside_ticks / (max_spread + 1e-8), 0.0, 1.0)

    # One-sided posting constraint
    if abs(z) > d:
        if z > 0:
            phi_bid = np.zeros(K, dtype=np.float32)
            phi_ask = np.ones(K, dtype=np.float32) / K
        else:
            phi_bid = np.ones(K, dtype=np.float32) / K
            phi_ask = np.zeros(K, dtype=np.float32)
    else:
        phi_bid = np.ones(K, dtype=np.float32) / K
        phi_ask = np.ones(K, dtype=np.float32) / K

    return np.concatenate([
        [np.clip(mid_star, -1.0, 1.0)],
        [np.clip(spread_norm, 0.0, 1.0)],
        phi_bid,
        phi_ask,
    ]).astype(np.float32)

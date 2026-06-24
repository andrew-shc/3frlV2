"""Architecture tensor-shape tracing and W&B logging."""
from __future__ import annotations

import numpy as np
import torch
import wandb

from pm.model.actor import Actor
from pm.model.critic import Critic


def _fmt(shape: list[int]) -> str:
    return "[" + ", ".join(str(d) for d in shape) + "]"


def log_shapes(
    actor: Actor,
    critic: Critic,
    cfg: dict,
    device: torch.device,
    T: int,
) -> None:
    """
    Trace every tensor through the full pipeline and log to W&B.
    Prints each stage; also writes a structured W&B Table.
    """
    k      = cfg["n_indicators"]
    n      = cfg["n_assets"]
    m      = cfg["m_days"]
    ranks  = cfg["tucker_ranks"]
    n_act  = cfg["n_actions"]
    flat   = actor.feat.output_dim
    hidden = cfg["fc_hidden"]

    SEP  = "═" * 62
    SEP2 = "─" * 62

    obs_v_size = k * n * m
    obs_c_size = k * n * n
    obs_dim = obs_v_size + obs_c_size

    print(f"\n{SEP}")
    print("  Tucker-DDPG  —  Full Tensor Shape Trace")
    print(SEP)
    print(f"\n  [DATA]  T={T} bars,  n={n} assets,  k={k} indicators")
    print(f"\n  [ENV OBS]")
    print(f"  V_t   {_fmt([k, n, m])}  → {obs_v_size:,} values")
    print(f"  Cor_t {_fmt([k, n, n])}  → {obs_c_size:,} values")
    print(f"  obs flat                        [{obs_dim:,}]")
    print(f"\n  [FEATURE EXTRACTOR]")
    print(f"  {SEP2}")

    v_t   = torch.zeros(1, k, n, m, device=device)
    cor_t = torch.zeros(1, k, n, n, device=device)

    with torch.no_grad():
        feat_shapes = actor.feat.trace_shapes(v_t, cor_t)

    for label, shape in feat_shapes.items():
        if label in ("V_t", "Cor_t"):
            continue
        dims = shape[1:]
        print(f"  {label:<34}  {_fmt(dims):28s}  ({int(np.prod(dims)):,} values)")

    print(f"\n  [ACTOR]  flatten → FC({flat:,}→{hidden}) → FC({hidden}→{n_act}) + softmax")
    with torch.no_grad():
        actor_out = actor(v_t, cor_t)
    print(f"  actor output (verified)         {_fmt(list(actor_out.shape[1:]))}")

    critic_in = flat + n_act
    print(f"\n  [CRITIC]  concat → FC({critic_in:,}→{hidden}) → FC({hidden}→1)")
    dummy_a = torch.zeros(1, n_act, device=device)
    with torch.no_grad():
        critic_out = critic(v_t, cor_t, dummy_a)
    print(f"  critic output (verified)        {_fmt(list(critic_out.shape[1:]))}")

    def count_params(mod: torch.nn.Module) -> int:
        return sum(p.numel() for p in mod.parameters() if p.requires_grad)

    a_params = count_params(actor)
    c_params = count_params(critic)
    print(f"\n  Actor  params: {a_params:>12,}")
    print(f"  Critic params: {c_params:>12,}")
    print(f"  Tucker factors (actor):  {count_params(actor.feat.tucker):>8,}")
    print(f"  Tucker factors (critic): {count_params(critic.feat.tucker):>8,}")
    print(f"\n{SEP}\n")

    rows = []
    for label, shape in feat_shapes.items():
        dims = shape[1:]
        rows.append([label, _fmt(shape), _fmt(dims), int(np.prod(dims))])
    rows.append(["actor output (softmax)", _fmt(list(actor_out.shape)),
                 _fmt(list(actor_out.shape[1:])), int(np.prod(actor_out.shape[1:]))])
    rows.append(["critic output (Q-value)", _fmt(list(critic_out.shape)),
                 _fmt(list(critic_out.shape[1:])), 1])

    wandb.log({
        "architecture/tensor_shapes": wandb.Table(
            columns=["layer", "shape (w/ batch)", "shape (no batch)", "elements/sample"],
            data=rows,
        ),
        "architecture/actor_params":        a_params,
        "architecture/critic_params":       c_params,
        "architecture/obs_dim":             obs_dim,
        "architecture/flat_dim":            flat,
        "architecture/tucker_rank_product": int(np.prod(ranks)),
    })

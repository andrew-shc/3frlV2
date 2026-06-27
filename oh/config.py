from dataclasses import dataclass, field


@dataclass
class HedgeConfig:
    # ── Market ────────────────────────────────────────────────────────────────
    S0: float = 100.0
    r: float = 0.0
    sigma: float = 0.2          # GBM / reference vol

    # ── Client-option flow ────────────────────────────────────────────────────
    client_lambda: float = 1.0  # Poisson arrivals per day
    client_maturity_days: int = 60
    contract_size: int = 100    # shares per contract

    # ── Hedging option ────────────────────────────────────────────────────────
    hedge_maturity_days: int = 30   # T_hedge in trading days

    # ── Episode ───────────────────────────────────────────────────────────────
    episode_len: int = 30       # trading days
    tc_ratio: float = 0.01      # κ: transaction cost as fraction of option value

    # ── Algorithm ─────────────────────────────────────────────────────────────
    n_quantiles: int = 32
    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    tau_soft: float = 0.005     # Polyak target update
    gamma: float = 1.0          # discount (=1 for short horizon)

    # ── Objective ─────────────────────────────────────────────────────────────
    objective: str = "cvar"     # "mean_std" | "var" | "cvar"
    risk_lambda: float = 1.0    # λ for mean−λ·std
    risk_alpha: float = 0.05    # α for VaR/CVaR tail level

    # ── OU exploration noise ──────────────────────────────────────────────────
    ou_theta: float = 0.15
    ou_sigma: float = 0.2

    # ── Training ──────────────────────────────────────────────────────────────
    total_timesteps: int = 1_000_000
    learning_starts: int = 10_000
    train_freq: int = 1
    market: str = "gbm"         # "gbm" | "heston" | "vg"
    seed: int = 42

    # ── Heston params ─────────────────────────────────────────────────────────
    heston_kappa: float = 2.0
    heston_theta: float = 0.04
    heston_xi: float = 0.3
    heston_rho: float = -0.7
    heston_v0: float = 0.04

    # ── Variance-Gamma params ─────────────────────────────────────────────────
    vg_sigma: float = 0.2
    vg_theta: float = -0.1
    vg_nu: float = 0.3

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_project: str = "3frlV2_gv"
    run_name: str = "d4pg_qr"
    wandb_mode: str = "online"   # "online" | "offline" | "disabled"

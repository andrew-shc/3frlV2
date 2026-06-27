"""Default hyperparameter configuration for IMM (TD3 + BC)."""

DEFAULT_CFG: dict = {
    # --- LOB / data ---
    "n_levels": 10,          # quote levels per side tracked in state
    "lob_levels": 10,        # levels in LOBSTER data used for features
    "tcsa_seq_len": 50,      # L: TCSA temporal window (ticks)
    "tick_size": 0.01,
    "total_volume": 20,      # V: fixed total volume per side per step
    "episode_ticks": 3_600,  # episode length in ticks (~5 min at AMZN tick rate)
    # --- architecture ---
    "f_dim": 184,            # F: feature dim per tick (LOB + lookbacks)
    "tcsa_channels": 64,     # TCN hidden channels
    "tcsa_layers": 4,        # TCN dilation layers
    "tcsa_out_dim": 64,      # s^m dimension
    "sl_hidden": 128,        # SL MLP hidden size
    "n_horizons": 4,         # number of signal horizons (20/120/240/600)
    "fc_hidden": 256,        # actor/critic FC hidden size
    # --- action ---
    "n_quote_levels": 2,     # K: levels per side the agent quotes
    "max_spread_ticks": 20,  # clamp for δ* output
    "max_offset_ticks": 10,  # clamp for m* output
    # --- RL / training ---
    "total_steps": 300_000,
    "learning_starts": 2_000,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "sl_lr": 1e-3,
    "exploration_noise": 0.1,
    "policy_freq": 2,
    "buffer_size": 100_000,
    "target_noise": 0.02,
    "noise_clip": 0.05,
    # --- BC (imitation) ---
    "bc_coef": 1.0,          # λ: initial BC loss weight (higher to stay near LTIIC)
    "bc_decay": 0.999995,    # very slow decay; ~0.47 at 300k steps
    # --- reward ---
    "inventory_limit": 10,   # C: threshold for inventory penalty
    "inventory_penalty": 0.005,  # halved so inventory fear doesn't dominate
    "fill_bonus": 0.01,      # 10x increase so fills are worth more than penalties
    # --- env ---
    "cost_bps": 0.0,
    "min_bc_coef": 0.05,     # BC floor — never let imitation fully vanish
    # --- misc ---
    "seed": 42,
    "log_interval": 100,
    "train_split": 0.8,   # fraction of ticks used for training; rest = held-out test
}

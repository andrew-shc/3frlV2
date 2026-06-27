"""Default hyperparameter configuration for Tucker-DDPG."""

DEFAULT_CFG: dict = {
    # architecture
    "n_assets": 29,
    "m_days": 28,
    "cor_window": 50,
    "n_indicators": 4,
    "conv_filters": 32,
    "tucker_ranks": [8, 8, 8, 8],
    "fc_hidden": 256,
    "n_actions": 30,  # n_assets + 1 cash
    # training
    "total_steps": 20_000,
    "learning_starts": 500,
    "batch_size": 64,
    "gamma": 0.99,
    "tau": 0.005,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "exploration_noise": 0.1,
    "policy_freq": 2,
    "buffer_size": 10_000,
    # env
    "cost_bps": 5.0,
    # misc
    "seed": 42,
    "log_interval": 50,
}

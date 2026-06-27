import numpy as np


def gbm_step(S: float, r: float, sigma: float, dt: float,
             rng: np.random.Generator) -> float:
    Z = rng.standard_normal()
    return S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)


def gbm_paths(S0: float, r: float, sigma: float, dt: float,
              n_steps: int, n_paths: int,
              rng: np.random.Generator) -> np.ndarray:
    """Returns (n_steps+1, n_paths) price matrix."""
    Z = rng.standard_normal((n_steps, n_paths))
    log_ret = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_S = np.concatenate([np.zeros((1, n_paths)),
                             np.cumsum(log_ret, axis=0)], axis=0)
    return S0 * np.exp(log_S)

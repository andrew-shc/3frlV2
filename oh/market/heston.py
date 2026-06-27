import numpy as np


def heston_step(S: float, v: float, r: float,
                kappa: float, theta: float, xi: float, rho: float,
                dt: float, rng: np.random.Generator):
    """Euler-Maruyama step for the Heston (1993) model.
    Returns (S_{t+dt}, v_{t+dt}).
    Uses the absorption fix v → max(v, 0) to prevent negative variance.
    """
    Z1 = rng.standard_normal()
    Z2 = rho * Z1 + np.sqrt(max(1 - rho**2, 0.0)) * rng.standard_normal()
    v_pos = max(v, 0.0)
    sqrt_v_dt = np.sqrt(v_pos * dt)
    v_new = max(v + kappa * (theta - v_pos) * dt + xi * sqrt_v_dt * Z2, 0.0)
    S_new = S * np.exp((r - 0.5 * v_pos) * dt + sqrt_v_dt * Z1)
    return S_new, v_new


def heston_paths(S0: float, v0: float, r: float,
                 kappa: float, theta: float, xi: float, rho: float,
                 dt: float, n_steps: int, n_paths: int,
                 rng: np.random.Generator):
    """Vectorised Euler-Maruyama for Heston.
    Returns S (n_steps+1, n_paths), v (n_steps+1, n_paths).
    """
    S = np.full(n_paths, S0, dtype=np.float64)
    v = np.full(n_paths, v0, dtype=np.float64)
    S_path = np.empty((n_steps + 1, n_paths))
    v_path = np.empty((n_steps + 1, n_paths))
    S_path[0] = S
    v_path[0] = v

    corr_mat = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(corr_mat)

    for i in range(n_steps):
        Z = rng.standard_normal((2, n_paths))
        W = L @ Z          # correlated (2, n_paths)
        W1, W2 = W[0], W[1]
        v_pos = np.maximum(v, 0.0)
        sqrt_v_dt = np.sqrt(v_pos * dt)
        v = np.maximum(v + kappa * (theta - v_pos) * dt + xi * sqrt_v_dt * W2, 0.0)
        S = S * np.exp((r - 0.5 * v_pos) * dt + sqrt_v_dt * W1)
        S_path[i + 1] = S
        v_path[i + 1] = v

    return S_path, v_path

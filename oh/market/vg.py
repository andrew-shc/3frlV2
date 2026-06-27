"""
Variance-Gamma process simulation.

Risk-neutral dynamics:
  S(t+dt) = S(t) * exp((r + omega)*dt + theta*dg + sigma*sqrt(dg)*Z)

where:
  dg  ~ Gamma(shape=dt/nu, scale=nu)   — VG time increment
  Z   ~ N(0,1) independent of dg
  omega = (1/nu) * log(1 - nu*theta - 0.5*nu*sigma^2)  — martingale correction

Parameters follow Madan, Carr & Chang (1998):
  sigma: vol of the subordinated BM
  theta: drift in subordinated BM  (< 0 gives left skew)
  nu:    variance rate of the Gamma clock
"""
from __future__ import annotations
import numpy as np


def _omega(sigma: float, theta: float, nu: float) -> float:
    arg = 1.0 - nu * theta - 0.5 * nu * sigma**2
    if arg <= 0:
        raise ValueError(f"VG params violate martingale condition: {arg=}")
    return np.log(arg) / nu


def vg_step(S: float, r: float, sigma: float, theta: float, nu: float,
            dt: float, rng: np.random.Generator) -> float:
    om = _omega(sigma, theta, nu)
    dg = rng.gamma(dt / nu, nu)
    Z = rng.standard_normal()
    dX = theta * dg + sigma * np.sqrt(dg) * Z
    return S * np.exp((r + om) * dt + dX)


def vg_paths(S0: float, r: float, sigma: float, theta: float, nu: float,
             dt: float, n_steps: int, n_paths: int,
             rng: np.random.Generator) -> np.ndarray:
    """Returns (n_steps+1, n_paths) price matrix."""
    om = _omega(sigma, theta, nu)
    dg = rng.gamma(dt / nu, nu, size=(n_steps, n_paths))
    Z  = rng.standard_normal((n_steps, n_paths))
    dX = theta * dg + sigma * np.sqrt(dg) * Z
    log_ret = (r + om) * dt + dX
    log_S = np.concatenate([np.zeros((1, n_paths)),
                             np.cumsum(log_ret, axis=0)], axis=0)
    return S0 * np.exp(log_S)

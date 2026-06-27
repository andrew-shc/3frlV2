"""
Analytical Black-Scholes formulas (vectorised via NumPy) and
QuantLib wrappers for Heston / Variance-Gamma pricing.

Convention: vega is ∂C/∂σ (per 1 unit of vol, i.e. ≈ per 100 vega pts).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ── Black-Scholes ─────────────────────────────────────────────────────────────

def _d1_d2(S, K, T, r, sigma):
    sigma = np.maximum(sigma, 1e-8)
    T = np.maximum(T, 1e-8)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs_call_price(S, K, T, r, sigma):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_call_intrinsic(S, K):
    return np.maximum(S - K, 0.0)


def bs_delta(S, K, T, r, sigma):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1)


def bs_gamma(S, K, T, r, sigma):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    sigma = np.maximum(sigma, 1e-8)
    T = np.maximum(T, 1e-8)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, T, r, sigma):
    """∂C/∂σ  (per unit of σ, NOT per vol-point)."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    T = np.maximum(T, 1e-8)
    return S * norm.pdf(d1) * np.sqrt(T)


def bs_implied_vol(price, S, K, T, r, tol=1e-6, max_iter=100):
    """Brentq implied vol from a call price (scalar)."""
    from scipy.optimize import brentq
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if price <= intrinsic:
        return 0.0
    try:
        iv = brentq(lambda s: bs_call_price(S, K, T, r, s) - price,
                    1e-4, 5.0, xtol=tol, maxiter=max_iter)
    except ValueError:
        iv = float(np.sqrt(abs(2 * np.log(S / K) / T))) if T > 0 else 0.2
    return iv


# ── QuantLib helpers ──────────────────────────────────────────────────────────

def _ql_today():
    import QuantLib as ql
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    return today


def _ql_maturity(today, T_years):
    import QuantLib as ql
    days = max(int(round(T_years * 365)), 1)
    return today + ql.Period(days, ql.Days)


def ql_heston_call(S, K, T, r, kappa, theta, xi, rho, v0):
    """Heston call price via QuantLib AnalyticHestonEngine (scalar)."""
    import QuantLib as ql
    today = _ql_today()
    mat = _ql_maturity(today, T)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, float(K))
    exercise = ql.EuropeanExercise(mat)
    option = ql.VanillaOption(payoff, exercise)

    spot = ql.QuoteHandle(ql.SimpleQuote(float(S)))
    rts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, float(r), ql.Actual365Fixed()))
    div = ql.YieldTermStructureHandle(
        ql.FlatForward(today, 0.0, ql.Actual365Fixed()))

    process = ql.HestonProcess(rts, div, spot,
                                float(v0), float(kappa),
                                float(theta), float(xi), float(rho))
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)
    option.setPricingEngine(engine)
    return option.NPV()


def ql_heston_greeks(S, K, T, r, kappa, theta, xi, rho, v0, bump=1e-2):
    """
    Heston Greeks via finite differences on ql_heston_call.
    Returns (price, delta, gamma, vega).
    """
    p = ql_heston_call(S, K, T, r, kappa, theta, xi, rho, v0)
    p_up = ql_heston_call(S + bump, K, T, r, kappa, theta, xi, rho, v0)
    p_dn = ql_heston_call(S - bump, K, T, r, kappa, theta, xi, rho, v0)
    delta = (p_up - p_dn) / (2 * bump)
    gamma = (p_up - 2 * p + p_dn) / bump**2

    vol_bump = 0.001
    sigma_iv = bs_implied_vol(p, S, K, T, r)
    p_vega = ql_heston_call(S, K, T, r, kappa, theta, xi, rho, v0 + vol_bump**2)
    vega = (p_vega - p) / vol_bump

    return p, delta, gamma, vega


def ql_vg_call(S, K, T, r, vg_sigma, vg_theta, vg_nu):
    """Variance-Gamma call price via QuantLib (scalar)."""
    import QuantLib as ql
    today = _ql_today()
    mat = _ql_maturity(today, T)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, float(K))
    exercise = ql.EuropeanExercise(mat)
    option = ql.VanillaOption(payoff, exercise)

    spot = ql.QuoteHandle(ql.SimpleQuote(float(S)))
    rts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, float(r), ql.Actual365Fixed()))
    div = ql.YieldTermStructureHandle(
        ql.FlatForward(today, 0.0, ql.Actual365Fixed()))

    process = ql.VarianceGammaProcess(
        spot, div, rts,
        float(vg_sigma), float(vg_nu), float(vg_theta))
    engine = ql.VarianceGammaEngine(process)
    option.setPricingEngine(engine)
    return option.NPV()


def ql_vg_greeks(S, K, T, r, vg_sigma, vg_theta, vg_nu, bump=1e-2):
    """VG Greeks via finite differences. Returns (price, delta, gamma, vega)."""
    p = ql_vg_call(S, K, T, r, vg_sigma, vg_theta, vg_nu)
    p_up = ql_vg_call(S + bump, K, T, r, vg_sigma, vg_theta, vg_nu)
    p_dn = ql_vg_call(S - bump, K, T, r, vg_sigma, vg_theta, vg_nu)
    delta = (p_up - p_dn) / (2 * bump)
    gamma = (p_up - 2 * p + p_dn) / bump**2

    p_v = ql_vg_call(S, K, T, r, vg_sigma + 0.001, vg_theta, vg_nu)
    vega = (p_v - p) / 0.001
    return p, delta, gamma, vega

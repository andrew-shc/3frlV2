"""
Compute per-tick feature matrix x ∈ R[T, F] from aligned LOBSTER DataFrames.

Feature layout (F = 40 + 3×48 = 184 per tick):
  [0:40]    Current LOB: 10 levels × (ask_p, ask_s, bid_p, bid_s), prices
            normalised relative to mid in units of tick_size, sizes / 1000
  [40:88]   1-min lookback: closing 10-level LOB (40) + OHLC(4) + ntx(1)
            + total_vol(1) + order_flow(1) + vwap(1)
  [88:136]  5-min lookback (same layout)
  [136:184] 10-min lookback (same layout)

build_feature_matrix() uses vectorised rolling operations — O(T) per horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOB_LEVELS = 10
LOB_DIM = LOB_LEVELS * 4           # 40
AGG_DIM = LOB_DIM + 8              # 48 per lookback horizon
HORIZONS_SEC = [60.0, 300.0, 600.0]
F_DIM = LOB_DIM + len(HORIZONS_SEC) * AGG_DIM  # 184


# ------------------------------------------------------------------
# Current LOB features  [T, LOB_DIM]
# ------------------------------------------------------------------

def _lob_features(ob: pd.DataFrame, mid: np.ndarray, tick: float) -> np.ndarray:
    T = len(ob)
    out = np.zeros((T, LOB_DIM), dtype=np.float32)
    denom = tick * np.ones(T, dtype=np.float64)

    for i in range(1, LOB_LEVELS + 1):
        base = (i - 1) * 4
        ap = ob[f"ask_p{i}"].values.astype(np.float64)
        as_ = ob[f"ask_s{i}"].values.astype(np.float64)
        bp = ob[f"bid_p{i}"].values.astype(np.float64)
        bs = ob[f"bid_s{i}"].values.astype(np.float64)

        valid_a = np.isfinite(ap)
        valid_b = np.isfinite(bp)

        out[:, base]     = np.where(valid_a, (ap - mid) / denom, LOB_LEVELS + 1)
        out[:, base + 1] = np.where(np.isfinite(as_), as_ / 1000.0, 0.0)
        out[:, base + 2] = np.where(valid_b, (mid - bp) / denom, LOB_LEVELS + 1)
        out[:, base + 3] = np.where(np.isfinite(bs), bs / 1000.0, 0.0)

    return out


# ------------------------------------------------------------------
# Aggregate lookback features  [T, AGG_DIM] for one horizon
# ------------------------------------------------------------------

def _agg_features_vectorised(
    ob: pd.DataFrame,
    msg: pd.DataFrame,
    time_sec: np.ndarray,
    horizon_sec: float,
    tick: float,
) -> np.ndarray:
    """
    Build [T, AGG_DIM] using pandas rolling over a time-based window.
    Rolling window is open on the left: [t - horizon_sec, t].
    """
    T = len(ob)
    mid = ob["mid"].values.astype(np.float64)
    spread = ob["spread"].values.astype(np.float64)

    # Use DatetimeIndex trick: convert seconds to pseudo-timestamps for rolling
    idx = pd.to_datetime(time_sec, unit="s")

    # -- Closing LOB at the trailing edge of the window --
    # For each row we want the LOB values at that row (it IS the closing LOB)
    lob_feats = np.zeros((T, LOB_DIM), dtype=np.float32)
    for i in range(1, LOB_LEVELS + 1):
        base = (i - 1) * 4
        ap = ob[f"ask_p{i}"].values.astype(np.float64)
        as_ = ob[f"ask_s{i}"].values.astype(np.float64)
        bp = ob[f"bid_p{i}"].values.astype(np.float64)
        bs = ob[f"bid_s{i}"].values.astype(np.float64)
        denom = tick
        lob_feats[:, base]     = np.where(np.isfinite(ap), (ap - mid) / denom, LOB_LEVELS + 1)
        lob_feats[:, base + 1] = np.where(np.isfinite(as_), as_ / 1000.0, 0.0)
        lob_feats[:, base + 2] = np.where(np.isfinite(bp), (mid - bp) / denom, LOB_LEVELS + 1)
        lob_feats[:, base + 3] = np.where(np.isfinite(bs), bs / 1000.0, 0.0)

    # -- OHLC of mid over rolling window --
    mid_s = pd.Series(mid, index=idx)
    window = f"{int(horizon_sec)}s"
    mid_open  = mid_s.rolling(window, min_periods=1).apply(lambda x: x.iloc[0],  raw=False).values
    mid_high  = mid_s.rolling(window, min_periods=1).max().values
    mid_low   = mid_s.rolling(window, min_periods=1).min().values
    mid_close = mid  # current row
    denom_ohlc = tick * 10 + 1e-8
    ohlc = np.stack([
        (mid_open  - mid_close) / denom_ohlc,
        (mid_high  - mid_close) / denom_ohlc,
        (mid_low   - mid_close) / denom_ohlc,
        np.zeros(T),   # close relative to itself = 0
    ], axis=1).astype(np.float32)

    # -- Trade statistics --
    is_trade = msg["type"].isin([4, 5]).astype(np.float64)
    buy_vol  = np.where((msg["type"].isin([4, 5])) & (msg["direction"] == 1), msg["size"].values, 0.0).astype(np.float64)
    sell_vol = np.where((msg["type"].isin([4, 5])) & (msg["direction"] == -1), msg["size"].values, 0.0).astype(np.float64)
    px_vol   = np.where(msg["type"].isin([4, 5]), msg["price"].values * msg["size"].values, 0.0).astype(np.float64)

    def _roll(arr):
        return pd.Series(arr, index=idx).rolling(window, min_periods=1).sum().values

    ntx_roll     = _roll(is_trade)
    total_vol    = _roll(buy_vol + sell_vol)
    buy_roll     = _roll(buy_vol)
    px_vol_roll  = _roll(px_vol)

    flow = (buy_roll - (total_vol - buy_roll)) / (total_vol + 1e-8)
    vwap = px_vol_roll / (total_vol + 1e-8)
    vwap_norm = (vwap - mid_close) / denom_ohlc

    stats = np.stack([
        np.log1p(ntx_roll) / 10.0,
        np.log1p(total_vol) / 10.0,
        flow,
        vwap_norm,
    ], axis=1).astype(np.float32)

    return np.concatenate([lob_feats, ohlc, stats], axis=1)  # [T, 48]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def build_feature_matrix(
    msg: pd.DataFrame,
    ob: pd.DataFrame,
    tick_size: float | None = None,
) -> np.ndarray:
    """Return float32 array [T, F_DIM].  Runs in O(T × n_horizons) time."""
    if tick_size is None:
        spreads = ob["spread"].dropna()
        spreads = spreads[spreads > 0]
        tick_size = float(np.percentile(spreads, 5)) if len(spreads) > 0 else 0.01
        tick_size = max(tick_size, 1e-4)

    mid      = ob["mid"].ffill().fillna(0).values.astype(np.float64)
    time_sec = msg["time"].values.astype(np.float64)

    ob   = ob.copy()
    msg  = msg.copy()

    parts = [_lob_features(ob, mid, tick_size)]
    for h_sec in HORIZONS_SEC:
        parts.append(_agg_features_vectorised(ob, msg, time_sec, h_sec, tick_size))

    X = np.concatenate(parts, axis=1).astype(np.float32)
    # Replace NaN/Inf that can arise in early-window rolling stats or missing LOB levels
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X

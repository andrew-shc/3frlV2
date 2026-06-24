"""Load OHLCV parquet and return TA feature arrays ready for model training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from pm.data.ta import MA_PERIOD, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL


def load_indicators(
    parquet_path: str,
    n_assets: int,
) -> tuple[dict[str, np.ndarray], pd.DatetimeIndex, pd.DataFrame]:
    """
    Load parquet, select top-n assets alphabetically, compute TA features.

    Returns:
        indicators : dict[str, np.ndarray]  float32, shape (T, n)
        dates      : pd.DatetimeIndex       length T
        close_df   : pd.DataFrame           shape (T, n), close prices
    """
    df = pd.read_parquet(parquet_path)
    pivot = (
        df.pivot(index="date", columns="symbol", values="close")
        .sort_index()
        .ffill()
        .bfill()
    )
    cols = sorted(pivot.columns)[:n_assets]
    pivot = pivot[cols]
    n = pivot.shape[1]

    close_arr = pivot.values.astype(np.float64)
    ma_arr    = np.full_like(close_arr, np.nan)
    rsi_arr   = np.full_like(close_arr, np.nan)
    macd_arr  = np.full_like(close_arr, np.nan)

    for i in range(n):
        c = close_arr[:, i]
        ma_arr[:, i]  = talib.SMA(c, timeperiod=MA_PERIOD)
        rsi_arr[:, i] = talib.RSI(c, timeperiod=RSI_PERIOD)
        macd_arr[:, i], _, _ = talib.MACD(c, fastperiod=MACD_FAST,
                                            slowperiod=MACD_SLOW,
                                            signalperiod=MACD_SIGNAL)

    for arr in (ma_arr, rsi_arr, macd_arr):
        col_means = np.nanmean(arr, axis=0)
        nan_idx = np.isnan(arr)
        arr[nan_idx] = np.take(col_means, np.where(nan_idx)[1])

    indicators = {
        "close": close_arr.astype(np.float32),
        "ma":    ma_arr.astype(np.float32),
        "rsi":   rsi_arr.astype(np.float32),
        "macd":  macd_arr.astype(np.float32),
    }
    return indicators, pivot.index, pivot

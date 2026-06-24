"""TA indicator computation from a wide-format close-price DataFrame."""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib

MA_PERIOD   = 28
RSI_PERIOD  = 14
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9


def compute_ta_df(close_wide: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compute TA indicators from a (T × n) close-price DataFrame.
    Returns one same-shaped DataFrame per indicator key.
    """
    ma_df   = pd.DataFrame(np.nan, index=close_wide.index, columns=close_wide.columns)
    rsi_df  = pd.DataFrame(np.nan, index=close_wide.index, columns=close_wide.columns)
    macd_df = pd.DataFrame(np.nan, index=close_wide.index, columns=close_wide.columns)

    for sym in close_wide.columns:
        c = close_wide[sym].values.astype(np.float64)
        ma_df[sym]  = talib.SMA(c, timeperiod=MA_PERIOD)
        rsi_df[sym] = talib.RSI(c, timeperiod=RSI_PERIOD)
        macd_line, _, _ = talib.MACD(c, fastperiod=MACD_FAST,
                                       slowperiod=MACD_SLOW,
                                       signalperiod=MACD_SIGNAL)
        macd_df[sym] = macd_line

    return {"close": close_wide, "ma": ma_df, "rsi": rsi_df, "macd": macd_df}

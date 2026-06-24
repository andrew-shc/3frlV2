"""Rolling mean pairwise correlation across assets."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_mean_corr(df: pd.DataFrame, window: int) -> pd.Series:
    """
    At each bar t >= window-1, compute the mean upper-triangle Pearson
    correlation across all assets using a rolling window of `window` bars.
    """
    T = len(df)
    results: dict = {}

    for t in range(window - 1, T):
        X    = df.iloc[t - window + 1 : t + 1].values
        mask = ~np.isnan(X).any(axis=0)
        X_c  = X[:, mask]
        if X_c.shape[1] < 2:
            results[df.index[t]] = np.nan
            continue
        C    = np.corrcoef(X_c.T)
        iu   = np.triu_indices(C.shape[0], k=1)
        vals = C[iu][np.isfinite(C[iu])]
        results[df.index[t]] = float(vals.mean()) if len(vals) else np.nan

    return pd.Series(results)

"""Load LOBSTER message + orderbook CSV files into aligned DataFrames."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_MSG_COLS = ["time", "type", "order_id", "size", "price", "direction"]


def load_lobster(
    msg_path: str | Path,
    ob_path: str | Path,
    n_levels: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (msg, ob) DataFrames with prices in dollars."""
    msg = pd.read_csv(msg_path, header=None, names=_MSG_COLS)

    ob_raw = pd.read_csv(ob_path, header=None)
    if n_levels is None:
        n_levels = ob_raw.shape[1] // 4
    elif n_levels * 4 < ob_raw.shape[1]:
        ob_raw = ob_raw.iloc[:, : n_levels * 4].copy()

    ob_cols: list[str] = []
    for i in range(1, n_levels + 1):
        ob_cols += [f"ask_p{i}", f"ask_s{i}", f"bid_p{i}", f"bid_s{i}"]
    ob_raw.columns = ob_cols

    # Remove trading-halt rows (type == 7)
    mask = msg["type"] != 7
    msg = msg[mask].reset_index(drop=True)
    ob = ob_raw[mask].reset_index(drop=True)

    # Prices are stored as integer × 10000
    msg["price"] = msg["price"] / 10_000.0
    for i in range(1, n_levels + 1):
        ob[f"ask_p{i}"] = ob[f"ask_p{i}"] / 10_000.0
        ob[f"bid_p{i}"] = ob[f"bid_p{i}"] / 10_000.0

    # Replace dummy sentinel prices with NaN
    dummy_ask = 9_999_999_999 / 10_000.0
    dummy_bid = -9_999_999_999 / 10_000.0
    for i in range(1, n_levels + 1):
        ob.loc[ob[f"ask_p{i}"] >= dummy_ask, [f"ask_p{i}", f"ask_s{i}"]] = np.nan
        ob.loc[ob[f"bid_p{i}"] <= dummy_bid, [f"bid_p{i}", f"bid_s{i}"]] = np.nan

    ob["mid"] = (ob["ask_p1"] + ob["bid_p1"]) / 2.0
    ob["spread"] = ob["ask_p1"] - ob["bid_p1"]

    return msg, ob

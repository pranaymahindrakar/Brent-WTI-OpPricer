"""Compute the Brent-WTI spread and rolling statistics.

The integrity rule that governs this whole module: never difference a stale leg
against a fresh one. Both legs are resampled onto a common grid and inner-joined
on timestamp before any subtraction happens.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, store

# Plausible USD/bbl band, used as a cheap unit and sanity guard.
PLAUSIBLE_RANGE = (0.0, 400.0)


def align(brent: pd.DataFrame, wti: pd.DataFrame, freq: str = None) -> pd.DataFrame:
    """Align two single-leg bar frames onto a common grid.

    Each input needs columns ts and close. Returns ts, brent, wti on shared
    timestamps only. Rows outside the plausible price band are dropped.
    """
    freq = freq or config.DEFAULT_FREQ
    b = (
        brent[["ts", "close"]].rename(columns={"close": "brent"})
        .dropna().set_index("ts").sort_index()
    )
    w = (
        wti[["ts", "close"]].rename(columns={"close": "wti"})
        .dropna().set_index("ts").sort_index()
    )
    # Common grid via last observation in each bucket.
    b = b.resample(freq).last()
    w = w.resample(freq).last()
    out = b.join(w, how="inner").dropna()
    for col in ("brent", "wti"):
        in_band = out[col].between(*PLAUSIBLE_RANGE)
        out = out[in_band]
    return out.reset_index()


def compute_spread(df: pd.DataFrame, window: int = None) -> pd.DataFrame:
    """Given aligned ts/brent/wti, compute the spread and rolling statistics."""
    window = window or config.ZSCORE_WINDOW
    min_p = max(2, window // 2)
    out = df.copy()
    out["spread"] = out["brent"] - out["wti"]

    roll = out["spread"].rolling(window, min_periods=min_p)
    out["roll_mean"] = roll.mean()
    out["roll_std"] = roll.std(ddof=1)
    # Replace a zero std with NaN so a constant spread yields NaN, never inf.
    out["zscore"] = (out["spread"] - out["roll_mean"]) / out["roll_std"].replace(0, np.nan)

    out["corr"] = (
        out["brent"].rolling(window, min_periods=min_p).corr(out["wti"])
    )
    rmin = out["spread"].rolling(window, min_periods=2).min()
    rmax = out["spread"].rolling(window, min_periods=2).max()
    out["pct_range"] = (out["spread"] - rmin) / (rmax - rmin).replace(0, np.nan)
    return out


def build(con, freq: str = None, window: int = None) -> pd.DataFrame:
    """Read bars from the store, compute the spread, persist it, and return it."""
    brent = store.read_bars(con, symbol=config.BRENT)
    wti = store.read_bars(con, symbol=config.WTI)
    if brent.empty or wti.empty:
        return pd.DataFrame()
    aligned = align(brent, wti, freq=freq)
    if aligned.empty:
        return pd.DataFrame()
    spread = compute_spread(aligned, window=window)
    store.write_spread(con, spread)
    return spread

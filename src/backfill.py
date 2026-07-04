"""Backfill daily history from FRED and reconcile it against the live feed.

A spread tracker with no past cannot contextualize its present, so this seeds a
real distribution of history behind the z-score and gives a settlement truth
source to catch live-feed drift.
"""
from __future__ import annotations

import pandas as pd

from src import config, store


def fetch_fred_series(symbol: str) -> pd.DataFrame:
    """Pull one daily FRED settlement series into the canonical bars schema."""
    from fredapi import Fred

    if not config.FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY not set; cannot backfill history.")
    fred = Fred(api_key=config.FRED_API_KEY)
    series = fred.get_series(config.FRED_SERIES[symbol]).dropna()
    df = series.reset_index()
    df.columns = ["ts", "close"]
    df["ts"] = pd.to_datetime(df["ts"])
    df["symbol"] = symbol
    # Daily settlement has no intraday OHLC, so mirror close and leave volume null.
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]
    df["volume"] = float("nan")
    df["source"] = "fred"
    return df[store.BARS_COLS]


def run(years: int = 5, con=None) -> int:
    """Backfill the last `years` of daily history for both legs.

    If `con` is provided, uses it directly (caller is responsible for the
    lifecycle). Otherwise opens and closes its own connection.
    """
    _own = con is None
    _con = store.connect() if _own else con
    total = 0
    cutoff = pd.Timestamp.now("UTC").replace(tzinfo=None) - pd.Timedelta(days=365 * years)
    for leg in config.LEGS:
        df = fetch_fred_series(leg)
        df = df[df["ts"] >= cutoff]
        total += store.write_bars(_con, df)
    if _own:
        _con.close()
    return total


def reconcile(con, tol: float = None) -> pd.DataFrame:
    """Compare the latest live daily close against FRED for each leg.

    Returns a small frame flagging any leg whose live close diverges from the
    FRED settlement by more than the tolerance.
    """
    tol = config.RECONCILE_TOL if tol is None else tol
    rows = []
    for leg in config.LEGS:
        allbars = store.read_bars(con, symbol=leg)
        if allbars.empty:
            continue
        live = allbars[allbars["source"] != "fred"]
        fred = allbars[allbars["source"] == "fred"]
        if live.empty or fred.empty:
            continue
        lv = float(live.sort_values("ts").iloc[-1]["close"])
        fv = float(fred.sort_values("ts").iloc[-1]["close"])
        rows.append({
            "leg": leg, "live": lv, "fred": fv,
            "diff": abs(lv - fv), "drift": abs(lv - fv) > tol,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    n = run()
    print(f"Backfilled {n} daily bars from FRED.")

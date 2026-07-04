"""Live ingestion from yfinance with market-hours awareness and tick filtering.

All timestamps are stored as UTC-naive so the intraday live feed and the daily
FRED backfill share one consistent time basis.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pandas_market_calendars as mcal

from src import config, store

# CMEGlobex_CL is the authoritative calendar for WTI crude (CL) futures.
# It correctly handles CME holidays, early closes (e.g. Juneteenth), and the
# daily 17:00-18:00 ET maintenance break via its session open/close windows.
_CME_CAL = mcal.get_calendar("CMEGlobex_CL")
_UTC = timezone.utc


def is_market_open(now: datetime = None) -> bool:
    """Return True if the CME Globex CL (WTI crude) market is currently open.

    Uses the pandas_market_calendars CMEGlobex_CL schedule, which covers
    regular hours, the 17:00-18:00 ET daily maintenance break, exchange
    holidays, and early-close days such as Juneteenth.
    """
    now_utc = (now or datetime.now(_UTC)).astimezone(_UTC).replace(tzinfo=None)
    # Check a 3-day window centred on today to capture overnight sessions.
    start = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")
    end = (now_utc + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        sched = _CME_CAL.schedule(start, end, tz="UTC")
    except Exception:
        return False
    if sched.empty:
        return False
    # market_open and market_close are already tz-aware (UTC).
    now_ts = pd.Timestamp(now_utc, tz="UTC")
    for _, row in sched.iterrows():
        if row["market_open"] <= now_ts < row["market_close"]:
            return True
    return False


def filter_bad_ticks(df: pd.DataFrame, pct: float = None) -> pd.DataFrame:
    """Drop rows whose close jumps more than `pct` from the last accepted close.

    The comparison is against the last accepted value rather than the immediately
    prior row, so a single bad print does not drag down the following good ones.
    """
    pct = config.BAD_TICK_PCT if pct is None else pct
    if df is None or df.empty:
        return df
    df = df.sort_values("ts").reset_index(drop=True)
    keep, last_good = [], None
    for i, c in enumerate(df["close"].values):
        if last_good is None or last_good == 0:
            keep.append(i)
            last_good = c
            continue
        if abs(c - last_good) / abs(last_good) <= pct:
            keep.append(i)
            last_good = c
        # otherwise skip this row and keep last_good unchanged
    return df.iloc[keep].reset_index(drop=True)


def _normalize(raw: pd.DataFrame, symbol: str, source: str = "yfinance") -> pd.DataFrame:
    """Turn a yfinance history frame into the canonical bars schema, UTC-naive."""
    cols = store.BARS_COLS
    if raw is None or raw.empty:
        return pd.DataFrame(columns=cols)
    df = raw.reset_index()
    tcol = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={
        tcol: "ts", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    ts = pd.to_datetime(df["ts"])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    df["ts"] = ts
    df["symbol"] = symbol
    df["source"] = source
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce").astype(float)
    return df[cols]


def fetch_leg(symbol: str, period: str = "1d", interval: str = None) -> pd.DataFrame:
    """Fetch one leg from yfinance and clean it. Imported lazily to keep tests light."""
    import yfinance as yf

    interval = interval or config.BAR_INTERVAL
    yf_symbol = config.YF_SYMBOLS[symbol]
    raw = yf.Ticker(yf_symbol).history(period=period, interval=interval, auto_adjust=False)
    df = _normalize(raw, symbol)
    return filter_bad_ticks(df)


def poll_once(con, period: str = "1d", interval: str = None) -> int:
    """Fetch both legs once and write new bars. No-op when the market is closed."""
    if not is_market_open():
        return 0
    written = 0
    for leg in config.LEGS:
        df = fetch_leg(leg, period=period, interval=interval)
        written += store.write_bars(con, df)
    return written


def seed_recent(con, period: str = "5d", interval: str = "1d") -> int:
    """Fetch recent daily bars from yfinance without a market-hours check.

    Useful on first run or when the market is closed. Uses daily bars so that
    the seed data aligns cleanly with FRED history and avoids intraday noise.
    """
    written = 0
    for leg in config.LEGS:
        df = fetch_leg(leg, period=period, interval=interval)
        written += store.write_bars(con, df)
    return written

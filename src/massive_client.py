"""Thin REST client for the massive.com market-data API.

This is a keyed, paid data source used as the *primary* source for the
Options and Energy Trends tabs, never for the core WTI/Brent ingestion path
(`ingest.py`), which stays on the CLAUDE.md-mandated free stack. Every
function here returns None (or an empty DataFrame) on any failure, missing
key, timeout, or non-200 response, so callers can always fall back to
yfinance without special-casing exceptions.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import requests

from src import config

_TIMEOUT = 10


def _get(path: str, params: Optional[dict] = None) -> Optional[dict]:
    """GET a massive.com endpoint. Returns the parsed JSON body, or None."""
    if not config.MASSIVE_API_KEY:
        return None
    try:
        resp = requests.get(
            f"{config.MASSIVE_BASE_URL}{path}",
            params=params or {},
            headers={"Authorization": f"Bearer {config.MASSIVE_API_KEY}"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def option_chain_snapshot(underlying: str, **filters) -> Optional[dict]:
    """Full options chain snapshot for `underlying`: greeks, IV, open interest.

    GET /v3/snapshot/options/{underlyingAsset}. `filters` maps directly to
    the endpoint's query parameters (e.g. expiration_date, contract_type,
    strike_price.gte). Returns the raw JSON body (a `results` list of
    contract snapshots) or None on failure.
    """
    return _get(f"/v3/snapshot/options/{underlying}", params=filters)


def aggs(ticker: str, start: str, end: str, timespan: str = "day") -> pd.DataFrame:
    """Daily (or other timespan) OHLCV bars for `ticker` between start/end.

    GET /v2/aggs/ticker/{ticker}/range/1/{timespan}/{start}/{end}. Dates are
    YYYY-MM-DD. Returns an empty DataFrame (never None) on failure, so
    callers can treat it like any other price frame.
    """
    body = _get(f"/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start}/{end}")
    results = (body or {}).get("results") or []
    if not results:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(results).rename(
        columns={"t": "ts", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df[["ts", "open", "high", "low", "close", "volume"]].sort_values("ts").reset_index(drop=True)


def futures_snapshot(tickers: list[str]) -> Optional[dict]:
    """Real-time snapshot (latest trade, quote, session stats) for futures contracts.

    GET /futures/v1/snapshot. Returns the raw JSON body or None on failure.
    """
    return _get("/futures/v1/snapshot", params={"ticker.any_of": ",".join(tickers)})

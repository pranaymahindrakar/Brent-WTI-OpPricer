"""Live options chains and Greeks for the Options tab.

Underlyings are liquid crude/energy ETFs (USO, XLE); options on the CL=F/BZ=F
futures contracts themselves aren't covered by massive or yfinance. Massive is
tried first (real greeks, IV, and open interest straight from its snapshot);
on any failure or missing key, yfinance's chain is used instead and Greeks are
computed locally via `pricing.py` so downstream code sees the same shape
either way.
"""
from __future__ import annotations

import sys
from datetime import datetime
from typing import Optional

import pandas as pd

from src import config, massive_client, pricing

CHAIN_COLS = [
    "expiration", "strike", "contract_type", "iv", "delta", "gamma", "theta",
    "vega", "open_interest", "volume", "bid", "ask", "last",
]


def get_expiries(ticker: str) -> list[str]:
    """List available expiration dates for `ticker`, via yfinance (no key needed)."""
    import yfinance as yf

    try:
        expiries = list(yf.Ticker(ticker).options)
    except Exception as exc:
        print(f"options.get_expiries({ticker}): yfinance raised {exc}", file=sys.stderr)
        return []
    if not expiries:
        print(
            f"options.get_expiries({ticker}): yfinance returned no expiries "
            "(often a transient Yahoo Finance rate limit; retrying later usually works)",
            file=sys.stderr,
        )
    return expiries


def _parse_massive_chain(body: dict) -> tuple[pd.DataFrame, Optional[float]]:
    """Parse a massive.com option chain snapshot into the unified schema."""
    rows = []
    underlying_price = None
    for c in (body or {}).get("results") or []:
        details = c.get("details") or {}
        greeks = c.get("greeks") or {}
        day = c.get("day") or {}
        quote = c.get("last_quote") or {}
        underlying = c.get("underlying_asset") or {}
        if underlying.get("price") is not None:
            underlying_price = float(underlying["price"])
        rows.append({
            "expiration": details.get("expiration_date"),
            "strike": details.get("strike_price"),
            "contract_type": details.get("contract_type"),
            "iv": c.get("implied_volatility"),
            "delta": greeks.get("delta"),
            "gamma": greeks.get("gamma"),
            "theta": greeks.get("theta"),
            "vega": greeks.get("vega"),
            "open_interest": c.get("open_interest"),
            "volume": day.get("volume"),
            "bid": quote.get("bid"),
            "ask": quote.get("ask"),
            "last": day.get("close"),
        })
    if not rows:
        return pd.DataFrame(columns=CHAIN_COLS), underlying_price
    return pd.DataFrame(rows)[CHAIN_COLS], underlying_price


def _fallback_yfinance_chain(ticker: str, expiration: str) -> tuple[pd.DataFrame, Optional[float]]:
    """Fetch a chain via yfinance and compute Greeks locally with Black-Scholes."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    try:
        underlying_price = float(tk.fast_info["last_price"])
    except Exception:
        underlying_price = None
    try:
        chain = tk.option_chain(expiration)
    except Exception:
        return pd.DataFrame(columns=CHAIN_COLS), underlying_price

    years_to_expiry = max(
        (datetime.strptime(expiration, "%Y-%m-%d") - datetime.utcnow()).days, 0
    ) / 365.0

    rows = []
    for contract_type, df in (("call", chain.calls), ("put", chain.puts)):
        for _, r in df.iterrows():
            iv = float(r["impliedVolatility"]) if pd.notna(r.get("impliedVolatility")) else None
            greeks = None
            if underlying_price and iv and iv > 0 and years_to_expiry > 0:
                greeks = pricing.bs_greeks(
                    underlying_price, float(r["strike"]), years_to_expiry,
                    config.OPTIONS_RISK_FREE_RATE, iv, contract_type,
                )
            rows.append({
                "expiration": expiration,
                "strike": r.get("strike"),
                "contract_type": contract_type,
                "iv": iv,
                "delta": greeks.delta if greeks else None,
                "gamma": greeks.gamma if greeks else None,
                "theta": greeks.theta if greeks else None,
                "vega": greeks.vega if greeks else None,
                "open_interest": r.get("openInterest"),
                "volume": r.get("volume"),
                "bid": r.get("bid"),
                "ask": r.get("ask"),
                "last": r.get("lastPrice"),
            })
    if not rows:
        return pd.DataFrame(columns=CHAIN_COLS), underlying_price
    return pd.DataFrame(rows)[CHAIN_COLS], underlying_price


def fetch_chain(ticker: str, expiration: str) -> dict:
    """Fetch an options chain for `ticker`/`expiration`, massive first.

    Returns {"chain": DataFrame, "underlying_price": float | None,
    "source": "massive" | "yfinance"}. The chain DataFrame is always
    CHAIN_COLS-shaped, even if empty, so callers never branch on source.
    """
    body = massive_client.option_chain_snapshot(ticker, expiration_date=expiration)
    if body:
        chain, underlying_price = _parse_massive_chain(body)
        if not chain.empty:
            return {"chain": chain, "underlying_price": underlying_price, "source": "massive"}

    chain, underlying_price = _fallback_yfinance_chain(ticker, expiration)
    return {"chain": chain, "underlying_price": underlying_price, "source": "yfinance"}


def atm_summary(chain: pd.DataFrame, underlying_price: Optional[float]) -> dict:
    """Summarize a chain into an ATM IV and put/call open-interest ratio.

    Both figures are plain computed numbers (no LLM involvement), suitable
    for the grounding payload passed to the insight note or chatbot.
    """
    if chain.empty or not underlying_price:
        return {"available": False}

    chain = chain.dropna(subset=["strike"])
    chain["dist"] = (chain["strike"] - underlying_price).abs()
    atm_strike = chain.loc[chain["dist"].idxmin(), "strike"]
    atm_rows = chain[chain["strike"] == atm_strike]
    ivs = atm_rows["iv"].dropna()
    atm_iv = float(ivs.mean()) if not ivs.empty else None

    call_oi = chain.loc[chain["contract_type"] == "call", "open_interest"].dropna().sum()
    put_oi = chain.loc[chain["contract_type"] == "put", "open_interest"].dropna().sum()
    put_call_oi_ratio = float(put_oi / call_oi) if call_oi else None

    return {
        "available": atm_iv is not None,
        "atm_strike": float(atm_strike),
        "atm_iv": atm_iv,
        "put_call_oi_ratio": put_call_oi_ratio,
        "call_open_interest": float(call_oi),
        "put_open_interest": float(put_oi),
    }

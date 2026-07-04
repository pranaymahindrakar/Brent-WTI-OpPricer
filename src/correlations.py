"""Fetch daily prices for correlated assets and compute rolling spread correlations.

Correlations are computed on daily percentage returns (not price levels) to
avoid spurious correlation from shared trends. The rolling window matches the
z-score window so both metrics describe the same historical regime.
"""
from __future__ import annotations

import pandas as pd

ASSETS: dict[str, dict] = {
    "DX-Y.NYB": {
        "name": "US Dollar Index (DXY)",
        "note": (
            "Oil is priced globally in USD. Dollar strength tends to compress oil prices "
            "in USD terms. The effect on the spread is asymmetric: WTI has more sensitivity "
            "to US domestic flows while Brent reflects global dollar liquidity, so a strong "
            "dollar often widens the Brent premium."
        ),
    },
    "XLE": {
        "name": "Energy Select SPDR (XLE)",
        "note": (
            "Broad energy sector ETF tracking the S&P 500 energy companies. Follows overall "
            "oil sentiment. Spread divergence from XLE can signal sector rotation or "
            "company-specific factors decoupling from the commodity."
        ),
    },
    "VLO": {
        "name": "Valero Energy (VLO)",
        "note": (
            "Largest US independent refiner. Buys WTI feedstock and sells refined products "
            "priced closer to Brent. A wider spread (cheap WTI) directly improves crack "
            "margins, which is typically reflected in VLO's share price."
        ),
    },
    "HO=F": {
        "name": "Heating Oil / Diesel (HO=F)",
        "note": (
            "NYMEX ultra-low sulfur diesel futures, the primary US distillate crack spread "
            "proxy. Reflects global diesel and jet fuel demand. Heating oil is priced off "
            "Brent; strength here often accompanies a widening Brent premium."
        ),
    },
    "RB=F": {
        "name": "RBOB Gasoline (RB=F)",
        "note": (
            "US conventional gasoline futures, the gasoline crack spread proxy. Seasonal "
            "demand peaks (summer driving) can tighten the spread as US refiners aggressively "
            "bid for WTI feedstock, narrowing the Brent premium."
        ),
    },
    "SPY": {
        "name": "S&P 500 (SPY)",
        "note": (
            "Risk appetite proxy. Recessions compress global oil demand and hit Brent harder "
            "than WTI given its broader demand base. Risk-on environments often coincide with "
            "a widening Brent premium as emerging market demand recovers faster."
        ),
    },
    "NG=F": {
        "name": "Natural Gas (NG=F)",
        "note": (
            "Henry Hub natural gas futures. As a competing fuel in power generation and "
            "heating, gas price spikes can shift industrial demand away from oil. "
            "LNG export growth has also increasingly linked US gas and global crude markets."
        ),
    },
}


def fetch_prices(period: str = "2y") -> pd.DataFrame:
    """Fetch daily adjusted closing prices for all correlated assets from yfinance."""
    import yfinance as yf

    frames = []
    for ticker in ASSETS:
        try:
            raw = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
            if raw.empty:
                continue
            s = raw["Close"].rename(ticker)
            s.index = pd.to_datetime(s.index).tz_localize(None)
            frames.append(s)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def _spread_daily(spread_df: pd.DataFrame) -> pd.Series:
    """Resample the stored spread frame to one daily close per calendar day."""
    if spread_df.empty:
        return pd.Series(dtype=float, name="spread")
    s = spread_df.set_index("ts")["spread"]
    s.index = pd.to_datetime(s.index).normalize()
    return s.resample("1D").last().dropna()


def compute_rolling_corr(
    spread_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """Rolling `window`-day Pearson correlation of spread returns vs each asset's returns."""
    if spread_df.empty or prices_df.empty:
        return pd.DataFrame()
    sp = _spread_daily(spread_df).pct_change().dropna()
    ar = prices_df.pct_change().dropna()
    idx = sp.index.intersection(ar.index)
    if len(idx) < window:
        return pd.DataFrame()
    s, a = sp.loc[idx], ar.loc[idx]
    result = pd.DataFrame(index=idx)
    for col in a.columns:
        result[col] = s.rolling(window).corr(a[col])
    return result.dropna(how="all")


def compute_current_corr(
    spread_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    window: int = 60,
) -> pd.Series:
    """Scalar Pearson correlation for each asset over the most recent `window` trading days."""
    if spread_df.empty or prices_df.empty:
        return pd.Series(dtype=float)
    sp = _spread_daily(spread_df).pct_change().dropna()
    ar = prices_df.pct_change().dropna()
    idx = sp.index.intersection(ar.index)
    s, a = sp.loc[idx].tail(window), ar.loc[idx].tail(window)
    return a.corrwith(s).rename("correlation")

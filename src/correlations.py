"""Fetch daily prices for correlated assets and compute rolling spread correlations.

Correlations are computed on daily percentage returns (not price levels) to
avoid spurious correlation from shared trends. The rolling window matches the
z-score window so both metrics describe the same historical regime.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta

import pandas as pd

from src import massive_client

ASSETS: dict[str, dict] = {
    "DX-Y.NYB": {
        "name": "US Dollar Index (DXY)",
        "category": "Macro",
        "note": (
            "Oil is priced globally in USD. Dollar strength tends to compress oil prices "
            "in USD terms. The effect on the spread is asymmetric: WTI has more sensitivity "
            "to US domestic flows while Brent reflects global dollar liquidity, so a strong "
            "dollar often widens the Brent premium."
        ),
    },
    "XLE": {
        "name": "Energy Select SPDR (XLE)",
        "category": "Sector ETF",
        "note": (
            "Broad energy sector ETF tracking the S&P 500 energy companies. Follows overall "
            "oil sentiment. Spread divergence from XLE can signal sector rotation or "
            "company-specific factors decoupling from the commodity."
        ),
    },
    "VLO": {
        "name": "Valero Energy (VLO)",
        "category": "Refiners",
        "note": (
            "Largest US independent refiner. Buys WTI feedstock and sells refined products "
            "priced closer to Brent. A wider spread (cheap WTI) directly improves crack "
            "margins, which is typically reflected in VLO's share price."
        ),
    },
    "MPC": {
        "name": "Marathon Petroleum (MPC)",
        "category": "Refiners",
        "note": (
            "Second-largest US refiner by capacity, heavily weighted toward Midwest and "
            "Gulf Coast WTI-linked feedstock. Moves with the spread for the same reason "
            "as Valero: a wider spread expands crack margins."
        ),
    },
    "PSX": {
        "name": "Phillips 66 (PSX)",
        "category": "Refiners",
        "note": (
            "Diversified refiner with midstream and chemicals segments alongside refining, "
            "so its correlation to the spread is typically weaker than pure-play refiners "
            "like Valero or Marathon."
        ),
    },
    "HO=F": {
        "name": "Heating Oil / Diesel (HO=F)",
        "category": "Products",
        "note": (
            "NYMEX ultra-low sulfur diesel futures, the primary US distillate crack spread "
            "proxy. Reflects global diesel and jet fuel demand. Heating oil is priced off "
            "Brent; strength here often accompanies a widening Brent premium."
        ),
    },
    "RB=F": {
        "name": "RBOB Gasoline (RB=F)",
        "category": "Products",
        "note": (
            "US conventional gasoline futures, the gasoline crack spread proxy. Seasonal "
            "demand peaks (summer driving) can tighten the spread as US refiners aggressively "
            "bid for WTI feedstock, narrowing the Brent premium."
        ),
    },
    "SPY": {
        "name": "S&P 500 (SPY)",
        "category": "Macro",
        "note": (
            "Risk appetite proxy. Recessions compress global oil demand and hit Brent harder "
            "than WTI given its broader demand base. Risk-on environments often coincide with "
            "a widening Brent premium as emerging market demand recovers faster."
        ),
    },
    "TLT": {
        "name": "20+ Year Treasury Bond ETF (TLT)",
        "category": "Macro",
        "note": (
            "Long-duration rates proxy. Falling yields (rising TLT) often coincide with "
            "growth scares that compress global oil demand expectations, hitting Brent's "
            "broader demand base harder than WTI's more domestic one."
        ),
    },
    "DBC": {
        "name": "Invesco DB Commodity Index (DBC)",
        "category": "Macro",
        "note": (
            "Broad commodity index (energy, metals, agriculture). Tracks whether crude "
            "moves are a crude-specific story or part of a broader commodity cycle."
        ),
    },
    "NG=F": {
        "name": "Natural Gas (NG=F)",
        "category": "Energy Complex",
        "note": (
            "Henry Hub natural gas futures. As a competing fuel in power generation and "
            "heating, gas price spikes can shift industrial demand away from oil. "
            "LNG export growth has also increasingly linked US gas and global crude markets."
        ),
    },
    "USO": {
        "name": "United States Oil Fund (USO)",
        "category": "Crude ETFs",
        "note": (
            "Front-month WTI futures ETF. Tracks WTI's leg of the spread directly rather "
            "than a downstream or macro proxy; used here as the Options tab's underlying."
        ),
    },
    "BNO": {
        "name": "United States Brent Oil Fund (BNO)",
        "category": "Crude ETFs",
        "note": (
            "Front-month Brent futures ETF. Tracks Brent's leg of the spread directly; "
            "comparing USO and BNO moves is a rough equity-market mirror of the spread itself."
        ),
    },
    "XOM": {
        "name": "ExxonMobil (XOM)",
        "category": "Majors",
        "note": (
            "Integrated major with both upstream production (benefits from high absolute "
            "oil prices) and downstream refining (benefits from a wide spread), so its "
            "spread correlation is usually weaker and less directional than a pure refiner."
        ),
    },
    "CVX": {
        "name": "Chevron (CVX)",
        "category": "Majors",
        "note": (
            "Integrated major with significant Permian Basin (WTI-linked) upstream "
            "production, giving it somewhat more spread sensitivity than Exxon's more "
            "globally diversified portfolio."
        ),
    },
    "COP": {
        "name": "ConocoPhillips (COP)",
        "category": "Majors",
        "note": (
            "Pure exploration and production major, no downstream refining. Tracks "
            "absolute oil price levels more than the spread itself; included as a "
            "contrast case against the refiners above."
        ),
    },
    "SLB": {
        "name": "SLB (Schlumberger) (SLB)",
        "category": "Services",
        "note": (
            "Largest oilfield services company. Revenue follows upstream capital spending "
            "decisions, which respond to absolute price levels and drilling economics more "
            "than the Brent-WTI spread specifically."
        ),
    },
}

# One consistent icon per category, used wherever a category label is shown
# (Market Lens filter, Energy Trends section headers) so the same category
# reads the same way across every tab.
CATEGORY_ICONS: dict[str, str] = {
    "Majors": "🛢️",
    "Refiners": "⚙️",
    "Services": "🔧",
    "Products": "⛽",
    "Crude ETFs": "🛢️",
    "Macro": "🌐",
    "Sector ETF": "📊",
    "Energy Complex": "🔥",
}


def _to_daily_index(s: pd.Series) -> pd.Series:
    """Snap a daily close series onto a tz-naive calendar-date index.

    Every daily series here must key off the same thing: the calendar date of
    the session. The two sources disagree on how they express that. yfinance
    returns a tz-aware midnight-in-exchange-time index, while massive returns
    the aggregate window's start as epoch milliseconds, which lands at 04:00 or
    05:00 UTC (midnight ET, shifting with US daylight saving). Left alone, a
    massive-sourced series and a yfinance-sourced series share no timestamps at
    all, so they union into a frame of holes instead of aligning, and neither
    one intersects the midnight-normalized spread index from _spread_daily.
    That is silent: the correlation just comes back NaN and the UI prints N/A.
    Normalizing both to midnight here is what makes the two sources
    interchangeable, which is the whole premise of the massive/yfinance
    fallback.
    """
    s = s.copy()
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    # A normalize() can collide two bars onto one date (a DST-straddling pair,
    # or an intraday bar slipping in); keep the last close for that session.
    return s[~s.index.duplicated(keep="last")]


def _fetch_one_price_series(ticker: str, period: str, start, end) -> pd.Series | None:
    """Fetch one ticker's daily close series, massive first, yfinance fallback.

    Shared by fetch_prices (the curated universe) and fetch_single_price (an
    arbitrary user-searched ticker), so both go through the same resilience
    path. Returns None if neither source has data.
    """
    import yfinance as yf

    massive_df = massive_client.aggs(ticker, start.isoformat(), end.isoformat())
    if not massive_df.empty:
        return _to_daily_index(massive_df.set_index("ts")["close"].rename(ticker))
    try:
        raw = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        if not raw.empty:
            return _to_daily_index(raw["Close"].rename(ticker))
    except Exception:
        pass
    return None


def _period_bounds(period: str):
    end = datetime.utcnow().date()
    start = end - timedelta(days=int(365 * float(period.rstrip("y")))) if period.endswith("y") else end - timedelta(days=730)
    return start, end


def fetch_prices(period: str = "2y") -> pd.DataFrame:
    """Fetch daily adjusted closing prices for all correlated assets.

    Tries massive.com first per ticker (real aggregates, no adjustment for
    splits/dividends applied). Whatever massive doesn't cover is fetched from
    yfinance in a single batched yf.download() call rather than one request
    per ticker: Yahoo Finance rate-limits shared cloud IPs (Streamlit Cloud
    included) much more aggressively against a burst of ~18 sequential
    per-ticker requests than a single batched one, so this is a real
    reliability fix, not just a speed one.
    """
    start, end = _period_bounds(period)
    frames: dict[str, pd.Series] = {}
    remaining = []
    for ticker in ASSETS:
        massive_df = massive_client.aggs(ticker, start.isoformat(), end.isoformat())
        if not massive_df.empty:
            frames[ticker] = _to_daily_index(massive_df.set_index("ts")["close"])
        else:
            remaining.append(ticker)

    if remaining:
        import yfinance as yf

        try:
            batch = yf.download(
                tickers=remaining, period=period, interval="1d",
                auto_adjust=True, group_by="ticker", threads=True, progress=False,
            )
        except Exception as exc:
            print(f"correlations.fetch_prices: yf.download failed: {exc}", file=sys.stderr)
            batch = pd.DataFrame()

        if not batch.empty:
            if isinstance(batch.columns, pd.MultiIndex):
                have_tickers = set(batch.columns.get_level_values(0))
                for t in remaining:
                    if t in have_tickers and "Close" in batch[t]:
                        s = batch[t]["Close"].dropna()
                        if not s.empty:
                            frames[t] = _to_daily_index(s)
            elif len(remaining) == 1 and "Close" in batch:
                s = batch["Close"].dropna()
                if not s.empty:
                    frames[remaining[0]] = _to_daily_index(s)

        missing = [t for t in remaining if t not in frames]
        if missing:
            print(
                f"correlations.fetch_prices: no data from massive or yfinance for {missing} "
                "(likely rate-limited or delisted; check Streamlit Cloud logs for the yf.download error above)",
                file=sys.stderr,
            )

    if not frames:
        return pd.DataFrame()
    # Every series is already on a tz-naive calendar-date index via
    # _to_daily_index, so this unions cleanly instead of interleaving.
    return pd.DataFrame(frames).sort_index()


def fetch_single_price(ticker: str, period: str = "2y") -> pd.Series:
    """Fetch one arbitrary ticker's daily close series (for the search-any-asset flow).

    Same massive-first, yfinance-fallback resilience as fetch_prices. Returns
    an empty Series (never None) on failure, so callers can check `.empty`.
    """
    start, end = _period_bounds(period)
    s = _fetch_one_price_series(ticker, period, start, end)
    return s if s is not None else pd.Series(dtype=float, name=ticker)


def _domain_from_website(website: str | None) -> str | None:
    if not website:
        return None
    domain = website.replace("https://", "").replace("http://", "").split("/")[0]
    return domain[4:] if domain.startswith("www.") else domain or None


def search_tickers(query: str, max_results: int = 6) -> list[dict]:
    """Free-text search for a stock, ETF, or index via yfinance.

    Returns a list of {"symbol", "name", "exchange", "sector", "industry",
    "logo_url"} dicts, empty on any failure (no network, bad query, yfinance
    error). `logo_url` is best-effort (Clearbit's public logo API, keyed off
    the company's website domain from yfinance's info); it's None when no
    domain can be found, and callers should show a fallback avatar then.
    Capped at 6 results by default (rather than the API's max of 250-ish) to
    keep the extra per-result info lookup for logos snappy.
    """
    if not query or not query.strip():
        return []
    import yfinance as yf

    try:
        results = yf.Search(query.strip(), max_results=max_results).quotes or []
    except Exception:
        return []

    out = []
    for r in results:
        symbol = r.get("symbol")
        if not symbol:
            continue
        website = None
        try:
            website = yf.Ticker(symbol).get_info().get("website")
        except Exception:
            pass
        domain = _domain_from_website(website)
        out.append({
            "symbol": symbol,
            "name": r.get("longname") or r.get("shortname") or symbol,
            "exchange": r.get("exchDisp") or r.get("exchange") or "",
            "sector": r.get("sectorDisp") or r.get("sector") or "",
            "industry": r.get("industryDisp") or r.get("industry") or "",
            "logo_url": f"https://logo.clearbit.com/{domain}" if domain else None,
        })
    return out


def get_ticker_profile_info(ticker: str) -> dict:
    """Company/asset reference info for the profile dialog, via yfinance.

    Returns an empty dict on any failure. Fields are used only as narrative
    grounding (name, sector, description) or plain display, never as
    something the LLM is asked to compute from.
    """
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).get_info() or {}
    except Exception:
        return {}
    domain = _domain_from_website(info.get("website"))
    return {
        "symbol": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "summary": info.get("longBusinessSummary"),
        "currency": info.get("currency"),
        "exchange": info.get("exchange"),
        "market_cap": info.get("marketCap"),
        "last_price": info.get("regularMarketPrice") or info.get("currentPrice"),
        "day_change_pct": info.get("regularMarketChangePercent"),
        "logo_url": f"https://logo.clearbit.com/{domain}" if domain else None,
    }


# Fewest overlapping return observations we will report a correlation from.
# Below this, a Pearson r is noise dressed up as a number: it stays in [-1, 1]
# and looks authoritative no matter how thin the sample behind it is.
MIN_CORR_OBS = 20


def _spread_daily(spread_df: pd.DataFrame) -> pd.Series:
    """Resample the stored spread frame to one daily close per calendar day."""
    if spread_df.empty:
        return pd.Series(dtype=float, name="spread")
    s = spread_df.set_index("ts")["spread"]
    s.index = pd.to_datetime(s.index).normalize()
    return s.resample("1D").last().dropna()


def _returns(s: pd.Series) -> pd.Series:
    """Daily percentage returns, with gaps left as gaps.

    fill_method=None is load-bearing, not deprecation housekeeping. pandas'
    pct_change still defaults to fill_method='pad', which forward-fills a
    missing close and then reports the resulting flat day as a real 0.0%
    return. That is fabricated data feeding a correlation, and it biases r
    toward zero rather than failing loudly. A day with no close has no return;
    say so, and let the pairwise alignment below drop it.
    """
    return s.pct_change(fill_method=None)


def _aligned_returns(
    spread_df: pd.DataFrame, prices_df: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Spread returns and asset returns on a shared calendar-date index.

    Alignment is pairwise, deliberately. Listwise deletion here (a plain
    .dropna() across the asset frame) means one asset's missing session drops
    that day for all ~17 of them, and the universe deliberately mixes trading
    calendars: futures like HO=F and NG=F keep different holidays from the
    equities, so listwise deletion silently shrinks everyone's sample to the
    intersection of every calendar at once. Each asset is instead paired with
    the spread on its own good days, in compute_current_corr/compute_rolling_corr.
    """
    if spread_df.empty or prices_df.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    sp = _returns(_spread_daily(spread_df)).dropna()
    ar = _returns(prices_df)
    idx = sp.index.intersection(ar.index)
    if idx.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    return sp.loc[idx], ar.loc[idx]


def compute_rolling_corr(
    spread_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """Rolling `window`-day Pearson correlation of spread returns vs each asset's returns.

    Returns an empty frame when no asset has `window` overlapping observations.
    """
    s, a = _aligned_returns(spread_df, prices_df)
    if s.empty or a.empty or len(s) < window:
        return pd.DataFrame()
    result = pd.DataFrame(index=s.index)
    for col in a.columns:
        result[col] = s.rolling(window).corr(a[col])
    return result.dropna(how="all")


def compute_current_corr(
    spread_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    window: int = 60,
    min_obs: int = MIN_CORR_OBS,
) -> pd.Series:
    """Scalar Pearson correlation for each asset over the most recent `window` trading days.

    An asset with fewer than `min_obs` overlapping return observations gets NaN
    rather than a number computed from a handful of days. The result carries
    `.attrs["n_obs"]`, a per-asset count of the observations actually used, so
    callers can label the figure with the window it really covers instead of
    asserting a flat "60-day".
    """
    s, a = _aligned_returns(spread_df, prices_df)
    if s.empty or a.empty:
        out = pd.Series(dtype=float, name="correlation")
        out.attrs["n_obs"] = {}
        return out

    s, a = s.tail(window), a.tail(window)
    corr, n_obs = {}, {}
    for col in a.columns:
        pair = pd.concat([s, a[col]], axis=1).dropna()
        n_obs[col] = len(pair)
        corr[col] = pair.iloc[:, 0].corr(pair.iloc[:, 1]) if len(pair) >= min_obs else float("nan")
    out = pd.Series(corr, dtype=float).rename("correlation")
    out.attrs["n_obs"] = n_obs
    return out

import pandas as pd
import pytest

from src import correlations, massive_client


def test_every_asset_has_a_category():
    for ticker, meta in correlations.ASSETS.items():
        assert meta.get("category"), f"{ticker} is missing a category"
        assert meta.get("name")
        assert meta.get("note")


def test_fetch_prices_uses_massive_when_it_returns_data(monkeypatch):
    ts = pd.date_range("2026-01-01", periods=3, freq="D")

    def fake_aggs(ticker, start, end, timespan="day"):
        return pd.DataFrame({
            "ts": ts, "open": [1, 2, 3], "high": [1, 2, 3],
            "low": [1, 2, 3], "close": [10.0, 11.0, 12.0], "volume": [1, 1, 1],
        })

    monkeypatch.setattr(massive_client, "aggs", fake_aggs)

    called_yfinance = {"count": 0}

    class _FakeTicker:
        def history(self, *a, **k):
            called_yfinance["count"] += 1
            return pd.DataFrame()

    monkeypatch.setattr("yfinance.Ticker", lambda t: _FakeTicker())

    prices = correlations.fetch_prices(period="1y")
    assert not prices.empty
    # massive served every ticker successfully, so yfinance should never be called.
    assert called_yfinance["count"] == 0
    assert list(prices["DX-Y.NYB"]) == [10.0, 11.0, 12.0]


def _fake_batch(tickers, ts, close):
    """Build a yf.download-shaped MultiIndex frame: (ticker, field) columns."""
    return pd.DataFrame(
        {(t, "Close"): close for t in tickers},
        index=ts,
    ).rename_axis(columns=["Ticker", "Price"])


def test_fetch_prices_falls_back_to_yfinance_when_massive_empty(monkeypatch):
    monkeypatch.setattr(massive_client, "aggs", lambda *a, **k: pd.DataFrame())

    ts = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")

    def fake_download(tickers, **kwargs):
        return _fake_batch(tickers, ts, [50.0, 51.0])

    monkeypatch.setattr("yfinance.download", fake_download)

    prices = correlations.fetch_prices(period="1y")
    assert not prices.empty
    assert list(prices["DX-Y.NYB"]) == [50.0, 51.0]


def test_fetch_prices_batches_yfinance_into_one_download_call(monkeypatch):
    """The batched call is a rate-limit fix; one request, not ~18."""
    monkeypatch.setattr(massive_client, "aggs", lambda *a, **k: pd.DataFrame())
    ts = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    calls = []

    def fake_download(tickers, **kwargs):
        calls.append(list(tickers))
        return _fake_batch(tickers, ts, [50.0, 51.0])

    monkeypatch.setattr("yfinance.download", fake_download)
    correlations.fetch_prices(period="1y")
    assert len(calls) == 1
    assert set(calls[0]) == set(correlations.ASSETS)


def test_fetch_prices_aligns_massive_and_yfinance_onto_one_date_index(monkeypatch):
    """massive and yfinance must land on the same index, or the frame is holes.

    massive returns the bar's window start in epoch ms, which decodes to 04:00
    UTC (midnight ET); yfinance returns tz-aware midnight. Mixed unnormalized,
    the two sources share no timestamps and every row carries a NaN.
    """
    massive_ticker = "XLE"
    ts_massive = pd.to_datetime(["2026-01-01 04:00", "2026-01-02 04:00"])
    ts_yf = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")

    def fake_aggs(ticker, start, end, timespan="day"):
        if ticker != massive_ticker:
            return pd.DataFrame()
        return pd.DataFrame({
            "ts": ts_massive, "open": [1, 2], "high": [1, 2],
            "low": [1, 2], "close": [10.0, 11.0], "volume": [1, 1],
        })

    monkeypatch.setattr(massive_client, "aggs", fake_aggs)
    monkeypatch.setattr(
        "yfinance.download",
        lambda tickers, **k: _fake_batch(tickers, ts_yf, [50.0, 51.0]),
    )

    prices = correlations.fetch_prices(period="1y")
    # One row per calendar date, with both sources populated on every row.
    assert len(prices) == 2
    assert not prices.isna().any().any()
    assert list(prices[massive_ticker]) == [10.0, 11.0]
    assert list(prices["DX-Y.NYB"]) == [50.0, 51.0]


def test_fetch_single_price_uses_shared_fallback_path(monkeypatch):
    monkeypatch.setattr(massive_client, "aggs", lambda *a, **k: pd.DataFrame())
    ts = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")

    class _FakeTicker:
        def history(self, *a, **k):
            return pd.DataFrame({"Close": [390.0, 395.0]}, index=ts)

    monkeypatch.setattr("yfinance.Ticker", lambda t: _FakeTicker())

    s = correlations.fetch_single_price("TSLA", period="1y")
    assert list(s.values) == [390.0, 395.0]


def test_fetch_single_price_empty_on_total_failure(monkeypatch):
    monkeypatch.setattr(massive_client, "aggs", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(
        "yfinance.Ticker",
        lambda t: (_ for _ in ()).throw(RuntimeError("network down")),
    )
    s = correlations.fetch_single_price("BADTICKER")
    assert s.empty


def test_search_tickers_empty_query_returns_empty_list():
    assert correlations.search_tickers("") == []
    assert correlations.search_tickers("   ") == []


def test_search_tickers_parses_yfinance_search_results(monkeypatch):
    class _FakeSearch:
        def __init__(self, query, max_results=8):
            self.quotes = [
                {"symbol": "TSLA", "longname": "Tesla, Inc.", "exchDisp": "NASDAQ",
                 "sectorDisp": "Consumer Cyclical", "industryDisp": "Auto Manufacturers"},
                {"symbol": "", "longname": "No symbol, should be skipped"},
            ]

    class _FakeTicker:
        def get_info(self):
            return {"website": "https://www.tesla.com"}

    monkeypatch.setattr("yfinance.Search", _FakeSearch)
    monkeypatch.setattr("yfinance.Ticker", lambda t: _FakeTicker())
    results = correlations.search_tickers("Tesla")
    assert len(results) == 1
    assert results[0]["symbol"] == "TSLA"
    assert results[0]["name"] == "Tesla, Inc."
    assert results[0]["exchange"] == "NASDAQ"
    assert results[0]["logo_url"] == "https://logo.clearbit.com/tesla.com"


def test_search_tickers_missing_website_gives_no_logo(monkeypatch):
    class _FakeSearch:
        def __init__(self, query, max_results=8):
            self.quotes = [{"symbol": "TSLA", "longname": "Tesla, Inc."}]

    class _FakeTicker:
        def get_info(self):
            return {}

    monkeypatch.setattr("yfinance.Search", _FakeSearch)
    monkeypatch.setattr("yfinance.Ticker", lambda t: _FakeTicker())
    results = correlations.search_tickers("Tesla")
    assert results[0]["logo_url"] is None


def test_search_tickers_per_result_info_failure_still_returns_result(monkeypatch):
    class _FakeSearch:
        def __init__(self, query, max_results=8):
            self.quotes = [{"symbol": "TSLA", "longname": "Tesla, Inc."}]

    def _boom_ticker(t):
        raise RuntimeError("info lookup failed")

    monkeypatch.setattr("yfinance.Search", _FakeSearch)
    monkeypatch.setattr("yfinance.Ticker", _boom_ticker)
    results = correlations.search_tickers("Tesla")
    assert len(results) == 1
    assert results[0]["logo_url"] is None


def test_search_tickers_returns_empty_on_exception(monkeypatch):
    def _boom(query, max_results=8):
        raise RuntimeError("network down")

    monkeypatch.setattr("yfinance.Search", _boom)
    assert correlations.search_tickers("Tesla") == []


def test_get_ticker_profile_info_shapes_fields(monkeypatch):
    class _FakeTicker:
        def get_info(self):
            return {
                "longName": "Tesla, Inc.", "sector": "Consumer Cyclical",
                "industry": "Auto Manufacturers", "longBusinessSummary": "Makes cars.",
                "currency": "USD", "exchange": "NMS", "marketCap": 123,
                "regularMarketPrice": 390.0, "regularMarketChangePercent": -1.1,
                "website": "https://www.tesla.com",
            }

    monkeypatch.setattr("yfinance.Ticker", lambda t: _FakeTicker())
    info = correlations.get_ticker_profile_info("TSLA")
    assert info["name"] == "Tesla, Inc."
    assert info["last_price"] == 390.0
    assert info["symbol"] == "TSLA"
    assert info["logo_url"] == "https://logo.clearbit.com/tesla.com"


def test_get_ticker_profile_info_empty_on_exception(monkeypatch):
    def _boom(t):
        raise RuntimeError("network down")

    monkeypatch.setattr("yfinance.Ticker", _boom)
    assert correlations.get_ticker_profile_info("BADTICKER") == {}


def _spread_frame(n=120, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    ts = pd.bdate_range("2026-01-01", periods=n)
    return pd.DataFrame({"ts": ts, "spread": 4.0 + rng.normal(0, 0.2, n).cumsum()})


def test_current_corr_matches_across_massive_and_yfinance_timestamp_conventions():
    """A massive-sourced series must correlate identically to a yfinance one.

    This is the N/A bug: the massive index sat at 04:00 UTC and never
    intersected the midnight-normalized spread index, so r came back NaN.
    """
    import numpy as np

    spread = _spread_frame()
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2026-01-01", periods=120)
    px = pd.Series(100 + rng.normal(0, 1, 120).cumsum(), index=dates)

    yf_style = correlations._to_daily_index(px.tz_localize("UTC"))
    massive_style = correlations._to_daily_index(px.set_axis(dates + pd.Timedelta(hours=4)))

    both = pd.DataFrame({"yf": yf_style, "massive": massive_style})
    corr = correlations.compute_current_corr(spread, both)
    assert corr.notna().all()
    assert corr["massive"] == pytest.approx(corr["yf"])


def test_current_corr_does_not_fabricate_returns_across_price_gaps():
    """A padded close is not a 0% return day; gaps must not become data."""
    spread = _spread_frame()
    dates = pd.bdate_range("2026-01-01", periods=120)
    px = pd.Series(range(100, 220), index=dates, dtype=float)
    gapped = px.copy()
    # Inside the trailing 60-day window, so tail() actually sees the gap.
    gapped.iloc[80:100] = float("nan")

    corr = correlations.compute_current_corr(spread, gapped.to_frame("GAPPY"))
    # The gap days are excluded rather than forward-filled into flat 0% returns.
    assert corr.attrs["n_obs"]["GAPPY"] < 60


def test_current_corr_pairs_each_asset_independently():
    """One asset's missing sessions must not shrink another asset's sample.

    The universe mixes trading calendars on purpose (futures vs equities), so
    listwise deletion across the frame would penalise every asset for one
    asset's holiday.
    """
    spread = _spread_frame()
    dates = pd.bdate_range("2026-01-01", periods=120)
    clean = pd.Series(range(100, 220), index=dates, dtype=float)
    holey = clean.copy()
    holey.iloc[::2] = float("nan")

    corr = correlations.compute_current_corr(spread, pd.DataFrame({"CLEAN": clean, "HOLEY": holey}))
    n = corr.attrs["n_obs"]
    assert n["CLEAN"] == 60, "a co-listed asset's gaps must not drop CLEAN's days"
    assert n["HOLEY"] < n["CLEAN"]


def test_current_corr_returns_nan_rather_than_a_number_from_too_few_days():
    spread = _spread_frame(n=5)
    dates = pd.bdate_range("2026-01-01", periods=5)
    px = pd.Series([100.0, 101.0, 99.0, 102.0, 103.0], index=dates)
    corr = correlations.compute_current_corr(spread, px.to_frame("THIN"))
    assert pd.isna(corr["THIN"])


def test_rolling_corr_empty_when_history_is_shorter_than_the_window():
    spread = _spread_frame(n=25)
    dates = pd.bdate_range("2026-01-01", periods=25)
    px = pd.Series(range(100, 125), index=dates, dtype=float)
    assert correlations.compute_rolling_corr(spread, px.to_frame("SHORT"), window=60).empty


def test_domain_from_website_variants():
    assert correlations._domain_from_website("https://www.tesla.com") == "tesla.com"
    assert correlations._domain_from_website("http://chevron.com/investors") == "chevron.com"
    assert correlations._domain_from_website(None) is None
    assert correlations._domain_from_website("") is None

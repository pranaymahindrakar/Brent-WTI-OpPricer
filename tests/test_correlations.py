import pandas as pd

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


def test_fetch_prices_falls_back_to_yfinance_when_massive_empty(monkeypatch):
    monkeypatch.setattr(massive_client, "aggs", lambda *a, **k: pd.DataFrame())

    ts = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")

    class _FakeTicker:
        def history(self, *a, **k):
            return pd.DataFrame({"Close": [50.0, 51.0]}, index=ts)

    monkeypatch.setattr("yfinance.Ticker", lambda t: _FakeTicker())

    prices = correlations.fetch_prices(period="1y")
    assert not prices.empty
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


def test_domain_from_website_variants():
    assert correlations._domain_from_website("https://www.tesla.com") == "tesla.com"
    assert correlations._domain_from_website("http://chevron.com/investors") == "chevron.com"
    assert correlations._domain_from_website(None) is None
    assert correlations._domain_from_website("") is None

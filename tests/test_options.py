import pandas as pd

from src import massive_client, options


def _fake_massive_body():
    return {
        "results": [
            {
                "details": {"expiration_date": "2026-08-21", "strike_price": 80.0, "contract_type": "call"},
                "greeks": {"delta": 0.55, "gamma": 0.02, "theta": -0.03, "vega": 0.12},
                "implied_volatility": 0.32,
                "open_interest": 500,
                "day": {"volume": 100, "close": 3.2},
                "last_quote": {"bid": 3.1, "ask": 3.3},
                "underlying_asset": {"price": 79.5},
            },
            {
                "details": {"expiration_date": "2026-08-21", "strike_price": 80.0, "contract_type": "put"},
                "greeks": {"delta": -0.45, "gamma": 0.02, "theta": -0.025, "vega": 0.11},
                "implied_volatility": 0.34,
                "open_interest": 300,
                "day": {"volume": 80, "close": 2.9},
                "last_quote": {"bid": 2.8, "ask": 3.0},
                "underlying_asset": {"price": 79.5},
            },
        ]
    }


def test_parse_massive_chain_shape_and_values():
    chain, underlying_price = options._parse_massive_chain(_fake_massive_body())
    assert underlying_price == 79.5
    assert list(chain.columns) == options.CHAIN_COLS
    assert set(chain["contract_type"]) == {"call", "put"}
    assert chain.loc[chain["contract_type"] == "call", "delta"].iloc[0] == 0.55


def test_parse_massive_chain_empty_results():
    chain, underlying_price = options._parse_massive_chain({"results": []})
    assert chain.empty
    assert underlying_price is None


def test_atm_summary_picks_nearest_strike_and_computes_ratio():
    chain = pd.DataFrame([
        {"expiration": "2026-08-21", "strike": 78.0, "contract_type": "call", "iv": 0.30,
         "delta": 0.6, "gamma": 0.02, "theta": -0.03, "vega": 0.1, "open_interest": 200,
         "volume": 10, "bid": 2.0, "ask": 2.1, "last": 2.05},
        {"expiration": "2026-08-21", "strike": 80.0, "contract_type": "call", "iv": 0.32,
         "delta": 0.5, "gamma": 0.02, "theta": -0.03, "vega": 0.1, "open_interest": 500,
         "volume": 10, "bid": 3.1, "ask": 3.3, "last": 3.2},
        {"expiration": "2026-08-21", "strike": 80.0, "contract_type": "put", "iv": 0.34,
         "delta": -0.5, "gamma": 0.02, "theta": -0.03, "vega": 0.1, "open_interest": 300,
         "volume": 10, "bid": 2.8, "ask": 3.0, "last": 2.9},
    ])
    summary = options.atm_summary(chain, underlying_price=79.8)
    assert summary["available"] is True
    assert summary["atm_strike"] == 80.0
    assert abs(summary["atm_iv"] - 0.33) < 1e-9
    # put/call OI ratio is chain-wide (a broader positioning signal), not
    # restricted to the ATM strike: puts=300, calls=200+500=700.
    assert abs(summary["put_call_oi_ratio"] - (300 / 700)) < 1e-9


def test_atm_summary_unavailable_without_underlying_price():
    chain = pd.DataFrame(columns=options.CHAIN_COLS)
    assert options.atm_summary(chain, underlying_price=None) == {"available": False}


def test_fetch_chain_uses_massive_when_available(monkeypatch):
    monkeypatch.setattr(massive_client, "option_chain_snapshot", lambda *a, **k: _fake_massive_body())
    result = options.fetch_chain("USO", "2026-08-21")
    assert result["source"] == "massive"
    assert not result["chain"].empty


def test_fetch_chain_falls_back_to_yfinance_when_massive_empty(monkeypatch):
    monkeypatch.setattr(massive_client, "option_chain_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(
        options, "_fallback_yfinance_chain",
        lambda ticker, expiration: (pd.DataFrame(columns=options.CHAIN_COLS), 80.0),
    )
    result = options.fetch_chain("USO", "2026-08-21")
    assert result["source"] == "yfinance"
    assert result["underlying_price"] == 80.0

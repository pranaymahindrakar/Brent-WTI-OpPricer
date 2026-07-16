import numpy as np
import pandas as pd
import pytest

from src import chatbot, config, store


def _con_with_spread():
    con = store.connect(":memory:")
    ts = pd.date_range("2026-01-01", periods=70, freq="D")
    spread_vals = 3.0 + 0.1 * (np.arange(70) % 10)
    df = pd.DataFrame({
        "ts": ts, "brent": 80.0, "wti": 80.0 - spread_vals, "spread": spread_vals,
        "zscore": (spread_vals - spread_vals.mean()) / spread_vals.std(),
        "roll_mean": spread_vals.mean(), "roll_std": spread_vals.std(),
        "corr": 0.9, "pct_range": 0.5,
    })
    store.write_spread(con, df)
    return con


def test_build_payload_merges_extra():
    con = _con_with_spread()
    payload = chatbot.build_payload(con, extra={"news_sentiment": {"label": "bullish"}})
    assert "spread" in payload
    assert "zscore" in payload
    assert payload["news_sentiment"] == {"label": "bullish"}


def test_answer_raises_without_key(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    con = _con_with_spread()
    with pytest.raises(RuntimeError):
        chatbot.answer(con, "what's the z-score?", history=[])

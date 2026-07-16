import pytest

from src import config, insights


def test_narrate_metric_raises_without_key(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    with pytest.raises(RuntimeError):
        insights.narrate_metric({"best_1d": {"name": "XLE", "pct": 1.2}}, "Explain this.")

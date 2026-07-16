import pandas as pd

from src import config, massive_client


def test_no_key_option_chain_snapshot_returns_none(monkeypatch):
    monkeypatch.setattr(config, "MASSIVE_API_KEY", "")
    assert massive_client.option_chain_snapshot("USO") is None


def test_no_key_aggs_returns_empty_dataframe(monkeypatch):
    monkeypatch.setattr(config, "MASSIVE_API_KEY", "")
    out = massive_client.aggs("USO", "2026-01-01", "2026-02-01")
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_no_key_futures_snapshot_returns_none(monkeypatch):
    monkeypatch.setattr(config, "MASSIVE_API_KEY", "")
    assert massive_client.futures_snapshot(["CL", "BZ"]) is None


def test_request_exception_falls_back_to_none(monkeypatch):
    monkeypatch.setattr(config, "MASSIVE_API_KEY", "fake-key-for-test")

    def _boom(*args, **kwargs):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(massive_client.requests, "get", _boom)
    assert massive_client.option_chain_snapshot("USO") is None
    assert massive_client.futures_snapshot(["CL"]) is None
    out = massive_client.aggs("USO", "2026-01-01", "2026-02-01")
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_non_200_response_falls_back_to_none(monkeypatch):
    monkeypatch.setattr(config, "MASSIVE_API_KEY", "fake-key-for-test")

    class _FakeResp:
        def raise_for_status(self):
            raise massive_client.requests.exceptions.HTTPError("500")

    monkeypatch.setattr(massive_client.requests, "get", lambda *a, **k: _FakeResp())
    assert massive_client.option_chain_snapshot("USO") is None


def test_aggs_parses_successful_response(monkeypatch):
    monkeypatch.setattr(config, "MASSIVE_API_KEY", "fake-key-for-test")

    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {"t": 1735689600000, "o": 80.0, "h": 81.0, "l": 79.5, "c": 80.5, "v": 1000},
                    {"t": 1735776000000, "o": 80.5, "h": 82.0, "l": 80.0, "c": 81.5, "v": 1200},
                ]
            }

    monkeypatch.setattr(massive_client.requests, "get", lambda *a, **k: _FakeResp())
    out = massive_client.aggs("USO", "2025-01-01", "2025-01-02")
    assert list(out["close"]) == [80.5, 81.5]
    assert out["ts"].is_monotonic_increasing

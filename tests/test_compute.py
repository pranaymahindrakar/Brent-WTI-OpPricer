import numpy as np
import pandas as pd

from src import compute


def test_align_drops_mismatched_timestamps():
    brent = pd.DataFrame({
        "ts": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:01"]),
        "close": [80.0, 81.0],
    })
    wti = pd.DataFrame({
        "ts": pd.to_datetime(["2026-01-01 00:01", "2026-01-01 00:02"]),
        "close": [75.0, 76.0],
    })
    out = compute.align(brent, wti, freq="1min")
    # Only 00:01 is shared, so the spread is computed on exactly one aligned row.
    assert len(out) == 1
    assert out.iloc[0]["brent"] == 81.0
    assert out.iloc[0]["wti"] == 75.0


def test_spread_and_zscore_math():
    n = 50
    ts = pd.date_range("2026-01-01", periods=n, freq="min")
    df = pd.DataFrame({
        "ts": ts,
        "brent": np.linspace(80, 85, n),
        "wti": np.linspace(75, 79, n),
    })
    out = compute.compute_spread(df, window=10)
    assert np.allclose(out["spread"], df["brent"] - df["wti"])
    last = out.iloc[-1]
    expected_z = (last["spread"] - last["roll_mean"]) / last["roll_std"]
    assert abs(last["zscore"] - expected_z) < 1e-9


def test_zscore_nan_when_spread_constant():
    ts = pd.date_range("2026-01-01", periods=30, freq="min")
    df = pd.DataFrame({"ts": ts, "brent": [80.0] * 30, "wti": [75.0] * 30})
    out = compute.compute_spread(df, window=10)
    # Constant spread gives zero std, which must yield NaN rather than inf.
    assert out["zscore"].dropna().empty


def test_units_band_filters_implausible_prices():
    brent = pd.DataFrame({
        "ts": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:01"]),
        "close": [80.0, 9999.0],  # second row is outside the plausible band
    })
    wti = pd.DataFrame({
        "ts": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:01"]),
        "close": [75.0, 76.0],
    })
    out = compute.align(brent, wti, freq="1min")
    assert (out["brent"] <= 400.0).all()
    assert len(out) == 1

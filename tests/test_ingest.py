from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src import ingest

ET = ZoneInfo("America/New_York")


def test_bad_tick_filter_rejects_spike_and_keeps_following_good_row():
    df = pd.DataFrame({
        "ts": pd.date_range("2026-01-01", periods=4, freq="min"),
        "close": [80.0, 80.5, 200.0, 81.0],  # 200 is a bad print
    })
    out = ingest.filter_bad_ticks(df, pct=0.10)
    assert 200.0 not in out["close"].values
    # 81.0 is measured against the last good value (80.5), not against the spike.
    assert 81.0 in out["close"].values
    assert len(out) == 3


def test_market_closed_on_saturday():
    sat = datetime(2026, 1, 3, 12, 0, tzinfo=ET)  # 2026-01-03 is a Saturday
    assert ingest.is_market_open(sat) is False


def test_market_open_weekday_midday():
    wed = datetime(2026, 1, 7, 12, 0, tzinfo=ET)  # Wednesday noon
    assert ingest.is_market_open(wed) is True


def test_market_closed_during_daily_break():
    wed = datetime(2026, 1, 7, 17, 30, tzinfo=ET)  # inside the 17:00 to 18:00 ET break
    assert ingest.is_market_open(wed) is False


def test_sunday_opens_after_break():
    sun_before = datetime(2026, 1, 4, 12, 0, tzinfo=ET)   # Sunday midday, still closed
    sun_after = datetime(2026, 1, 4, 19, 0, tzinfo=ET)    # Sunday 19:00 ET, open
    assert ingest.is_market_open(sun_before) is False
    assert ingest.is_market_open(sun_after) is True

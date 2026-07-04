"""Rule-based and calendar annotations.

Layered so the cheap, deterministic rules carry the load and the calendar adds
context. Rule and calendar annotations are rebuilt each run; manual annotations
added in the UI are preserved.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src import config, store

ANN_COLS = ["ts", "kind", "severity", "text", "source"]
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=ANN_COLS)


def rule_annotations(spread: pd.DataFrame) -> pd.DataFrame:
    """Generate annotations from the computed spread frame."""
    if spread is None or spread.empty:
        return _empty()
    s = spread.dropna(subset=["zscore"]).sort_values("ts").reset_index(drop=True)
    if s.empty:
        return _empty()
    rows = []

    # z-score threshold crossings (only the moment of crossing, not every bar over)
    z = s["zscore"]
    crossed = (z.abs() >= config.Z_ALERT) & (z.shift(1).abs() < config.Z_ALERT)
    for _, r in s[crossed].iterrows():
        rows.append({
            "ts": r["ts"], "kind": "zscore", "severity": "high",
            "text": f"Spread z-score crossed to {r['zscore']:.2f}", "source": "rule",
        })

    # new window high / low in the spread itself.
    # Compare against the max/min of the PRIOR w bars (shift(1) excludes the current
    # bar) so the annotation fires on a genuine breakout, not on every bar that is
    # merely equal to its own rolling extreme.  A cooldown of w//2 bars further
    # prevents continuous re-firing during a sustained trend without masking
    # distinct breakouts separated by a consolidation.
    w = config.NEW_EXTREME_WINDOW
    cooldown = max(1, w // 2)
    prior_max = s["spread"].rolling(w, min_periods=1).max().shift(1)
    prior_min = s["spread"].rolling(w, min_periods=1).min().shift(1)
    raw_new_high = s["spread"] > prior_max.fillna(-float("inf"))
    raw_new_low = s["spread"] < prior_min.fillna(float("inf"))

    def _apply_cooldown(mask: pd.Series, cd: int) -> list:
        out, last = [], -cd
        for i, v in enumerate(mask):
            if v and (i - last) >= cd:
                out.append(True); last = i
            else:
                out.append(False)
        return out

    hi_flags = _apply_cooldown(raw_new_high.reset_index(drop=True), cooldown)
    lo_flags = _apply_cooldown(raw_new_low.reset_index(drop=True), cooldown)
    for idx, r in s[hi_flags].iterrows():
        rows.append({
            "ts": r["ts"], "kind": "new_high", "severity": "info",
            "text": f"Spread new {w}-bar high at {r['spread']:.2f}", "source": "rule",
        })
    for idx, r in s[lo_flags].iterrows():
        rows.append({
            "ts": r["ts"], "kind": "new_low", "severity": "info",
            "text": f"Spread new {w}-bar low at {r['spread']:.2f}", "source": "rule",
        })

    # volatility spike: short-window dispersion well above the baseline
    short = s["spread"].rolling(5, min_periods=3).std()
    base = s["spread"].rolling(config.ZSCORE_WINDOW, min_periods=10).std()
    spike = short > (config.VOL_SPIKE_MULT * base)
    for _, r in s[spike].iterrows():
        rows.append({
            "ts": r["ts"], "kind": "vol_spike", "severity": "high",
            "text": "Spread volatility spike", "source": "rule",
        })

    return pd.DataFrame(rows, columns=ANN_COLS)


def _load_events(path=None) -> dict:
    path = Path(path or Path(__file__).parent / "events.yaml")
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def calendar_annotations(events: dict) -> pd.DataFrame:
    rows = []
    for ev in events.get("dated", []) or []:
        rows.append({
            "ts": pd.to_datetime(ev["date"]), "kind": ev.get("kind", "event"),
            "severity": ev.get("severity", "info"), "text": ev["text"], "source": "calendar",
        })
    return pd.DataFrame(rows, columns=ANN_COLS)


def recurring_annotations(spread: pd.DataFrame, events: dict) -> pd.DataFrame:
    """Expand recurring weekly rules (for example the EIA report) across the span."""
    if spread is None or spread.empty:
        return _empty()
    start = pd.to_datetime(spread["ts"]).min().normalize()
    end = pd.to_datetime(spread["ts"]).max().normalize()
    if pd.isna(start) or pd.isna(end) or start > end:
        return _empty()
    days = pd.date_range(start, end, freq="D")
    rows = []
    for rule in events.get("recurring", []) or []:
        if rule.get("rule") != "weekly":
            continue
        wd = _WEEKDAYS.get(str(rule.get("weekday", "")).lower())
        if wd is None:
            continue
        for d in days[days.weekday == wd]:
            rows.append({
                "ts": d, "kind": rule.get("kind", "event"), "severity": "info",
                "text": rule.get("text", "recurring event"), "source": "calendar",
            })
    return pd.DataFrame(rows, columns=ANN_COLS)


def build(con, spread: pd.DataFrame, events_path=None) -> pd.DataFrame:
    """Rebuild rule and calendar annotations, preserving manual ones."""
    events = _load_events(events_path)
    con.execute("DELETE FROM annotations WHERE source IN ('rule', 'calendar')")
    parts = [
        rule_annotations(spread),
        calendar_annotations(events),
        recurring_annotations(spread, events),
    ]
    parts = [p for p in parts if p is not None and not p.empty]
    ann = pd.concat(parts, ignore_index=True) if parts else _empty()
    for _, r in ann.iterrows():
        store.write_annotation(con, r["ts"], r["kind"], r["severity"], r["text"], r["source"])
    return ann

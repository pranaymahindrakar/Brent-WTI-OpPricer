"""DuckDB storage layer: schema, upserts, and range reads.

Four tables keep concerns separated so the deterministic numbers and the
generated narrative never get muddled together:
  bars        raw and backfilled OHLCV per leg
  spread      the aligned spread and its rolling statistics
  annotations rule, calendar, and manual notes pinned to timestamps
  insights    grounded LLM notes with their input payloads
"""
from __future__ import annotations

from typing import Optional

import duckdb
import pandas as pd

from src import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    ts        TIMESTAMP,
    symbol    VARCHAR,
    open      DOUBLE,
    high      DOUBLE,
    low       DOUBLE,
    close     DOUBLE,
    volume    DOUBLE,
    source    VARCHAR,
    PRIMARY KEY (ts, symbol)
);
CREATE TABLE IF NOT EXISTS spread (
    ts        TIMESTAMP PRIMARY KEY,
    brent     DOUBLE,
    wti       DOUBLE,
    spread    DOUBLE,
    zscore    DOUBLE,
    roll_mean DOUBLE,
    roll_std  DOUBLE,
    corr      DOUBLE,
    pct_range DOUBLE
);
CREATE TABLE IF NOT EXISTS annotations (
    ts        TIMESTAMP,
    kind      VARCHAR,
    severity  VARCHAR,
    text      VARCHAR,
    source    VARCHAR
);
CREATE TABLE IF NOT EXISTS insights (
    ts           TIMESTAMP,
    payload_json VARCHAR,
    note_text    VARCHAR,
    model        VARCHAR
);
"""

BARS_COLS = ["ts", "symbol", "open", "high", "low", "close", "volume", "source"]
SPREAD_COLS = ["ts", "brent", "wti", "spread", "zscore", "roll_mean", "roll_std", "corr", "pct_range"]


def connect(db_path: Optional[str] = None) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection and ensure the schema exists."""
    con = duckdb.connect(str(db_path or config.DB_PATH))
    for stmt in SCHEMA.strip().split(";"):
        s = stmt.strip()
        if s:
            con.execute(s)
    return con


def write_bars(con, df: pd.DataFrame) -> int:
    """Upsert bars. Duplicate (ts, symbol) rows are ignored."""
    if df is None or df.empty:
        return 0
    df = df[BARS_COLS]
    con.register("df_bars", df)
    con.execute(
        "INSERT INTO bars SELECT * FROM df_bars ON CONFLICT (ts, symbol) DO NOTHING"
    )
    con.unregister("df_bars")
    return len(df)


def write_spread(con, df: pd.DataFrame) -> int:
    """Upsert spread rows; existing timestamps are updated in place."""
    if df is None or df.empty:
        return 0
    df = df[SPREAD_COLS]
    con.register("df_spread", df)
    con.execute(
        "INSERT INTO spread SELECT * FROM df_spread "
        "ON CONFLICT (ts) DO UPDATE SET "
        "brent=excluded.brent, wti=excluded.wti, spread=excluded.spread, "
        "zscore=excluded.zscore, roll_mean=excluded.roll_mean, "
        "roll_std=excluded.roll_std, corr=excluded.corr, pct_range=excluded.pct_range"
    )
    con.unregister("df_spread")
    return len(df)


def write_annotation(con, ts, kind, severity, text, source) -> None:
    con.execute(
        "INSERT INTO annotations VALUES (?, ?, ?, ?, ?)",
        [ts, kind, severity, text, source],
    )


def write_insight(con, ts, payload_json, note_text, model) -> None:
    con.execute(
        "INSERT INTO insights VALUES (?, ?, ?, ?)",
        [ts, payload_json, note_text, model],
    )


def _range(con, table: str, extra: str = "", params=None) -> pd.DataFrame:
    params = params or []
    return con.execute(f"SELECT * FROM {table} {extra} ORDER BY ts", params).df()


def read_bars(con, symbol=None, start=None, end=None) -> pd.DataFrame:
    clauses, params = [], []
    if symbol:
        clauses.append("symbol = ?"); params.append(symbol)
    if start:
        clauses.append("ts >= ?"); params.append(start)
    if end:
        clauses.append("ts <= ?"); params.append(end)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return _range(con, "bars", where, params)


def read_spread(con, start=None, end=None) -> pd.DataFrame:
    clauses, params = [], []
    if start:
        clauses.append("ts >= ?"); params.append(start)
    if end:
        clauses.append("ts <= ?"); params.append(end)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return _range(con, "spread", where, params)


def read_annotations(con, start=None, end=None) -> pd.DataFrame:
    clauses, params = [], []
    if start:
        clauses.append("ts >= ?"); params.append(start)
    if end:
        clauses.append("ts <= ?"); params.append(end)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return _range(con, "annotations", where, params)


def read_insights(con, start=None, end=None) -> pd.DataFrame:
    """Read stored insight notes ordered by timestamp."""
    clauses, params = [], []
    if start:
        clauses.append("ts >= ?"); params.append(start)
    if end:
        clauses.append("ts <= ?"); params.append(end)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return _range(con, "insights", where, params)

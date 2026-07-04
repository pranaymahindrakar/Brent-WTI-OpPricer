# Brent-WTI Spread Tracker

A near-real-time tracker for the Brent-minus-WTI crude spread. It ingests
front-month WTI and Brent futures, computes the spread and its rolling z-score,
annotates the chart with rule-based and calendar events, and produces grounded
natural-language insight notes through the Anthropic API. v1 ships as a Streamlit
and Plotly dashboard backed by DuckDB.

This repo is structured to be handed to Claude Code. Read `CLAUDE.md` for the
project rules and `PLAN.md` for the phased build. The code here is a working v1
skeleton that already implements Phases 0 through 6 in a basic form; use the plan
to harden, extend, and validate each piece.

## Data sources

- Live legs: yfinance, symbols `CL=F` (WTI) and `BZ=F` (Brent Last Day Financial,
  a faithful proxy for ICE Brent). Delayed by roughly ten to fifteen minutes.
- History and reconciliation: FRED series `DCOILWTICO` and `DCOILBRENTEU`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then fill in ANTHROPIC_API_KEY and FRED_API_KEY
```

## Use

```bash
python -m src.backfill             # seed multi-year daily history from FRED
streamlit run src/app.py           # launch the dashboard, then click Refresh data
pytest -q                          # run the unit tests
```

## Layout

```
src/config.py     env and constants
src/store.py      DuckDB schema and read/write
src/ingest.py     yfinance pollers, market-hours logic, bad-tick filter
src/backfill.py   FRED history seed and reconciliation
src/compute.py    alignment, spread, z-score, rolling stats
src/annotate.py   rule-based and calendar annotations
src/insights.py   grounded Anthropic API note
src/app.py        Streamlit and Plotly dashboard
src/events.yaml   curated event calendar
tests/            unit tests for the compute and ingest logic
```

## Known v1 limitations, by design

- The yfinance feed is delayed and can be flaky at the one-minute interval. If it
  misbehaves, widen `BAR_INTERVAL` or add a freemium REST cross-check source.
- Refresh is manual via a button rather than a background scheduler, since the
  Streamlit execution model fights long-lived schedulers. Auto-refresh can be
  added with `streamlit-autorefresh` later.
- Calendar annotations align to spread timestamps by exact match, so daily-dated
  events may not pin onto intraday bars. A nearest-asof merge is a later refinement.
- `events.yaml` ships with placeholder dates. Fill in real OPEC, FOMC, and other
  verified events before relying on them.

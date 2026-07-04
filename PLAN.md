# PLAN.md — Brent-WTI Spread Tracker build plan

Execute one phase at a time. After each phase: run the listed checks, commit, then clear
context before starting the next phase. Do not start a phase until the previous one passes
its checks. Use plan mode to propose the work for each phase before implementing it.

---

## Phase 0 — Scaffold
Goal: an empty but runnable skeleton.
- Create the directory structure listed in CLAUDE.md.
- requirements.txt: yfinance, pandas, numpy, duckdb, apscheduler, anthropic, streamlit,
  plotly, pyyaml, python-dotenv, fredapi, pytest.
- .env.example with: ANTHROPIC_API_KEY, FRED_API_KEY (optional), EIA_API_KEY (optional).
- src/config.py: load env, define constants (symbols, DB path, z-score window, bad-tick
  threshold, CME Globex hours).
- README.md with setup and run instructions.
Checks: `pip install -r requirements.txt` succeeds; `python -c "import src.config"` runs.

## Phase 1 — Ingestion and storage
Goal: live bars landing in DuckDB, cleanly.
- src/store.py: DuckDB schema with four tables:
  - bars(ts, symbol, open, high, low, close, volume, source)
  - spread(ts, brent, wti, spread, zscore, roll_mean, roll_std)
  - annotations(ts, kind, severity, text, source)
  - insights(ts, payload_json, note_text, model)
  Provide upsert and range-read helpers.
- src/ingest.py: poll CL=F and BZ=F via yfinance on a chosen interval (default 1m bars,
  refreshed every 60s). Implement: market-hours check against CME Globex, bad-tick filter,
  dedup on (ts, symbol), and a freshness timestamp. Write bars to store.
Checks: running ingest for a few minutes during market hours populates bars; closed-market
runs log "market closed" rather than erroring.

## Phase 2 — History backfill and validation
Goal: a real distribution behind the statistics.
- src/backfill.py: pull DCOILWTICO and DCOILBRENTEU from FRED for several years, write daily
  bars to the store tagged source='fred'. If EIA_API_KEY present, cross-check.
- Add a reconciliation function that compares the latest live daily close against FRED/EIA and
  logs drift beyond a tolerance.
Checks: `python -m src.backfill` populates multi-year daily history; reconciliation runs.

## Phase 3 — Spread and z-score compute
Goal: the analytical core, correct and tested.
- src/compute.py:
  - align(brent_df, wti_df): resample both to a common grid, inner-join on ts, assert USD/bbl.
  - spread = brent.close - wti.close, computed only on aligned rows.
  - rolling mean, rolling std, and z-score = (spread - roll_mean) / roll_std over a configurable
    window. Add rolling correlation and percent-of-range.
  - write results to the spread table.
- tests/test_compute.py: alignment never differences mismatched timestamps; z-score math is
  correct on a synthetic series; bad-tick filter rejects an injected spike.
Checks: `pytest -q` green; spread table populates from stored bars.

## Phase 4 — Annotations
Goal: the chart explains itself.
- src/events.yaml: curated calendar. Seed with the EIA Weekly Petroleum Status Report
  (Wednesdays 10:30 ET), known OPEC and OPEC+ meeting dates, and FOMC dates.
- src/annotate.py:
  - rule-based: fire on z-score threshold crossings, volatility spikes, and new N-day highs or
    lows in the spread.
  - calendar: map events.yaml entries onto the timeline.
  - write annotations to the store with kind and severity.
Checks: feeding historical spread data produces sensible annotations at known volatile dates.

## Phase 5 — Dashboard
Goal: a usable Streamlit + Plotly tracker.
- src/app.py:
  - Plotly figure: spread line with rolling bands, z-score subplot, annotations rendered as
    markers with hover text.
  - APScheduler job refreshing ingest and compute on an interval.
  - A visible data-freshness indicator and a manual annotation input that persists to the store.
Checks: `streamlit run src/app.py` shows a live-updating spread with z-score and annotations.

## Phase 6 — Grounded LLM insight
Goal: trustworthy narrative, zero invented numbers.
- src/insights.py:
  - assemble a payload: current spread, z-score, regime stats, recent annotations.
  - call the Anthropic API with a system prompt instructing: use only supplied numbers, mark
    uncertainty, return JSON pairing each claim with its supporting datapoint, no arithmetic.
  - parse the JSON, store the note, render it in an insight panel in the app.
Checks: the note references only passed-in figures; malformed JSON is handled gracefully.

---

## Later phases (do not build in v1, listed for direction)
- Term structure: pull M1..M6 for both benchmarks and chart the spread across the curve.
- News-pinned annotations via a news API (GDELT free, or Marketaux/Finnhub).
- Mean-reversion half-life via an Ornstein-Uhlenbeck fit and a paper-trading signal overlay.
- Product-grade frontend: React + Vite + TypeScript on Supabase realtime, charts via
  TradingView Lightweight Charts.
- Real-time data upgrade: swap the yfinance poller for a Databento or Polygon websocket. Keep
  ingest.py modular so this is a one-module change.

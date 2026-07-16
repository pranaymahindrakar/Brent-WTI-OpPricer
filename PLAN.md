# PLAN.md — Brent-WTI Spread Tracker build plan

Execute one phase at a time. After each phase: run the listed checks, commit, then clear
context before starting the next phase. Do not start a phase until the previous one passes
its checks. Use plan mode to propose the work for each phase before implementing it.

---

## Phase 0 — Scaffold
Goal: an empty but runnable skeleton.
- Create the directory structure listed in CLAUDE.md.
- requirements.txt: yfinance, pandas, numpy, duckdb, apscheduler, google-genai, streamlit,
  plotly, pyyaml, python-dotenv, fredapi, pytest.
- .env.example with: GEMINI_API_KEY, FRED_API_KEY (optional), EIA_API_KEY (optional).
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
  - call the Google Gemini API with a system prompt instructing: use only supplied numbers,
    mark uncertainty, return JSON pairing each claim with its supporting datapoint, no
    arithmetic. (Originally built against the Anthropic API; migrated to Gemini, see git log.)
  - parse the JSON, store the note, render it in an insight panel in the app.
Checks: the note references only passed-in figures; malformed JSON is handled gracefully.

---

## v2: Options, energy trends, research, and chatbot (phases 7-13)

Built in response to a request to substantially expand the tracker: full price/data
enablement, broader oil and energy market context, live options data with Greeks, a
documented pricing methodology, more correlated instruments, visual/animation polish, and a
persistent chatbot. Superseded the "Later phases" stub that used to live here (term structure
and mean-reversion direction folded into Phase 7's OU fit; news-pinned annotations already
covered by the existing Marketaux integration in news.py).

Key architectural note: massive.com (api.massive.com, keyed via MASSIVE_API_KEY) is used as a
primary, defensive data source for the new tabs only, always with a yfinance fallback. It is
a different thing from the Claude Code MCP connector for massive (mcp.massive.com), which is
only reachable from an interactive coding session and is not part of the running app. See
CLAUDE.md's tech stack and forbidden-actions sections for the exact boundary.

### Phase 7 — Pricing engine + Research/Methodology tab
- src/pricing.py: Black-Scholes price + Greeks (delta, gamma, theta, vega, rho), an
  implied-vol bisection solver, and an OU mean-reversion fit (AR(1) least squares, no
  statsmodels) with a half-life guard that refuses fits spanning fewer than 5 estimated
  half-lives (avoids reading finite-sample AR(1) bias on a random walk as real reversion).
- New Research tab in app.py: alignment/z-score/correlation formulas as `st.latex`, a live OU
  half-life fit on the actual spread history, and the Black-Scholes formula set with a worked
  example (real once Phase 9's options snapshot exists, synthetic otherwise).
Checks: `pytest -q` (tests/test_pricing.py) green; Research tab renders a finite half-life.

### Phase 8 — massive REST client
- src/massive_client.py: thin `requests`-based wrapper (option chain snapshots, aggs,
  futures snapshots), `Authorization: Bearer` auth, every function fails safe to None/empty
  on missing key, timeout, or non-200, so callers always have a working yfinance fallback.
Checks: tests/test_massive_client.py green with no key set (fallback contract) and with a
real key (manual check).

### Phase 9 — Options & Greeks tab
- src/options.py: live chains for USO/XLE, massive first (real greeks/IV/open interest),
  yfinance + local Black-Scholes fallback otherwise.
- store.py: options_snapshot table, written once per (ticker, day) so a rolling
  IV-vs-spread-z-score correlation can accumulate over time.
- New Options tab: chain table, Greeks by strike, IV skew, accumulating correlation chart.
Checks: tests/test_options.py green; a real snapshot lands in DuckDB when the tab runs.

### Phase 10 — Energy Trends tab + Market Lens expansion
- correlations.py: ASSETS expanded to ~18 tickers across Majors/Refiners/Services/Products/
  Crude ETFs/Macro/Sector ETF/Energy Complex categories; fetch_prices tries massive per
  ticker, falls back to yfinance per ticker.
- New Energy Trends tab: performance heatmap, rolling 20-day annualized volatility (assets
  vs. the spread itself), spread distribution histogram, per-category movement cards.
- Market Lens: category filter, movement-card row.
Checks: tests/test_correlations.py green; both tabs render with and without MASSIVE_API_KEY.

### Phase 11 — Floating chatbot
- src/chatbot.py: Gemini-grounded multi-turn Q&A, same non-negotiable rules as insights.py
  (use only supplied numbers, no arithmetic, mark uncertainty).
- app.py: bottom-left floating panel via `st.container(key=...)` + CSS targeting the
  `st-key-*` class, toggle button, `st.chat_message`/`st.chat_input`.
Checks: tests/test_chatbot.py green; AppTest round-trip (open panel, ask, get grounded
answer) produces zero exceptions.

### Phase 12 — Visual and animation polish pass
- Global CSS: gradient hover-lift on every `st.metric` app-wide, a pulse animation on the
  z-score card when `|zscore| > Z_ALERT`, hover-lift on movement cards, a category icon set,
  a spread sparkline on the Dashboard metrics row.
Checks: AppTest zero exceptions across all tabs; no literal screenshot taken (no headless
browser in the dev sandbox this was built in) — worth a manual look before relying on it.

### Phase 13 — Docs
- This section of PLAN.md, plus CLAUDE.md's tech stack/directory notes/forbidden actions,
  plus README.md's tab list and data-sources section, updated to match what's actually built.

---

## Later phases (do not build yet, listed for direction)
- Term structure: pull M1..M6 for both benchmarks and chart the spread across the curve
  (massive's /futures/v1/aggs endpoint could support this; not yet wired up).
- Product-grade frontend: React + Vite + TypeScript on Supabase realtime, charts via
  TradingView Lightweight Charts. Still explicitly out of scope per CLAUDE.md.
- Real-time data upgrade: swap the yfinance poller for a Databento or Polygon websocket in
  the core ingestion path. Keep ingest.py modular so this is a one-module change; note this
  is a different, larger step than massive's current defensive/supplementary role.
- A paper-trading signal overlay on top of the Phase 7 OU mean-reversion fit.

# Brent-WTI Spread Tracker

A near-real-time tracker for the Brent-minus-WTI crude spread. It ingests
front-month WTI and Brent futures, computes the spread and its rolling z-score,
annotates the chart with rule-based and calendar events, and produces grounded
natural-language insight notes and chatbot answers through the Google Gemini
API. v1 ships as a Streamlit and Plotly dashboard backed by DuckDB; v2 adds
live options/Greeks, a broader energy-market view, a pricing-methodology tab,
and a floating chatbot, all on the same accuracy guardrails as v1.

This repo is structured to be handed to Claude Code. Read `CLAUDE.md` for the
project rules and `PLAN.md` for the phased build. The code here already
implements PLAN.md's Phases 0-13; use the plan to harden, extend, and validate
further pieces.

## Tabs

- **Dashboard**: live spread chart with z-score, rolling bands, and annotations.
- **Market Lens**: rolling correlations against ~18 categorized crude/energy/
  macro assets, with a category filter and movement cards.
- **Energy Trends**: performance heatmap, rolling volatility, and spread
  distribution across the same asset universe.
- **Options**: live option chains, Greeks, and IV skew for USO/XLE.
- **Research**: the pricing methodology, live: alignment/z-score formulas, an
  Ornstein-Uhlenbeck mean-reversion half-life fit, and the Black-Scholes math
  behind the Options tab.
- **News**: Yahoo Finance, EIA RSS, NewsAPI, and Marketaux headlines with
  per-entity sentiment.
- **About**: educational background on WTI, Brent, the spread, and market context.
- A floating chatbot (bottom-left toggle) answers ad-hoc questions grounded in
  whatever's already computed on the other tabs.

## Data sources

- Live legs: yfinance, symbols `CL=F` (WTI) and `BZ=F` (Brent Last Day Financial,
  a faithful proxy for ICE Brent). Delayed by roughly ten to fifteen minutes.
- History and reconciliation: FRED series `DCOILWTICO` and `DCOILBRENTEU`.
- News: EIA RSS (no key), NewsAPI and Marketaux (optional keys).
- Options, Energy Trends, and the expanded Market Lens universe: massive.com
  (`api.massive.com`, optional `MASSIVE_API_KEY`), with an automatic yfinance
  fallback so these tabs work fully without a key, just with less detailed
  Greeks/IV data. This is a plain REST API, unrelated to and not to be
  confused with the separate Claude Code MCP connector for massive
  (`mcp.massive.com`), which only exists inside an interactive coding session.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then fill in GEMINI_API_KEY and FRED_API_KEY
                                    # MASSIVE_API_KEY, NEWSAPI_KEY, MARKETAUX_KEY are optional
```

## Use

```bash
python -m src.backfill             # seed multi-year daily history from FRED
streamlit run src/app.py           # launch the dashboard, then click Refresh data
pytest -q                          # run the unit tests
```

## Layout

```
src/config.py          env and constants
src/store.py            DuckDB schema and read/write
src/ingest.py            yfinance pollers, market-hours logic, bad-tick filter
src/backfill.py          FRED history seed and reconciliation
src/compute.py           alignment, spread, z-score, rolling stats
src/annotate.py          rule-based and calendar annotations
src/correlations.py      correlated-asset universe (categorized), rolling correlation
src/news.py              yfinance/EIA RSS/NewsAPI/Marketaux headline fetchers
src/insights.py          grounded Gemini insight note (single-shot summary)
src/chatbot.py           grounded Gemini multi-turn Q&A (floating chatbot)
src/pricing.py           Black-Scholes pricing/Greeks, IV solver, OU mean-reversion fit
src/massive_client.py    REST client for api.massive.com, fails safe to None/empty
src/options.py           live option chains (massive first, yfinance + local BS fallback)
src/app.py               Streamlit and Plotly dashboard
src/events.yaml          curated event calendar
tests/                   unit tests for every module above
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

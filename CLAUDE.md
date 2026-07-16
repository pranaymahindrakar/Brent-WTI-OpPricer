# CLAUDE.md

## Project summary
Brent-WTI Spread Tracker. A near-real-time tool that ingests front-month WTI and Brent
crude futures, computes the Brent-minus-WTI spread and its rolling z-score, annotates the
chart with rule-based and calendar events, and produces grounded natural-language insight
notes and chatbot answers via the Google Gemini API. v1 ships as a Streamlit + Plotly
dashboard backed by DuckDB; v2 (PLAN.md phases 7-13) adds options/Greeks, a broader
energy-market view, a pricing-methodology tab, and a grounded chatbot on top of the same
accuracy guardrails. Build it phase by phase as defined in PLAN.md. Do not skip ahead.

## Tech stack (locked for v1, do not substitute without asking)
- Python 3.11+
- Data feed: yfinance, symbols CL=F (WTI front month) and BZ=F (Brent Last Day Financial front month)
- History + validation: FRED series DCOILWTICO and DCOILBRENTEU (and EIA if a key is present)
- Storage: DuckDB, single local file at data/tracker.duckdb
- Compute: pandas + numpy
- Scheduling: APScheduler inside the app
- LLM insights and chatbot: Google Gemini Python SDK (google-genai), model gemini-2.0-flash
- UI: Streamlit + Plotly
- Tests: pytest
- Supplementary data (Options, Energy Trends, expanded Market Lens tabs only):
  massive.com REST API (api.massive.com, keyed via MASSIVE_API_KEY), with
  yfinance as an automatic fallback on any failure or missing key. Never used
  in the core WTI/Brent ingestion path (ingest.py, backfill.py, compute.py);
  see "Forbidden actions" below for the boundary this crosses and why it's
  scoped the way it is.
- Note: the Claude Code MCP connector for massive (mcp.massive.com) is a
  separate thing, only reachable from an interactive Claude Code session, and
  is not part of the running app. Do not attempt to wire MCP calls into
  src/*.py; use src/massive_client.py's REST client instead.

## Commands
- Install: `pip install -r requirements.txt`
- Run app: `streamlit run src/app.py`
- Backfill history: `python -m src.backfill`
- Tests: `pytest -q`

## Directory notes
- src/config.py          env + constants
- src/ingest.py          yfinance pollers, market-hours logic, bad-tick filter
- src/backfill.py        FRED/EIA history seed
- src/store.py           DuckDB schema and read/write
- src/compute.py         alignment, spread, z-score, rolling stats
- src/annotate.py        rule-based + calendar annotations
- src/correlations.py    correlated-asset universe (categorized), rolling correlation
- src/news.py            yfinance/EIA RSS/NewsAPI/Marketaux headline fetchers
- src/insights.py        grounded Gemini insight note (single-shot summary)
- src/chatbot.py         grounded Gemini multi-turn Q&A (floating chatbot)
- src/pricing.py         Black-Scholes pricing/Greeks, IV solver, OU mean-reversion fit
- src/massive_client.py  REST client for api.massive.com, fails safe to None/empty
- src/options.py         live option chains (massive first, yfinance + local BS fallback)
- src/app.py             Streamlit + Plotly dashboard
- src/events.yaml        curated event calendar

## Accuracy guardrails (these are hard rules, never violate)
- Never difference a stale leg against a fresh one. Resample both legs to a common time
  grid and align on timestamp before computing the spread. Carry an as-of timestamp through
  every calculation and surface it in the UI.
- Both legs must be the same continuous front-month contract with a documented roll. Note in
  code comments that BZ=F is the NYMEX Brent Last Day Financial proxy for ICE Brent.
- Assert both legs are USD per barrel before differencing.
- Filter implausible ticks at ingest. Reject single-bar jumps beyond a configurable threshold.
- Distinguish "market closed" from "feed broken" using CME Globex hours. Show a data-freshness
  indicator in the UI.
- Backfill real history before trusting any z-score. A z-score with no distribution behind it
  is meaningless.
- Reconcile the live feed daily close against FRED/EIA settlement and flag drift.

## LLM rules (non-negotiable)
- The language model narrates, it never computes. Every number it speaks must be one Python
  already calculated and passed into its context.
- Pass it the computed spread, z-score, regime stats, and any pulled headlines. Instruct it to
  use only those numbers, to mark uncertainty honestly, and to return structured JSON pairing
  each claim with the datapoint that supports it.
- Temperature low. Grounding context mandatory. No invented figures.

## Conventions
- No em dashes anywhere in code comments, docstrings, or generated text. Use commas and
  semicolons.
- Type hints on all functions. Docstrings on every module and public function.
- Secrets only via .env, never hardcoded. Provide .env.example.
- Small, reviewable commits, one per completed phase.

## Forbidden actions
- Paid market-data APIs are permitted only in the scoped, defensive way
  massive.com is used today (src/massive_client.py): every call has a
  yfinance fallback, it only feeds the Options/Energy Trends/Market Lens
  tabs, and it is never a dependency of the core WTI/Brent ingestion or
  spread/z-score computation. Do not add a new paid data source, or expand
  massive's role into the core ingestion path, without asking first.
- Do not add a web React frontend in v1. That is a later phase.
- Do not let the LLM perform arithmetic on prices.
- Never write a real API key/secret into any file, commit, or code sample.
  If a key is ever pasted into a chat or terminal, treat it as compromised
  and ask the user to rotate it rather than persisting it anywhere.

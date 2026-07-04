"""Configuration and constants for the Brent-WTI Spread Tracker.

All tunables live here and are overridable via environment variables so the
behaviour can change without touching code.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Use the operating system trust store for TLS verification. On machines where a
# corporate proxy or antivirus intercepts HTTPS, Python's bundled CA list rejects
# the substituted certificate while the OS already trusts it. This keeps full
# certificate verification on (we never disable it) while sourcing roots from the
# OS, so requests, yfinance, and the news fetchers can reach their endpoints.
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

# Canonical leg names used everywhere in the codebase.
WTI = "WTI"
BRENT = "BRENT"
LEGS = (WTI, BRENT)

# Source symbols.
# yfinance front-month continuous contracts. BZ=F is the NYMEX Brent Last Day
# Financial future, a faithful proxy for ICE Brent settlement.
YF_SYMBOLS = {WTI: "CL=F", BRENT: "BZ=F"}
# FRED daily settlement series, used for history backfill and reconciliation.
FRED_SERIES = {WTI: "DCOILWTICO", BRENT: "DCOILBRENTEU"}

# Secrets (never hardcode; supply via .env).
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
EIA_API_KEY = os.getenv("EIA_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
MARKETAUX_KEY = os.getenv("MARKETAUX_KEY", "")

# Marketaux news + entity-sentiment feed tunables.
# Free tier returns 3 articles per request and allows 100 requests per day, so
# keep the page count modest to stay within quota.
MARKETAUX_SEARCH = os.getenv(
    "MARKETAUX_SEARCH",
    "crude oil OR Brent OR WTI OR OPEC OR refinery OR shale",
)
MARKETAUX_PAGES = int(os.getenv("MARKETAUX_PAGES", "3"))          # requests per refresh
# Energy-industry articles carrying entity sentiment are sparser than raw
# headlines, so bound recency generously and let the client sort surface newest.
MARKETAUX_LOOKBACK_DAYS = int(os.getenv("MARKETAUX_LOOKBACK_DAYS", "365"))

# LLM. Gemini is the active insight engine; Anthropic key kept for reference.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Storage.
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DB_PATH = DATA_DIR / "tracker.duckdb"

# Compute.
ZSCORE_WINDOW = int(os.getenv("ZSCORE_WINDOW", "60"))        # rolling window in bars
NEW_EXTREME_WINDOW = int(os.getenv("NEW_EXTREME_WINDOW", "20"))  # new high/low rule
DEFAULT_FREQ = os.getenv("DEFAULT_FREQ", "1min")            # common resample grid

# Ingestion.
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))
BAR_INTERVAL = os.getenv("BAR_INTERVAL", "1m")             # yfinance interval
BAD_TICK_PCT = float(os.getenv("BAD_TICK_PCT", "0.10"))    # reject single-bar jumps over this fraction

# Annotation thresholds.
Z_ALERT = float(os.getenv("Z_ALERT", "2.0"))              # absolute z-score crossing alert
VOL_SPIKE_MULT = float(os.getenv("VOL_SPIKE_MULT", "3.0"))

# Reconciliation tolerance in USD/bbl between live close and FRED settlement.
RECONCILE_TOL = float(os.getenv("RECONCILE_TOL", "1.50"))

# Ensure the data directory exists.
DATA_DIR.mkdir(parents=True, exist_ok=True)

"""Streamlit + Plotly dashboard for the Brent-WTI spread tracker.

Run from the project root with:  streamlit run src/app.py

Tabs:
  Dashboard      live spread chart with z-score, auto-refreshes every POLL_SECONDS
  Market Lens    rolling correlations against ~18 categorized crude/energy/macro
                 assets (majors, refiners, services, products, crude ETFs, macro)
  Energy Trends  performance heatmap, rolling volatility, spread distribution,
                 and per-asset movement cards across the same asset universe
  Options        live option chains, Greeks, and IV skew for USO/XLE (massive.com,
                 falling back to yfinance + local Black-Scholes)
  Research       pricing methodology: alignment/z-score formulas, an OU
                 mean-reversion half-life fit, and the Black-Scholes option
                 pricing math used on the Options tab
  News           yfinance headlines, EIA RSS feed, NewsAPI headlines
  About          educational content on WTI, Brent, the spread, and market context

A floating chatbot (bottom-left, toggle button) answers ad-hoc questions
grounded in whatever's already computed on the other tabs (spread, z-score,
correlations, news sentiment, options positioning); see chatbot.py.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import annotate, backfill, chatbot, compute, config, correlations, ingest, insights, news, options, pricing, store

st.set_page_config(page_title="Brent-WTI Spread Tracker", layout="wide", page_icon="")

_GLOBAL_CSS = """
<style>
/* Gradient KPI cards, applied to every st.metric across every tab so the
   whole app reads as one system rather than each tab having its own look. */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(76,155,232,0.12) 0%, rgba(19,23,32,0.6) 60%);
    border: 1px solid #2A2F3A;
    border-radius: 12px;
    padding: 0.9rem 1rem 0.7rem 1rem;
    transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.35);
}

/* Movement cards (Market Lens / Energy Trends sparkline tiles) get the same
   hover-lift treatment via their container key prefix. */
div[class*="st-key-movecard_"] {
    border: 1px solid #2A2F3A;
    border-radius: 10px;
    padding: 0.4rem 0.5rem;
    transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out, border-color 0.15s ease-in-out;
}
div[class*="st-key-movecard_"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.3);
    border-color: #4C9BE8;
}

/* Pulse on the z-score metric card when |zscore| exceeds the alert threshold. */
@keyframes zscore-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(232,85,78,0.55); }
    70%  { box-shadow: 0 0 0 10px rgba(232,85,78,0); }
    100% { box-shadow: 0 0 0 0 rgba(232,85,78,0); }
}
.st-key-zscore_alert [data-testid="stMetric"] {
    border-color: #E8554E;
    animation: zscore-pulse 2s infinite;
}

/* Correlation cells on Market Lens: the whole cell is a button styled as a
   plain card, so clicking anywhere on it opens the profile, no separate
   "View profile" button needed. */
div[class*="st-key-corrcard_"] {
    border: 1px solid #2A2F3A;
    border-radius: 10px;
    margin-bottom: 0.6rem;
    overflow: hidden;
    transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out, border-color 0.15s ease-in-out;
}
div[class*="st-key-corrcard_"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.3);
    border-color: #4C9BE8;
}
div[class*="st-key-corrcard_"] button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    padding: 0.65rem 0.85rem;
    color: #FAFAFA;
    font-weight: 500;
    border-radius: 10px;
    white-space: pre-line;
}
div[class*="st-key-corrcard_"] button:hover {
    background: rgba(76,155,232,0.08);
    border: none;
    color: #FAFAFA;
}
div[class*="st-key-corrcard_"] button p {
    text-align: left;
}
</style>
"""
st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_con():
    """One DuckDB connection shared across all sessions in this container.

    cache_resource is process-wide, so every browser session reuses this single
    connection. DuckDB serves concurrent reads fine but a write from two threads
    at once can error, so all writers must hold _ingest_gate()['lock'].
    """
    return store.connect()


@st.cache_resource
def _ingest_gate() -> dict:
    """Process-wide guard shared by every session (survives Streamlit reruns).

    'lock' serialises DuckDB writes; 'ts' throttles ingestion so ten open tabs
    do not each hit yfinance every POLL_SECONDS, they cooperate on one poll.
    """
    return {"lock": threading.Lock(), "ts": 0.0}


@st.cache_resource
def _bootstrap(_con) -> bool:
    """Populate an empty database once per container on cold start.

    Streamlit Community Cloud wipes the local disk on every restart, so the
    gitignored DuckDB file is gone and a fresh boot would render a blank chart.
    Seed a month of daily bars from yfinance, then backfill five years from FRED
    when a key is present, so the deployed app is self-populating and never blank.
    Runs once per process thanks to cache_resource; the leading underscore on
    _con tells Streamlit not to hash the connection.
    """
    if not store.read_spread(_con).empty:
        return False
    try:
        ingest.seed_recent(_con, period="1mo", interval="1d")
        if config.FRED_API_KEY:
            backfill.run(years=5, con=_con)
        spread_df = compute.build(_con)
        annotate.build(_con, spread_df)
    except Exception:
        return False
    return True


def _ingest_and_compute(con) -> tuple[int, pd.DataFrame]:
    n = ingest.poll_once(con)
    spread_df = compute.build(con)
    if n > 0:
        annotate.build(con, spread_df)
    return n, spread_df


# ---------------------------------------------------------------------------
# Tab 1: Dashboard (auto-refreshing fragment)
# ---------------------------------------------------------------------------

@st.fragment(run_every=config.POLL_SECONDS)
def _dashboard_tab() -> None:
    con = get_con()

    # Auto-ingest during market hours. Only one session per POLL_SECONDS window
    # actually polls yfinance (throttle), and it holds the shared write lock so
    # concurrent sessions never write to DuckDB at the same instant.
    status_slot = st.empty()
    gate = _ingest_gate()
    now = time.time()
    if ingest.is_market_open() and now - gate["ts"] >= config.POLL_SECONDS:
        with gate["lock"]:
            if now - gate["ts"] >= config.POLL_SECONDS:  # re-check inside lock
                gate["ts"] = now
                try:
                    n, _ = _ingest_and_compute(con)
                    if n:
                        status_slot.success(f"Auto-fetched {n} new bars", icon="")
                except Exception as exc:
                    status_slot.warning(f"Auto-ingest error: {exc}")

    spread = store.read_spread(con)
    ann = store.read_annotations(con)

    if spread.empty:
        st.info(
            "No data yet. Use **Seed recent (yfinance)** in the sidebar to load the last "
            "5 days of bars instantly (no API key needed), or **Backfill from FRED** for "
            "five years of daily history."
        )
        return

    # Metrics row
    last_ts = pd.to_datetime(spread["ts"]).max()
    now_utc = pd.Timestamp.now("UTC").replace(tzinfo=None)
    age_min = int((now_utc - last_ts).total_seconds() // 60)
    latest = spread.dropna(subset=["spread"]).iloc[-1]

    zscore_alert = pd.notna(latest["zscore"]) and abs(latest["zscore"]) > config.Z_ALERT

    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("Spread (USD/bbl)", f"{latest['spread']:.2f}" if pd.notna(latest["spread"]) else "N/A")
    with mc2:
        with st.container(key="zscore_alert" if zscore_alert else "zscore_normal"):
            st.metric(
                "Z-score", f"{latest['zscore']:.2f}" if pd.notna(latest["zscore"]) else "N/A",
                delta="Alert" if zscore_alert else None,
                delta_color="inverse" if zscore_alert else "normal",
            )
    mc3.metric("Roll mean", f"{latest['roll_mean']:.2f}" if pd.notna(latest["roll_mean"]) else "N/A")
    mc4.metric("Roll std", f"{latest['roll_std']:.2f}" if pd.notna(latest["roll_std"]) else "N/A")
    mc5.metric("Pct of range", f"{latest['pct_range']:.1%}" if pd.notna(latest["pct_range"]) else "N/A")
    mc6.metric(
        "Data age",
        f"{age_min} min",
        delta="Market open" if ingest.is_market_open() else "Market closed",
        delta_color="normal" if ingest.is_market_open() else "off",
    )
    st.caption(f"Last data point: {last_ts!s:.19} UTC | auto-refresh every {config.POLL_SECONDS}s")

    spark_series = spread.dropna(subset=["spread"]).set_index("ts")["spread"].tail(30)
    if len(spark_series) >= 2:
        rising = spark_series.iloc[-1] >= spark_series.iloc[0]
        spark_color = "#4CAF50" if rising else "#E8554E"
        spark_fill = "rgba(76,175,80,0.08)" if rising else "rgba(232,85,78,0.08)"
        spark_fig = go.Figure(go.Scatter(
            y=spark_series.values, mode="lines", line=dict(color=spark_color, width=1.5),
            fill="tozeroy", fillcolor=spark_fill,
        ))
        spark_fig.update_layout(
            height=60, margin=dict(t=0, l=0, r=0, b=0), showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        st.plotly_chart(
            spark_fig, use_container_width=True, config={"displayModeBar": False},
            key="dashboard_spread_sparkline",
        )
        st.caption("Spread, last 30 observations")

    # Date range + chart toggles
    min_date = pd.to_datetime(spread["ts"]).min().date()
    max_date = pd.to_datetime(spread["ts"]).max().date()

    fc1, fc2, fc3, fc4 = st.columns([2, 1, 1, 1])
    date_range = fc1.date_input(
        "View window",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        label_visibility="collapsed",
    )
    show_bands = fc2.checkbox("Show ±2σ bands", value=True)
    show_ann = fc3.checkbox("Show annotations", value=True)
    show_zscore = fc4.checkbox("Show z-score panel", value=True)

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d = end_d = date_range if not isinstance(date_range, (list, tuple)) else date_range[0]

    ts_dates = pd.to_datetime(spread["ts"]).dt.date
    view = spread[(ts_dates >= start_d) & (ts_dates <= end_d)].copy()
    if view.empty:
        st.warning("No data in the selected date range.")
        return

    # Build chart
    rows = 2 if show_zscore else 1
    row_h = [0.65, 0.35] if show_zscore else [1.0]
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=row_h, vertical_spacing=0.06,
        subplot_titles=(
            ["Brent-WTI spread (USD/bbl)", "Z-score"] if show_zscore
            else ["Brent-WTI spread (USD/bbl)"]
        ),
    )

    fig.add_trace(
        go.Scatter(x=view["ts"], y=view["spread"], name="Spread",
                   line=dict(color="#4C9BE8", width=2)),
        row=1, col=1,
    )
    if show_bands:
        fig.add_trace(
            go.Scatter(x=view["ts"], y=view["roll_mean"], name="Rolling mean",
                       line=dict(color="#AAB7C4", width=1, dash="dot")),
            row=1, col=1,
        )
        upper = view["roll_mean"] + 2 * view["roll_std"]
        lower = view["roll_mean"] - 2 * view["roll_std"]
        fig.add_trace(
            go.Scatter(x=view["ts"], y=upper, name="+2σ",
                       line=dict(color="#A8D5A2", width=0.5), showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=view["ts"], y=lower, name="±2σ band",
                       line=dict(color="#A8D5A2", width=0.5),
                       fill="tonexty", fillcolor="rgba(168,213,162,0.12)"),
            row=1, col=1,
        )

    if show_ann and not ann.empty:
        ann_v = ann.copy()
        ann_v["ts"] = pd.to_datetime(ann_v["ts"])
        ann_v = ann_v[
            (ann_v["ts"].dt.date >= start_d) & (ann_v["ts"].dt.date <= end_d)
        ]
        if not ann_v.empty:
            view_ts = view[["ts", "spread"]].copy()
            view_ts["ts"] = pd.to_datetime(view_ts["ts"])
            merged = pd.merge_asof(
                ann_v.sort_values("ts"),
                view_ts.sort_values("ts"),
                on="ts", direction="nearest",
            ).dropna(subset=["spread"])
            if not merged.empty:
                color_map = {"high": "#E8554E", "medium": "#F5A623", "info": "#7B8D8E"}
                for sev in merged["severity"].unique():
                    sub = merged[merged["severity"] == sev]
                    fig.add_trace(
                        go.Scatter(
                            x=sub["ts"], y=sub["spread"],
                            mode="markers",
                            name=f"Event ({sev})",
                            marker=dict(
                                size=9, symbol="diamond",
                                color=color_map.get(sev, "#F5A623"),
                                line=dict(width=1, color="white"),
                            ),
                            text=sub["text"], hoverinfo="text+x+y",
                        ),
                        row=1, col=1,
                    )

    if show_zscore:
        z_col = view["zscore"].copy()
        pos_mask = z_col >= 0
        fig.add_trace(
            go.Bar(
                x=view.loc[pos_mask, "ts"], y=z_col[pos_mask],
                name="Z-score +", marker_color="rgba(76,155,232,0.6)",
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=view.loc[~pos_mask, "ts"], y=z_col[~pos_mask],
                name="Z-score -", marker_color="rgba(232,85,78,0.6)",
            ),
            row=2, col=1,
        )
        fig.add_hline(y=config.Z_ALERT, line=dict(width=1, dash="dash", color="#E8554E"), row=2, col=1)
        fig.add_hline(y=-config.Z_ALERT, line=dict(width=1, dash="dash", color="#E8554E"), row=2, col=1)
        fig.add_hline(y=0, line=dict(width=0.5, color="#AAB7C4"), row=2, col=1)

    fig.update_layout(
        height=580 if show_zscore else 400,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(t=50, l=10, r=10, b=10),
        barmode="overlay",
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        xaxis=dict(gridcolor="#1E2329"),
        yaxis=dict(gridcolor="#1E2329", title="USD/bbl"),
    )
    if show_zscore:
        fig.update_layout(xaxis2=dict(gridcolor="#1E2329"), yaxis2=dict(gridcolor="#1E2329"))
    st.plotly_chart(fig, use_container_width=True)

    # Grounded insight
    st.subheader("Grounded insight")
    if config.GEMINI_API_KEY:
        if st.button("Generate insight note"):
            with st.spinner("Calling Gemini API..."):
                try:
                    # Ground the note with the latest Marketaux news sentiment if available.
                    mtx = _load_news().get("Marketaux Sentiment", [])
                    sent = news.marketaux_sentiment_summary(mtx)
                    extra = {"news_sentiment": sent} if sent.get("available") else None
                    note = insights.generate(con, extra=extra)
                    st.write(note.get("summary", ""))
                    for claim in note.get("claims", []):
                        st.markdown(
                            f"- {claim.get('claim', '')}  \n"
                            f"  _support: {claim.get('support', '')}_"
                        )
                    if note.get("caveats"):
                        st.caption("Caveats: " + "; ".join(note["caveats"]))
                except Exception as exc:
                    st.error(f"Insight failed: {exc}")
    else:
        st.caption("Set GEMINI_API_KEY in .env to enable grounded LLM insight notes.")

    # Annotations table
    if not ann.empty:
        with st.expander(f"Recent annotations ({len(ann)} total)"):
            display = ann.sort_values("ts", ascending=False).head(30).copy()
            display["ts"] = pd.to_datetime(display["ts"]).dt.strftime("%Y-%m-%d")
            st.dataframe(display, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 2: Market Lens (correlations)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Fetching correlated asset prices...")
def _load_prices() -> pd.DataFrame:
    return correlations.fetch_prices(period="2y")


def _movement_card(col, ticker: str, prices: pd.DataFrame, key_prefix: str = "") -> None:
    """Render a sparkline + %change badge for one asset into a Streamlit column.

    `key_prefix` disambiguates calls from different tabs/sections so repeated
    sparklines for the same ticker (e.g. shown in both Market Lens and Energy
    Trends) don't collide on Streamlit's auto-generated element ID.
    """
    if ticker not in prices.columns:
        return
    series = prices[ticker].dropna()
    if len(series) < 2:
        return
    meta = correlations.ASSETS.get(ticker, {})
    last, prev = series.iloc[-1], series.iloc[-2]
    pct = (last / prev - 1.0) * 100 if prev else 0.0
    color = "#4CAF50" if pct >= 0 else "#E8554E"
    arrow = "▲" if pct >= 0 else "▼"
    with col, st.container(key=f"movecard_{key_prefix}_{ticker}"):
        spark = go.Figure(go.Scatter(
            y=series.tail(30).values, mode="lines", line=dict(color=color, width=1.5),
        ))
        spark.update_layout(
            height=48, margin=dict(t=0, l=0, r=0, b=0), showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        st.plotly_chart(
            spark, use_container_width=True, config={"displayModeBar": False},
            key=f"spark_{key_prefix}_{ticker}",
        )
        st.markdown(
            f"**{meta.get('name', ticker)}**  \n"
            f"${last:,.2f} &nbsp; <span style='color:{color}'>{arrow} {pct:+.2f}%</span>",
            unsafe_allow_html=True,
        )


def _performance_table(prices: pd.DataFrame) -> pd.DataFrame:
    """1D/1W/1M/YTD percent change per asset, from the cached daily price frame."""
    if prices.empty:
        return pd.DataFrame()
    last = prices.iloc[-1]

    def _pct_ago(n: int) -> pd.Series:
        if len(prices) <= n:
            return pd.Series(index=prices.columns, dtype=float)
        return (last / prices.iloc[-1 - n] - 1.0) * 100

    same_year = prices.index[prices.index.year == prices.index[-1].year]
    ytd_base = prices.loc[same_year[0]] if len(same_year) else prices.iloc[0]
    return pd.DataFrame({
        "1D": _pct_ago(1), "1W": _pct_ago(5), "1M": _pct_ago(21),
        "YTD": (last / ytd_base - 1.0) * 100,
    })


def _search_tickers_cached(query: str) -> list[dict]:
    """Search cache keyed manually via session_state rather than st.cache_data.

    A plain @st.cache_data(ttl=...) would also cache a transient empty result
    (a momentary network hiccup, a rate limit), poisoning that exact query
    text for the full TTL even after the underlying issue clears. Only
    caching non-empty, successful results avoids that trap; a query that
    fails is simply retried the next time it's run, not stuck showing "no
    results" for several minutes.
    """
    cache_key = f"_search_cache_{query.strip().lower()}"
    cached = st.session_state.get(cache_key)
    if cached:
        return cached
    results = correlations.search_tickers(query)
    if results:
        st.session_state[cache_key] = results
    return results


@st.cache_data(ttl=3600, show_spinner="Fetching price history...")
def _load_single_price(ticker: str) -> pd.Series:
    return correlations.fetch_single_price(ticker, period="2y")


@st.cache_data(ttl=3600, show_spinner=False)
def _load_ticker_info(ticker: str) -> dict:
    return correlations.get_ticker_profile_info(ticker)


@st.dialog("Asset profile", width="large")
def _ticker_profile_dialog(ticker: str) -> None:
    """Modal profile for any curated or searched asset: price, correlation, blurb.

    Works for both the curated ASSETS universe (which also gets its static
    `note`) and an arbitrary user-searched ticker (info-only, no curated note).
    """
    con = get_con()
    spread = store.read_spread(con)
    meta = correlations.ASSETS.get(ticker, {})
    info = _load_ticker_info(ticker)
    name = meta.get("name") or info.get("name") or ticker

    head_logo, head_text = st.columns([1, 8])
    with head_logo:
        if info.get("logo_url"):
            st.image(info["logo_url"], width=48)
    with head_text:
        st.subheader(f"{name} ({ticker})")
        sub_bits = [b for b in [info.get("sector"), info.get("industry"), info.get("exchange")] if b]
        if sub_bits:
            st.caption(" · ".join(sub_bits))

    series = _load_single_price(ticker)
    if series.empty:
        st.warning("Couldn't fetch price history for this ticker.")
        return

    last = float(series.iloc[-1])
    prev = float(series.iloc[-2]) if len(series) > 1 else last
    day_change = (last / prev - 1.0) * 100 if prev else 0.0

    current_corr = None
    if not spread.empty:
        corr_series = correlations.compute_current_corr(spread, series.to_frame(ticker))
        if not corr_series.empty and pd.notna(corr_series.iloc[0]):
            current_corr = float(corr_series.iloc[0])

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Last price", f"${last:,.2f}", f"{day_change:+.2f}%")
    mcol2.metric(
        "60-day correlation with spread",
        f"{current_corr:+.2f}" if current_corr is not None else "N/A",
    )
    mcol3.metric("Price history points", f"{len(series)}")

    fig_price = go.Figure(go.Scatter(
        x=series.index, y=series.values, line=dict(color="#4C9BE8", width=1.5),
    ))
    fig_price.update_layout(
        height=260, margin=dict(t=20, l=10, r=10, b=10),
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font=dict(color="#FAFAFA"),
        xaxis=dict(gridcolor="#1E2329"), yaxis=dict(gridcolor="#1E2329", title="Price"),
    )
    st.plotly_chart(fig_price, use_container_width=True, key=f"profile_price_{ticker}")

    if not spread.empty:
        rolling = correlations.compute_rolling_corr(spread, series.to_frame(ticker))
        if not rolling.empty and ticker in rolling.columns:
            fig_roll = go.Figure(go.Scatter(
                x=rolling.index, y=rolling[ticker], line=dict(color="#F5A623", width=1.5),
            ))
            fig_roll.add_hline(y=0, line=dict(width=0.8, color="#AAB7C4", dash="dot"))
            fig_roll.update_layout(
                height=220, margin=dict(t=20, l=10, r=10, b=10),
                plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font=dict(color="#FAFAFA"),
                xaxis=dict(gridcolor="#1E2329"),
                yaxis=dict(range=[-1.1, 1.1], gridcolor="#1E2329", title="Rolling correlation"),
            )
            st.plotly_chart(fig_roll, use_container_width=True, key=f"profile_corr_{ticker}")

    if meta.get("note"):
        st.markdown(f"**Why this might correlate with the spread:** {meta['note']}")

    if config.GEMINI_API_KEY:
        payload = {
            "ticker": ticker, "name": name, "sector": info.get("sector"),
            "industry": info.get("industry"), "last_price": round(last, 2),
            "day_change_pct": round(day_change, 2),
            "correlation_with_spread_60d": round(current_corr, 3) if current_corr is not None else None,
            "business_summary": (info.get("summary") or "")[:600] or None,
        }
        _grounded_paragraph(
            "Write a short profile paragraph (2-3 sentences) for this asset for a "
            "reader looking at its correlation with the Brent-WTI oil spread. "
            "Mention what the business does if a summary is given, and comment on "
            "the correlation figure honestly, including saying so if it's weak, "
            "near zero, or unavailable. Use only the numbers and text given.",
            payload,
        )
    else:
        st.caption("Set GEMINI_API_KEY in .env for a written profile summary.")


def _market_lens_tab() -> None:
    con = get_con()
    spread = store.read_spread(con)

    st.subheader("Spread correlations with related markets")
    st.caption(
        "Correlation is computed on daily percentage returns over a rolling 60-day window. "
        "A positive value means the asset tends to move in the same direction as the spread "
        "(Brent premium widening); negative means they move oppositely."
    )

    categories = sorted({m["category"] for m in correlations.ASSETS.values()})
    selected_categories = st.multiselect(
        "Filter by category", categories, default=categories, key="market_lens_categories",
    )
    selected_tickers = {
        t for t, m in correlations.ASSETS.items() if m["category"] in selected_categories
    }

    rcol1, rcol2 = st.columns([5, 1])
    if rcol2.button("Refresh data"):
        _load_prices.clear()
        st.rerun()

    if spread.empty:
        st.info("No spread data. Seed or backfill data from the sidebar first.")
        return

    prices_all = _load_prices()
    if prices_all.empty:
        st.warning("Could not fetch correlated asset prices from massive or yfinance.")
        return
    prices = prices_all[[c for c in prices_all.columns if c in selected_tickers]]
    if prices.empty:
        st.info("No assets match the selected categories.")
        return

    current = correlations.compute_current_corr(spread, prices)
    rolling = correlations.compute_rolling_corr(spread, prices)

    if current.empty:
        st.info("Not enough overlapping history to compute correlations yet.")
        return

    # Current correlation bar chart: full width, up top, so it isn't left
    # towering over empty space next to a much taller list beside it.
    sorted_corr = current.dropna().sort_values()
    bar_colors = [
        "#E8554E" if v < 0 else "#4CAF50" for v in sorted_corr.values
    ]
    labels = [
        correlations.ASSETS.get(t, {}).get("name", t) for t in sorted_corr.index
    ]
    fig_bar = go.Figure(go.Bar(
        x=sorted_corr.values,
        y=labels,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+.2f}" for v in sorted_corr.values],
        textposition="outside",
    ))
    fig_bar.update_layout(
        title="Current 60-day correlation with Brent-WTI spread",
        height=max(340, 26 * len(labels)),
        margin=dict(t=40, l=10, r=60, b=10),
        xaxis=dict(range=[-1.1, 1.1], gridcolor="#1E2329", zeroline=True,
                   zerolinecolor="#AAB7C4"),
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        yaxis=dict(gridcolor="#1E2329"),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Compact grid below the chart instead of a long single-column list, so
    # 17 assets take a handful of short rows rather than a page-length scroll.
    # Each cell is itself the click target (styled as a plain card via CSS),
    # with a trailing arrow signaling it opens a profile.
    st.markdown("**What each correlation means**")
    ranked = list(current.dropna().sort_values(ascending=False).items())
    n_cols = 3
    for i in range(0, len(ranked), n_cols):
        row = ranked[i:i + n_cols]
        for col, (ticker, val) in zip(st.columns(n_cols), row):
            meta = correlations.ASSETS.get(ticker, {})
            name = meta.get("name", ticker)
            with col, st.container(key=f"corrcard_{ticker}"):
                if st.button(f"{name}  \n{val:+.2f}  ›", key=f"default_profile_{ticker}", use_container_width=True):
                    _ticker_profile_dialog(ticker)

    # Rolling correlation time series
    if not rolling.empty:
        st.subheader("Rolling 60-day correlation over time")
        fig_roll = go.Figure()
        palette = [
            "#4C9BE8", "#E8554E", "#F5A623", "#A8D5A2",
            "#B39DDB", "#80CBC4", "#FFCC80",
        ]
        for i, col in enumerate(rolling.columns):
            name = correlations.ASSETS.get(col, {}).get("name", col)
            fig_roll.add_trace(go.Scatter(
                x=rolling.index, y=rolling[col],
                name=name,
                line=dict(width=1.5, color=palette[i % len(palette)]),
            ))
        fig_roll.add_hline(y=0, line=dict(width=0.8, color="#AAB7C4", dash="dot"))
        fig_roll.add_hline(y=0.5, line=dict(width=0.5, color="#4CAF50", dash="dash"))
        fig_roll.add_hline(y=-0.5, line=dict(width=0.5, color="#E8554E", dash="dash"))
        fig_roll.update_layout(
            height=380,
            legend=dict(orientation="h"),
            margin=dict(t=20, l=10, r=10, b=10),
            yaxis=dict(range=[-1.1, 1.1], gridcolor="#1E2329", title="Correlation"),
            xaxis=dict(gridcolor="#1E2329"),
            plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
            font=dict(color="#FAFAFA"),
        )
        st.plotly_chart(fig_roll, use_container_width=True)
        st.caption(
            "Dashed green/red lines at ±0.5 mark the threshold for a strong correlation. "
            "Correlations that flip sign over time indicate regime changes."
        )

    st.subheader("Movement")
    tickers_in_view = [t for t in prices.columns if t in selected_tickers]
    for i in range(0, len(tickers_in_view), 6):
        row = tickers_in_view[i:i + 6]
        for c, t in zip(st.columns(len(row)), row):
            _movement_card(c, t, prices, key_prefix="marketlens")

    st.divider()
    st.subheader("Search any stock or organization")
    st.caption(
        "Not limited to the curated list above: look up any ticker or company "
        "to see its price history and correlation with the Brent-WTI spread."
    )
    scol1, scol2 = st.columns([4, 1])
    query = scol1.text_input(
        "Search by ticker or company name", key="market_lens_search_query",
        label_visibility="collapsed", placeholder="e.g. Tesla, AAPL, Chevron, Delta Air Lines...",
    )
    # The button is a visual affordance; pressing Enter in the text input
    # already triggers a rerun, and that rerun hits this same search call
    # below, so both paths work without requiring an explicit click.
    scol2.button("Search", key="market_lens_search_btn")

    results = []
    if query:
        with st.spinner("Searching..."):
            results = _search_tickers_cached(query)
        if not results:
            st.caption("No matches. Try the exact ticker symbol, or a different spelling.")

    for r in results:
        logo_col, info_col, btn_col = st.columns([1, 4, 1])
        with logo_col:
            if r.get("logo_url"):
                st.image(r["logo_url"], width=40)
            else:
                initial = (r["name"] or r["symbol"])[0].upper()
                st.markdown(
                    "<div style='width:40px;height:40px;border-radius:50%;"
                    "background:#2A2F3A;display:flex;align-items:center;"
                    "justify-content:center;font-weight:600;color:#FAFAFA;'>"
                    f"{initial}</div>",
                    unsafe_allow_html=True,
                )
        detail = " · ".join(b for b in [r["exchange"], r["sector"]] if b)
        info_col.markdown(f"**{r['symbol']}** — {r['name']}" + (f"  \n_{detail}_" if detail else ""))
        if btn_col.button("View profile", key=f"search_profile_{r['symbol']}"):
            _ticker_profile_dialog(r["symbol"])


# ---------------------------------------------------------------------------
# Tab 2b: Energy Trends (broader crude/energy context)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _narrate_section(instruction: str, payload_json: str) -> str:
    """Cached wrapper around insights.narrate_metric, shared across tabs.

    Cached on (instruction, payload_json) so the same data doesn't re-hit
    Gemini on every rerun. Returns "" (never raises) on any failure so a
    narration outage never breaks the chart or panel above it.
    """
    try:
        return insights.narrate_metric(json.loads(payload_json), instruction)
    except Exception:
        return ""


def _grounded_paragraph(instruction: str, payload: dict) -> None:
    """Render a short Gemini-authored plain-English paragraph, grounded in payload."""
    if not config.GEMINI_API_KEY:
        return
    text = _narrate_section(instruction, json.dumps(payload, default=str, sort_keys=True))
    if text:
        st.markdown(f"*{text}*")


def _energy_trends_tab() -> None:
    con = get_con()
    spread = store.read_spread(con)

    st.subheader("Oil and energy market trends")
    st.caption(
        "Broader crude and energy-complex context beyond the Brent-WTI spread itself: "
        "relative performance, volatility regime, and the spread's own distribution."
    )
    if not config.GEMINI_API_KEY:
        st.caption("Set GEMINI_API_KEY in .env to also get a plain-English explanation under each chart below.")

    prices = _load_prices()
    if prices.empty:
        st.info("No correlated asset prices yet. Open Market Lens and click Refresh first.")
        return

    st.markdown("##### Performance heatmap")
    perf = _performance_table(prices)
    if not perf.empty:
        z = perf.values
        labels = [correlations.ASSETS.get(t, {}).get("name", t) for t in perf.index]
        fig_heat = go.Figure(go.Heatmap(
            z=z, x=list(perf.columns), y=labels,
            colorscale="RdYlGn", zmid=0,
            text=[[f"{v:+.1f}%" if pd.notna(v) else "" for v in row] for row in z],
            texttemplate="%{text}", showscale=True,
        ))
        fig_heat.update_layout(
            height=max(320, 24 * len(labels)), margin=dict(t=10, l=10, r=10, b=10),
            plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font=dict(color="#FAFAFA"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        movers = perf["1D"].dropna().sort_values(ascending=False)
        ytd_movers = perf["YTD"].dropna().sort_values(ascending=False)
        if not movers.empty:
            def _name(t):
                return correlations.ASSETS.get(t, {}).get("name", t)
            _grounded_paragraph(
                "This heatmap shows 1-day, 1-week, 1-month, and year-to-date percent "
                "changes for a set of crude/energy/macro assets. Explain in plain "
                "English what the biggest movers mean, using only the numbers given.",
                {
                    "best_1d": {"name": _name(movers.index[0]), "pct": round(float(movers.iloc[0]), 2)},
                    "worst_1d": {"name": _name(movers.index[-1]), "pct": round(float(movers.iloc[-1]), 2)},
                    "best_ytd": (
                        {"name": _name(ytd_movers.index[0]), "pct": round(float(ytd_movers.iloc[0]), 2)}
                        if not ytd_movers.empty else None
                    ),
                    "worst_ytd": (
                        {"name": _name(ytd_movers.index[-1]), "pct": round(float(ytd_movers.iloc[-1]), 2)}
                        if not ytd_movers.empty else None
                    ),
                },
            )

    st.markdown("##### Rolling 20-day annualized volatility")
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    vol = (returns.rolling(20).std().iloc[-1] * (252 ** 0.5) * 100).dropna()
    spread_daily = _spread_daily_series(spread)
    spread_vol_series = spread_daily.pct_change(fill_method=None).rolling(20).std() * (252 ** 0.5) * 100
    spread_vol = spread_vol_series.dropna().iloc[-1] if not spread_vol_series.dropna().empty else None
    if not vol.empty:
        vol_labeled = vol.rename(index=lambda t: correlations.ASSETS.get(t, {}).get("name", t))
        if spread_vol is not None:
            vol_labeled["Brent-WTI spread"] = spread_vol
        vol_labeled = vol_labeled.sort_values()
        colors = ["#F5A623" if name == "Brent-WTI spread" else "#4C9BE8" for name in vol_labeled.index]
        fig_vol = go.Figure(go.Bar(
            x=vol_labeled.values, y=vol_labeled.index, orientation="h", marker_color=colors,
            text=[f"{v:.0f}%" for v in vol_labeled.values], textposition="outside",
        ))
        fig_vol.update_layout(
            height=max(320, 22 * len(vol_labeled)), margin=dict(t=10, l=10, r=40, b=10),
            plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font=dict(color="#FAFAFA"),
            xaxis=dict(gridcolor="#1E2329", title="Annualized volatility"),
            yaxis=dict(gridcolor="#1E2329"), showlegend=False,
        )
        st.plotly_chart(fig_vol, use_container_width=True)
        st.caption(
            "Annualized volatility of daily returns over the trailing 20 trading days "
            "(std x sqrt(252)). The spread's own volatility (highlighted) uses its daily "
            "percent change, the same basis as every other bar."
        )
        _grounded_paragraph(
            "This bar chart ranks assets by trailing 20-day annualized volatility, "
            "including the Brent-WTI spread itself. Explain in plain English what a "
            "reader should take from where the spread ranks versus the other assets, "
            "using only the numbers given.",
            {
                "spread_volatility_pct": round(float(spread_vol), 1) if spread_vol is not None else None,
                "most_volatile": {"name": vol_labeled.index[-1], "pct": round(float(vol_labeled.iloc[-1]), 1)},
                "least_volatile": {"name": vol_labeled.index[0], "pct": round(float(vol_labeled.iloc[0]), 1)},
                "spread_rank_from_least_volatile": int(vol_labeled.index.get_loc("Brent-WTI spread")) + 1
                if "Brent-WTI spread" in vol_labeled.index else None,
                "n_assets": len(vol_labeled),
            },
        )

    st.markdown("##### Spread distribution")
    if not spread_daily.empty:
        current_val = spread_daily.iloc[-1]
        fig_hist = go.Figure(go.Histogram(
            x=spread_daily.values, nbinsx=40, marker_color="#4C9BE8", opacity=0.85,
        ))
        fig_hist.add_vline(
            x=current_val, line=dict(color="#F5A623", width=2, dash="dash"),
            annotation_text=f"Current: ${current_val:.2f}",
        )
        fig_hist.update_layout(
            height=300, margin=dict(t=10, l=10, r=10, b=10),
            plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font=dict(color="#FAFAFA"),
            xaxis=dict(gridcolor="#1E2329", title="Spread (USD/bbl)"),
            yaxis=dict(gridcolor="#1E2329", title="Days"), showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        _grounded_paragraph(
            "This histogram shows the distribution of the Brent-WTI spread's daily "
            "closes, with the current value marked. Explain in plain English where "
            "today's spread sits relative to its own history, using only the numbers "
            "given.",
            {
                "current_spread": round(float(current_val), 2),
                "historical_mean": round(float(spread_daily.mean()), 2),
                "historical_std": round(float(spread_daily.std()), 2),
                "historical_min": round(float(spread_daily.min()), 2),
                "historical_max": round(float(spread_daily.max()), 2),
                "percentile_of_current": round(
                    float((spread_daily <= current_val).mean() * 100), 1
                ),
                "n_days": int(len(spread_daily)),
            },
        )

    st.markdown("##### Movement by category")
    for cat in sorted({m["category"] for m in correlations.ASSETS.values()}):
        cat_tickers = [
            t for t, m in correlations.ASSETS.items()
            if m["category"] == cat and t in prices.columns
        ]
        if not cat_tickers:
            continue
        st.caption(f"{correlations.CATEGORY_ICONS.get(cat, '')} {cat}".strip())
        cols = st.columns(min(len(cat_tickers), 6))
        for c, t in zip(cols, cat_tickers):
            _movement_card(c, t, prices, key_prefix="energytrends")


# ---------------------------------------------------------------------------
# Tab 3: News
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner="Fetching news feeds...")
def _load_news() -> dict:
    return news.fetch_all(
        newsapi_key=config.NEWSAPI_KEY,
        marketaux_key=config.MARKETAUX_KEY,
        marketaux_search=config.MARKETAUX_SEARCH,
        marketaux_pages=config.MARKETAUX_PAGES,
        marketaux_lookback_days=config.MARKETAUX_LOOKBACK_DAYS,
    )


def _age_str(ts: datetime, now: datetime) -> str:
    age = now - ts
    if age.days > 0:
        return f"{age.days}d ago"
    if age.seconds > 3600:
        return f"{age.seconds // 3600}h ago"
    return f"{age.seconds // 60}m ago"


def _sentiment_badge(score: float | None) -> str:
    if score is None:
        return ""
    if score > 0.15:
        return f"🟢 bullish ({score:+.2f})"
    if score < -0.15:
        return f"🔴 bearish ({score:+.2f})"
    return f"⚪ neutral ({score:+.2f})"


def _render_articles(articles: list[dict]) -> None:
    if not articles:
        st.caption("No articles available.")
        return
    now = datetime.utcnow()
    for art in articles[:20]:
        st.markdown(f"**[{art['title']}]({art['link']})**")
        meta = f"{art['publisher']} · {_age_str(art['ts'], now)}"
        badge = _sentiment_badge(art.get("sentiment"))
        if badge:
            meta += f" · {badge}"
        st.caption(meta)
        if art.get("snippet"):
            st.caption(art["snippet"][:280])
        ents = [e for e in (art.get("entities") or []) if e.get("symbol")]
        if ents:
            chips = ", ".join(
                f"{e['symbol']} ({e['sentiment']:+.2f})" if e.get("sentiment") is not None
                else e["symbol"]
                for e in ents[:6]
            )
            st.caption(f"Tagged: {chips}")


def _news_tab() -> None:
    ncol1, ncol2 = st.columns([5, 1])
    ncol1.subheader("Energy market news")
    if ncol2.button("Refresh news"):
        _load_news.clear()
        st.rerun()

    # Upcoming events from events.yaml
    from src.annotate import _load_events
    events = _load_events()
    today = datetime.utcnow().date()
    horizon = today + timedelta(days=60)
    upcoming = [
        e for e in (events.get("dated") or [])
        if today <= pd.to_datetime(e["date"]).date() <= horizon
    ]
    if upcoming:
        with st.expander(f"Upcoming scheduled events (next 60 days) — {len(upcoming)} events"):
            for e in sorted(upcoming, key=lambda x: x["date"]):
                badge = {"opec": "OPEC+", "fomc": "FOMC", "eia_report": "EIA"}.get(
                    e.get("kind", ""), e.get("kind", "EVENT").upper()
                )
                st.markdown(f"`{e['date']}` **{badge}** — {e['text']}")

    all_news = _load_news()

    # Marketaux carries per-entity sentiment; surface an aggregate snapshot up top.
    mtx = all_news.get("Marketaux Sentiment", [])
    summary = news.marketaux_sentiment_summary(mtx)
    if summary.get("available"):
        st.markdown("##### Energy news sentiment (Marketaux)")
        scol1, scol2, scol3 = st.columns(3)
        scol1.metric(
            "Mean sentiment",
            f"{summary['mean_sentiment']:+.2f}",
            summary["label"],
        )
        scol2.metric(
            "Bullish / Bearish",
            f"{summary['bullish']} / {summary['bearish']}",
            f"{summary['neutral']} neutral",
        )
        scol3.metric(
            "Scored articles",
            f"{summary['n_scored']} / {summary['n_articles']}",
        )
        st.caption(
            "Sentiment is the mean of Marketaux per-entity scores across recent "
            "Energy-industry headlines, in [-1, 1]. Computed in Python, not by the LLM."
        )

    source_icons = {
        "Marketaux Sentiment": "Marketaux (energy headlines with entity sentiment)",
        "Yahoo Finance": "Yahoo Finance",
        "EIA Official Feed": "EIA (US Energy Information Administration)",
        "NewsAPI Headlines": "NewsAPI (Reuters, Bloomberg, FT, etc.)",
    }
    for source_key, display_name in source_icons.items():
        articles = all_news.get(source_key, [])
        label = f"{display_name} — {len(articles)} articles" if articles else f"{display_name} — unavailable"
        with st.expander(label, expanded=(source_key == "Marketaux Sentiment")):
            _render_articles(articles)


# ---------------------------------------------------------------------------
# Tab 4: About
# ---------------------------------------------------------------------------

_ABOUT_WTI = """
## West Texas Intermediate (WTI)

West Texas Intermediate is the primary crude oil benchmark for North America,
produced mainly from the **Permian Basin** (West Texas and New Mexico), the
**Bakken shale** (North Dakota), and the **Eagle Ford** formation (South Texas).

| Property | Value |
|---|---|
| API gravity | ~39.6 (light crude) |
| Sulfur content | ~0.24% (sweet crude) |
| Delivery point | Cushing, Oklahoma |
| Exchange | NYMEX (CME Group), symbol CL |
| This app | CL=F (front-month continuous contract) |

**Cushing, Oklahoma** is known as the "Pipeline Crossroads of the World" and
is a major storage hub where dozens of pipelines converge. Because WTI settles
with physical delivery at this inland, landlocked location, its price is highly
sensitive to:

- US domestic production volumes (Permian, Bakken output)
- Cushing crude inventory levels (reported weekly by EIA)
- Pipeline capacity from producing regions to Cushing
- US crude export capacity (Gulf Coast terminals)

Before 2015, US crude exports were legally banned. Any US supply surplus had
nowhere to go except Cushing storage, which depressed WTI sharply relative to
globally-traded Brent during the 2010-2014 shale boom.
"""

_ABOUT_BRENT = """
## Brent Crude

Brent crude is the **leading international benchmark**, used to price
approximately 70-80% of the world's crude oil. Despite its name referencing the
Brent oilfield in the North Sea, the benchmark now represents a blend of five
crude streams collectively called **BFOET**:

- **B**rent (UK)
- **F**orties (UK, largest stream)
- **O**seberg (Norway)
- **E**kofisk (Norway)
- **T**roll (Norway)

| Property | Value |
|---|---|
| API gravity | ~38.3 (slightly heavier than WTI) |
| Sulfur content | ~0.37% (slightly more sour than WTI) |
| Delivery | Waterborne, at North Sea loading terminals |
| Primary exchange | ICE Futures Europe (London) |
| This app | BZ=F — NYMEX Brent Last Day Financial, a faithful proxy for ICE Brent settlement |

Because Brent is **waterborne** (loaded onto tankers at sea), it has global
delivery optionality: any refinery in the world can receive a Brent-linked cargo
by tanker. This makes Brent more sensitive to:

- Global demand conditions (China, India, Europe, emerging markets)
- Geopolitical risk premiums (Middle East, Russia, West Africa)
- VLCC tanker freight rates (shipping costs affect the net-back price)
- OPEC+ production decisions (most OPEC members sell Brent-linked crudes)
"""

_ABOUT_SPREAD = """
## The Brent-WTI Spread

The spread is defined as **Brent price minus WTI price**. A positive value means
Brent is more expensive (which is the normal state today).

### Historical regimes

| Period | Spread range | Driver |
|---|---|---|
| Pre-2010 | WTI +$1-3 premium | WTI slightly lighter/sweeter; lower US production |
| 2010-2014 | WTI -$5 to -$28 discount | US shale boom + crude export ban = Cushing overflow |
| 2015-2019 | Brent +$2-6 premium | Export ban lifted; infrastructure catch-up |
| 2020 (COVID) | Brent +$0-3, volatile | Simultaneous demand shock hit both legs |
| 2021-present | Brent +$2-7 premium | Shale recovery + OPEC discipline + Russia risk premium |

### What moves the spread

**Widens (Brent >> WTI):**
- US production surge or Cushing inventory build
- Geopolitical disruptions in Middle East, Russia, or West Africa (lifts Brent)
- OPEC+ production cuts (members sell Brent-linked crudes)
- Weak US export infrastructure (Gulf Coast congestion)

**Narrows (WTI approaches Brent):**
- Strong US crude export demand (Gulf Coast exports absorbing surplus)
- Global recession fears (hits Brent-exposed global demand harder)
- High VLCC shipping rates (makes waterborne cargo less competitive)
- Permian pipeline bottlenecks resolved (removes WTI discount)

### Why it matters to different market participants

- **US refiners** (Valero, Marathon, Phillips 66): buy WTI, sell Brent-linked products. Wide spread = better margins.
- **International oil companies**: sell Brent-linked production; wide spread = relatively lower WTI hurts US onshore economics.
- **Airlines**: jet fuel priced off Brent globally. Wide spread = slight cost advantage for US carriers with WTI-linked contracts.
- **Traders and hedge funds**: spread is a mean-reverting statistical series, making it attractive for pairs strategies.
"""

_ABOUT_ZSCORE = """
## Reading the Z-Score

The z-score answers: **how unusual is the current spread relative to recent history?**

Formula:

```
z = (spread - rolling_mean) / rolling_std
```

The rolling window defaults to **60 bars** (approximately 3 months with daily data).

| Z-score | Interpretation |
|---|---|
| > +2.0 | Spread is statistically wide (>2 standard deviations above mean) |
| +1.0 to +2.0 | Moderately elevated |
| -1.0 to +1.0 | Near-normal range |
| -1.0 to -2.0 | Moderately compressed |
| < -2.0 | Spread is statistically narrow (>2 standard deviations below mean) |

**Important caveats:**
- The z-score describes the spread's position relative to the *recent* distribution,
  not an absolute fundamental anchor. If a new regime has started (e.g., a structural
  shift in US export capacity), the mean will drift and the z-score will reset.
- With only daily FRED data, the z-score requires at least 60 trading days (about
  3 months) to be meaningful. With intraday yfinance bars, it populates faster.
- A z-score of -1.14 (today's reading) means the spread is about 1.14 standard
  deviations below its recent rolling mean, indicating the Brent premium is
  somewhat compressed relative to recent history.
"""

_ABOUT_MARKETS = """
## How the Spread Informs Broader Markets

### US Refiners
Companies like **Valero (VLO)**, **Marathon Petroleum (MPC)**, and
**Phillips 66 (PSX)** buy WTI as their primary crude feedstock and sell
refined products (gasoline, diesel, jet fuel) priced at international Brent
parity. A wider spread directly expands their gross refining margin ("crack
spread"). Historically, VLO shares and the Brent-WTI spread show a moderately
positive correlation during periods of spread widening.

### Energy Sector (XLE)
The S&P 500 Energy sector ETF broadly tracks oil price moves. Its relationship
with the spread is complex: exploration and production (E&P) companies prefer high
oil prices in general, while refiners prefer a wide spread. The net effect on XLE
depends on the sector composition at any given time.

### US Dollar (DXY)
Oil is denominated in USD globally. Dollar strengthening compresses oil prices
in USD terms for non-US buyers. The effect on the spread is asymmetric: WTI,
priced in a US domestic context, can be more sensitive to local supply/demand,
while Brent reflects global dollar liquidity. A strong dollar rally often
accompanies a widening Brent-WTI spread.

### Refined Products (HO=F, RB=F)
- **Heating oil (HO=F)** is the primary US distillate futures contract, a proxy
  for global diesel and jet fuel. It is effectively priced off Brent via
  international product markets.
- **RBOB gasoline (RB=F)** reflects US driving demand. Summer peak demand
  compresses the spread as refiners aggressively bid for WTI feedstock.
- The 3-2-1 crack spread (3 barrels crude into 2 gasoline + 1 distillate) is
  the key refinery margin metric. A wide Brent-WTI spread boosts crack margins
  when feedstock is WTI.

### Natural Gas (NG=F)
Henry Hub natural gas is a competing fuel in power generation and industrial
heating. Gas price spikes can shift demand away from oil at the margin.
Additionally, the growth of US LNG exports has increasingly linked Henry Hub
prices to global energy markets, creating indirect correlation pathways with
Brent.

### S&P 500 (SPY)
Equity risk appetite influences crude demand expectations. In risk-off
environments, global demand fears compress Brent harder than WTI, narrowing the
spread. In risk-on environments, EM demand recovery expectations tend to widen
the Brent premium.

### Shipping
VLCC (Very Large Crude Carrier) freight rates affect the economics of moving
crude from the Middle East and West Africa to refineries globally. High freight
rates reduce the net-back value of waterborne Brent cargoes, effectively
narrowing the spread from Brent's side.
"""

_ABOUT_GLOSSARY = """
## Quick Reference Glossary

| Term | Definition |
|---|---|
| **API gravity** | Measure of crude density. Higher API = lighter crude. Light crude (>31 API) is easier to refine into gasoline. |
| **Sweet crude** | Crude with sulfur content below 0.5%. Less processing needed. WTI and Brent are both sweet. |
| **Front-month contract** | The nearest-expiry futures contract. CL=F and BZ=F are front-month continuations. |
| **Backwardation** | When front-month price > later months. Indicates tight near-term supply. |
| **Contango** | When front-month price < later months. Indicates oversupply or high storage costs. |
| **Crack spread** | The price differential between crude oil and its refined products. Measures refinery margin. |
| **OPEC+** | OPEC members plus allied producers (Russia, Kazakhstan, etc.). Controls ~40% of global oil supply. |
| **Cushing** | Delivery point for WTI futures in Cushing, Oklahoma. Storage levels here drive WTI pricing. |
| **VLCC** | Very Large Crude Carrier. Supertanker carrying 2 million barrels. Key vessel for Brent/Middle East crude. |
| **EIA** | US Energy Information Administration. Publishes weekly Petroleum Status Reports every Wednesday 10:30 ET. |
| **JMMC** | OPEC+ Joint Ministerial Monitoring Committee. Meets regularly to assess production compliance. |
| **Z-score** | Standard deviations from rolling mean. Values beyond ±2 indicate statistically unusual conditions. |
| **Roll yield** | Gain or loss from rolling a futures position to the next contract at expiry. |
| **Henry Hub** | Delivery point for US natural gas futures (NG=F), in Louisiana. |
"""


def _about_tab() -> None:
    st.subheader("Understanding Brent, WTI, and the Spread")
    sections = {
        "West Texas Intermediate (WTI)": _ABOUT_WTI,
        "Brent Crude": _ABOUT_BRENT,
        "The Brent-WTI Spread": _ABOUT_SPREAD,
        "Reading the Z-Score": _ABOUT_ZSCORE,
        "How the Spread Informs Broader Markets": _ABOUT_MARKETS,
        "Quick Reference Glossary": _ABOUT_GLOSSARY,
    }
    for title, content in sections.items():
        with st.expander(title, expanded=(title == "The Brent-WTI Spread")):
            st.markdown(content)


# ---------------------------------------------------------------------------
# Tab 3: Options (live chains, Greeks, IV skew)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner="Fetching options chain...")
def _load_chain(ticker: str, expiration: str) -> dict:
    return options.fetch_chain(ticker, expiration)


def _options_tab() -> None:
    con = get_con()
    spread = store.read_spread(con)

    st.subheader("Live options and Greeks")
    st.caption(
        "Underlyings are liquid crude/energy ETFs (USO, XLE); options on the "
        "CL=F/BZ=F futures themselves aren't covered by either data source. "
        "Data comes from massive.com when a key is configured, with a "
        "yfinance chain plus locally-computed Black-Scholes Greeks as a fallback."
    )

    ocol1, ocol2, ocol3 = st.columns([1, 2, 1])
    ticker = ocol1.selectbox("Underlying", config.OPTIONS_UNDERLYINGS)
    expiries = options.get_expiries(ticker)
    if not expiries:
        st.warning(f"No option expiries available for {ticker} right now.")
        return
    expiration = ocol2.selectbox("Expiration", expiries)
    if ocol3.button("Refresh chain"):
        _load_chain.clear()
        st.rerun()

    result = _load_chain(ticker, expiration)
    chain, underlying_price, source = result["chain"], result["underlying_price"], result["source"]

    if chain.empty:
        st.info("No contracts returned for this expiration.")
        return

    badge = "massive" if source == "massive" else "yfinance + local Black-Scholes"
    price_str = f"${underlying_price:.2f}" if underlying_price else "N/A"
    st.caption(f"Source: **{badge}** | Underlying price: {price_str}")

    summary = options.atm_summary(chain, underlying_price)
    if summary.get("available"):
        scol1, scol2, scol3, scol4 = st.columns(4)
        scol1.metric("ATM strike", f"${summary['atm_strike']:.2f}")
        scol2.metric("ATM IV", f"{summary['atm_iv']:.1%}")
        pc_ratio = summary.get("put_call_oi_ratio")
        scol3.metric("Put/Call OI ratio", f"{pc_ratio:.2f}" if pc_ratio is not None else "N/A")
        scol4.metric("Contracts", f"{len(chain)}")

        # Write one snapshot per (ticker, calendar day) so a rolling
        # correlation between options positioning and the spread's regime
        # can accumulate over time, and the Research tab can show a real
        # worked example instead of a synthetic one.
        valid_spread = spread.dropna(subset=["zscore"])
        if not valid_spread.empty:
            latest_spread = valid_spread.iloc[-1]
            today = datetime.utcnow().date()
            snap_key = f"_options_snap_{ticker}_{today}"
            if not st.session_state.get(snap_key):
                with _ingest_gate()["lock"]:
                    store.write_options_snapshot(
                        con, pd.Timestamp.utcnow().replace(tzinfo=None), ticker,
                        underlying_price, summary["atm_strike"], summary["atm_iv"],
                        pc_ratio, float(latest_spread["spread"]), float(latest_spread["zscore"]), source,
                    )
                st.session_state[snap_key] = True

    st.subheader("Greeks by strike")
    greek_choice = st.radio("Greek", ["delta", "gamma", "theta", "vega"], horizontal=True)
    fig = go.Figure()
    for ct, color in (("call", "#4CAF50"), ("put", "#E8554E")):
        sub = chain[chain["contract_type"] == ct].dropna(subset=[greek_choice]).sort_values("strike")
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["strike"], y=sub[greek_choice], name=ct.capitalize(),
            line=dict(color=color, width=2), mode="lines+markers",
        ))
    if underlying_price:
        fig.add_vline(x=underlying_price, line=dict(color="#AAB7C4", dash="dot"))
    fig.update_layout(
        height=340, margin=dict(t=20, l=10, r=10, b=10),
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font=dict(color="#FAFAFA"),
        xaxis=dict(gridcolor="#1E2329", title="Strike"),
        yaxis=dict(gridcolor="#1E2329", title=greek_choice.capitalize()),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Implied volatility skew")
    fig_iv = go.Figure()
    for ct, color in (("call", "#4CAF50"), ("put", "#E8554E")):
        sub = chain[chain["contract_type"] == ct].dropna(subset=["iv"]).sort_values("strike")
        if sub.empty:
            continue
        fig_iv.add_trace(go.Scatter(
            x=sub["strike"], y=sub["iv"], name=ct.capitalize(),
            line=dict(color=color, width=2), mode="lines+markers",
        ))
    fig_iv.update_layout(
        height=300, margin=dict(t=20, l=10, r=10, b=10),
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font=dict(color="#FAFAFA"),
        xaxis=dict(gridcolor="#1E2329", title="Strike"),
        yaxis=dict(gridcolor="#1E2329", title="IV", tickformat=".0%"),
    )
    st.plotly_chart(fig_iv, use_container_width=True)

    with st.expander(f"Full chain ({len(chain)} contracts)"):
        st.dataframe(
            chain.sort_values(["contract_type", "strike"]),
            use_container_width=True, hide_index=True,
        )

    hist = store.read_options_snapshot(con, ticker=ticker)
    hist = hist.dropna(subset=["atm_iv", "spread_zscore"]) if not hist.empty else hist
    st.subheader(f"{ticker} ATM IV vs. spread z-score (accumulating history)")
    if len(hist) < 10:
        st.caption(
            f"History starts building the first time this tab runs each day: "
            f"{len(hist)}/10 snapshots collected so far. Shown once there's "
            "enough to compute a meaningful correlation."
        )
    else:
        corr = float(hist["atm_iv"].corr(hist["spread_zscore"]))
        st.metric(f"{ticker} ATM IV vs. spread z-score correlation", f"{corr:+.2f}")
        fig_hist = make_subplots(specs=[[{"secondary_y": True}]])
        fig_hist.add_trace(go.Scatter(x=hist["ts"], y=hist["atm_iv"], name="ATM IV",
                                       line=dict(color="#4C9BE8", width=2)), secondary_y=False)
        fig_hist.add_trace(go.Scatter(x=hist["ts"], y=hist["spread_zscore"], name="Spread z-score",
                                       line=dict(color="#F5A623", width=2)), secondary_y=True)
        fig_hist.update_layout(
            height=320, margin=dict(t=20, l=10, r=10, b=10),
            plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font=dict(color="#FAFAFA"),
            xaxis=dict(gridcolor="#1E2329"),
        )
        fig_hist.update_yaxes(title_text="ATM IV", gridcolor="#1E2329", tickformat=".0%", secondary_y=False)
        fig_hist.update_yaxes(title_text="Spread z-score", gridcolor="#1E2329", secondary_y=True)
        st.plotly_chart(fig_hist, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 5: Research (pricing methodology, mean-reversion, options math)
# ---------------------------------------------------------------------------

def _spread_daily_series(spread_df: pd.DataFrame) -> pd.Series:
    """Resample the stored spread frame to one daily close per calendar day.

    The OU half-life is only meaningful in a fixed time unit; daily closes
    keep it comparable across both the 1-minute live feed and the daily FRED
    backfill.
    """
    if spread_df.empty:
        return pd.Series(dtype=float, name="spread")
    s = spread_df.set_index("ts")["spread"]
    s.index = pd.to_datetime(s.index).normalize()
    return s.resample("1D").last().dropna()


def _research_tab() -> None:
    con = get_con()
    spread = store.read_spread(con)

    st.subheader("Methodology")
    st.markdown(
        "Every figure elsewhere in this app traces back to one of the formulas "
        "below. Nothing here is computed by the LLM; the insight panel and "
        "chatbot only narrate numbers already produced by this module and "
        "`compute.py`."
    )

    with st.expander("Alignment and spread", expanded=True):
        st.markdown(
            "Both legs are resampled onto a common time grid and inner-joined "
            "on timestamp before any subtraction happens, so a stale quote is "
            "never differenced against a fresh one (`compute.align`)."
        )
        st.latex(r"\text{spread}_t = \text{Brent}_t - \text{WTI}_t")

    with st.expander("Rolling z-score, correlation, and percent-of-range"):
        st.latex(
            r"z_t = \frac{\text{spread}_t - \mu_{t,w}}{\sigma_{t,w}}"
            r"\qquad \mu_{t,w}, \sigma_{t,w} = \text{rolling mean/std over window } w"
        )
        st.latex(
            r"\rho_{t,w} = \text{corr}(\text{Brent}_{t-w:t}, \text{WTI}_{t-w:t})"
        )
        st.latex(
            r"\text{pct\_range}_t = \frac{\text{spread}_t - \min_{w}}{\max_{w} - \min_{w}}"
        )
        st.caption(f"Current window w = {config.ZSCORE_WINDOW} bars (`ZSCORE_WINDOW`).")

    st.divider()
    st.subheader("Mean-reversion: Ornstein-Uhlenbeck fit")
    st.markdown(
        "The spread is modeled as an OU process, "
        r"$dX_t = \kappa(\mu - X_t)\,dt + \sigma\,dW_t$, fit by discretizing it "
        "as an AR(1) regression on daily closes and solving in closed form "
        "(`pricing.fit_ou`, ordinary least squares, no external solver)."
    )
    daily = _spread_daily_series(spread)
    fit = pricing.fit_ou(daily, dt=1.0) if not daily.empty else None

    if fit is None:
        st.info(
            "Not enough daily history yet to trust a mean-reversion fit. The "
            "fit requires the sample to span at least five estimated "
            "half-lives; a short or too-random-looking series is refused "
            "rather than shown as a misleading number."
        )
    else:
        oc1, oc2, oc3, oc4 = st.columns(4)
        oc1.metric("Mean-reversion speed (κ)", f"{fit.kappa:.4f} /day")
        oc2.metric("Long-run mean (μ)", f"${fit.mu:.2f}/bbl")
        oc3.metric("Half-life", f"{fit.half_life:.1f} days")
        oc4.metric("Fit R²", f"{fit.r_squared:.3f}")
        st.caption(
            f"Fit on {fit.n_obs} daily closes. Half-life is the time for a "
            "deviation from the long-run mean to decay by half, holding the "
            "fitted dynamics constant; it is descriptive of the fitted "
            "sample, not a forecast guarantee."
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily.index, y=daily.values, name="Spread (daily)",
                                  line=dict(color="#4C9BE8", width=1.5)))
        fig.add_hline(y=fit.mu, line=dict(color="#F5A623", width=1.5, dash="dash"),
                      annotation_text="Fitted long-run mean")
        fig.update_layout(
            height=320, margin=dict(t=20, l=10, r=10, b=10),
            plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font=dict(color="#FAFAFA"),
            xaxis=dict(gridcolor="#1E2329"), yaxis=dict(gridcolor="#1E2329", title="USD/bbl"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Options pricing: Black-Scholes")
    st.markdown(
        "Greeks shown on the Options tab come from the closed-form "
        "Black-Scholes formulas below (`pricing.bs_price`, `pricing.bs_greeks`), "
        "computed in Python; implied volatility is solved by bisection on "
        "market price (`pricing.implied_vol`)."
    )
    st.latex(
        r"C = S\,N(d_1) - K e^{-rT} N(d_2), \qquad "
        r"P = K e^{-rT} N(-d_2) - S\,N(-d_1)"
    )
    st.latex(
        r"d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, "
        r"\qquad d_2 = d_1 - \sigma\sqrt{T}"
    )
    opt_hist = store.read_options_snapshot(con)
    live_row = None
    if not opt_hist.empty:
        candidates = opt_hist.dropna(subset=["underlying_price", "atm_strike", "atm_iv"])
        if not candidates.empty:
            live_row = candidates.sort_values("ts").iloc[-1]

    if live_row is not None:
        ex_S = float(live_row["underlying_price"])
        ex_K = float(live_row["atm_strike"])
        ex_sigma = float(live_row["atm_iv"])
        ex_r = config.OPTIONS_RISK_FREE_RATE
        ex_T = 30 / 365  # Options tab records the ATM snapshot, not the specific expiry's tenor.
        g = pricing.bs_greeks(ex_S, ex_K, ex_T, ex_r, ex_sigma, "call")
        ec1, ec2, ec3, ec4, ec5, ec6 = st.columns(6)
        ec1.metric("Price", f"${g.price:.2f}")
        ec2.metric("Delta", f"{g.delta:.3f}")
        ec3.metric("Gamma", f"{g.gamma:.4f}")
        ec4.metric("Theta/day", f"{g.theta:.3f}")
        ec5.metric("Vega/1pt", f"{g.vega:.3f}")
        ec6.metric("Rho/1pt", f"{g.rho:.3f}")
        st.caption(
            f"Live ATM call on {live_row['ticker']} as of {live_row['ts']!s:.19} UTC: "
            f"S=${ex_S:.2f}, K=${ex_K:.2f}, σ={ex_sigma:.1%} (from the Options tab snapshot), "
            f"r={ex_r:.0%}, assumed 30-day tenor (the snapshot doesn't record the exact "
            "expiry, only the ATM contract's strike/IV)."
        )
    else:
        st.caption(
            "A live worked example appears here once the Options tab has been "
            "opened at least once this session. Illustrative example below "
            "with synthetic inputs in the meantime."
        )
        ex_S, ex_K, ex_T, ex_r, ex_sigma = 80.0, 80.0, 30 / 365, 0.05, 0.35
        g = pricing.bs_greeks(ex_S, ex_K, ex_T, ex_r, ex_sigma, "call")
        ec1, ec2, ec3, ec4, ec5, ec6 = st.columns(6)
        ec1.metric("Price", f"${g.price:.2f}")
        ec2.metric("Delta", f"{g.delta:.3f}")
        ec3.metric("Gamma", f"{g.gamma:.4f}")
        ec4.metric("Theta/day", f"{g.theta:.3f}")
        ec5.metric("Vega/1pt", f"{g.vega:.3f}")
        ec6.metric("Rho/1pt", f"{g.rho:.3f}")
        st.caption(
            f"Synthetic ATM 30-day call: S=K=${ex_S:.0f}, r={ex_r:.0%}, σ={ex_sigma:.0%}. "
            "Not a live quote."
        )


# ---------------------------------------------------------------------------
# Floating chatbot (grounded Q&A over everything computed elsewhere)
# ---------------------------------------------------------------------------

_CHATBOT_CSS = """
<style>
.st-key-chatbot_closed, .st-key-chatbot_open {
    position: fixed;
    bottom: 1.25rem;
    left: 1.25rem;
    z-index: 9999;
    background: #131720;
    border: 1px solid #2A2F3A;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.45);
    transition: width 0.2s ease-in-out, height 0.2s ease-in-out, padding 0.2s ease-in-out;
    overflow: hidden;
}
.st-key-chatbot_closed {
    width: 56px;
    height: 56px;
    padding: 0;
}
.st-key-chatbot_closed button {
    border-radius: 50%;
    width: 56px;
    height: 56px;
    font-size: 1.4rem;
}
.st-key-chatbot_open {
    width: 360px;
    max-height: 520px;
    padding: 0.75rem 0.75rem 0.25rem 0.75rem;
}
</style>
"""


def _chatbot_extra_context(con) -> dict:
    """Assemble correlations/news/options context for the chatbot's grounding payload.

    Reuses whatever's already cached from the other tabs rather than issuing
    fresh network calls, so opening the chatbot never triggers new fetches.
    """
    extra: dict = {}
    spread = store.read_spread(con)
    prices = _load_prices()
    if not spread.empty and not prices.empty:
        current_corr = correlations.compute_current_corr(spread, prices)
        top = current_corr.dropna().reindex(
            current_corr.dropna().abs().sort_values(ascending=False).index
        ).head(5)
        if not top.empty:
            extra["top_correlations"] = {
                correlations.ASSETS.get(t, {}).get("name", t): round(float(v), 3)
                for t, v in top.items()
            }

    mtx = _load_news().get("Marketaux Sentiment", [])
    sentiment = news.marketaux_sentiment_summary(mtx)
    if sentiment.get("available"):
        extra["news_sentiment"] = sentiment

    opt_hist = store.read_options_snapshot(con)
    if not opt_hist.empty:
        latest = opt_hist.sort_values("ts").iloc[-1]
        extra["latest_options_snapshot"] = {
            "ticker": latest["ticker"],
            "atm_iv": None if pd.isna(latest["atm_iv"]) else round(float(latest["atm_iv"]), 4),
            "put_call_oi_ratio": (
                None if pd.isna(latest["put_call_oi_ratio"]) else round(float(latest["put_call_oi_ratio"]), 3)
            ),
        }
    return extra


def _chatbot_widget() -> None:
    con = get_con()
    st.session_state.setdefault("chatbot_open", False)
    st.session_state.setdefault("chatbot_history", [])

    st.markdown(_CHATBOT_CSS, unsafe_allow_html=True)
    panel_key = "chatbot_open" if st.session_state.chatbot_open else "chatbot_closed"

    with st.container(key=panel_key):
        if not st.session_state.chatbot_open:
            if st.button("💬", key="chatbot_open_btn", help="Ask the tracker a question"):
                st.session_state.chatbot_open = True
                st.rerun()
            return

        head1, head2 = st.columns([5, 1])
        head1.markdown("**Ask the tracker**")
        if head2.button("✕", key="chatbot_close_btn"):
            st.session_state.chatbot_open = False
            st.rerun()

        if not config.GEMINI_API_KEY:
            st.caption("Set GEMINI_API_KEY in .env to enable the chatbot.")
            return

        history_box = st.container(height=280)
        with history_box:
            for turn in st.session_state.chatbot_history:
                with st.chat_message(turn["role"]):
                    st.write(turn["content"])

        prompt = st.chat_input("Ask about the spread, correlations, options...")
        if prompt:
            st.session_state.chatbot_history.append({"role": "user", "content": prompt})
            try:
                extra = _chatbot_extra_context(con)
                reply = chatbot.answer(
                    con, prompt, st.session_state.chatbot_history[:-1], extra=extra,
                )
            except Exception as exc:
                reply = f"Sorry, I couldn't generate an answer right now: {exc}"
            st.session_state.chatbot_history.append({"role": "assistant", "content": reply})
            st.rerun()


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

con = get_con()
_bootstrap(con)  # self-seed on first boot; no-op once data exists
st.title("Brent-WTI Spread Tracker")

with st.sidebar:
    st.header("Data controls")

    if st.button("Seed recent (yfinance)", help="Fetch last 5 days of daily bars. No API key required."):
        with st.spinner("Fetching from yfinance..."):
            try:
                with _ingest_gate()["lock"]:
                    n = ingest.seed_recent(con, period="5d", interval="1d")
                    spread_df = compute.build(con)
                    annotate.build(con, spread_df)
                st.success(f"Seeded {n} bars.")
                st.rerun()
            except Exception as exc:
                st.error(f"Seed failed: {exc}")

    if config.FRED_API_KEY:
        if st.button("Backfill from FRED (5 yr)", help="Pull 5 years of daily settlement history."):
            with st.spinner("Backfilling from FRED..."):
                try:
                    with _ingest_gate()["lock"]:
                        n = backfill.run(years=5, con=con)
                        spread_df = compute.build(con)
                        annotate.build(con, spread_df)
                    st.success(f"Backfilled {n} daily bars.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Backfill failed: {exc}")
    else:
        st.caption("Set FRED_API_KEY in .env for 5-year FRED backfill.")

    st.divider()
    with st.expander("Add manual annotation"):
        note_text = st.text_input("Note text")
        if st.button("Save note") and note_text:
            latest_spread = store.read_spread(con)
            if not latest_spread.empty:
                store.write_annotation(
                    con,
                    pd.to_datetime(latest_spread["ts"]).max(),
                    "manual", "info", note_text, "manual",
                )
                st.success("Saved.")
            else:
                st.warning("No spread data to anchor the annotation to.")

tab_dash, tab_market, tab_energy, tab_options, tab_research, tab_news, tab_about = st.tabs(
    ["Dashboard", "Market Lens", "Energy Trends", "Options", "Research", "News", "About"]
)

with tab_dash:
    _dashboard_tab()

with tab_market:
    _market_lens_tab()

with tab_energy:
    _energy_trends_tab()

with tab_options:
    _options_tab()

with tab_research:
    _research_tab()

with tab_news:
    _news_tab()

with tab_about:
    _about_tab()

_chatbot_widget()

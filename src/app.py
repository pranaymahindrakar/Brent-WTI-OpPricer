"""Streamlit + Plotly dashboard for the Brent-WTI spread tracker.

Run from the project root with:  streamlit run src/app.py

Four tabs:
  Dashboard    live spread chart with z-score, auto-refreshes every POLL_SECONDS
  Market Lens  rolling correlations against DXY, XLE, VLO, HO=F, RB=F, SPY, NG=F
  News         yfinance headlines, EIA RSS feed, NewsAPI headlines
  About        educational content on WTI, Brent, the spread, and market context
"""
from __future__ import annotations

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

from src import annotate, backfill, compute, config, correlations, ingest, insights, news, store

st.set_page_config(page_title="Brent-WTI Spread Tracker", layout="wide", page_icon="")


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

    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("Spread (USD/bbl)", f"{latest['spread']:.2f}" if pd.notna(latest["spread"]) else "N/A")
    mc2.metric("Z-score", f"{latest['zscore']:.2f}" if pd.notna(latest["zscore"]) else "N/A")
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
    if config.ANTHROPIC_API_KEY:
        if st.button("Generate insight note"):
            with st.spinner("Calling Anthropic API..."):
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
        st.caption("Set ANTHROPIC_API_KEY in .env to enable grounded LLM insight notes.")

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


def _market_lens_tab() -> None:
    con = get_con()
    spread = store.read_spread(con)

    st.subheader("Spread correlations with related markets")
    st.caption(
        "Correlation is computed on daily percentage returns over a rolling 60-day window. "
        "A positive value means the asset tends to move in the same direction as the spread "
        "(Brent premium widening); negative means they move oppositely."
    )

    rcol1, rcol2 = st.columns([5, 1])
    if rcol2.button("Refresh data"):
        _load_prices.clear()
        st.rerun()

    if spread.empty:
        st.info("No spread data. Seed or backfill data from the sidebar first.")
        return

    prices = _load_prices()
    if prices.empty:
        st.warning("Could not fetch correlated asset prices from yfinance.")
        return

    current = correlations.compute_current_corr(spread, prices)
    rolling = correlations.compute_rolling_corr(spread, prices)

    if current.empty:
        st.info("Not enough overlapping history to compute correlations yet.")
        return

    # Current correlation bar chart
    chart_col, text_col = st.columns([3, 2])
    with chart_col:
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
            height=340,
            margin=dict(t=40, l=10, r=60, b=10),
            xaxis=dict(range=[-1.1, 1.1], gridcolor="#1E2329", zeroline=True,
                       zerolinecolor="#AAB7C4"),
            plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
            font=dict(color="#FAFAFA"),
            yaxis=dict(gridcolor="#1E2329"),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with text_col:
        st.markdown("**What each correlation means**")
        for ticker, val in current.dropna().sort_values(ascending=False).items():
            meta = correlations.ASSETS.get(ticker, {})
            strength = "strong" if abs(val) > 0.5 else "moderate" if abs(val) > 0.3 else "weak"
            direction = "positive" if val > 0 else "negative"
            st.markdown(
                f"**{meta.get('name', ticker)}** `{val:+.2f}`  \n"
                f"_{strength.capitalize()} {direction}._ {meta.get('note', '')}"
            )
            st.divider()

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

tab_dash, tab_market, tab_news, tab_about = st.tabs(
    ["Dashboard", "Market Lens", "News", "About"]
)

with tab_dash:
    _dashboard_tab()

with tab_market:
    _market_lens_tab()

with tab_news:
    _news_tab()

with tab_about:
    _about_tab()

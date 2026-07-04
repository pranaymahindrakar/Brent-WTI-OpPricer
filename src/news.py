"""News feed aggregation from three free sources.

All fetchers return a uniform list of article dicts:
  {"title": str, "publisher": str, "link": str, "ts": datetime}

Callers should cache the results; these functions make network requests.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Optional

EIA_RSS_URL = "https://www.eia.gov/rss/todayinenergy.xml"
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
MARKETAUX_ENDPOINT = "https://api.marketaux.com/v1/news/all"


def _parse_rfc2822(s: str) -> datetime:
    """Parse an RFC 2822 date string (pubDate in RSS 2.0) to a naive UTC datetime."""
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(s).replace(tzinfo=None)
    except Exception:
        return datetime.utcnow()


def fetch_yfinance_news(tickers: list[str] | None = None) -> list[dict]:
    """Pull recent headlines from Yahoo Finance for crude oil and energy tickers."""
    import yfinance as yf

    tickers = tickers or ["CL=F", "BZ=F", "XLE"]
    seen: set[str] = set()
    articles: list[dict] = []
    for t in tickers:
        try:
            items = yf.Ticker(t).news or []
            for item in items:
                url = item.get("link") or item.get("url", "")
                if not url or url in seen:
                    continue
                seen.add(url)
                ts_raw = item.get("providerPublishTime") or item.get("publishedAt", 0)
                if isinstance(ts_raw, str):
                    try:
                        ts = datetime.fromisoformat(
                            ts_raw.replace("Z", "+00:00")
                        ).replace(tzinfo=None)
                    except ValueError:
                        ts = datetime.utcnow()
                else:
                    ts = datetime.utcfromtimestamp(int(ts_raw)) if ts_raw else datetime.utcnow()
                articles.append({
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", "Yahoo Finance"),
                    "link": url,
                    "ts": ts,
                })
        except Exception:
            pass
    return sorted(articles, key=lambda x: x["ts"], reverse=True)


def fetch_eia_rss() -> list[dict]:
    """Pull EIA official energy news from their public RSS 2.0 feed (no key required)."""
    import requests

    try:
        resp = requests.get(EIA_RSS_URL, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        articles: list[dict] = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            if not title or not link:
                continue
            articles.append({
                "title": title,
                "publisher": "EIA",
                "link": link,
                "ts": _parse_rfc2822(pub),
            })
        return sorted(articles, key=lambda x: x["ts"], reverse=True)
    except Exception:
        return []


def fetch_newsapi(
    query: str = "crude oil Brent WTI spread energy",
    api_key: Optional[str] = None,
) -> list[dict]:
    """Pull headlines from NewsAPI.org. Free tier: 100 req/day, last 30 days of news."""
    if not api_key:
        return []
    import requests

    try:
        resp = requests.get(
            NEWSAPI_ENDPOINT,
            params={
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 25,
                "apiKey": api_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        articles: list[dict] = []
        for art in data.get("articles", []):
            title = (art.get("title") or "").strip()
            url = (art.get("url") or "").strip()
            if not title or not url or title == "[Removed]":
                continue
            pub_raw = art.get("publishedAt", "")
            try:
                ts = datetime.fromisoformat(
                    pub_raw.replace("Z", "+00:00")
                ).replace(tzinfo=None)
            except Exception:
                ts = datetime.utcnow()
            articles.append({
                "title": title,
                "publisher": art.get("source", {}).get("name", "NewsAPI"),
                "link": url,
                "ts": ts,
            })
        return articles
    except Exception:
        return []


def _parse_iso8601(s: str) -> datetime:
    """Parse an ISO 8601 timestamp (Marketaux published_at) to a naive UTC datetime."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return datetime.utcnow()


def fetch_marketaux(
    api_key: Optional[str] = None,
    search: str = "crude oil OR Brent OR WTI OR OPEC OR refinery OR shale",
    pages: int = 3,
    lookback_days: int = 365,
) -> list[dict]:
    """Pull crude oil headlines with per-entity sentiment from Marketaux.

    Marketaux enriches each article with the publicly traded entities it mentions
    (oil majors and energy ETFs such as XLE and XOP for crude headlines) and a
    sentiment_score in [-1, 1] for each. We average the per-entity scores into a
    single article sentiment; articles with no scored entity keep sentiment None
    and are still shown as headlines. Results are requested newest first
    (sort=published_at, descending) and bounded to the recent window.

    Free tier returns three articles per request and caps daily requests, so
    `pages` controls how many sequential requests are issued per refresh. Each
    returned dict carries the uniform keys plus `snippet`, `source`, `entities`
    (list of {symbol, name, sentiment}), and `sentiment` (float or None).
    """
    if not api_key:
        return []
    import requests

    published_after = (
        datetime.utcnow() - timedelta(days=lookback_days)
    ).strftime("%Y-%m-%dT%H:%M")
    seen: set[str] = set()
    articles: list[dict] = []
    for page in range(1, max(1, pages) + 1):
        try:
            resp = requests.get(
                MARKETAUX_ENDPOINT,
                params={
                    "search": search,
                    "language": "en",
                    "published_after": published_after,
                    "sort": "published_at",
                    "sort_order": "desc",
                    "limit": 3,
                    "page": page,
                    "api_token": api_key,
                },
                timeout=12,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            break

        rows = data.get("data") or []
        if not rows:
            break
        for art in rows:
            url = (art.get("url") or "").strip()
            title = (art.get("title") or "").strip()
            if not url or not title or url in seen:
                continue
            seen.add(url)

            entities: list[dict] = []
            scores: list[float] = []
            for ent in art.get("entities") or []:
                score = ent.get("sentiment_score")
                if isinstance(score, (int, float)):
                    scores.append(float(score))
                entities.append({
                    "symbol": ent.get("symbol", ""),
                    "name": ent.get("name", ""),
                    "sentiment": float(score) if isinstance(score, (int, float)) else None,
                })
            sentiment = round(sum(scores) / len(scores), 4) if scores else None

            articles.append({
                "title": title,
                "publisher": art.get("source", "Marketaux"),
                "link": url,
                "ts": _parse_iso8601(art.get("published_at", "")),
                "snippet": (art.get("snippet") or art.get("description") or "").strip(),
                "source": art.get("source", ""),
                "entities": entities,
                "sentiment": sentiment,
            })

        # Stop early once we have paged past the available results.
        meta = data.get("meta") or {}
        if meta.get("returned", 0) < 3:
            break

    return sorted(articles, key=lambda x: x["ts"], reverse=True)


def marketaux_sentiment_summary(articles: list[dict]) -> dict:
    """Aggregate Marketaux per-article sentiment into a single grounded snapshot.

    Returns the mean sentiment across articles that carry a score, the count of
    scored articles, and a bullish/neutral/bearish tally using a +/-0.15 band.
    All figures are computed here in Python so they can be passed verbatim to the
    LLM without it performing any arithmetic.
    """
    scored = [a["sentiment"] for a in articles if a.get("sentiment") is not None]
    if not scored:
        return {"available": False, "n_scored": 0}
    bullish = sum(1 for s in scored if s > 0.15)
    bearish = sum(1 for s in scored if s < -0.15)
    neutral = len(scored) - bullish - bearish
    mean = sum(scored) / len(scored)
    if mean > 0.15:
        label = "bullish"
    elif mean < -0.15:
        label = "bearish"
    else:
        label = "neutral"
    return {
        "available": True,
        "n_articles": len(articles),
        "n_scored": len(scored),
        "mean_sentiment": round(mean, 4),
        "label": label,
        "bullish": bullish,
        "neutral": neutral,
        "bearish": bearish,
    }


def fetch_all(
    newsapi_key: Optional[str] = None,
    marketaux_key: Optional[str] = None,
    marketaux_search: str = "crude oil OR Brent OR WTI OR OPEC OR refinery OR shale",
    marketaux_pages: int = 3,
    marketaux_lookback_days: int = 365,
) -> dict[str, list[dict]]:
    """Fetch from all configured sources. Returns a dict keyed by source display name."""
    return {
        "Yahoo Finance": fetch_yfinance_news(),
        "EIA Official Feed": fetch_eia_rss(),
        "NewsAPI Headlines": fetch_newsapi(api_key=newsapi_key),
        "Marketaux Sentiment": fetch_marketaux(
            api_key=marketaux_key,
            search=marketaux_search,
            pages=marketaux_pages,
            lookback_days=marketaux_lookback_days,
        ),
    }

"""
news_sentiment.py — Financial news sentiment analysis for Indian markets.

Scrapes headlines from RSS feeds (Economic Times, MoneyControl, etc.)
and uses Groq AI to classify sentiment (Bullish / Bearish / Neutral).

Also provides per-stock news scanning and market mood aggregation.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import requests
from openai import OpenAI

import config

logger = logging.getLogger(__name__)


# ── RSS Feed Sources ─────────────────────────────────────────

RSS_FEEDS: dict[str, str] = {
    "ET Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "ET Stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "Moneycontrol": "https://www.moneycontrol.com/rss/marketreports.xml",
    "LiveMint Markets": "https://www.livemint.com/rss/markets",
}


# ── Data Structures ──────────────────────────────────────────

@dataclass
class NewsItem:
    """A single news headline."""

    title: str
    source: str
    link: str = ""
    published: str = ""
    sentiment: str = ""     # BULLISH / BEARISH / NEUTRAL
    score: float = 0.0      # -1.0 to +1.0
    relevant_stocks: list[str] = field(default_factory=list)


@dataclass
class MarketSentiment:
    """Aggregated market sentiment."""

    overall: str = "NEUTRAL"       # BULLISH / BEARISH / NEUTRAL
    score: float = 0.0             # -1.0 to +1.0
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    total_headlines: int = 0
    top_bullish: list[str] = field(default_factory=list)
    top_bearish: list[str] = field(default_factory=list)
    news_items: list[NewsItem] = field(default_factory=list)


# ── RSS Fetching ─────────────────────────────────────────────

def _fetch_rss(url: str, source_name: str, max_items: int = 15) -> list[NewsItem]:
    """Fetch and parse an RSS feed."""
    items = []
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StockAgent/1.0"
        })
        resp.raise_for_status()

        root = ET.fromstring(resp.content)

        # Handle both RSS 2.0 and Atom formats
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        # RSS 2.0
        for item in root.findall(".//item")[:max_items]:
            title = item.findtext("title", "").strip()
            link = item.findtext("link", "").strip()
            pub_date = item.findtext("pubDate", "").strip()

            if title:
                items.append(NewsItem(
                    title=title,
                    source=source_name,
                    link=link,
                    published=pub_date,
                ))

        # Atom fallback
        if not items:
            for entry in root.findall("atom:entry", ns)[:max_items]:
                title = entry.findtext("atom:title", "", ns).strip()
                link_el = entry.find("atom:link", ns)
                link = link_el.get("href", "") if link_el is not None else ""

                if title:
                    items.append(NewsItem(
                        title=title,
                        source=source_name,
                        link=link,
                    ))

    except Exception as exc:
        logger.warning("Failed to fetch RSS from %s: %s", source_name, exc)

    logger.debug("Fetched %d headlines from %s", len(items), source_name)
    return items


def fetch_all_news(max_per_source: int = 12) -> list[NewsItem]:
    """Fetch headlines from all configured RSS feeds."""
    all_items: list[NewsItem] = []
    for name, url in RSS_FEEDS.items():
        all_items.extend(_fetch_rss(url, name, max_per_source))

    # De-duplicate by title similarity
    seen_titles: set[str] = set()
    unique: list[NewsItem] = []
    for item in all_items:
        key = re.sub(r"[^a-z0-9]", "", item.title.lower())[:60]
        if key not in seen_titles:
            seen_titles.add(key)
            unique.append(item)

    logger.info("Fetched %d unique headlines from %d sources.", len(unique), len(RSS_FEEDS))
    return unique


# ── AI Sentiment Analysis ───────────────────────────────────

def _analyze_batch_sentiment(headlines: list[str]) -> list[dict]:
    """Use Groq AI to classify a batch of headlines.

    Returns list of {"sentiment": "BULLISH/BEARISH/NEUTRAL", "score": float}.
    """
    if not headlines:
        return []

    client = OpenAI(
        api_key=config.GROQ_API_KEY,
        base_url=config.GROQ_BASE_URL,
    )

    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))

    prompt = f"""Analyse these Indian stock market news headlines for sentiment.
For EACH headline, classify as BULLISH, BEARISH, or NEUTRAL.
Also give a score from -1.0 (very bearish) to +1.0 (very bullish).

Headlines:
{numbered}

Reply ONLY in this exact format, one line per headline:
1|BULLISH|0.7
2|NEUTRAL|0.0
3|BEARISH|-0.5
...

No explanations. Just the numbered lines."""

    try:
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a financial sentiment analyst for Indian stock markets."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
        )

        text = response.choices[0].message.content.strip()
        results = []
        for line in text.split("\n"):
            parts = line.strip().split("|")
            if len(parts) >= 3:
                sentiment = parts[1].strip().upper()
                if sentiment not in ("BULLISH", "BEARISH", "NEUTRAL"):
                    sentiment = "NEUTRAL"
                try:
                    score = float(parts[2].strip())
                    score = max(-1.0, min(1.0, score))
                except ValueError:
                    score = 0.0
                results.append({"sentiment": sentiment, "score": score})

        # Pad if AI returned fewer lines
        while len(results) < len(headlines):
            results.append({"sentiment": "NEUTRAL", "score": 0.0})

        return results[:len(headlines)]

    except Exception as exc:
        logger.error("AI sentiment analysis failed: %s", exc)
        return [{"sentiment": "NEUTRAL", "score": 0.0}] * len(headlines)


def _find_stock_mentions(title: str, universe: list[str] | None = None) -> list[str]:
    """Find stock names mentioned in a headline."""
    stocks = universe or config.NIFTY_50
    title_upper = title.upper()
    found = []
    for symbol in stocks:
        # Match full symbol name or common company names
        if symbol.upper() in title_upper:
            found.append(symbol)
    # Also check common company name mappings
    COMPANY_NAMES = {
        "RELIANCE": ["RELIANCE", "RIL", "MUKESH AMBANI"],
        "TCS": ["TCS", "TATA CONSULTANCY"],
        "INFY": ["INFOSYS", "INFY"],
        "HDFCBANK": ["HDFC BANK"],
        "ICICIBANK": ["ICICI BANK"],
        "SBIN": ["SBI", "STATE BANK"],
        "WIPRO": ["WIPRO"],
        "BAJFINANCE": ["BAJAJ FINANCE"],
        "ITC": ["ITC"],
        "BHARTIARTL": ["AIRTEL", "BHARTI AIRTEL"],
        "HINDUNILVR": ["HUL", "HINDUSTAN UNILEVER"],
        "MARUTI": ["MARUTI", "MARUTI SUZUKI"],
        "SUNPHARMA": ["SUN PHARMA"],
        "TITAN": ["TITAN"],
        "ADANIENT": ["ADANI"],
        "LT": ["L&T", "LARSEN"],
        "KOTAKBANK": ["KOTAK"],
        "AXISBANK": ["AXIS BANK"],
    }
    for symbol, names in COMPANY_NAMES.items():
        for name in names:
            if name in title_upper and symbol not in found:
                found.append(symbol)
    return found


# ── Main Analysis Function ───────────────────────────────────

def analyze_market_sentiment(
    universe: list[str] | None = None,
) -> MarketSentiment:
    """Fetch news and run AI sentiment analysis.

    Parameters
    ----------
    universe : list[str] | None
        Stock universe for mention detection.

    Returns
    -------
    MarketSentiment
        Complete sentiment analysis with individual headlines.
    """
    news_items = fetch_all_news()
    if not news_items:
        logger.warning("No news headlines fetched.")
        return MarketSentiment()

    # Batch analyse in chunks of 20
    BATCH_SIZE = 20
    for i in range(0, len(news_items), BATCH_SIZE):
        batch = news_items[i:i + BATCH_SIZE]
        headlines = [item.title for item in batch]
        results = _analyze_batch_sentiment(headlines)

        for item, result in zip(batch, results):
            item.sentiment = result["sentiment"]
            item.score = result["score"]
            item.relevant_stocks = _find_stock_mentions(item.title, universe)

    # Aggregate
    sentiment = MarketSentiment()
    sentiment.news_items = news_items
    sentiment.total_headlines = len(news_items)

    scores = []
    for item in news_items:
        scores.append(item.score)
        if item.sentiment == "BULLISH":
            sentiment.bullish_count += 1
        elif item.sentiment == "BEARISH":
            sentiment.bearish_count += 1
        else:
            sentiment.neutral_count += 1

    if scores:
        sentiment.score = sum(scores) / len(scores)

    if sentiment.score >= 0.15:
        sentiment.overall = "BULLISH"
    elif sentiment.score <= -0.15:
        sentiment.overall = "BEARISH"
    else:
        sentiment.overall = "NEUTRAL"

    # Top bullish/bearish headlines
    sorted_items = sorted(news_items, key=lambda x: x.score, reverse=True)
    sentiment.top_bullish = [
        item.title for item in sorted_items[:3] if item.score > 0
    ]
    sentiment.top_bearish = [
        item.title for item in sorted_items[-3:] if item.score < 0
    ]

    logger.info(
        "Sentiment: %s (%.2f) — %d bullish, %d bearish, %d neutral",
        sentiment.overall, sentiment.score,
        sentiment.bullish_count, sentiment.bearish_count, sentiment.neutral_count,
    )
    return sentiment


def format_sentiment_report(sentiment: MarketSentiment) -> str:
    """Format sentiment analysis as Telegram-friendly HTML."""
    if sentiment.total_headlines == 0:
        return "📰 No news headlines available."

    # Mood emoji
    if sentiment.overall == "BULLISH":
        mood = "🟢 BULLISH"
    elif sentiment.overall == "BEARISH":
        mood = "🔴 BEARISH"
    else:
        mood = "⚪ NEUTRAL"

    # Score bar (visual)
    bar_pos = int((sentiment.score + 1) / 2 * 10)  # 0-10
    bar = "▓" * bar_pos + "░" * (10 - bar_pos)

    lines = [
        "<b>📰 NEWS SENTIMENT ANALYSIS</b>",
        "",
        f"<b>Market Mood:</b> {mood}",
        f"<b>Score:</b> [{bar}] {sentiment.score:+.2f}",
        "",
        f"📊 Headlines Analysed: {sentiment.total_headlines}",
        f"🟢 Bullish: {sentiment.bullish_count} | "
        f"🔴 Bearish: {sentiment.bearish_count} | "
        f"⚪ Neutral: {sentiment.neutral_count}",
    ]

    if sentiment.top_bullish:
        lines.append("")
        lines.append("<b>🟢 Top Bullish:</b>")
        for h in sentiment.top_bullish[:3]:
            lines.append(f"  • {h[:80]}")

    if sentiment.top_bearish:
        lines.append("")
        lines.append("<b>🔴 Top Bearish:</b>")
        for h in sentiment.top_bearish[:3]:
            lines.append(f"  • {h[:80]}")

    # Stock mentions
    stock_mentions: dict[str, int] = {}
    for item in sentiment.news_items:
        for s in item.relevant_stocks:
            stock_mentions[s] = stock_mentions.get(s, 0) + 1

    if stock_mentions:
        top_mentioned = sorted(stock_mentions.items(), key=lambda x: x[1], reverse=True)[:5]
        lines.append("")
        lines.append("<b>🔥 Most Mentioned:</b>")
        lines.append("  " + " | ".join(f"{s} ({c})" for s, c in top_mentioned))

    return "\n".join(lines)


# ── Quick Test ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    sentiment = analyze_market_sentiment()
    print(format_sentiment_report(sentiment))

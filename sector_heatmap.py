"""
sector_heatmap.py — Visual sector-wise performance analysis.

Groups NSE stocks by sector and calculates:
  - Sector average daily/weekly change
  - Top/bottom performers per sector
  - Money flow (volume × price change)
  - Sector rotation signals

Output: Telegram-friendly text heatmap + sector scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

import config
from data_fetcher import fetch_ohlcv, fetch_fundamentals

logger = logging.getLogger(__name__)


# ── Sector Mapping for Indian Markets ────────────────────────

SECTOR_MAP: dict[str, str] = {
    # Banking & Finance
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
    "KOTAKBANK": "Banking", "AXISBANK": "Banking", "BAJFINANCE": "Banking",
    "BAJAJFINSV": "Banking", "INDUSINDBK": "Banking", "BANKBARODA": "Banking",
    "PNB": "Banking", "CANBK": "Banking", "IDFCFIRSTB": "Banking",
    "YESBANK": "Banking", "SBILIFE": "Insurance", "HDFCLIFE": "Insurance",
    "ICICIPRULI": "Insurance",

    # IT & Tech
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT",
    "TECHM": "IT", "LTIM": "IT", "PERSISTENT": "IT", "COFORGE": "IT",
    "MPHASIS": "IT", "TATAELXSI": "IT", "NAUKRI": "IT", "DIXON": "IT",

    # Oil & Energy
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy",
    "IOC": "Energy", "GAIL": "Energy", "NTPC": "Power",
    "POWERGRID": "Power", "TATAPOWER": "Power", "NHPC": "Power",
    "RECLTD": "Power", "PFC": "Power", "COALINDIA": "Mining",
    "ADANIENT": "Energy", "ADANIPORTS": "Ports",

    # Metals & Mining
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "VEDL": "Metals", "SAIL": "Metals", "NATIONALUM": "Metals",
    "JINDALSTEL": "Metals",

    # Pharma & Healthcare
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "APOLLOHOSP": "Healthcare", "TORNTPHARM": "Pharma",

    # Auto
    "MARUTI": "Auto", "BAJAJ-AUTO": "Auto", "EICHERMOT": "Auto",
    "HEROMOTOCO": "Auto",

    # FMCG & Consumer
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "TATACONSUM": "FMCG", "GODREJCP": "FMCG",
    "COLPAL": "FMCG", "MARICO": "FMCG", "DMART": "Retail",

    # Cement & Construction
    "ULTRACEMCO": "Cement", "GRASIM": "Cement", "AMBUJACEM": "Cement",
    "SHREECEM": "Cement", "LT": "Construction", "DLF": "Realty",

    # Telecom
    "BHARTIARTL": "Telecom", "IDEA": "Telecom",

    # Others
    "TITAN": "Consumer", "ASIANPAINT": "Consumer", "PIDILITIND": "Chemicals",
    "SIEMENS": "Capital Goods", "ABB": "Capital Goods", "HAL": "Defence",
    "BEL": "Defence", "BHEL": "Capital Goods", "IRCTC": "Railway",
    "RVNL": "Railway", "IRFC": "Railway",
    "SUZLON": "Renewable", "ZOMATO": "Tech Platform", "TRENT": "Retail",
    "INDIGO": "Aviation", "POLYCAB": "Capital Goods",
    "HAVELLS": "Capital Goods", "ASTRAL": "Chemicals",
    "DEEPAKNTR": "Chemicals", "TIINDIA": "Capital Goods",
    "MCDOWELL-N": "FMCG",
}


@dataclass
class SectorPerformance:
    """Performance data for a single sector."""

    name: str
    stocks_count: int = 0
    avg_change_1d: float = 0.0          # avg 1-day % change
    avg_change_5d: float = 0.0          # avg 5-day % change
    total_volume_ratio: float = 0.0     # avg volume vs 20-day avg
    top_gainer: str = ""
    top_gainer_pct: float = 0.0
    top_loser: str = ""
    top_loser_pct: float = 0.0
    sentiment: str = "NEUTRAL"          # BULLISH / BEARISH / NEUTRAL
    heat_score: int = 0                 # -100 to +100


def _get_sector(symbol: str) -> str:
    """Look up sector for a symbol."""
    return SECTOR_MAP.get(symbol, "Other")


def calculate_sector_performance(
    universe: list[str] | None = None,
) -> list[SectorPerformance]:
    """Calculate performance for all sectors.

    Parameters
    ----------
    universe : list[str] | None
        Stocks to scan. Defaults to FULL_UNIVERSE.

    Returns
    -------
    list[SectorPerformance]
        Sector performance data sorted by heat_score.
    """
    stocks = universe or config.FULL_UNIVERSE

    logger.info("Calculating sector heatmap for %d stocks …", len(stocks))

    # Collect per-stock data
    stock_data: dict[str, dict] = {}
    for symbol in stocks:
        try:
            df = fetch_ohlcv(symbol, period_days=30)
            if df.empty or len(df) < 6:
                continue

            last_close = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else last_close
            close_5d = float(df["Close"].iloc[-6]) if len(df) >= 6 else last_close
            avg_vol = float(df["Volume"].rolling(20).mean().dropna().iloc[-1]) if len(df) >= 20 else float(df["Volume"].mean())
            cur_vol = float(df["Volume"].iloc[-1])

            change_1d = ((last_close - prev_close) / prev_close) * 100
            change_5d = ((last_close - close_5d) / close_5d) * 100
            vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0

            stock_data[symbol] = {
                "sector": _get_sector(symbol),
                "change_1d": change_1d,
                "change_5d": change_5d,
                "vol_ratio": vol_ratio,
                "last_close": last_close,
            }
        except Exception as exc:
            logger.debug("Sector scan failed for %s: %s", symbol, exc)

    # Aggregate by sector
    sectors: dict[str, list[dict]] = {}
    for symbol, data in stock_data.items():
        sector = data["sector"]
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append({"symbol": symbol, **data})

    results: list[SectorPerformance] = []
    for sector_name, stocks_in_sector in sectors.items():
        perf = SectorPerformance(name=sector_name)
        perf.stocks_count = len(stocks_in_sector)

        changes_1d = [s["change_1d"] for s in stocks_in_sector]
        changes_5d = [s["change_5d"] for s in stocks_in_sector]
        vol_ratios = [s["vol_ratio"] for s in stocks_in_sector]

        perf.avg_change_1d = np.mean(changes_1d)
        perf.avg_change_5d = np.mean(changes_5d)
        perf.total_volume_ratio = np.mean(vol_ratios)

        # Top gainer/loser
        sorted_by_1d = sorted(stocks_in_sector, key=lambda s: s["change_1d"], reverse=True)
        if sorted_by_1d:
            perf.top_gainer = sorted_by_1d[0]["symbol"]
            perf.top_gainer_pct = sorted_by_1d[0]["change_1d"]
            perf.top_loser = sorted_by_1d[-1]["symbol"]
            perf.top_loser_pct = sorted_by_1d[-1]["change_1d"]

        # Heat score: weighted combination
        score = 0
        score += perf.avg_change_1d * 10    # 1-day change (heavy weight)
        score += perf.avg_change_5d * 5     # 5-day trend
        if perf.total_volume_ratio > 1.5:
            score += 10                     # High volume boost
        elif perf.total_volume_ratio < 0.5:
            score -= 5

        perf.heat_score = max(-100, min(100, int(score)))

        if perf.heat_score >= 20:
            perf.sentiment = "BULLISH"
        elif perf.heat_score <= -20:
            perf.sentiment = "BEARISH"
        else:
            perf.sentiment = "NEUTRAL"

        results.append(perf)

    results.sort(key=lambda s: s.heat_score, reverse=True)
    logger.info("Sector heatmap calculated for %d sectors.", len(results))
    return results


def format_sector_heatmap(sectors: list[SectorPerformance]) -> str:
    """Format sector data as a Telegram/text-friendly heatmap."""
    if not sectors:
        return "No sector data available."

    lines = ["<b>🗺️ SECTOR HEATMAP</b>", ""]

    # Heatmap bars
    for s in sectors:
        # Visual bar
        if s.heat_score >= 30:
            emoji = "🟢🟢🟢"
        elif s.heat_score >= 15:
            emoji = "🟢🟢"
        elif s.heat_score >= 5:
            emoji = "🟢"
        elif s.heat_score >= -5:
            emoji = "⚪"
        elif s.heat_score >= -15:
            emoji = "🔴"
        elif s.heat_score >= -30:
            emoji = "🔴🔴"
        else:
            emoji = "🔴🔴🔴"

        sign = "+" if s.avg_change_1d >= 0 else ""
        lines.append(
            f"{emoji} <b>{s.name}</b> ({s.stocks_count}) "
            f"— {sign}{s.avg_change_1d:.1f}% today | "
            f"{'+' if s.avg_change_5d >= 0 else ''}{s.avg_change_5d:.1f}% (5d)"
        )
        lines.append(
            f"   ↑ {s.top_gainer} +{s.top_gainer_pct:.1f}% | "
            f"↓ {s.top_loser} {s.top_loser_pct:.1f}%"
        )

    # Summary
    bullish = [s for s in sectors if s.sentiment == "BULLISH"]
    bearish = [s for s in sectors if s.sentiment == "BEARISH"]
    lines.append("")
    lines.append(
        f"<b>Market Mood:</b> "
        f"🟢 {len(bullish)} bullish | "
        f"🔴 {len(bearish)} bearish | "
        f"⚪ {len(sectors) - len(bullish) - len(bearish)} neutral"
    )

    if bullish:
        lines.append(f"<b>Hot Sectors:</b> {', '.join(s.name for s in bullish[:3])}")
    if bearish:
        lines.append(f"<b>Cold Sectors:</b> {', '.join(s.name for s in bearish[:3])}")

    return "\n".join(lines)


# ── Quick Test ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    sectors = calculate_sector_performance(config.NIFTY_50[:20])
    print(format_sector_heatmap(sectors))

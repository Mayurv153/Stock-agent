"""
portfolio_tracker.py — Track real holdings, P&L, and exit signals.

Reads portfolio from portfolio.json (or creates a sample one).
Calculates:
  - Unrealised P&L per stock and total
  - Day change, overall return %
  - Target / stop-loss hit detection
  - Trailing stop updates

JSON format for portfolio.json:
[
  {
    "symbol": "RELIANCE",
    "buy_price": 2450.0,
    "qty": 10,
    "buy_date": "2024-12-01",
    "target": 2700.0,
    "stop_loss": 2350.0
  }
]
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import config
from data_fetcher import fetch_ohlcv

logger = logging.getLogger(__name__)

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "portfolio.json")


# ── Data Structures ──────────────────────────────────────────

@dataclass
class Holding:
    """A single portfolio holding."""

    symbol: str
    buy_price: float
    qty: int
    buy_date: str
    target: float = 0.0
    stop_loss: float = 0.0

    # Calculated fields
    current_price: float = 0.0
    day_change_pct: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    invested: float = 0.0
    current_value: float = 0.0
    status: str = "OPEN"  # OPEN / TARGET_HIT / STOP_HIT / TRAILING_STOP


@dataclass
class PortfolioSummary:
    """Aggregate portfolio metrics."""

    total_invested: float = 0.0
    total_current: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    total_day_change: float = 0.0
    winners: int = 0
    losers: int = 0
    holdings: list[Holding] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)


# ── Portfolio I/O ────────────────────────────────────────────

SAMPLE_PORTFOLIO = [
    {
        "symbol": "RELIANCE",
        "buy_price": 1250.0,
        "qty": 10,
        "buy_date": "2024-10-15",
        "target": 1450.0,
        "stop_loss": 1150.0,
    },
    {
        "symbol": "TCS",
        "buy_price": 3700.0,
        "qty": 5,
        "buy_date": "2024-11-01",
        "target": 4200.0,
        "stop_loss": 3500.0,
    },
    {
        "symbol": "HDFCBANK",
        "buy_price": 1650.0,
        "qty": 15,
        "buy_date": "2024-11-20",
        "target": 1850.0,
        "stop_loss": 1550.0,
    },
]


def load_portfolio() -> list[dict]:
    """Load portfolio from JSON file. Create sample if missing."""
    if not os.path.exists(PORTFOLIO_FILE):
        logger.info("No portfolio.json found — creating sample portfolio.")
        save_portfolio(SAMPLE_PORTFOLIO)
        return SAMPLE_PORTFOLIO

    with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d holdings from portfolio.json", len(data))
    return data


def save_portfolio(holdings: list[dict]) -> None:
    """Save portfolio to JSON."""
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(holdings, f, indent=2, ensure_ascii=False)
    logger.info("Portfolio saved to %s", PORTFOLIO_FILE)


def add_holding(
    symbol: str,
    buy_price: float,
    qty: int,
    buy_date: str | None = None,
    target: float = 0.0,
    stop_loss: float = 0.0,
) -> None:
    """Add a new holding to the portfolio."""
    portfolio = load_portfolio()
    portfolio.append({
        "symbol": symbol.upper(),
        "buy_price": buy_price,
        "qty": qty,
        "buy_date": buy_date or datetime.now().strftime("%Y-%m-%d"),
        "target": target,
        "stop_loss": stop_loss,
    })
    save_portfolio(portfolio)
    logger.info("Added %s × %d @ ₹%.1f", symbol, qty, buy_price)


def remove_holding(symbol: str) -> bool:
    """Remove a holding by symbol."""
    portfolio = load_portfolio()
    filtered = [h for h in portfolio if h["symbol"].upper() != symbol.upper()]
    if len(filtered) == len(portfolio):
        return False
    save_portfolio(filtered)
    logger.info("Removed %s from portfolio.", symbol.upper())
    return True


# ── Portfolio Analysis ───────────────────────────────────────

def track_portfolio() -> PortfolioSummary:
    """Fetch live prices and calculate P&L for all holdings.

    Returns
    -------
    PortfolioSummary
        Full portfolio analysis with alerts.
    """
    raw = load_portfolio()
    summary = PortfolioSummary()

    for entry in raw:
        symbol = entry["symbol"]
        h = Holding(
            symbol=symbol,
            buy_price=entry["buy_price"],
            qty=entry["qty"],
            buy_date=entry.get("buy_date", ""),
            target=entry.get("target", 0.0),
            stop_loss=entry.get("stop_loss", 0.0),
        )

        try:
            df = fetch_ohlcv(symbol, period_days=5)
            if df.empty:
                logger.warning("No data for %s", symbol)
                continue

            h.current_price = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else h.current_price
            h.day_change_pct = ((h.current_price - prev_close) / prev_close) * 100

            h.invested = h.buy_price * h.qty
            h.current_value = h.current_price * h.qty
            h.pnl = h.current_value - h.invested
            h.pnl_pct = ((h.current_price - h.buy_price) / h.buy_price) * 100

            # Check target / stop-loss
            if h.target > 0 and h.current_price >= h.target:
                h.status = "TARGET_HIT"
                summary.alerts.append(
                    f"🎯 {symbol}: Target ₹{h.target:.0f} HIT! "
                    f"CMP ₹{h.current_price:.1f} (+{h.pnl_pct:.1f}%)"
                )
            elif h.stop_loss > 0 and h.current_price <= h.stop_loss:
                h.status = "STOP_HIT"
                summary.alerts.append(
                    f"🚨 {symbol}: Stop Loss ₹{h.stop_loss:.0f} HIT! "
                    f"CMP ₹{h.current_price:.1f} ({h.pnl_pct:.1f}%)"
                )
            else:
                h.status = "OPEN"

            # P&L tracking
            if h.pnl >= 0:
                summary.winners += 1
            else:
                summary.losers += 1

            summary.total_invested += h.invested
            summary.total_current += h.current_value

        except Exception as exc:
            logger.error("Portfolio tracking failed for %s: %s", symbol, exc)
            continue

        summary.holdings.append(h)

    summary.total_pnl = summary.total_current - summary.total_invested
    if summary.total_invested > 0:
        summary.total_pnl_pct = (summary.total_pnl / summary.total_invested) * 100
    summary.total_day_change = sum(h.day_change_pct for h in summary.holdings) / max(len(summary.holdings), 1)

    logger.info(
        "Portfolio: ₹%.0f invested → ₹%.0f current (%.1f%%)",
        summary.total_invested,
        summary.total_current,
        summary.total_pnl_pct,
    )
    return summary


def format_portfolio_report(summary: PortfolioSummary) -> str:
    """Format portfolio as Telegram-friendly HTML."""
    if not summary.holdings:
        return "📭 No holdings in portfolio."

    lines = ["<b>💼 PORTFOLIO TRACKER</b>", ""]

    # Alerts first
    if summary.alerts:
        lines.append("<b>⚡ ALERTS:</b>")
        for alert in summary.alerts:
            lines.append(f"  {alert}")
        lines.append("")

    # Holdings table
    lines.append("<b>Holdings:</b>")
    for h in sorted(summary.holdings, key=lambda x: x.pnl_pct, reverse=True):
        pnl_emoji = "🟢" if h.pnl >= 0 else "🔴"
        sign = "+" if h.pnl >= 0 else ""
        status_tag = ""
        if h.status == "TARGET_HIT":
            status_tag = " 🎯"
        elif h.status == "STOP_HIT":
            status_tag = " 🚨"

        lines.append(
            f"{pnl_emoji} <b>{h.symbol}</b>{status_tag}"
        )
        lines.append(
            f"   Buy: ₹{h.buy_price:.0f} × {h.qty} | "
            f"CMP: ₹{h.current_price:.1f}"
        )
        lines.append(
            f"   P&L: {sign}₹{h.pnl:.0f} ({sign}{h.pnl_pct:.1f}%) | "
            f"Today: {'+' if h.day_change_pct >= 0 else ''}{h.day_change_pct:.1f}%"
        )
        if h.target > 0 or h.stop_loss > 0:
            lines.append(
                f"   SL: ₹{h.stop_loss:.0f} | "
                f"TGT: ₹{h.target:.0f}"
            )
        lines.append("")

    # Summary
    pnl_emoji = "🟢" if summary.total_pnl >= 0 else "🔴"
    sign = "+" if summary.total_pnl >= 0 else ""
    lines.append("<b>📊 SUMMARY</b>")
    lines.append(
        f"  Invested: ₹{summary.total_invested:,.0f} | "
        f"Current: ₹{summary.total_current:,.0f}"
    )
    lines.append(
        f"  {pnl_emoji} Total P&L: {sign}₹{summary.total_pnl:,.0f} "
        f"({sign}{summary.total_pnl_pct:.1f}%)"
    )
    lines.append(
        f"  Winners: {summary.winners} | Losers: {summary.losers}"
    )

    return "\n".join(lines)


# ── Quick Test ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    summary = track_portfolio()
    print(format_portfolio_report(summary))

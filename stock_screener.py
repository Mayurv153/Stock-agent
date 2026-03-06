"""
stock_screener.py — Auto-discover breakout stocks from a large universe.

Scans Nifty 50 + Next 50 + Midcaps for momentum, volume spikes,
breakouts, and oversold bounces.  Returns a curated list of the most
promising stocks for detailed AI analysis.

This is what separates retail traders from pros — finding opportunities
before the crowd.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

import config
from data_fetcher import fetch_ohlcv, fetch_fundamentals, StockFundamentals
from indicators import (
    calculate_rsi, calculate_macd, calculate_ema,
    calculate_volume_analysis, calculate_vwap, calculate_supertrend,
    _ema,
)

logger = logging.getLogger(__name__)


# ── Screening Criteria ───────────────────────────────────────


@dataclass
class ScreenResult:
    """A screened stock with its scores and signals."""

    symbol: str
    last_close: float = 0.0
    change_pct: float = 0.0            # 1-day % change
    change_5d_pct: float = 0.0         # 5-day % change
    rsi: float = 50.0
    volume_ratio: float = 1.0
    macd_crossover: str = "NEUTRAL"
    ema_crossover: str = "NEUTRAL"
    supertrend: str = "NEUTRAL"
    vwap_signal: str = "NEUTRAL"
    momentum_score: int = 0            # -100 to +100
    signals: list[str] = field(default_factory=list)  # reasons it was picked
    sector: str = ""
    market_cap: Optional[float] = None


# ── Scanning Functions ───────────────────────────────────────


def _quick_scan_stock(symbol: str) -> Optional[ScreenResult]:
    """Quickly scan a single stock for screening signals.

    Uses only 30 days of OHLCV data — no fundamentals fetch to keep it fast.
    """
    try:
        df = fetch_ohlcv(symbol, period_days=60)
        if df.empty or len(df) < 15:
            return None

        result = ScreenResult(symbol=symbol)
        result.last_close = float(df["Close"].iloc[-1])

        # 1-day change
        if len(df) >= 2:
            prev_close = float(df["Close"].iloc[-2])
            result.change_pct = ((result.last_close - prev_close) / prev_close) * 100

        # 5-day change
        if len(df) >= 6:
            close_5d = float(df["Close"].iloc[-6])
            result.change_5d_pct = ((result.last_close - close_5d) / close_5d) * 100

        # RSI
        rsi_val, rsi_signal = calculate_rsi(df)
        result.rsi = rsi_val if rsi_val is not None else 50.0

        # Volume
        _, _, vol_ratio, vol_signal = calculate_volume_analysis(df)
        result.volume_ratio = vol_ratio if vol_ratio is not None else 1.0

        # MACD
        _, _, _, macd_cross = calculate_macd(df)
        result.macd_crossover = macd_cross

        # EMA
        _, _, ema_cross = calculate_ema(df)
        result.ema_crossover = ema_cross

        # VWAP
        _, vwap_sig = calculate_vwap(df)
        result.vwap_signal = vwap_sig

        # Supertrend
        _, st_dir = calculate_supertrend(df)
        result.supertrend = st_dir

        # ── Calculate momentum score ─────────────────────────
        score = 0
        signals = []

        # Volume spike (strong signal)
        if result.volume_ratio >= 2.5:
            score += 25
            signals.append(f"🔥 Volume spike {result.volume_ratio:.1f}x")
        elif result.volume_ratio >= 1.5:
            score += 15
            signals.append(f"📈 High volume {result.volume_ratio:.1f}x")

        # RSI oversold bounce
        if result.rsi <= 30 and result.change_pct > 0:
            score += 20
            signals.append(f"💎 Oversold bounce RSI={result.rsi:.0f}")
        elif result.rsi <= 25:
            score += 15
            signals.append(f"⚡ Deeply oversold RSI={result.rsi:.0f}")
        elif result.rsi >= 70 and result.change_pct > 0:
            score += 10
            signals.append(f"🚀 Momentum RSI={result.rsi:.0f}")

        # MACD bullish crossover
        if result.macd_crossover == "BULLISH":
            score += 15
            signals.append("✅ MACD bullish crossover")

        # EMA golden cross
        if result.ema_crossover == "BULLISH":
            score += 15
            signals.append("✅ EMA 9/21 golden cross")

        # Supertrend bullish
        if result.supertrend == "BULLISH":
            score += 10
            signals.append("📊 Supertrend bullish")

        # VWAP
        if result.vwap_signal == "ABOVE_VWAP":
            score += 5
            signals.append("💰 Above VWAP")

        # Strong price momentum
        if result.change_5d_pct >= 5:
            score += 15
            signals.append(f"🔥 5-day momentum +{result.change_5d_pct:.1f}%")
        elif result.change_5d_pct >= 3:
            score += 10
            signals.append(f"📈 5-day up +{result.change_5d_pct:.1f}%")

        # Bearish signals (reduce score)
        if result.macd_crossover == "BEARISH":
            score -= 10
        if result.ema_crossover == "BEARISH":
            score -= 10
        if result.supertrend == "BEARISH":
            score -= 10
        if result.change_5d_pct <= -5:
            score -= 15

        result.momentum_score = max(-100, min(100, score))
        result.signals = signals

        return result

    except Exception as exc:
        logger.debug("Scan failed for %s: %s", symbol, exc)
        return None


# ── Main Screening Functions ─────────────────────────────────


def scan_breakout_stocks(
    universe: list[str] | None = None,
    top_n: int = 15,
) -> list[ScreenResult]:
    """Scan the full stock universe and return top breakout candidates.

    Parameters
    ----------
    universe : list[str] | None
        Stocks to scan. Defaults to ``config.FULL_UNIVERSE``.
    top_n : int
        Number of top picks to return.

    Returns
    -------
    list[ScreenResult]
        Top screened stocks sorted by momentum score (highest first).
    """
    if universe is None:
        universe = config.FULL_UNIVERSE

    logger.info("🔍 SCREENER: Scanning %d stocks for breakout signals …", len(universe))

    results: list[ScreenResult] = []
    for idx, symbol in enumerate(universe, start=1):
        if idx % 20 == 0:
            logger.info("   Scanned %d/%d stocks …", idx, len(universe))
        result = _quick_scan_stock(symbol)
        if result is not None and result.momentum_score > 0:
            results.append(result)

    # Sort by momentum score descending
    results.sort(key=lambda r: r.momentum_score, reverse=True)
    top_picks = results[:top_n]

    logger.info(
        "🔍 SCREENER: Found %d positive signals, returning top %d",
        len(results), len(top_picks),
    )
    for r in top_picks:
        logger.info(
            "   %s: score=%+d, RSI=%.0f, Vol=%.1fx | %s",
            r.symbol, r.momentum_score, r.rsi, r.volume_ratio,
            " | ".join(r.signals[:3]),
        )

    return top_picks


def scan_oversold_gems(
    universe: list[str] | None = None,
    top_n: int = 10,
) -> list[ScreenResult]:
    """Find deeply oversold stocks that may bounce — value picks.

    Looks for RSI < 30, price near support, with any early reversal sign.
    """
    if universe is None:
        universe = config.FULL_UNIVERSE

    logger.info("💎 SCREENER: Scanning for oversold gems …")

    results: list[ScreenResult] = []
    for symbol in universe:
        result = _quick_scan_stock(symbol)
        if result is not None and result.rsi <= 35:
            results.append(result)

    results.sort(key=lambda r: r.rsi)  # Most oversold first
    top_picks = results[:top_n]

    logger.info("💎 SCREENER: Found %d oversold stocks", len(top_picks))
    for r in top_picks:
        logger.info("   %s: RSI=%.1f, 5d=%+.1f%%", r.symbol, r.rsi, r.change_5d_pct)

    return top_picks


def scan_volume_spikes(
    universe: list[str] | None = None,
    threshold: float = 2.0,
    top_n: int = 10,
) -> list[ScreenResult]:
    """Find stocks with unusual volume — something is happening."""
    if universe is None:
        universe = config.FULL_UNIVERSE

    logger.info("🔥 SCREENER: Scanning for volume spikes (>%.1fx) …", threshold)

    results: list[ScreenResult] = []
    for symbol in universe:
        result = _quick_scan_stock(symbol)
        if result is not None and result.volume_ratio >= threshold:
            results.append(result)

    results.sort(key=lambda r: r.volume_ratio, reverse=True)
    top_picks = results[:top_n]

    logger.info("🔥 SCREENER: Found %d volume spikes", len(top_picks))
    for r in top_picks:
        logger.info(
            "   %s: Volume %.1fx, Change %+.1f%%",
            r.symbol, r.volume_ratio, r.change_pct,
        )

    return top_picks


def format_screen_results(results: list[ScreenResult], title: str = "Screener Results") -> str:
    """Format screening results for Telegram/text output."""
    if not results:
        return f"📊 {title}\n\nNo stocks matched the screening criteria."

    lines = [f"📊 <b>{title}</b>", ""]
    for i, r in enumerate(results, 1):
        emoji = "🟢" if r.momentum_score >= 30 else "🟡" if r.momentum_score >= 10 else "⚪"
        lines.append(
            f"{emoji} <b>{i}. {r.symbol}</b> — ₹{r.last_close:.2f} "
            f"({'+' if r.change_pct >= 0 else ''}{r.change_pct:.1f}%)"
        )
        lines.append(
            f"   Score: {r.momentum_score:+d} | RSI: {r.rsi:.0f} | Vol: {r.volume_ratio:.1f}x"
        )
        if r.signals:
            lines.append(f"   {' | '.join(r.signals[:3])}")
        lines.append("")

    return "\n".join(lines)


# ── Quick Test ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    # Quick test with a small universe
    test_universe = config.NIFTY_50[:10]
    results = scan_breakout_stocks(test_universe, top_n=5)
    print(format_screen_results(results, "Top Breakout Picks"))

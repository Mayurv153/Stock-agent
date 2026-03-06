"""
options_chain.py — NSE Options Chain Analysis (Nifty & BankNifty).

Fetches options chain data from NSE India website and calculates:
  - Put/Call Ratio (PCR)
  - Max Pain strike
  - Highest OI at strikes (support/resistance)
  - Change in OI (money flow)
  - FII/DII sentiment from OI patterns

Uses NSE's public JSON endpoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import requests
import numpy as np

import config

logger = logging.getLogger(__name__)


# NSE endpoints
NSE_BASE = "https://www.nseindia.com"
NSE_OPTIONS_URL = "https://www.nseindia.com/api/option-chain-indices"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
}


# ── Data Structures ──────────────────────────────────────────

@dataclass
class StrikeData:
    """OI data for a single strike price."""

    strike: float
    call_oi: int = 0
    call_change_oi: int = 0
    call_volume: int = 0
    call_ltp: float = 0.0
    put_oi: int = 0
    put_change_oi: int = 0
    put_volume: int = 0
    put_ltp: float = 0.0


@dataclass
class OptionsAnalysis:
    """Complete options chain analysis."""

    index: str = ""                     # NIFTY / BANKNIFTY
    spot_price: float = 0.0
    expiry: str = ""
    total_call_oi: int = 0
    total_put_oi: int = 0
    pcr: float = 0.0                    # Put/Call Ratio
    max_pain: float = 0.0
    max_call_oi_strike: float = 0.0     # Resistance
    max_put_oi_strike: float = 0.0      # Support
    max_call_oi: int = 0
    max_put_oi: int = 0
    sentiment: str = ""                 # BULLISH / BEARISH / NEUTRAL
    support_levels: list[float] = field(default_factory=list)
    resistance_levels: list[float] = field(default_factory=list)
    strikes: list[StrikeData] = field(default_factory=list)
    error: str = ""


# ── NSE Data Fetching ────────────────────────────────────────

def _get_nse_session() -> requests.Session:
    """Create a session with NSE cookies."""
    session = requests.Session()
    session.headers.update(HEADERS)
    # First visit main page to get cookies
    try:
        session.get(NSE_BASE, timeout=10)
    except Exception:
        pass
    return session


def _fetch_options_chain(index: str = "NIFTY") -> dict | None:
    """Fetch raw options chain data from NSE.

    Parameters
    ----------
    index : str
        "NIFTY" or "BANKNIFTY"

    Returns
    -------
    dict | None
        Raw JSON data or None on failure.
    """
    session = _get_nse_session()

    try:
        resp = session.get(
            NSE_OPTIONS_URL,
            params={"symbol": index},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return data
    except requests.exceptions.HTTPError as exc:
        logger.warning("NSE returned HTTP %s for %s", exc.response.status_code, index)
        return None
    except Exception as exc:
        logger.error("Failed to fetch options chain for %s: %s", index, exc)
        return None


def _parse_options_data(raw: dict, index: str) -> OptionsAnalysis:
    """Parse NSE JSON into OptionsAnalysis."""
    analysis = OptionsAnalysis(index=index)

    records = raw.get("records", {})
    data = records.get("data", [])
    expiry_dates = records.get("expiryDates", [])

    if not data:
        analysis.error = "No options data available"
        return analysis

    # Use nearest expiry
    if expiry_dates:
        analysis.expiry = expiry_dates[0]

    # Get spot price
    underlying = records.get("underlyingValue", 0)
    analysis.spot_price = float(underlying)

    # Filter for nearest expiry only
    nearest_expiry = analysis.expiry
    filtered_data = [
        d for d in data
        if d.get("expiryDate") == nearest_expiry
    ]

    if not filtered_data:
        filtered_data = data  # fallback: use all

    strikes: list[StrikeData] = []
    total_call_oi = 0
    total_put_oi = 0

    for record in filtered_data:
        strike_price = record.get("strikePrice", 0)
        sd = StrikeData(strike=float(strike_price))

        ce = record.get("CE", {})
        pe = record.get("PE", {})

        if ce:
            sd.call_oi = int(ce.get("openInterest", 0))
            sd.call_change_oi = int(ce.get("changeinOpenInterest", 0))
            sd.call_volume = int(ce.get("totalTradedVolume", 0))
            sd.call_ltp = float(ce.get("lastPrice", 0))
            total_call_oi += sd.call_oi

        if pe:
            sd.put_oi = int(pe.get("openInterest", 0))
            sd.put_change_oi = int(pe.get("changeinOpenInterest", 0))
            sd.put_volume = int(pe.get("totalTradedVolume", 0))
            sd.put_ltp = float(pe.get("lastPrice", 0))
            total_put_oi += sd.put_oi

        strikes.append(sd)

    analysis.strikes = strikes
    analysis.total_call_oi = total_call_oi
    analysis.total_put_oi = total_put_oi

    # PCR
    if total_call_oi > 0:
        analysis.pcr = total_put_oi / total_call_oi
    else:
        analysis.pcr = 0.0

    # Max OI strikes (resistance/support)
    if strikes:
        max_call = max(strikes, key=lambda s: s.call_oi)
        max_put = max(strikes, key=lambda s: s.put_oi)
        analysis.max_call_oi_strike = max_call.strike
        analysis.max_call_oi = max_call.call_oi
        analysis.max_put_oi_strike = max_put.strike
        analysis.max_put_oi = max_put.put_oi

        # Top 3 support/resistance levels
        sorted_by_put_oi = sorted(strikes, key=lambda s: s.put_oi, reverse=True)
        sorted_by_call_oi = sorted(strikes, key=lambda s: s.call_oi, reverse=True)
        analysis.support_levels = [s.strike for s in sorted_by_put_oi[:3]]
        analysis.resistance_levels = [s.strike for s in sorted_by_call_oi[:3]]

    # Max Pain calculation
    analysis.max_pain = _calculate_max_pain(strikes)

    # Sentiment from PCR
    if analysis.pcr >= 1.2:
        analysis.sentiment = "BULLISH"     # High put writing = bullish
    elif analysis.pcr <= 0.7:
        analysis.sentiment = "BEARISH"     # Low PCR = bearish
    elif analysis.pcr >= 1.0:
        analysis.sentiment = "MILDLY BULLISH"
    elif analysis.pcr >= 0.8:
        analysis.sentiment = "NEUTRAL"
    else:
        analysis.sentiment = "MILDLY BEARISH"

    return analysis


def _calculate_max_pain(strikes: list[StrikeData]) -> float:
    """Calculate max pain — strike where total options loss is minimum.

    At each strike, calculate total intrinsic value writers would pay,
    and find the strike with minimum total pain.
    """
    if not strikes:
        return 0.0

    strike_prices = [s.strike for s in strikes]
    min_pain = float("inf")
    max_pain_strike = 0.0

    for test_strike in strike_prices:
        total_pain = 0.0
        for s in strikes:
            # Call pain: call buyers lose if strike > test_strike
            if s.strike < test_strike:
                call_itm = (test_strike - s.strike) * s.call_oi
                total_pain += call_itm

            # Put pain: put buyers lose if strike < test_strike
            if s.strike > test_strike:
                put_itm = (s.strike - test_strike) * s.put_oi
                total_pain += put_itm

        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = test_strike

    return max_pain_strike


# ── Public API ───────────────────────────────────────────────

def analyze_options(index: str = "NIFTY") -> OptionsAnalysis:
    """Fetch and analyze options chain for an index.

    Parameters
    ----------
    index : str
        "NIFTY" or "BANKNIFTY"

    Returns
    -------
    OptionsAnalysis
    """
    logger.info("Fetching options chain for %s …", index)

    raw = _fetch_options_chain(index)
    if raw is None:
        analysis = OptionsAnalysis(index=index)
        analysis.error = f"Failed to fetch data from NSE for {index}"
        return analysis

    analysis = _parse_options_data(raw, index)
    logger.info(
        "%s Options — Spot: %.0f, PCR: %.2f, MaxPain: %.0f, Sentiment: %s",
        index, analysis.spot_price, analysis.pcr, analysis.max_pain, analysis.sentiment
    )
    return analysis


def format_options_report(analyses: list[OptionsAnalysis]) -> str:
    """Format options analysis as Telegram-friendly HTML."""
    lines = ["<b>📊 OPTIONS CHAIN ANALYSIS</b>", ""]

    for a in analyses:
        if a.error:
            lines.append(f"⚠️ <b>{a.index}:</b> {a.error}")
            lines.append("")
            continue

        # Sentiment emoji
        if "BULLISH" in a.sentiment:
            emoji = "🟢"
        elif "BEARISH" in a.sentiment:
            emoji = "🔴"
        else:
            emoji = "⚪"

        lines.append(f"<b>{'━' * 25}</b>")
        lines.append(f"{emoji} <b>{a.index}</b> — Expiry: {a.expiry}")
        lines.append(f"<b>Spot:</b> {a.spot_price:,.0f}")
        lines.append("")

        # Key metrics
        lines.append(f"<b>PCR:</b> {a.pcr:.2f} → {a.sentiment}")
        lines.append(f"<b>Max Pain:</b> {a.max_pain:,.0f}")
        lines.append("")

        # OI data
        lines.append(
            f"<b>Call OI:</b> {a.total_call_oi:,} | "
            f"<b>Put OI:</b> {a.total_put_oi:,}"
        )
        lines.append(
            f"<b>Max Call OI:</b> {a.max_call_oi_strike:,.0f} "
            f"({a.max_call_oi:,}) ← Resistance"
        )
        lines.append(
            f"<b>Max Put OI:</b> {a.max_put_oi_strike:,.0f} "
            f"({a.max_put_oi:,}) ← Support"
        )

        # Support/Resistance
        if a.support_levels:
            lines.append(
                f"\n<b>Support:</b> "
                + " | ".join(f"{s:,.0f}" for s in a.support_levels)
            )
        if a.resistance_levels:
            lines.append(
                f"<b>Resistance:</b> "
                + " | ".join(f"{r:,.0f}" for r in a.resistance_levels)
            )

        # Interpretation
        lines.append("")
        if a.pcr >= 1.2:
            lines.append("📝 High PCR = Heavy put writing = Writers confident market won't fall")
        elif a.pcr <= 0.7:
            lines.append("📝 Low PCR = Heavy call writing = Bears dominating")
        else:
            lines.append("📝 PCR in neutral zone = Market consolidating")

        if a.max_pain != 0 and a.spot_price != 0:
            diff = ((a.max_pain - a.spot_price) / a.spot_price) * 100
            if abs(diff) > 0.5:
                direction = "UP" if diff > 0 else "DOWN"
                lines.append(
                    f"📝 Max Pain {direction} bias: {diff:+.1f}% "
                    f"(Spot → {a.max_pain:,.0f})"
                )

        lines.append("")

    return "\n".join(lines)


# ── Quick Test ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    nifty = analyze_options("NIFTY")
    banknifty = analyze_options("BANKNIFTY")
    print(format_options_report([nifty, banknifty]))

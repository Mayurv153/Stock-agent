"""
risk_manager.py — Position sizing, portfolio risk, and correlation analysis.

THIS IS WHAT SEPARATES TOP 1% TRADERS FROM THE REST.

Most retail traders fail because they:
  1. Risk too much per trade (>5% of capital)
  2. Don't calculate position size based on stop-loss
  3. Overload one sector (e.g. 5 IT stocks at once)
  4. Never use trailing stop-losses

This module solves all of the above.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import config
from analyzer import StockRecommendation

logger = logging.getLogger(__name__)


# ── Data Models ──────────────────────────────────────────────


@dataclass
class PositionSize:
    """Calculated position sizing for a trade."""

    symbol: str
    capital_available: float
    entry_price: float
    stop_loss: float
    target_price: float

    # Calculated fields
    risk_per_share: float = 0.0          # entry - stop_loss
    reward_per_share: float = 0.0        # target - entry
    risk_reward_ratio: float = 0.0       # reward / risk
    max_risk_amount: float = 0.0         # capital × max_risk_pct
    shares_to_buy: int = 0               # max_risk_amount / risk_per_share
    position_value: float = 0.0          # shares × entry_price
    position_pct: float = 0.0            # position_value / capital
    potential_profit: float = 0.0        # shares × reward_per_share
    potential_loss: float = 0.0          # shares × risk_per_share
    verdict: str = ""                    # GOOD_TRADE / RISKY / SKIP


@dataclass
class PortfolioRisk:
    """Overall portfolio risk assessment."""

    total_capital: float
    positions: list[PositionSize] = field(default_factory=list)
    total_invested: float = 0.0
    total_risk: float = 0.0
    cash_remaining: float = 0.0
    portfolio_risk_pct: float = 0.0
    sector_exposure: dict = field(default_factory=dict)
    diversification_score: int = 0       # 0-100
    warnings: list[str] = field(default_factory=list)


@dataclass
class TrailingStop:
    """Trailing stop-loss recommendation."""

    symbol: str
    current_price: float
    entry_price: float
    initial_stop: float
    trailing_stop: float
    trailing_pct: float
    profit_locked: float                 # guaranteed profit if stop hits
    action: str                          # HOLD / TIGHTEN / EXIT


# ── Position Sizing Calculator ───────────────────────────────


def calculate_position_size(
    rec: StockRecommendation,
    capital: float | None = None,
    max_risk_pct: float | None = None,
) -> PositionSize:
    """Calculate optimal position size based on risk management rules.

    The Kelly Criterion / Fixed Risk approach:
    - Never risk more than 2% of capital on a single trade
    - Position size = (Capital × Risk%) / (Entry - StopLoss)
    - Ensures one bad trade doesn't wipe you out

    Parameters
    ----------
    rec : StockRecommendation
        The AI recommendation with entry, target, stop-loss.
    capital : float | None
        Available capital. Defaults to ``config.PORTFOLIO_CAPITAL``.
    max_risk_pct : float | None
        Max risk per trade. Defaults to ``config.MAX_RISK_PER_TRADE``.

    Returns
    -------
    PositionSize
        Calculated position sizing with verdict.
    """
    cap = capital or config.PORTFOLIO_CAPITAL
    risk_pct = max_risk_pct or config.MAX_RISK_PER_TRADE

    pos = PositionSize(
        symbol=rec.symbol,
        capital_available=cap,
        entry_price=rec.entry_price,
        stop_loss=rec.stop_loss,
        target_price=rec.target_price,
    )

    # Guard against bad data
    if rec.entry_price <= 0 or rec.stop_loss <= 0 or rec.target_price <= 0:
        pos.verdict = "SKIP"
        return pos

    if rec.stop_loss >= rec.entry_price:
        pos.verdict = "SKIP"
        return pos

    # Core calculations
    pos.risk_per_share = rec.entry_price - rec.stop_loss
    pos.reward_per_share = rec.target_price - rec.entry_price
    pos.risk_reward_ratio = pos.reward_per_share / pos.risk_per_share if pos.risk_per_share > 0 else 0

    # Maximum risk in rupees
    pos.max_risk_amount = cap * risk_pct

    # Number of shares we can buy
    pos.shares_to_buy = int(pos.max_risk_amount / pos.risk_per_share)

    # Cap position to MAX_POSITION_SIZE of capital
    max_position = cap * config.MAX_POSITION_SIZE
    max_shares_by_position = int(max_position / rec.entry_price) if rec.entry_price > 0 else 0
    pos.shares_to_buy = min(pos.shares_to_buy, max_shares_by_position)

    # At least 1 share
    pos.shares_to_buy = max(pos.shares_to_buy, 0)

    pos.position_value = pos.shares_to_buy * rec.entry_price
    pos.position_pct = (pos.position_value / cap) * 100 if cap > 0 else 0
    pos.potential_profit = pos.shares_to_buy * pos.reward_per_share
    pos.potential_loss = pos.shares_to_buy * pos.risk_per_share

    # Verdict
    if pos.risk_reward_ratio >= 2.0 and rec.confidence_score >= 70:
        pos.verdict = "STRONG_TRADE"
    elif pos.risk_reward_ratio >= 1.5 and rec.confidence_score >= 60:
        pos.verdict = "GOOD_TRADE"
    elif pos.risk_reward_ratio >= 1.0:
        pos.verdict = "MODERATE"
    else:
        pos.verdict = "RISKY"

    return pos


def calculate_all_positions(
    recommendations: list[StockRecommendation],
    capital: float | None = None,
) -> list[PositionSize]:
    """Calculate position sizes for all BUY recommendations.

    Only processes BUY recommendations with valid prices.
    """
    cap = capital or config.PORTFOLIO_CAPITAL
    positions = []

    buy_recs = [r for r in recommendations
                if r.analysis_success and r.recommendation == "BUY"]

    remaining_capital = cap
    for rec in buy_recs:
        pos = calculate_position_size(rec, capital=remaining_capital)
        if pos.shares_to_buy > 0:
            positions.append(pos)
            remaining_capital -= pos.position_value

    return positions


# ── Portfolio Risk Assessment ────────────────────────────────


def assess_portfolio_risk(
    positions: list[PositionSize],
    capital: float | None = None,
    sector_map: dict[str, str] | None = None,
) -> PortfolioRisk:
    """Assess overall portfolio risk across all positions.

    Parameters
    ----------
    positions : list[PositionSize]
        All calculated positions.
    capital : float | None
        Total capital.
    sector_map : dict[str, str] | None
        Mapping of symbol → sector (for diversification check).

    Returns
    -------
    PortfolioRisk
        Complete portfolio risk analysis with warnings.
    """
    cap = capital or config.PORTFOLIO_CAPITAL

    portfolio = PortfolioRisk(total_capital=cap)
    portfolio.positions = positions

    total_invested = sum(p.position_value for p in positions)
    total_risk = sum(p.potential_loss for p in positions)
    portfolio.total_invested = total_invested
    portfolio.total_risk = total_risk
    portfolio.cash_remaining = cap - total_invested
    portfolio.portfolio_risk_pct = (total_risk / cap) * 100 if cap > 0 else 0

    # Sector exposure
    if sector_map:
        for p in positions:
            sector = sector_map.get(p.symbol, "Unknown")
            portfolio.sector_exposure[sector] = (
                portfolio.sector_exposure.get(sector, 0) + p.position_value
            )

    # Warnings
    warnings = []

    if portfolio.portfolio_risk_pct > config.MAX_PORTFOLIO_RISK * 100:
        warnings.append(
            f"⚠️ Total portfolio risk ({portfolio.portfolio_risk_pct:.1f}%) "
            f"exceeds {config.MAX_PORTFOLIO_RISK * 100:.0f}% limit!"
        )

    if total_invested > cap * 0.8:
        warnings.append(
            f"⚠️ Over-invested! {total_invested/cap*100:.0f}% of capital deployed. "
            f"Keep 20% cash reserve."
        )

    for sector, exposure in portfolio.sector_exposure.items():
        sector_pct = exposure / cap * 100
        if sector_pct > config.MAX_SECTOR_EXPOSURE * 100:
            warnings.append(
                f"⚠️ {sector} sector exposure ({sector_pct:.0f}%) "
                f"exceeds {config.MAX_SECTOR_EXPOSURE * 100:.0f}% limit!"
            )

    if len(positions) > 10:
        warnings.append(
            "⚠️ Too many positions! Hard to track >10 stocks. Focus on best picks."
        )

    portfolio.warnings = warnings

    # Diversification score
    n_positions = len(positions)
    n_sectors = len(portfolio.sector_exposure)
    if n_positions == 0:
        portfolio.diversification_score = 0
    else:
        # More sectors + reasonable number of positions = better
        sector_score = min(n_sectors / 5 * 50, 50)  # Max 50 for 5+ sectors
        position_score = min(n_positions / 5 * 30, 30) if n_positions <= 10 else max(30 - (n_positions - 10) * 5, 0)
        cash_score = 20 if portfolio.cash_remaining / cap >= 0.2 else (portfolio.cash_remaining / cap / 0.2) * 20
        portfolio.diversification_score = int(sector_score + position_score + cash_score)

    return portfolio


# ── Trailing Stop-Loss Calculator ────────────────────────────


def calculate_trailing_stop(
    symbol: str,
    current_price: float,
    entry_price: float,
    initial_stop: float,
    highest_since_entry: float | None = None,
) -> TrailingStop:
    """Calculate trailing stop-loss based on price movement since entry.

    Rules:
    - If in profit, trail the stop at 2-3% below the highest price
    - If near breakeven, tighten stop to entry price (risk-free trade)
    - If losing, keep original stop-loss

    Parameters
    ----------
    symbol : str
        Stock symbol.
    current_price : float
        Current market price.
    entry_price : float
        Original entry price.
    initial_stop : float
        Original stop-loss price.
    highest_since_entry : float | None
        Highest price since entry (for trailing). Defaults to current_price.

    Returns
    -------
    TrailingStop
        Updated stop-loss recommendation.
    """
    highest = max(highest_since_entry or current_price, current_price)
    profit_pct = ((current_price - entry_price) / entry_price) * 100

    # Determine trailing percentage based on profit
    if profit_pct >= 10:
        trail_pct = 3.0   # Lock in most profits
    elif profit_pct >= 5:
        trail_pct = 4.0   # Give some room
    elif profit_pct >= 2:
        trail_pct = 5.0   # Wider trail
    else:
        trail_pct = 0.0   # Use initial stop

    if trail_pct > 0:
        trailing_stop = highest * (1 - trail_pct / 100)
        trailing_stop = max(trailing_stop, initial_stop)  # Never lower than initial
    else:
        trailing_stop = initial_stop

    # If in good profit, never let it go below entry (risk-free trade)
    if profit_pct >= 5:
        trailing_stop = max(trailing_stop, entry_price * 1.01)  # +1% above entry

    profit_locked = max(0, (trailing_stop - entry_price) * 1)  # per share

    # Action
    if current_price <= trailing_stop:
        action = "EXIT"
    elif profit_pct >= 8:
        action = "TIGHTEN"
    else:
        action = "HOLD"

    return TrailingStop(
        symbol=symbol,
        current_price=current_price,
        entry_price=entry_price,
        initial_stop=initial_stop,
        trailing_stop=round(trailing_stop, 2),
        trailing_pct=trail_pct,
        profit_locked=round(profit_locked, 2),
        action=action,
    )


# ── Formatting Functions ─────────────────────────────────────


def format_position_sizing(positions: list[PositionSize], capital: float | None = None) -> str:
    """Format position sizing results for Telegram/text output."""
    cap = capital or config.PORTFOLIO_CAPITAL

    lines = [
        f"<b>💰 POSITION SIZING (Capital: ₹{cap:,.0f})</b>",
        f"<i>Max risk per trade: {config.MAX_RISK_PER_TRADE*100:.0f}%</i>",
        "",
    ]

    if not positions:
        lines.append("No BUY recommendations to size.")
        return "\n".join(lines)

    total_invested = 0
    total_risk = 0

    for p in positions:
        emoji = "🟢" if p.verdict in ("STRONG_TRADE", "GOOD_TRADE") else "🟡" if p.verdict == "MODERATE" else "🔴"
        lines.append(f"{emoji} <b>{p.symbol}</b> — {p.verdict}")
        lines.append(
            f"   Buy {p.shares_to_buy} shares @ ₹{p.entry_price:.2f}"
        )
        lines.append(
            f"   Investment: ₹{p.position_value:,.0f} ({p.position_pct:.1f}% of capital)"
        )
        lines.append(
            f"   Risk:Reward = 1:{p.risk_reward_ratio:.1f} | "
            f"Profit: ₹{p.potential_profit:,.0f} | Loss: ₹{p.potential_loss:,.0f}"
        )
        lines.append(
            f"   Entry: ₹{p.entry_price:.2f} → Target: ₹{p.target_price:.2f} "
            f"| SL: ₹{p.stop_loss:.2f}"
        )
        lines.append("")

        total_invested += p.position_value
        total_risk += p.potential_loss

    lines.append("<b>📊 Summary</b>")
    lines.append(f"   Total invested: ₹{total_invested:,.0f} ({total_invested/cap*100:.0f}%)")
    lines.append(f"   Total at risk : ₹{total_risk:,.0f} ({total_risk/cap*100:.1f}%)")
    lines.append(f"   Cash remaining: ₹{cap - total_invested:,.0f}")

    return "\n".join(lines)


def format_portfolio_risk(portfolio: PortfolioRisk) -> str:
    """Format portfolio risk assessment for display."""
    lines = [
        "<b>🛡️ PORTFOLIO RISK ASSESSMENT</b>",
        f"Capital: ₹{portfolio.total_capital:,.0f}",
        "",
        f"Invested: ₹{portfolio.total_invested:,.0f} ({portfolio.total_invested/portfolio.total_capital*100:.0f}%)" if portfolio.total_capital > 0 else "",
        f"At Risk : ₹{portfolio.total_risk:,.0f} ({portfolio.portfolio_risk_pct:.1f}%)",
        f"Cash    : ₹{portfolio.cash_remaining:,.0f}",
        f"Diversification Score: {portfolio.diversification_score}/100",
        "",
    ]

    if portfolio.warnings:
        lines.append("<b>⚠️ Warnings:</b>")
        for w in portfolio.warnings:
            lines.append(f"  {w}")
        lines.append("")

    if portfolio.sector_exposure:
        lines.append("<b>Sector Exposure:</b>")
        for sector, amount in sorted(portfolio.sector_exposure.items(), key=lambda x: x[1], reverse=True):
            pct = amount / portfolio.total_capital * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            lines.append(f"  {sector}: {bar} {pct:.0f}%")

    return "\n".join(lines)


# ── Quick Test ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    # Test with dummy recommendation
    dummy = StockRecommendation(
        symbol="RELIANCE", recommendation="BUY", trade_type="SHORT_TERM",
        entry_price=1420.0, target_price=1530.0, stop_loss=1380.0,
        risk_level="MEDIUM", confidence_score=70,
    )
    pos = calculate_position_size(dummy, capital=100000)
    print(f"\n{pos.symbol}: Buy {pos.shares_to_buy} shares")
    print(f"Investment: ₹{pos.position_value:,.0f} ({pos.position_pct:.1f}%)")
    print(f"Risk:Reward = 1:{pos.risk_reward_ratio:.1f}")
    print(f"Verdict: {pos.verdict}")

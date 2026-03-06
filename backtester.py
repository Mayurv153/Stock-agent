"""
backtester.py — Backtest AI recommendations against historical data.

Answers the critical question: "If I followed the AI signals for the
past year, would I be profitable?"

Tracks:
- Win rate (% of profitable trades)
- Average return per trade
- Maximum drawdown
- Sharpe ratio estimate
- Best and worst trades
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

import config
from data_fetcher import fetch_ohlcv

logger = logging.getLogger(__name__)


# ── Data Models ──────────────────────────────────────────────


@dataclass
class BacktestTrade:
    """A single simulated trade in the backtest."""

    symbol: str
    entry_date: str
    entry_price: float
    exit_date: str = ""
    exit_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    shares: int = 0
    pnl: float = 0.0                    # profit/loss in rupees
    return_pct: float = 0.0             # % return
    outcome: str = ""                    # WIN / LOSS / OPEN
    exit_reason: str = ""               # TARGET_HIT / STOP_HIT / TIME_EXIT
    holding_days: int = 0


@dataclass
class BacktestResult:
    """Complete backtest results summary."""

    # Settings
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 0.0
    final_capital: float = 0.0

    # Performance
    total_return_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0               # 0-100%
    avg_return_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0          # gross profit / gross loss
    sharpe_ratio: float = 0.0

    # Trade details
    trades: list[BacktestTrade] = field(default_factory=list)
    best_trade: Optional[BacktestTrade] = None
    worst_trade: Optional[BacktestTrade] = None
    equity_curve: list[float] = field(default_factory=list)

    # Verdict
    verdict: str = ""                    # PROFITABLE / BREAKEVEN / UNPROFITABLE
    grade: str = ""                      # A+ / A / B / C / D / F


# ── Technical Signal Generator (for historical simulation) ───


def _generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate BUY/SELL signals based on technical indicators.

    Uses a simple but effective multi-indicator strategy:
    - BUY when: RSI < 35 AND MACD bullish crossover
    - SELL (target): +5% from entry
    - SELL (stop): -3% from entry
    - SELL (time): 10 trading days max hold

    This simulates what the AI would recommend.
    """
    close = df["Close"].copy()

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    # EMA crossover
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()

    df = df.copy()
    df["rsi"] = rsi
    df["macd"] = macd
    df["macd_signal"] = signal
    df["ema9"] = ema9
    df["ema21"] = ema21

    # Generate buy signals
    df["buy_signal"] = False
    for i in range(1, len(df)):
        rsi_ok = df["rsi"].iloc[i] < 35
        macd_cross = (df["macd"].iloc[i] > df["macd_signal"].iloc[i] and
                      df["macd"].iloc[i - 1] <= df["macd_signal"].iloc[i - 1])
        ema_near = abs(df["ema9"].iloc[i] - df["ema21"].iloc[i]) / df["ema21"].iloc[i] < 0.02

        # Buy on RSI oversold + MACD bullish crossover
        if rsi_ok and macd_cross:
            df.iloc[i, df.columns.get_loc("buy_signal")] = True
        # Also buy on EMA golden cross with low RSI
        elif df["rsi"].iloc[i] < 40 and df["ema9"].iloc[i] > df["ema21"].iloc[i] and df["ema9"].iloc[i - 1] <= df["ema21"].iloc[i - 1]:
            df.iloc[i, df.columns.get_loc("buy_signal")] = True

    return df


# ── Core Backtesting Engine ──────────────────────────────────


def backtest_stock(
    symbol: str,
    lookback_days: int = config.BACKTEST_DAYS,
    target_pct: float = 5.0,
    stop_pct: float = 3.0,
    max_hold_days: int = 10,
    capital_per_trade: float = 20000.0,
) -> list[BacktestTrade]:
    """Backtest a single stock using technical signals.

    Parameters
    ----------
    symbol : str
        NSE stock symbol.
    lookback_days : int
        Days of historical data to backtest.
    target_pct : float
        Target profit percentage.
    stop_pct : float
        Stop-loss percentage.
    max_hold_days : int
        Maximum holding period in trading days.
    capital_per_trade : float
        Capital allocated per trade.

    Returns
    -------
    list[BacktestTrade]
        List of simulated trades.
    """
    try:
        df = fetch_ohlcv(symbol, period_days=lookback_days)
        if df.empty or len(df) < 30:
            return []

        df = _generate_signals(df)
        trades: list[BacktestTrade] = []
        in_trade = False
        trade: Optional[BacktestTrade] = None

        for i in range(len(df)):
            date_str = df.index[i].strftime("%Y-%m-%d")
            close = float(df["Close"].iloc[i])
            high = float(df["High"].iloc[i])
            low = float(df["Low"].iloc[i])

            if in_trade and trade is not None:
                trade.holding_days += 1
                target = trade.entry_price * (1 + target_pct / 100)
                stop = trade.entry_price * (1 - stop_pct / 100)

                # Check stop-loss first (conservative)
                if low <= stop:
                    trade.exit_price = stop
                    trade.exit_date = date_str
                    trade.exit_reason = "STOP_HIT"
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.shares
                    trade.return_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
                    trade.outcome = "LOSS"
                    trades.append(trade)
                    in_trade = False
                    trade = None

                # Check target
                elif high >= target:
                    trade.exit_price = target
                    trade.exit_date = date_str
                    trade.exit_reason = "TARGET_HIT"
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.shares
                    trade.return_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
                    trade.outcome = "WIN"
                    trades.append(trade)
                    in_trade = False
                    trade = None

                # Check max hold
                elif trade.holding_days >= max_hold_days:
                    trade.exit_price = close
                    trade.exit_date = date_str
                    trade.exit_reason = "TIME_EXIT"
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.shares
                    trade.return_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
                    trade.outcome = "WIN" if trade.pnl > 0 else "LOSS"
                    trades.append(trade)
                    in_trade = False
                    trade = None

            elif not in_trade and df["buy_signal"].iloc[i]:
                shares = max(1, int(capital_per_trade / close))
                trade = BacktestTrade(
                    symbol=symbol,
                    entry_date=date_str,
                    entry_price=close,
                    target_price=close * (1 + target_pct / 100),
                    stop_loss=close * (1 - stop_pct / 100),
                    shares=shares,
                )
                in_trade = True

        return trades

    except Exception as exc:
        logger.warning("Backtest failed for %s: %s", symbol, exc)
        return []


def run_full_backtest(
    symbols: list[str] | None = None,
    lookback_days: int = config.BACKTEST_DAYS,
    initial_capital: float = config.BACKTEST_INITIAL_CAPITAL,
) -> BacktestResult:
    """Run a backtest across multiple stocks.

    Parameters
    ----------
    symbols : list[str] | None
        Stocks to backtest. Defaults to ``config.WATCHLIST``.
    lookback_days : int
        Historical period.
    initial_capital : float
        Starting capital.

    Returns
    -------
    BacktestResult
        Complete backtest results with metrics and trade history.
    """
    if symbols is None:
        symbols = config.WATCHLIST

    logger.info("📊 BACKTEST: Running %d-day backtest on %d stocks …", lookback_days, len(symbols))

    all_trades: list[BacktestTrade] = []
    capital_per_trade = initial_capital / min(len(symbols), 5)  # Spread across positions

    for idx, symbol in enumerate(symbols, 1):
        if idx % 10 == 0:
            logger.info("   Backtesting %d/%d …", idx, len(symbols))
        trades = backtest_stock(symbol, lookback_days, capital_per_trade=capital_per_trade)
        all_trades.extend(trades)

    if not all_trades:
        logger.warning("No trades generated during backtest period.")
        return BacktestResult(
            initial_capital=initial_capital,
            final_capital=initial_capital,
            verdict="NO_DATA",
            grade="N/A",
        )

    # Sort trades by entry date
    all_trades.sort(key=lambda t: t.entry_date)

    # Calculate metrics
    result = BacktestResult()
    result.initial_capital = initial_capital
    result.trades = all_trades
    result.start_date = all_trades[0].entry_date
    result.end_date = all_trades[-1].exit_date or all_trades[-1].entry_date
    result.total_trades = len(all_trades)

    wins = [t for t in all_trades if t.outcome == "WIN"]
    losses = [t for t in all_trades if t.outcome == "LOSS"]
    result.winning_trades = len(wins)
    result.losing_trades = len(losses)
    result.win_rate = (len(wins) / len(all_trades) * 100) if all_trades else 0

    returns = [t.return_pct for t in all_trades]
    result.avg_return_pct = float(np.mean(returns)) if returns else 0
    result.avg_win_pct = float(np.mean([t.return_pct for t in wins])) if wins else 0
    result.avg_loss_pct = float(np.mean([t.return_pct for t in losses])) if losses else 0

    result.best_trade_pct = max(returns) if returns else 0
    result.worst_trade_pct = min(returns) if returns else 0

    # Find best and worst trades
    if returns:
        best_idx = returns.index(max(returns))
        worst_idx = returns.index(min(returns))
        result.best_trade = all_trades[best_idx]
        result.worst_trade = all_trades[worst_idx]

    # Profit factor
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Equity curve & drawdown
    equity = initial_capital
    peak_equity = initial_capital
    max_dd = 0
    equity_curve = [initial_capital]

    for trade in all_trades:
        equity += trade.pnl
        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity * 100
        max_dd = max(max_dd, dd)

    result.final_capital = equity
    result.total_return_pct = ((equity - initial_capital) / initial_capital) * 100
    result.max_drawdown_pct = max_dd
    result.equity_curve = equity_curve

    # Sharpe ratio (simplified — daily returns annualised)
    if len(returns) > 1:
        avg_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret > 0:
            # Annualise: ~250 trading days, avg hold ~5 days → ~50 trades/year
            result.sharpe_ratio = (avg_ret / std_ret) * np.sqrt(50)
        else:
            result.sharpe_ratio = 0
    else:
        result.sharpe_ratio = 0

    # Verdict & Grade
    if result.total_return_pct >= 20 and result.win_rate >= 60:
        result.verdict = "PROFITABLE"
        result.grade = "A+"
    elif result.total_return_pct >= 10 and result.win_rate >= 55:
        result.verdict = "PROFITABLE"
        result.grade = "A"
    elif result.total_return_pct >= 5:
        result.verdict = "PROFITABLE"
        result.grade = "B"
    elif result.total_return_pct >= 0:
        result.verdict = "BREAKEVEN"
        result.grade = "C"
    elif result.total_return_pct >= -5:
        result.verdict = "SLIGHTLY_UNPROFITABLE"
        result.grade = "D"
    else:
        result.verdict = "UNPROFITABLE"
        result.grade = "F"

    logger.info(
        "📊 BACKTEST COMPLETE: %d trades, %.1f%% win rate, %.1f%% total return → Grade %s",
        result.total_trades, result.win_rate, result.total_return_pct, result.grade,
    )

    return result


# ── Formatting ───────────────────────────────────────────────


def format_backtest_results(result: BacktestResult) -> str:
    """Format backtest results for Telegram/text output."""
    lines = [
        "<b>📊 BACKTEST RESULTS</b>",
        f"<i>{result.start_date} → {result.end_date}</i>",
        "",
        f"<b>Grade: {result.grade}</b> ({result.verdict})",
        "",
        f"Starting Capital : ₹{result.initial_capital:,.0f}",
        f"Final Capital    : ₹{result.final_capital:,.0f}",
        f"<b>Total Return     : {'+' if result.total_return_pct >= 0 else ''}{result.total_return_pct:.1f}%</b>",
        "",
        f"Total Trades     : {result.total_trades}",
        f"Win Rate         : {result.win_rate:.1f}%",
        f"Avg Win          : +{result.avg_win_pct:.1f}%",
        f"Avg Loss         : {result.avg_loss_pct:.1f}%",
        f"Profit Factor    : {result.profit_factor:.2f}",
        f"Max Drawdown     : {result.max_drawdown_pct:.1f}%",
        f"Sharpe Ratio     : {result.sharpe_ratio:.2f}",
        "",
    ]

    if result.best_trade:
        lines.append(
            f"🏆 Best Trade: {result.best_trade.symbol} "
            f"+{result.best_trade.return_pct:.1f}% "
            f"({result.best_trade.entry_date})"
        )
    if result.worst_trade:
        lines.append(
            f"💀 Worst Trade: {result.worst_trade.symbol} "
            f"{result.worst_trade.return_pct:.1f}% "
            f"({result.worst_trade.entry_date})"
        )

    lines.append("")
    lines.append("<i>⚠️ Past performance ≠ future results. Backtest uses simplified signals.</i>")

    return "\n".join(lines)


# ── Quick Test ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    # Quick test with a few stocks
    result = run_full_backtest(
        symbols=["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN"],
        lookback_days=180,
    )
    print(format_backtest_results(result))

#!/usr/bin/env python3
"""
main.py — Entry point for the AI Stock Market Analysis Agent.

Usage
-----
    python main.py                          # Full Nifty 50 analysis
    python main.py --stocks RELIANCE TCS    # Analyse specific stocks
    python main.py --scan                   # Run momentum screener on full universe
    python main.py --backtest               # Backtest strategy on watchlist
    python main.py --alerts                 # Start real-time alert monitoring
    python main.py --alerts --once          # Single alert scan (no loop)
    python main.py --heatmap                # Sector heatmap
    python main.py --portfolio              # Portfolio P&L tracker
    python main.py --news                   # News sentiment analysis
    python main.py --options                # Nifty/BankNifty options chain
    python main.py --email                  # Send report via email  
    python main.py --bot                    # Start interactive Telegram bot
    python main.py --schedule               # Start daily scheduler (08:30 IST)
    python main.py --no-telegram            # Skip Telegram notification
    python main.py --capital 500000         # Set portfolio capital (₹)
    python main.py --universe all           # Nifty50 + Next50 + Midcaps

The pipeline:
    1. Fetch OHLCV + fundamentals from Yahoo Finance (NSE)
    2. Compute technical indicators (RSI, MACD, Bollinger, EMA, VWAP, Supertrend, Fibonacci)
    3. Send data to Groq AI for structured BUY/SELL/HOLD recommendation
    4. Calculate position sizing & risk management
    5. Generate beautiful TXT + HTML reports
    6. (Optional) Push report to Telegram / Email
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import config
from data_fetcher import fetch_all_stocks, StockData
from indicators import compute_all_indicators, TechnicalSummary
from analyzer import analyse_all, StockRecommendation
from report_generator import save_reports
from notifier import send_telegram_report

logger = logging.getLogger(__name__)


# ── Logging Setup ────────────────────────────────────────────


def setup_logging(verbose: bool = False) -> None:
    """Configure logging to console + rotating log file."""
    import io
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers on re-init
    if root.handlers:
        return

    fmt = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)

    # Console — force UTF-8 on Windows to avoid cp1252 emoji crashes
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    console = logging.StreamHandler(utf8_stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File
    config.LOGS_DIR.mkdir(exist_ok=True)
    fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ── Universe Selection ───────────────────────────────────────


def _resolve_watchlist(universe: str | None, stocks: list[str] | None) -> list[str]:
    """Resolve which stocks to analyse based on CLI arguments."""
    if stocks:
        return stocks

    if universe:
        u = universe.lower()
        if u == "nifty50":
            return config.NIFTY_50.copy()
        elif u == "next50":
            return config.NIFTY_NEXT50.copy()
        elif u == "midcap":
            return config.MIDCAP_GEMS.copy()
        elif u == "all":
            return config.FULL_UNIVERSE.copy()
        else:
            logger.warning("Unknown universe '%s', using default watchlist.", universe)

    return config.WATCHLIST.copy()


# ── Core Pipeline ────────────────────────────────────────────


def run_pipeline(
    stocks: list[str] | None = None,
    send_telegram: bool = True,
    verbose: bool = False,
    capital: float | None = None,
    universe: str | None = None,
) -> list[StockRecommendation]:
    """Execute the full analysis pipeline.

    Parameters
    ----------
    stocks : list[str] | None
        Specific symbols to analyse.  ``None`` → use ``config.WATCHLIST``.
    send_telegram : bool
        Whether to push the report to Telegram.
    verbose : bool
        Enable debug-level logging.
    capital : float | None
        Portfolio capital override.
    universe : str | None
        Which stock universe to use (nifty50/next50/midcap/all).

    Returns
    -------
    list[StockRecommendation]
        All generated recommendations.
    """
    setup_logging(verbose)
    start = datetime.now()

    # Override capital if specified
    if capital is not None:
        config.PORTFOLIO_CAPITAL = capital

    watchlist = _resolve_watchlist(universe, stocks)

    logger.info("=" * 60)
    logger.info("🚀  AI Stock Market Analysis Agent — Pipeline Start")
    logger.info("=" * 60)
    logger.info("Watchlist: %d stocks | Capital: ₹%.0f", len(watchlist), config.PORTFOLIO_CAPITAL)

    # ── Step 1: Fetch data ───────────────────────────────────
    logger.info("")
    logger.info("📥  STEP 1/6 — Fetching stock data from Yahoo Finance …")
    all_stock_data: list[StockData] = fetch_all_stocks(watchlist)
    successful_stocks = [s for s in all_stock_data if s.fetch_success]
    failed_stocks = [s for s in all_stock_data if not s.fetch_success]

    logger.info(
        "   ✅ Fetched: %d  |  ❌ Failed: %d",
        len(successful_stocks), len(failed_stocks),
    )
    for fs in failed_stocks:
        logger.warning("   ⚠️  %s — %s", fs.symbol, fs.error_message)

    if not successful_stocks:
        logger.error("No stocks fetched successfully — aborting pipeline.")
        return []

    # ── Step 2: Technical indicators ─────────────────────────
    logger.info("")
    logger.info("📊  STEP 2/6 — Computing technical indicators (RSI, MACD, VWAP, Supertrend, Fib) …")
    stock_indicator_pairs: list[tuple[StockData, TechnicalSummary]] = []
    for sd in successful_stocks:
        indicators = compute_all_indicators(sd.symbol, sd.ohlcv)
        stock_indicator_pairs.append((sd, indicators))
    logger.info("   Indicators computed for %d stocks.", len(stock_indicator_pairs))

    # ── Step 3: Groq AI analysis ─────────────────────────────
    logger.info("")
    logger.info("🤖  STEP 3/6 — Running Groq AI analysis …")
    if not config.GROQ_API_KEY:
        logger.error(
            "   GROQ_API_KEY not set!  "
            "Please add it to your .env file.  Aborting AI analysis."
        )
        recommendations = [
            StockRecommendation(
                symbol=sd.symbol, analysis_success=False,
                error_message="API key missing",
            )
            for sd, _ in stock_indicator_pairs
        ]
    else:
        recommendations = analyse_all(stock_indicator_pairs)
        ok = sum(1 for r in recommendations if r.analysis_success)
        logger.info("   ✅ Analysed: %d  |  ❌ Failed: %d", ok, len(recommendations) - ok)

    # ── Step 4: Position sizing & risk ───────────────────────
    logger.info("")
    logger.info("💰  STEP 4/6 — Calculating position sizing & portfolio risk …")
    try:
        from risk_manager import calculate_all_positions, assess_portfolio_risk, format_position_sizing, format_portfolio_risk

        buy_recs = [r for r in recommendations if r.analysis_success and r.recommendation == "BUY"]
        if buy_recs:
            positions = calculate_all_positions(buy_recs)
            risk = assess_portfolio_risk(buy_recs, positions)
            logger.info("   %d BUY positions sized.", len(positions))
            logger.info("   Portfolio risk: %.1f%% | Diversification: %.0f%%",
                        risk.total_risk * 100, risk.diversification_score * 100)
            for w in risk.warnings:
                logger.warning("   ⚠️  %s", w)

            # Attach position sizing info to pass downstream
            _position_info = format_position_sizing(positions)
            _risk_info = format_portfolio_risk(risk)
        else:
            logger.info("   No BUY picks — position sizing skipped.")
            _position_info = ""
            _risk_info = ""
            positions = []
            risk = None
    except Exception as exc:
        logger.warning("   Position sizing failed: %s", exc)
        _position_info = ""
        _risk_info = ""
        positions = []
        risk = None

    # ── Step 5: Generate reports ─────────────────────────────
    logger.info("")
    logger.info("📝  STEP 5/6 — Generating reports …")
    txt_path, html_path = save_reports(recommendations)
    logger.info("   TXT  → %s", txt_path)
    logger.info("   HTML → %s", html_path)

    # ── Step 6: Telegram notification ────────────────────────
    logger.info("")
    if send_telegram and config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        logger.info("📨  STEP 6/6 — Sending Telegram notification …")
        success = send_telegram_report(recommendations, extra_sections=[
            _position_info, _risk_info,
        ])
        if success:
            logger.info("   ✅ Telegram report sent!")
        else:
            logger.warning("   ⚠️  Telegram delivery failed.")
    else:
        if not send_telegram:
            logger.info("📨  STEP 6/6 — Telegram skipped (--no-telegram flag).")
        else:
            logger.info(
                "📨  STEP 6/6 — Telegram not configured. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env."
            )

    # ── Done ─────────────────────────────────────────────────
    elapsed = (datetime.now() - start).total_seconds()
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅  Pipeline complete in %.1f seconds.", elapsed)
    logger.info("=" * 60)

    return recommendations


# ── Screener Mode ────────────────────────────────────────────


def run_screen_mode(
    universe: str | None = None,
    send_telegram: bool = True,
    verbose: bool = False,
) -> None:
    """Run the momentum stock screener."""
    setup_logging(verbose)
    from stock_screener import scan_breakout_stocks, scan_oversold_gems, scan_volume_spikes, format_screen_results

    stocks = _resolve_watchlist(universe, None)
    if universe and universe.lower() == "all":
        stocks = config.FULL_UNIVERSE.copy()
    elif not universe:
        # Default screener to full universe
        stocks = config.FULL_UNIVERSE.copy()

    logger.info("=" * 60)
    logger.info("🔍  STOCK SCREENER — Scanning %d stocks", len(stocks))
    logger.info("=" * 60)

    breakouts = scan_breakout_stocks(stocks, top_n=10)
    oversold = scan_oversold_gems(stocks, top_n=5)
    volume_spikes = scan_volume_spikes(stocks)

    # Build combined report
    report_parts = []
    report_parts.append(format_screen_results(breakouts, "TOP BREAKOUT PICKS"))
    report_parts.append(format_screen_results(oversold, "OVERSOLD GEMS (potential bounce)"))
    report_parts.append(format_screen_results(volume_spikes, "VOLUME SPIKE ALERTS"))
    report = "\n\n".join(report_parts)
    print(report)

    if send_telegram and config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        from notifier import _send_telegram_async
        import asyncio
        logger.info("📨 Sending screener results to Telegram …")
        asyncio.run(_send_telegram_async(report))
        logger.info("   ✅ Sent!")


# ── Backtest Mode ────────────────────────────────────────────


def run_backtest_mode(
    stocks: list[str] | None = None,
    universe: str | None = None,
    verbose: bool = False,
) -> None:
    """Run the backtesting engine."""
    setup_logging(verbose)
    from backtester import run_full_backtest, format_backtest_results

    watchlist = _resolve_watchlist(universe, stocks)
    # Cap backtest to top 20 for speed
    if len(watchlist) > 20:
        watchlist = watchlist[:20]

    logger.info("=" * 60)
    logger.info("📈  BACKTESTER — Testing %d stocks over %d days", len(watchlist), config.BACKTEST_DAYS)
    logger.info("=" * 60)

    result = run_full_backtest(watchlist)
    report = format_backtest_results(result)
    print(report)


# ── Alert Mode ───────────────────────────────────────────────


def run_alert_mode(
    stocks: list[str] | None = None,
    universe: str | None = None,
    send_telegram: bool = True,
    once: bool = False,
    verbose: bool = False,
) -> None:
    """Run alert monitoring."""
    setup_logging(verbose)
    from alerts import run_alert_monitor, run_single_alert_scan

    watchlist = _resolve_watchlist(universe, stocks)

    if once:
        run_single_alert_scan(watchlist, send_telegram=send_telegram)
    else:
        run_alert_monitor(watchlist, send_telegram=send_telegram)


# ── Heatmap Mode ─────────────────────────────────────────────


def run_heatmap_mode(
    universe: str | None = None,
    send_telegram: bool = True,
    verbose: bool = False,
) -> None:
    """Run sector heatmap analysis."""
    setup_logging(verbose)
    from sector_heatmap import calculate_sector_performance, format_sector_heatmap

    stocks = _resolve_watchlist(universe, None)
    if not universe:
        stocks = config.FULL_UNIVERSE.copy()

    logger.info("=" * 60)
    logger.info("🗺️  SECTOR HEATMAP — Scanning %d stocks", len(stocks))
    logger.info("=" * 60)

    sectors = calculate_sector_performance(stocks)
    report = format_sector_heatmap(sectors)
    print(report)

    if send_telegram and config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        from notifier import _send_telegram_async
        import asyncio
        logger.info("📨 Sending heatmap to Telegram …")
        asyncio.run(_send_telegram_async(report))
        logger.info("   ✅ Sent!")


# ── Portfolio Mode ───────────────────────────────────────────


def run_portfolio_mode(
    send_telegram: bool = True,
    verbose: bool = False,
) -> None:
    """Run portfolio tracker."""
    setup_logging(verbose)
    from portfolio_tracker import track_portfolio, format_portfolio_report

    logger.info("=" * 60)
    logger.info("💼  PORTFOLIO TRACKER")
    logger.info("=" * 60)

    summary = track_portfolio()
    report = format_portfolio_report(summary)
    print(report)

    if send_telegram and config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        from notifier import _send_telegram_async
        import asyncio
        logger.info("📨 Sending portfolio report to Telegram …")
        asyncio.run(_send_telegram_async(report))
        logger.info("   ✅ Sent!")


# ── News Sentiment Mode ─────────────────────────────────────


def run_news_mode(
    send_telegram: bool = True,
    verbose: bool = False,
) -> None:
    """Run news sentiment analysis."""
    setup_logging(verbose)
    from news_sentiment import analyze_market_sentiment, format_sentiment_report

    logger.info("=" * 60)
    logger.info("📰  NEWS SENTIMENT ANALYSIS")
    logger.info("=" * 60)

    sentiment = analyze_market_sentiment()
    report = format_sentiment_report(sentiment)
    print(report)

    if send_telegram and config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        from notifier import _send_telegram_async
        import asyncio
        logger.info("📨 Sending news sentiment to Telegram …")
        asyncio.run(_send_telegram_async(report))
        logger.info("   ✅ Sent!")


# ── Options Chain Mode ───────────────────────────────────────


def run_options_mode(
    send_telegram: bool = True,
    verbose: bool = False,
) -> None:
    """Run Nifty/BankNifty options chain analysis."""
    setup_logging(verbose)
    from options_chain import analyze_options, format_options_report

    logger.info("=" * 60)
    logger.info("📊  OPTIONS CHAIN ANALYSIS")
    logger.info("=" * 60)

    nifty = analyze_options("NIFTY")
    banknifty = analyze_options("BANKNIFTY")
    report = format_options_report([nifty, banknifty])
    print(report)

    if send_telegram and config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        from notifier import _send_telegram_async
        import asyncio
        logger.info("📨 Sending options analysis to Telegram …")
        asyncio.run(_send_telegram_async(report))
        logger.info("   ✅ Sent!")


# ── Email Mode ───────────────────────────────────────────────


def run_email_mode(
    stocks: list[str] | None = None,
    universe: str | None = None,
    verbose: bool = False,
) -> None:
    """Run pipeline and send report via email."""
    setup_logging(verbose)

    logger.info("=" * 60)
    logger.info("📧  EMAIL REPORT MODE")
    logger.info("=" * 60)

    # Run pipeline first (no telegram)
    recs = run_pipeline(
        stocks=stocks,
        send_telegram=False,
        verbose=verbose,
        universe=universe,
    )

    if not recs:
        logger.warning("No recommendations to email.")
        return

    from email_notifier import send_email_report
    success = send_email_report(recs)
    if success:
        logger.info("✅ Email report sent successfully!")
    else:
        logger.error("❌ Email delivery failed.")


# ── Telegram Bot Mode ───────────────────────────────────────


def run_bot_mode(verbose: bool = False) -> None:
    """Start the interactive Telegram bot."""
    setup_logging(verbose)
    from telegram_bot import run_telegram_bot

    logger.info("=" * 60)
    logger.info("🤖  INTERACTIVE TELEGRAM BOT")
    logger.info("=" * 60)

    run_telegram_bot()


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    """Parse CLI arguments and dispatch."""
    parser = argparse.ArgumentParser(
        description="AI Stock Market Analysis Agent for Indian Markets (NSE/BSE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                                # Analyse Nifty 50\n"
            "  python main.py --stocks RELIANCE TCS          # Specific stocks\n"
            "  python main.py --universe all                 # Nifty50 + Next50 + Midcaps\n"
            "  python main.py --scan                         # Momentum screener\n"
            "  python main.py --backtest                     # Backtest strategy\n"
            "  python main.py --alerts                       # Real-time alerts\n"
            "  python main.py --alerts --once                # Single alert scan\n"
            "  python main.py --heatmap                      # Sector heatmap\n"
            "  python main.py --portfolio                    # Portfolio P&L tracker\n"
            "  python main.py --news                         # News sentiment analysis\n"
            "  python main.py --options                      # Options chain analysis\n"
            "  python main.py --email                        # Send report via email\n"
            "  python main.py --bot                          # Interactive Telegram bot\n"
            "  python main.py --capital 500000 --no-telegram # Set capital, skip Telegram\n"
            "  python main.py --schedule                     # Daily scheduler\n"
        ),
    )

    # Mode selection (mutually exclusive)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--schedule",
        action="store_true",
        help="Start the daily scheduler (runs at 08:30 IST, Mon-Fri).",
    )
    mode.add_argument(
        "--scan",
        action="store_true",
        help="Run the momentum stock screener on the full universe.",
    )
    mode.add_argument(
        "--backtest",
        action="store_true",
        help="Run the backtesting engine to evaluate strategy performance.",
    )
    mode.add_argument(
        "--alerts",
        action="store_true",
        help="Start real-time price/volume alert monitoring via Telegram.",
    )
    mode.add_argument(
        "--heatmap",
        action="store_true",
        help="Generate sector-wise performance heatmap.",
    )
    mode.add_argument(
        "--portfolio",
        action="store_true",
        help="Track portfolio holdings, P&L, and exit alerts.",
    )
    mode.add_argument(
        "--news",
        action="store_true",
        help="Run AI-powered news sentiment analysis.",
    )
    mode.add_argument(
        "--options",
        action="store_true",
        help="Analyse Nifty/BankNifty options chain (PCR, Max Pain, OI).",
    )
    mode.add_argument(
        "--email",
        action="store_true",
        help="Run full pipeline and send report via Gmail.",
    )
    mode.add_argument(
        "--bot",
        action="store_true",
        help="Start interactive Telegram bot with slash commands.",
    )

    # Stock selection
    parser.add_argument(
        "--stocks",
        nargs="+",
        metavar="SYMBOL",
        help="Analyse specific stock symbols instead of the full watchlist.",
    )
    parser.add_argument(
        "--universe",
        choices=["nifty50", "next50", "midcap", "all"],
        default=None,
        help="Which stock universe to use (default: nifty50).",
    )

    # Options
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        metavar="AMOUNT",
        help="Portfolio capital in ₹ (overrides .env setting).",
    )
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Skip sending the report via Telegram.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan (used with --alerts).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    args = parser.parse_args()

    if args.schedule:
        setup_logging(args.verbose)
        logger.info("Starting scheduler mode …")
        from scheduler import start_scheduler
        start_scheduler()
    elif args.scan:
        run_screen_mode(
            universe=args.universe,
            send_telegram=not args.no_telegram,
            verbose=args.verbose,
        )
    elif args.backtest:
        run_backtest_mode(
            stocks=args.stocks,
            universe=args.universe,
            verbose=args.verbose,
        )
    elif args.alerts:
        run_alert_mode(
            stocks=args.stocks,
            universe=args.universe,
            send_telegram=not args.no_telegram,
            once=args.once,
            verbose=args.verbose,
        )
    elif args.heatmap:
        run_heatmap_mode(
            universe=args.universe,
            send_telegram=not args.no_telegram,
            verbose=args.verbose,
        )
    elif args.portfolio:
        run_portfolio_mode(
            send_telegram=not args.no_telegram,
            verbose=args.verbose,
        )
    elif args.news:
        run_news_mode(
            send_telegram=not args.no_telegram,
            verbose=args.verbose,
        )
    elif args.options:
        run_options_mode(
            send_telegram=not args.no_telegram,
            verbose=args.verbose,
        )
    elif args.email:
        run_email_mode(
            stocks=args.stocks,
            universe=args.universe,
            verbose=args.verbose,
        )
    elif args.bot:
        run_bot_mode(verbose=args.verbose)
    else:
        run_pipeline(
            stocks=args.stocks,
            send_telegram=not args.no_telegram,
            verbose=args.verbose,
            capital=args.capital,
            universe=args.universe,
        )


if __name__ == "__main__":
    main()

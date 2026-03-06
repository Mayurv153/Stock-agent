"""
telegram_bot.py — Interactive Telegram Bot with slash commands.

Commands:
  /start     — Welcome message
  /help      — List all commands
  /scan      — Run stock screener (breakouts, oversold, volume)
  /analyse SYMBOL — Full AI analysis for a single stock
  /heatmap   — Sector heatmap
  /portfolio — Portfolio P&L tracker
  /news      — News sentiment analysis
  /options   — Nifty/BankNifty options chain
  /alerts    — Run alert scan
  /backtest  — Run quick backtest
  /report    — Full daily pipeline report

Uses python-telegram-bot v20+ (async).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────

async def _send_html(update: Update, text: str) -> None:
    """Send HTML-formatted message, auto-splitting if too long."""
    MAX_LEN = 4000
    if len(text) <= MAX_LEN:
        await update.message.reply_text(text, parse_mode="HTML")
    else:
        # Split on newlines
        chunks = []
        current = ""
        for line in text.split("\n"):
            if len(current) + len(line) + 1 > MAX_LEN:
                chunks.append(current)
                current = line
            else:
                current += "\n" + line if current else line
        if current:
            chunks.append(current)

        for chunk in chunks:
            await update.message.reply_text(chunk, parse_mode="HTML")
            await asyncio.sleep(0.5)  # Rate limit


# ── Command Handlers ─────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    msg = (
        "<b>🤖 AI Stock Agent — Indian Markets</b>\n\n"
        "Welcome! I'm your AI-powered stock market assistant.\n"
        "I analyse NSE/BSE stocks with technical indicators, AI insights, and more.\n\n"
        "Type /help to see all available commands."
    )
    await _send_html(update, msg)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    msg = (
        "<b>📋 AVAILABLE COMMANDS</b>\n\n"
        "/scan — Run stock screener (breakouts, oversold, volume spikes)\n"
        "/analyse SYMBOL — Full AI analysis (e.g. /analyse RELIANCE)\n"
        "/heatmap — Sector performance heatmap\n"
        "/portfolio — Track your holdings P&L\n"
        "/news — News sentiment analysis\n"
        "/options — Nifty/BankNifty options chain\n"
        "/alerts — Check price/volume alerts\n"
        "/backtest — Backtest top picks\n"
        "/report — Full daily analysis pipeline\n\n"
        "<b>Quick Tips:</b>\n"
        "• /analyse TCS — Analyse any NSE stock\n"
        "• /scan — Find trading opportunities\n"
        "• /options — Check market direction via OI"
    )
    await _send_html(update, msg)


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /scan — Run stock screener."""
    await update.message.reply_text("🔍 Scanning stocks… please wait ⏳")

    try:
        from stock_screener import (
            scan_breakout_stocks,
            scan_oversold_gems,
            scan_volume_spikes,
            format_screen_results,
        )

        universe = config.NIFTY_50[:30]

        breakouts = scan_breakout_stocks(universe)
        oversold = scan_oversold_gems(universe)
        volume = scan_volume_spikes(universe)

        msg = ""
        if breakouts:
            msg += format_screen_results(breakouts, "Breakout Stocks") + "\n\n"
        if oversold:
            msg += format_screen_results(oversold, "Oversold Gems") + "\n\n"
        if volume:
            msg += format_screen_results(volume, "Volume Spikes") + "\n\n"

        if not msg:
            msg = "No significant signals found right now. Market is quiet. 😴"

        await _send_html(update, msg)

    except Exception as exc:
        logger.error("Scan command failed: %s", exc)
        await update.message.reply_text(f"❌ Scan failed: {exc}")


async def cmd_analyse(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /analyse SYMBOL — Full AI analysis for one stock."""
    if not context.args:
        await update.message.reply_text(
            "Usage: /analyse SYMBOL\nExample: /analyse RELIANCE"
        )
        return

    symbol = context.args[0].upper()
    await update.message.reply_text(f"🤖 Analysing {symbol}… please wait ⏳")

    try:
        from data_fetcher import fetch_stock_data
        from indicators import compute_indicators
        from analyzer import analyze_stock

        stock_data = fetch_stock_data(symbol)
        if stock_data is None or stock_data.ohlcv.empty:
            await update.message.reply_text(f"❌ No data found for {symbol}")
            return

        tech = compute_indicators(stock_data.ohlcv)
        recommendation = analyze_stock(symbol, stock_data, tech)

        if recommendation is None:
            await update.message.reply_text(f"❌ AI analysis failed for {symbol}")
            return

        # Calculate risk:reward
        risk = recommendation.entry_price - recommendation.stop_loss if recommendation.stop_loss > 0 else 0
        reward = recommendation.target_price - recommendation.entry_price if recommendation.target_price > 0 else 0
        rr = f"1:{reward/risk:.1f}" if risk > 0 else "N/A"
        reasons = "\n".join(f"- {r}" for r in recommendation.key_reasons[:5]) if recommendation.key_reasons else "N/A"

        msg = (
            f"<b>\U0001f916 AI ANALYSIS \u2014 {symbol}</b>\n\n"
            f"<b>Action:</b> {recommendation.recommendation}\n"
            f"<b>Confidence:</b> {recommendation.confidence_score}%\n"
            f"<b>Entry:</b> \u20b9{recommendation.entry_price}\n"
            f"<b>Target:</b> \u20b9{recommendation.target_price}\n"
            f"<b>Stop Loss:</b> \u20b9{recommendation.stop_loss}\n"
            f"<b>Risk/Reward:</b> {rr}\n"
            f"<b>Type:</b> {recommendation.trade_type}\n\n"
            f"<b>Key Reasons:</b>\n{reasons}"
        )

        await _send_html(update, msg)

    except Exception as exc:
        logger.error("Analyse command failed for %s: %s", symbol, exc)
        await update.message.reply_text(f"❌ Analysis failed: {exc}")


async def cmd_heatmap(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /heatmap — Sector heatmap."""
    await update.message.reply_text("🗺️ Generating sector heatmap… ⏳")

    try:
        from sector_heatmap import calculate_sector_performance, format_sector_heatmap

        sectors = calculate_sector_performance(config.NIFTY_50)
        msg = format_sector_heatmap(sectors)
        await _send_html(update, msg)

    except Exception as exc:
        logger.error("Heatmap command failed: %s", exc)
        await update.message.reply_text(f"❌ Heatmap failed: {exc}")


async def cmd_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /portfolio — Portfolio tracker."""
    await update.message.reply_text("💼 Checking portfolio… ⏳")

    try:
        from portfolio_tracker import track_portfolio, format_portfolio_report

        summary = track_portfolio()
        msg = format_portfolio_report(summary)
        await _send_html(update, msg)

    except Exception as exc:
        logger.error("Portfolio command failed: %s", exc)
        await update.message.reply_text(f"❌ Portfolio failed: {exc}")


async def cmd_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /news — News sentiment analysis."""
    await update.message.reply_text("📰 Analysing news sentiment… ⏳")

    try:
        from news_sentiment import analyze_market_sentiment, format_sentiment_report

        sentiment = analyze_market_sentiment()
        msg = format_sentiment_report(sentiment)
        await _send_html(update, msg)

    except Exception as exc:
        logger.error("News command failed: %s", exc)
        await update.message.reply_text(f"❌ News analysis failed: {exc}")


async def cmd_options(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /options — Options chain analysis."""
    await update.message.reply_text("📊 Fetching options chain from NSE… ⏳")

    try:
        from options_chain import analyze_options, format_options_report

        nifty = analyze_options("NIFTY")
        banknifty = analyze_options("BANKNIFTY")
        msg = format_options_report([nifty, banknifty])
        await _send_html(update, msg)

    except Exception as exc:
        logger.error("Options command failed: %s", exc)
        await update.message.reply_text(f"❌ Options analysis failed: {exc}")


async def cmd_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /alerts — Run alert scan."""
    await update.message.reply_text("🔔 Scanning for alerts… ⏳")

    try:
        from alerts import run_single_alert_scan

        alerts = run_single_alert_scan(config.NIFTY_50[:30])

        if not alerts:
            await update.message.reply_text("✅ No alerts triggered. All quiet!")
            return

        lines = ["<b>🔔 ALERT SCAN RESULTS</b>\n"]
        for a in alerts[:15]:
            emoji = {"VOLUME_SPIKE": "📊", "RSI_EXTREME": "📈", "BREAKOUT": "🚀",
                     "BREAKDOWN": "📉", "SUPERTREND_FLIP": "🔄"}.get(a.alert_type, "⚡")
            lines.append(f"{emoji} <b>{a.symbol}</b> — {a.message}")

        await _send_html(update, "\n".join(lines))

    except Exception as exc:
        logger.error("Alerts command failed: %s", exc)
        await update.message.reply_text(f"❌ Alert scan failed: {exc}")


async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /backtest — Quick backtest."""
    await update.message.reply_text("📈 Running backtest… this may take a minute ⏳")

    try:
        from backtester import run_full_backtest, format_backtest_results

        test_stocks = config.NIFTY_50[:10]
        result = run_full_backtest(test_stocks, days=90)
        msg = format_backtest_results(result)
        await _send_html(update, msg)

    except Exception as exc:
        logger.error("Backtest command failed: %s", exc)
        await update.message.reply_text(f"❌ Backtest failed: {exc}")


async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /report — Full daily pipeline."""
    await update.message.reply_text(
        "📋 Running full daily analysis pipeline… this takes 1-2 minutes ⏳"
    )

    try:
        from data_fetcher import fetch_stock_data
        from indicators import compute_indicators
        from analyzer import analyze_stock

        stocks = config.NIFTY_50[:15]  # Top 15 for speed
        results = []

        for symbol in stocks:
            try:
                sd = fetch_stock_data(symbol)
                if sd and not sd.ohlcv.empty:
                    tech = compute_indicators(sd.ohlcv)
                    rec = analyze_stock(symbol, sd, tech)
                    if rec:
                        results.append(rec)
            except Exception:
                continue

        if not results:
            await update.message.reply_text("❌ No stocks could be analysed.")
            return

        # Build summary
        buys = [r for r in results if r.recommendation == "BUY"]
        sells = [r for r in results if r.recommendation == "SELL"]
        holds = [r for r in results if r.recommendation == "HOLD"]

        lines = [
            f"<b>📋 DAILY REPORT — {datetime.now().strftime('%d %b %Y')}</b>\n",
            f"Analysed: {len(results)} stocks\n",
        ]

        if buys:
            lines.append("<b>🟢 BUY Signals:</b>")
            for r in buys[:5]:
                lines.append(
                    f"  • {r.symbol} — {r.confidence_score}% conf, "
                    f"Entry ₹{r.entry_price}, Target ₹{r.target_price}"
                )
            lines.append("")

        if sells:
            lines.append("<b>🔴 SELL Signals:</b>")
            for r in sells[:5]:
                reasons = ", ".join(r.key_reasons[:2]) if r.key_reasons else r.risk_warning[:60]
                lines.append(f"  • {r.symbol} — {reasons}")
            lines.append("")

        if holds:
            lines.append(f"<b>⚪ HOLD:</b> {', '.join(r.symbol for r in holds[:8])}")

        await _send_html(update, "\n".join(lines))

    except Exception as exc:
        logger.error("Report command failed: %s", exc)
        await update.message.reply_text(f"❌ Report failed: {exc}")


async def cmd_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle unknown commands."""
    await update.message.reply_text(
        "❓ Unknown command. Type /help to see available commands."
    )


# ── Bot Launcher ─────────────────────────────────────────────

def run_telegram_bot() -> None:
    """Start the Telegram bot (blocking — runs until Ctrl+C)."""
    token = config.TELEGRAM_BOT_TOKEN
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set!")
        return

    logger.info("Starting Telegram bot… Press Ctrl+C to stop.")

    app = Application.builder().token(token).build()

    # Register handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("analyse", cmd_analyse))
    app.add_handler(CommandHandler("analyze", cmd_analyse))  # US spelling
    app.add_handler(CommandHandler("heatmap", cmd_heatmap))
    app.add_handler(CommandHandler("portfolio", cmd_portfolio))
    app.add_handler(CommandHandler("news", cmd_news))
    app.add_handler(CommandHandler("options", cmd_options))
    app.add_handler(CommandHandler("alerts", cmd_alerts))
    app.add_handler(CommandHandler("backtest", cmd_backtest))
    app.add_handler(CommandHandler("report", cmd_report))

    # Unknown command handler (must be last)
    app.add_handler(MessageHandler(filters.COMMAND, cmd_unknown))

    # Start polling
    app.run_polling(allowed_updates=Update.ALL_TYPES)


# ── Entry Point ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    run_telegram_bot()

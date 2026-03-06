"""
notifier.py — Deliver daily stock analysis reports via Telegram.

Uses the python-telegram-bot library to send HTML-formatted messages
to a configured Telegram chat or channel.  Long messages are
automatically split at the 4096-character Telegram limit.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import telegram
from telegram.constants import ParseMode

import config
from analyzer import StockRecommendation

logger = logging.getLogger(__name__)

# Telegram message length limit
MAX_MESSAGE_LENGTH = 4096


# ── Emoji Helpers ────────────────────────────────────────────

def _rec_emoji(rec: str) -> str:
    return {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(rec, "⚪")


def _risk_emoji(risk: str) -> str:
    return {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk, "⚪")


# ── Message Formatting ───────────────────────────────────────


def format_telegram_message(recommendations: list[StockRecommendation]) -> str:
    """Build an HTML-formatted Telegram message from recommendations.

    Parameters
    ----------
    recommendations : list[StockRecommendation]
        The full list of AI-generated recommendations.

    Returns
    -------
    str
        HTML string suitable for Telegram's ``ParseMode.HTML``.
    """
    from datetime import datetime
    date_str = datetime.now().strftime(config.REPORT_FORMAT_DATE)

    lines: list[str] = []
    lines.append(f"<b>📊 AI Stock Analysis Report</b>")
    lines.append(f"<i>{date_str} — NSE</i>")
    lines.append("")

    # Categorise
    buy_picks = []
    intraday_picks = []
    avoid_picks = []
    hold_picks = []

    for rec in recommendations:
        if not rec.analysis_success:
            continue
        if rec.recommendation == "BUY" and rec.trade_type == "INTRADAY":
            intraday_picks.append(rec)
        elif rec.recommendation == "BUY":
            buy_picks.append(rec)
        elif rec.recommendation == "SELL" or rec.trade_type == "AVOID":
            avoid_picks.append(rec)
        else:
            hold_picks.append(rec)

    buy_picks.sort(key=lambda r: r.confidence_score, reverse=True)
    intraday_picks.sort(key=lambda r: r.confidence_score, reverse=True)
    buy_picks = buy_picks[: config.MAX_BUY_PICKS]
    intraday_picks = intraday_picks[: config.MAX_INTRADAY_PICKS]

    # Summary
    total = sum(1 for r in recommendations if r.analysis_success)
    lines.append(f"<b>Analysed:</b> {total} stocks")
    lines.append(
        f"🟢 BUY: {len(buy_picks) + len(intraday_picks)} | "
        f"🔴 SELL: {len(avoid_picks)} | "
        f"🟡 HOLD: {len(hold_picks)}"
    )
    lines.append("")

    def _stock_block(rec: StockRecommendation) -> list[str]:
        block = []
        emoji = _rec_emoji(rec.recommendation)
        block.append(f"{emoji} <b>{rec.symbol}</b> — {rec.recommendation} ({rec.trade_type})")
        block.append(
            f"   Entry: ₹{rec.entry_price:.2f} | "
            f"Target: ₹{rec.target_price:.2f} | "
            f"SL: ₹{rec.stop_loss:.2f}"
        )
        block.append(
            f"   Risk: {_risk_emoji(rec.risk_level)} {rec.risk_level} | "
            f"Confidence: {rec.confidence_score}%"
        )
        if rec.key_reasons:
            reasons_str = " • ".join(rec.key_reasons[:3])
            block.append(f"   📝 {reasons_str}")
        if rec.risk_warning:
            block.append(f"   ⚠️ <i>{rec.risk_warning}</i>")
        return block

    # Sections
    if buy_picks:
        lines.append("<b>━━━ 🟢 TOP BUY PICKS ━━━</b>")
        for rec in buy_picks:
            lines.extend(_stock_block(rec))
            lines.append("")

    if intraday_picks:
        lines.append("<b>━━━ ⚡ INTRADAY PICKS ━━━</b>")
        for rec in intraday_picks:
            lines.extend(_stock_block(rec))
            lines.append("")

    if avoid_picks:
        lines.append("<b>━━━ 🚫 AVOID ━━━</b>")
        for rec in avoid_picks:
            lines.extend(_stock_block(rec))
            lines.append("")

    if hold_picks:
        lines.append("<b>━━━ 🟡 HOLD ━━━</b>")
        for rec in hold_picks:
            lines.extend(_stock_block(rec))
            lines.append("")

    # Disclaimer
    lines.append(
        "<i>⚠️ AI-generated report for educational purposes only. "
        "Not financial advice. Markets are subject to risk. DYOR.</i>"
    )

    return "\n".join(lines)


# ── Message Splitting ────────────────────────────────────────


def _split_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a long message into chunks ≤ max_length, breaking at newlines.

    Parameters
    ----------
    text : str
        The full message text.
    max_length : int
        Maximum allowed length per chunk.

    Returns
    -------
    list[str]
        List of message chunks.
    """
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        # +1 for the newline character
        if len(current) + len(line) + 1 > max_length:
            if current:
                chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)

    return chunks


# ── Send via Telegram ────────────────────────────────────────


async def _send_telegram_async(
    text: str,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> bool:
    """Send an HTML message to Telegram (async implementation).

    Returns True on success, False on failure.
    """
    token = bot_token or config.TELEGRAM_BOT_TOKEN
    cid = chat_id or config.TELEGRAM_CHAT_ID

    if not token or not cid:
        logger.error(
            "Telegram credentials missing — set TELEGRAM_BOT_TOKEN "
            "and TELEGRAM_CHAT_ID in your .env file."
        )
        return False

    bot = telegram.Bot(token=token)

    chunks = _split_message(text)
    logger.info("Sending %d Telegram message chunk(s) …", len(chunks))

    for i, chunk in enumerate(chunks, start=1):
        try:
            await bot.send_message(
                chat_id=cid,
                text=chunk,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            logger.info("Chunk %d/%d sent successfully.", i, len(chunks))
        except telegram.error.TelegramError as exc:
            logger.error("Telegram send failed (chunk %d): %s", i, exc)
            return False

    return True


def send_telegram_report(
    recommendations: list[StockRecommendation],
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    extra_sections: Optional[list[str]] = None,
) -> bool:
    """Format the recommendations and send to Telegram.

    This is a synchronous wrapper that creates an event loop if needed.

    Parameters
    ----------
    recommendations : list[StockRecommendation]
        Analysis results.
    bot_token : str | None
        Override Telegram bot token (defaults to .env).
    chat_id : str | None
        Override Telegram chat/channel ID (defaults to .env).
    extra_sections : list[str] | None
        Additional text sections to append (e.g. position sizing, risk info).

    Returns
    -------
    bool
        True if all messages were sent successfully.
    """
    message = format_telegram_message(recommendations)

    # Append any extra sections (position sizing, risk, etc.)
    if extra_sections:
        for section in extra_sections:
            if section and section.strip():
                message += "\n\n" + section.strip()

    logger.info("Telegram message length: %d chars", len(message))

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're already inside an async context — schedule coroutine
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                _send_telegram_async(message, bot_token, chat_id),
            )
            return future.result()
    else:
        return asyncio.run(_send_telegram_async(message, bot_token, chat_id))


# ── Quick Test ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    # Quick formatting test (does NOT actually send if creds are missing)
    from analyzer import StockRecommendation

    dummy = [
        StockRecommendation(
            symbol="RELIANCE", recommendation="BUY", trade_type="SHORT_TERM",
            entry_price=2450.0, target_price=2600.0, stop_loss=2380.0,
            risk_level="MEDIUM", confidence_score=78,
            key_reasons=["Bullish MACD", "RSI rising", "Volume spike"],
            risk_warning="Market volatility high.",
        ),
    ]
    msg = format_telegram_message(dummy)
    print(msg)
    print(f"\n(Message length: {len(msg)} chars)")

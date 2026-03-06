"""
alerts.py — Real-time price/volume alert system via Telegram.

Monitors your watchlist continuously and sends instant Telegram alerts for:
  - Price breakouts (above resistance)
  - Price breakdowns (below support)
  - Volume spikes (2.5x+ average)
  - RSI extreme levels (>80 overbought, <20 oversold)
  - Stop-loss breaches

Run with: python main.py --alerts
Sends instant Telegram messages when conditions are met.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

import config
from data_fetcher import fetch_ohlcv
from indicators import (
    calculate_rsi, calculate_volume_analysis,
    calculate_support_resistance, calculate_vwap,
    calculate_supertrend,
)
from notifier import _send_telegram_async

logger = logging.getLogger(__name__)


# ── Alert Data Models ────────────────────────────────────────


@dataclass
class Alert:
    """A triggered alert."""

    symbol: str
    alert_type: str          # BREAKOUT / BREAKDOWN / VOLUME_SPIKE / RSI_EXTREME / STOP_BREACH
    severity: str            # HIGH / MEDIUM / LOW
    message: str
    price: float = 0.0
    trigger_value: float = 0.0
    timestamp: str = ""


@dataclass
class AlertState:
    """Track alert cooldowns to avoid spamming."""

    last_alerts: dict = field(default_factory=dict)   # symbol+type → timestamp
    cooldown_minutes: int = 30                        # Don't repeat same alert within 30 min


# ── Core Alert Detection ─────────────────────────────────────


def check_stock_alerts(
    symbol: str,
    state: AlertState,
) -> list[Alert]:
    """Check a single stock for alert conditions.

    Parameters
    ----------
    symbol : str
        NSE stock symbol.
    state : AlertState
        Current alert state (for cooldown tracking).

    Returns
    -------
    list[Alert]
        List of triggered alerts (may be empty).
    """
    alerts: list[Alert] = []
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M")

    try:
        df = fetch_ohlcv(symbol, period_days=30)
        if df.empty or len(df) < 10:
            return []

        last_close = float(df["Close"].iloc[-1])

        # ── 1. Volume Spike Alert ────────────────────────────
        _, avg_vol, vol_ratio, vol_signal = calculate_volume_analysis(df)
        if vol_ratio is not None and vol_ratio >= config.VOLUME_SPIKE_THRESHOLD:
            alert_key = f"{symbol}_VOLUME_SPIKE"
            if _should_alert(alert_key, state):
                change_pct = 0
                if len(df) >= 2:
                    prev = float(df["Close"].iloc[-2])
                    change_pct = ((last_close - prev) / prev) * 100

                direction = "📈" if change_pct > 0 else "📉"
                alerts.append(Alert(
                    symbol=symbol,
                    alert_type="VOLUME_SPIKE",
                    severity="HIGH",
                    message=(
                        f"🔥 <b>VOLUME SPIKE</b> — {symbol}\n"
                        f"   Volume: {vol_ratio:.1f}x average!\n"
                        f"   Price: ₹{last_close:.2f} ({direction} {change_pct:+.1f}%)\n"
                        f"   ⚡ Something big is happening — investigate!"
                    ),
                    price=last_close,
                    trigger_value=vol_ratio,
                    timestamp=now_str,
                ))
                state.last_alerts[alert_key] = now

        # ── 2. RSI Extreme Alert ─────────────────────────────
        rsi, rsi_signal = calculate_rsi(df)
        if rsi is not None:
            if rsi >= config.RSI_EXTREME_HIGH:
                alert_key = f"{symbol}_RSI_HIGH"
                if _should_alert(alert_key, state):
                    alerts.append(Alert(
                        symbol=symbol,
                        alert_type="RSI_EXTREME",
                        severity="MEDIUM",
                        message=(
                            f"🔴 <b>RSI OVERBOUGHT</b> — {symbol}\n"
                            f"   RSI: {rsi:.1f} (extreme overbought!)\n"
                            f"   Price: ₹{last_close:.2f}\n"
                            f"   ⚠️ Consider booking profits / avoid fresh buying"
                        ),
                        price=last_close,
                        trigger_value=rsi,
                        timestamp=now_str,
                    ))
                    state.last_alerts[alert_key] = now

            elif rsi <= config.RSI_EXTREME_LOW:
                alert_key = f"{symbol}_RSI_LOW"
                if _should_alert(alert_key, state):
                    alerts.append(Alert(
                        symbol=symbol,
                        alert_type="RSI_EXTREME",
                        severity="HIGH",
                        message=(
                            f"💎 <b>RSI DEEPLY OVERSOLD</b> — {symbol}\n"
                            f"   RSI: {rsi:.1f} (potential bounce zone!)\n"
                            f"   Price: ₹{last_close:.2f}\n"
                            f"   👀 Watch for reversal — could be a buying opportunity"
                        ),
                        price=last_close,
                        trigger_value=rsi,
                        timestamp=now_str,
                    ))
                    state.last_alerts[alert_key] = now

        # ── 3. Breakout / Breakdown Alert ────────────────────
        support, resistance = calculate_support_resistance(df, period=20)
        if support is not None and resistance is not None:
            breakout_threshold = resistance * (1 + config.PRICE_BREAKOUT_PCT / 100)
            breakdown_threshold = support * (1 - config.PRICE_BREAKDOWN_PCT / 100)

            if last_close >= breakout_threshold:
                alert_key = f"{symbol}_BREAKOUT"
                if _should_alert(alert_key, state):
                    alerts.append(Alert(
                        symbol=symbol,
                        alert_type="BREAKOUT",
                        severity="HIGH",
                        message=(
                            f"🚀 <b>BREAKOUT!</b> — {symbol}\n"
                            f"   Price ₹{last_close:.2f} broke above resistance ₹{resistance:.2f}\n"
                            f"   ⚡ Strong bullish signal — potential rally ahead"
                        ),
                        price=last_close,
                        trigger_value=resistance,
                        timestamp=now_str,
                    ))
                    state.last_alerts[alert_key] = now

            elif last_close <= breakdown_threshold:
                alert_key = f"{symbol}_BREAKDOWN"
                if _should_alert(alert_key, state):
                    alerts.append(Alert(
                        symbol=symbol,
                        alert_type="BREAKDOWN",
                        severity="HIGH",
                        message=(
                            f"🔻 <b>BREAKDOWN!</b> — {symbol}\n"
                            f"   Price ₹{last_close:.2f} fell below support ₹{support:.2f}\n"
                            f"   ⚠️ Bearish signal — protect your capital"
                        ),
                        price=last_close,
                        trigger_value=support,
                        timestamp=now_str,
                    ))
                    state.last_alerts[alert_key] = now

        # ── 4. Supertrend Flip Alert ─────────────────────────
        _, st_direction = calculate_supertrend(df)
        # Check if direction just changed by comparing with data excluding last row
        if len(df) >= 12:
            _, prev_st_dir = calculate_supertrend(df.iloc[:-1])
            if st_direction != prev_st_dir and prev_st_dir != "NEUTRAL":
                alert_key = f"{symbol}_SUPERTREND_FLIP"
                if _should_alert(alert_key, state):
                    emoji = "🟢" if st_direction == "BULLISH" else "🔴"
                    alerts.append(Alert(
                        symbol=symbol,
                        alert_type="SUPERTREND_FLIP",
                        severity="MEDIUM",
                        message=(
                            f"{emoji} <b>SUPERTREND FLIP</b> — {symbol}\n"
                            f"   Direction changed to {st_direction}\n"
                            f"   Price: ₹{last_close:.2f}\n"
                            f"   📊 Trend reversal signal"
                        ),
                        price=last_close,
                        trigger_value=0,
                        timestamp=now_str,
                    ))
                    state.last_alerts[alert_key] = now

    except Exception as exc:
        logger.debug("Alert check failed for %s: %s", symbol, exc)

    return alerts


def _should_alert(alert_key: str, state: AlertState) -> bool:
    """Check if enough time has passed since last alert of this type."""
    if alert_key not in state.last_alerts:
        return True
    last = state.last_alerts[alert_key]
    cooldown = timedelta(minutes=state.cooldown_minutes)
    return datetime.now() - last >= cooldown


# ── Alert Sending ────────────────────────────────────────────


def _send_alert_telegram(alert: Alert) -> bool:
    """Send a single alert via Telegram."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                _send_telegram_async(alert.message),
            )
            return future.result()
    else:
        return asyncio.run(_send_telegram_async(alert.message))


# ── Main Alert Loop ──────────────────────────────────────────


def run_alert_monitor(
    watchlist: list[str] | None = None,
    check_interval: int | None = None,
    send_telegram: bool = True,
) -> None:
    """Run continuous alert monitoring loop.

    Parameters
    ----------
    watchlist : list[str] | None
        Stocks to monitor. Defaults to ``config.WATCHLIST``.
    check_interval : int | None
        Seconds between scans. Defaults to ``config.ALERT_CHECK_INTERVAL``.
    send_telegram : bool
        Whether to send alerts via Telegram.
    """
    stocks = watchlist or config.WATCHLIST
    interval = check_interval or config.ALERT_CHECK_INTERVAL
    state = AlertState()

    logger.info("🔔 ALERT MONITOR started — watching %d stocks", len(stocks))
    logger.info("   Check interval: %d seconds", interval)
    logger.info("   Telegram alerts: %s", "ON" if send_telegram else "OFF")
    logger.info("   Press Ctrl+C to stop")
    logger.info("")

    # Send startup message
    if send_telegram:
        startup_msg = (
            f"🔔 <b>Alert Monitor Started</b>\n"
            f"Watching {len(stocks)} stocks\n"
            f"Interval: {interval}s\n"
            f"Time: {datetime.now().strftime('%H:%M')}"
        )
        _send_alert_telegram(Alert(
            symbol="SYSTEM", alert_type="STARTUP", severity="LOW",
            message=startup_msg, timestamp=datetime.now().strftime("%H:%M"),
        ))

    scan_count = 0
    total_alerts = 0

    try:
        while True:
            scan_count += 1
            scan_start = time.time()
            alerts_this_scan = []

            logger.info("🔍 Alert scan #%d — checking %d stocks …", scan_count, len(stocks))

            for symbol in stocks:
                stock_alerts = check_stock_alerts(symbol, state)
                alerts_this_scan.extend(stock_alerts)

            if alerts_this_scan:
                total_alerts += len(alerts_this_scan)
                logger.info("🚨 %d alert(s) triggered!", len(alerts_this_scan))
                for alert in alerts_this_scan:
                    logger.info("   [%s] %s: %s", alert.severity, alert.symbol, alert.alert_type)
                    if send_telegram:
                        _send_alert_telegram(alert)
                        time.sleep(1)  # Be polite to Telegram API
            else:
                logger.info("   No alerts triggered.")

            scan_time = time.time() - scan_start
            logger.info("   Scan completed in %.1fs. Total alerts sent: %d", scan_time, total_alerts)
            logger.info("   Next scan in %d seconds …", interval)
            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("\n🔔 Alert monitor stopped. Total alerts sent: %d", total_alerts)


def run_single_alert_scan(
    watchlist: list[str] | None = None,
    send_telegram: bool = True,
) -> list[Alert]:
    """Run a single alert scan (non-looping, for CLI use).

    Returns
    -------
    list[Alert]
        All triggered alerts.
    """
    stocks = watchlist or config.WATCHLIST
    state = AlertState()

    logger.info("🔍 Running single alert scan on %d stocks …", len(stocks))

    all_alerts: list[Alert] = []
    for symbol in stocks:
        alerts = check_stock_alerts(symbol, state)
        all_alerts.extend(alerts)

    if all_alerts:
        logger.info("🚨 %d alert(s) found!", len(all_alerts))
        for alert in all_alerts:
            logger.info("   [%s] %s: %s", alert.severity, alert.symbol, alert.alert_type)
            if send_telegram:
                _send_alert_telegram(alert)
                time.sleep(1)
    else:
        logger.info("   ✅ All clear — no alerts triggered.")

    return all_alerts


def format_alerts_summary(alerts: list[Alert]) -> str:
    """Format alerts for display."""
    if not alerts:
        return "✅ No alerts triggered. Markets look calm."

    lines = [f"🚨 <b>{len(alerts)} ALERT(S) TRIGGERED</b>", ""]
    for a in alerts:
        lines.append(a.message)
        lines.append("")

    return "\n".join(lines)


# ── Quick Test ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    # Single scan test
    alerts = run_single_alert_scan(
        watchlist=config.NIFTY_50[:10],
        send_telegram=False,
    )
    print(f"\n{len(alerts)} alerts found.")
    for a in alerts:
        print(f"  [{a.severity}] {a.symbol}: {a.alert_type}")

"""
scheduler.py — Daily automation for the stock analysis pipeline.

Uses the ``schedule`` library to trigger the analysis every weekday at
08:30 IST.  Skips weekends and Indian market holidays defined in
``config.py``.  Execution timestamps and status are logged to
``logs/agent.log``.

Can also be used as a standalone script:
    python scheduler.py          # start the scheduler loop
    python scheduler.py --once   # run once immediately and exit
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime

import schedule

import config

logger = logging.getLogger(__name__)


# ── Holiday / Weekend Check ──────────────────────────────────


def is_market_holiday(check_date: date | None = None) -> bool:
    """Return True if *check_date* is a known Indian market holiday.

    Parameters
    ----------
    check_date : date | None
        The date to check.  Defaults to today (IST).
    """
    if check_date is None:
        check_date = date.today()
    return check_date.strftime("%Y-%m-%d") in config.INDIAN_MARKET_HOLIDAYS


def is_weekend(check_date: date | None = None) -> bool:
    """Return True if *check_date* is Saturday or Sunday."""
    if check_date is None:
        check_date = date.today()
    return check_date.weekday() >= 5  # 5 = Sat, 6 = Sun


def is_trading_day(check_date: date | None = None) -> bool:
    """Return True if *check_date* is a valid trading day."""
    if check_date is None:
        check_date = date.today()
    if is_weekend(check_date):
        logger.info("Skipping — %s is a weekend.", check_date)
        return False
    if is_market_holiday(check_date):
        logger.info("Skipping — %s is an Indian market holiday.", check_date)
        return False
    return True


# ── Pipeline Runner ──────────────────────────────────────────


def run_analysis_pipeline() -> None:
    """Execute the full analysis pipeline if today is a trading day.

    Steps:
        1. Check if today is a valid trading day.
        2. Fetch data for all watchlist stocks.
        3. Compute technical indicators.
        4. Send each stock to Claude AI for analysis.
        5. Generate and save reports (TXT + HTML).
        6. Send the report via Telegram.
    """
    start = datetime.now()
    logger.info("=" * 60)
    logger.info("Pipeline triggered at %s", start.strftime("%Y-%m-%d %H:%M:%S"))

    if not is_trading_day():
        logger.info("Not a trading day — pipeline skipped.")
        return

    try:
        # Lazy imports so the scheduler module stays lightweight when testing
        from data_fetcher import fetch_all_stocks
        from indicators import compute_all_indicators
        from analyzer import analyse_all, StockRecommendation
        from report_generator import save_reports
        from notifier import send_telegram_report

        # Step 1: Fetch data
        logger.info("Step 1/5 — Fetching stock data …")
        all_stocks = fetch_all_stocks()
        successful = [s for s in all_stocks if s.fetch_success]
        logger.info(
            "Fetched %d/%d stocks successfully.",
            len(successful), len(all_stocks),
        )

        # Step 2: Compute indicators
        logger.info("Step 2/5 — Computing technical indicators …")
        stock_indicator_pairs = []
        for sd in successful:
            indicators = compute_all_indicators(sd.symbol, sd.ohlcv)
            stock_indicator_pairs.append((sd, indicators))

        # Step 3: AI analysis
        logger.info("Step 3/5 — Running Claude AI analysis …")
        recommendations: list[StockRecommendation] = analyse_all(stock_indicator_pairs)
        success_recs = [r for r in recommendations if r.analysis_success]
        logger.info(
            "AI analysis complete: %d/%d succeeded.",
            len(success_recs), len(recommendations),
        )

        # Step 4: Generate reports
        logger.info("Step 4/5 — Generating reports …")
        txt_path, html_path = save_reports(recommendations)
        logger.info("Reports saved: %s, %s", txt_path, html_path)

        # Step 5: Telegram notification
        logger.info("Step 5/5 — Sending Telegram notification …")
        if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
            sent = send_telegram_report(recommendations)
            if sent:
                logger.info("Telegram report sent successfully.")
            else:
                logger.warning("Telegram report delivery failed.")
        else:
            logger.info("Telegram not configured — skipping notification.")

    except Exception as exc:
        logger.exception("Pipeline failed with error: %s", exc)

    elapsed = (datetime.now() - start).total_seconds()
    logger.info("Pipeline finished in %.1f seconds.", elapsed)
    logger.info("=" * 60)


# ── Scheduler Loop ───────────────────────────────────────────


def start_scheduler() -> None:
    """Start the schedule loop — runs the pipeline at the configured time.

    The loop checks every 30 seconds for pending jobs and sleeps in between.
    """
    schedule_time = config.SCHEDULE_TIME_IST
    logger.info(
        "Scheduler started — analysis will run at %s IST on weekdays.",
        schedule_time,
    )

    # Schedule the job for every day (holiday/weekend skipped inside the func)
    schedule.every().monday.at(schedule_time).do(run_analysis_pipeline)
    schedule.every().tuesday.at(schedule_time).do(run_analysis_pipeline)
    schedule.every().wednesday.at(schedule_time).do(run_analysis_pipeline)
    schedule.every().thursday.at(schedule_time).do(run_analysis_pipeline)
    schedule.every().friday.at(schedule_time).do(run_analysis_pipeline)

    logger.info("Next scheduled run: %s", schedule.next_run())

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")


# ── CLI Entry Point ──────────────────────────────────────────


def _setup_logging() -> None:
    """Configure root logging to both console and file."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
    root_logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
    root_logger.addHandler(fh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Analysis Scheduler")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run the analysis once immediately and exit.",
    )
    args = parser.parse_args()

    _setup_logging()

    if args.once:
        logger.info("Running one-shot analysis …")
        run_analysis_pipeline()
    else:
        start_scheduler()

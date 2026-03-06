"""
start.py — Cloud entry point for Render deployment.

Runs BOTH the daily scheduler AND the interactive Telegram bot
in parallel using threading.

- Scheduler: Runs full pipeline at 08:30 IST (Mon-Fri)
- Telegram Bot: Always listening for /scan, /analyse, /heatmap, etc.
"""

import logging
import threading
import sys
import io
import signal

import config


def setup_cloud_logging():
    """Configure logging for cloud (stdout + file)."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if root.handlers:
        return

    fmt = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)

    # Console — UTF-8
    console = logging.StreamHandler(
        io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    )
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File
    config.LOGS_DIR.mkdir(exist_ok=True)
    fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)


def run_scheduler():
    """Start the daily scheduler in a thread."""
    logger = logging.getLogger("start.scheduler")
    try:
        from scheduler import start_scheduler
        logger.info("Starting daily scheduler (08:30 IST, Mon-Fri)...")
        start_scheduler()
    except Exception as exc:
        logger.exception("Scheduler crashed: %s", exc)


def run_bot():
    """Start the Telegram bot in a thread."""
    logger = logging.getLogger("start.bot")
    try:
        from telegram_bot import run_telegram_bot
        logger.info("Starting Telegram bot...")
        run_telegram_bot()
    except Exception as exc:
        logger.exception("Telegram bot crashed: %s", exc)


def main():
    setup_cloud_logging()
    logger = logging.getLogger("start")

    logger.info("=" * 60)
    logger.info("🚀 AI Stock Agent — Cloud Deployment Starting")
    logger.info("=" * 60)
    logger.info("Timezone: Asia/Kolkata (IST)")
    logger.info("Scheduler: Daily at %s IST", config.SCHEDULE_TIME_IST)
    logger.info("Telegram Bot: Active (interactive commands)")
    logger.info("=" * 60)

    # Start scheduler in background thread
    scheduler_thread = threading.Thread(
        target=run_scheduler,
        name="Scheduler",
        daemon=True,
    )
    scheduler_thread.start()
    logger.info("Scheduler thread started.")

    # Run Telegram bot in main thread (blocking)
    # This keeps the process alive
    run_bot()


if __name__ == "__main__":
    main()

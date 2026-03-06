"""
config.py — Central configuration for the Stock Market Analysis Agent.

All tuneable parameters, watchlists, and path settings live here.
Sensitive credentials are loaded from the .env file at runtime.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ────────────────────────────────────────────────
load_dotenv()

# ── Directory Paths ──────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent
REPORTS_DIR: Path = BASE_DIR / "reports"
LOGS_DIR: Path = BASE_DIR / "logs"
TEMPLATES_DIR: Path = BASE_DIR / "templates"

# Create directories if they don't exist
REPORTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# ── API Keys (loaded from .env) ──────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Email Settings (Gmail SMTP) ──────────────────────────────
EMAIL_SENDER: str = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD: str = os.getenv("EMAIL_PASSWORD", "")    # Gmail App Password
EMAIL_RECEIVER: str = os.getenv("EMAIL_RECEIVER", "")    # comma-separated for multi

# ── Groq AI Settings ─────────────────────────────────────────
GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
GROQ_MODEL: str = "llama-3.3-70b-versatile"   # Free, fast Groq model
GROQ_MAX_TOKENS: int = 4096
GROQ_TEMPERATURE: float = 0.3              # lower → more deterministic
GROQ_MAX_RETRIES: int = 3                  # retry on transient API errors
GROQ_RETRY_DELAY: float = 2.0              # seconds between retries

# ── Stock Watchlist (NSE symbols) ────────────────────────────
# Add .NS suffix for NSE, .BO suffix for BSE when querying yfinance.

# Top 30 Nifty stocks (optimised for Groq 100K daily token limit)
NIFTY_50: list[str] = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "BHARTIARTL", "HINDUNILVR", "ITC", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "BAJAJFINSV", "MARUTI",
    "HCLTECH", "WIPRO", "TITAN", "SUNPHARMA", "NTPC",
    "TATASTEEL", "POWERGRID", "ONGC", "TECHM", "ADANIENT",
    "ADANIPORTS", "ASIANPAINT", "DRREDDY", "JSWSTEEL", "ULTRACEMCO",
]

# Nifty Next 50 (Midcap leaders)
NIFTY_NEXT50: list[str] = [
    "BANKBARODA", "VEDL", "TATAPOWER", "ZOMATO", "HAL",
    "IRCTC", "PIDILITIND", "SIEMENS", "ABB", "AMBUJACEM",
    "DLF", "GODREJCP", "ICICIPRULI", "INDIGO", "IOC",
    "MCDOWELL-N", "NAUKRI", "PNB", "SHREECEM", "TORNTPHARM",
    "TRENT", "PERSISTENT", "POLYCAB", "CANBK", "COLPAL",
    "DMART", "GAIL", "HAVELLS", "JINDALSTEL", "MARICO",
]

# High momentum midcap gems (potential multibaggers)
MIDCAP_GEMS: list[str] = [
    "RVNL", "IRFC", "SUZLON", "NHPC", "BEL",
    "RECLTD", "PFC", "SAIL", "NATIONALUM", "BHEL",
    "IDFCFIRSTB", "IDEA", "YESBANK", "TIINDIA", "DIXON",
    "ASTRAL", "DEEPAKNTR", "TATAELXSI", "MPHASIS", "COFORGE",
]

# Default watchlist (use Nifty 50 for daily analysis)
WATCHLIST: list[str] = NIFTY_50.copy()

# Full universe for screening (Nifty 50 + Next 50 + Midcaps)
FULL_UNIVERSE: list[str] = NIFTY_50 + NIFTY_NEXT50 + MIDCAP_GEMS

# Exchange suffix used when building the yfinance ticker
EXCHANGE_SUFFIX: str = ".NS"   # ".NS" for NSE, ".BO" for BSE

# ── Data Fetching Settings ───────────────────────────────────
DATA_LOOKBACK_DAYS: int = 30             # days of OHLCV history to fetch
YFINANCE_MAX_RETRIES: int = 3            # retry failed yfinance requests
YFINANCE_RETRY_DELAY: float = 1.5        # seconds between retries

# ── Technical Indicator Thresholds ───────────────────────────
RSI_PERIOD: int = 14
RSI_OVERBOUGHT: float = 70.0
RSI_OVERSOLD: float = 30.0

MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9

BOLLINGER_PERIOD: int = 20
BOLLINGER_STD_DEV: float = 2.0

EMA_SHORT: int = 9
EMA_LONG: int = 21

VOLUME_AVG_PERIOD: int = 20              # 20-day average volume comparison

# ── Advanced Indicator Settings ──────────────────────────────
# Supertrend
SUPERTREND_PERIOD: int = 10
SUPERTREND_MULTIPLIER: float = 3.0

# Fibonacci retracement levels
FIBONACCI_LEVELS: list[float] = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

# Multi-timeframe
DATA_LOOKBACK_WEEKLY: int = 180          # ~6 months of weekly data

# ── Portfolio & Risk Management Settings ─────────────────────
PORTFOLIO_CAPITAL: float = float(os.getenv("PORTFOLIO_CAPITAL", "100000"))  # ₹1 Lakh default
MAX_RISK_PER_TRADE: float = 0.02        # 2% of capital per trade
MAX_PORTFOLIO_RISK: float = 0.06        # 6% total portfolio risk
MAX_POSITION_SIZE: float = 0.20         # Max 20% of capital in one stock
MAX_SECTOR_EXPOSURE: float = 0.35       # Max 35% in one sector
MAX_CORRELATED_STOCKS: int = 3          # Max stocks from same sector

# ── Alert Settings ───────────────────────────────────────────
ALERT_CHECK_INTERVAL: int = 60          # seconds between alert checks
VOLUME_SPIKE_THRESHOLD: float = 2.5     # 2.5x avg volume = spike
PRICE_BREAKOUT_PCT: float = 2.0         # 2% above resistance = breakout
PRICE_BREAKDOWN_PCT: float = 2.0        # 2% below support = breakdown
RSI_EXTREME_HIGH: float = 80.0          # extreme overbought alert
RSI_EXTREME_LOW: float = 20.0           # extreme oversold alert

# ── Backtest Settings ────────────────────────────────────────
BACKTEST_DAYS: int = 365                # 1 year of historical data
BACKTEST_INITIAL_CAPITAL: float = 100000.0
BACKTEST_COMMISSION_PCT: float = 0.05   # brokerage + taxes ~0.05%

# ── Report / Recommendation Settings ─────────────────────────
MIN_CONFIDENCE_SCORE: int = 65           # minimum to qualify as a recommendation
MAX_BUY_PICKS: int = 5                   # top N buy picks in report
MAX_INTRADAY_PICKS: int = 5              # top N intraday picks
REPORT_FORMAT_DATE: str = "%d %B %Y"     # e.g. "06 March 2026"

# ── Scheduler Settings ───────────────────────────────────────
# Time in IST (Indian Standard Time) when the analysis should run
SCHEDULE_TIME_IST: str = "08:30"

# Indian market holidays for 2025-2026  (date strings: "YYYY-MM-DD")
INDIAN_MARKET_HOLIDAYS: list[str] = [
    # 2025
    "2025-01-26",   # Republic Day
    "2025-02-26",   # Maha Shivaratri
    "2025-03-14",   # Holi
    "2025-03-31",   # Id-Ul-Fitr (Eid)
    "2025-04-10",   # Shri Mahavir Jayanti
    "2025-04-14",   # Dr. Ambedkar Jayanti
    "2025-04-18",   # Good Friday
    "2025-05-01",   # Maharashtra Day
    "2025-06-07",   # Bakri Id (Eid ul-Adha)
    "2025-07-06",   # Muharram
    "2025-08-15",   # Independence Day
    "2025-08-16",   # Janmashtami
    "2025-09-05",   # Milad-un-Nabi
    "2025-10-02",   # Mahatma Gandhi Jayanti
    "2025-10-20",   # Diwali (Lakshmi Puja)
    "2025-10-21",   # Diwali Balipratipada
    "2025-11-05",   # Guru Nanak Jayanti
    "2025-12-25",   # Christmas
    # 2026
    "2026-01-26",   # Republic Day
    "2026-02-17",   # Maha Shivaratri
    "2026-03-03",   # Holi
    "2026-03-20",   # Id-Ul-Fitr (Eid)
    "2026-03-25",   # Shri Mahavir Jayanti
    "2026-04-03",   # Good Friday
    "2026-04-14",   # Dr. Ambedkar Jayanti
    "2026-05-01",   # Maharashtra Day
    "2026-05-27",   # Bakri Id (Eid ul-Adha)
    "2026-06-26",   # Muharram
    "2026-08-15",   # Independence Day
    "2026-08-25",   # Milad-un-Nabi
    "2026-09-04",   # Janmashtami
    "2026-10-02",   # Mahatma Gandhi Jayanti
    "2026-10-09",   # Diwali (Lakshmi Puja)
    "2026-10-10",   # Diwali Balipratipada
    "2026-10-26",   # Guru Nanak Jayanti
    "2026-12-25",   # Christmas
]

# ── Logging ──────────────────────────────────────────────────
LOG_FILE: Path = LOGS_DIR / "agent.log"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

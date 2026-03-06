"""
data_fetcher.py — Fetch stock market data from Yahoo Finance for NSE/BSE stocks.

Uses the yfinance library to pull OHLCV history and fundamental data.
All network calls include retry logic and graceful error handling so that
one bad ticker does not crash the entire pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)


# ── Data Models ──────────────────────────────────────────────


@dataclass
class StockFundamentals:
    """Key fundamental data points for a stock."""

    symbol: str
    current_price: Optional[float] = None
    pe_ratio: Optional[float] = None
    market_cap: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    avg_volume: Optional[int] = None
    sector: Optional[str] = None
    industry: Optional[str] = None


@dataclass
class StockData:
    """Complete data bundle for a single stock, ready for analysis."""

    symbol: str
    fundamentals: StockFundamentals
    ohlcv: pd.DataFrame = field(default_factory=pd.DataFrame)
    fetch_success: bool = True
    error_message: str = ""


# ── Helper Utilities ─────────────────────────────────────────


def _build_ticker(symbol: str) -> str:
    """Append the exchange suffix to a plain symbol name.

    Examples
    --------
    >>> _build_ticker("RELIANCE")
    'RELIANCE.NS'
    """
    return f"{symbol}{config.EXCHANGE_SUFFIX}"


def _retry(func, *args, max_retries: int = config.YFINANCE_MAX_RETRIES,
           delay: float = config.YFINANCE_RETRY_DELAY, **kwargs) -> Any:
    """Call *func* with retries on failure.

    Parameters
    ----------
    func : callable
        The function to call.
    max_retries : int
        Maximum number of attempts.
    delay : float
        Seconds to wait between retries.

    Returns
    -------
    Any
        The return value of *func*.

    Raises
    ------
    Exception
        Re-raises the last exception after all retries are exhausted.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Attempt %d/%d for %s failed: %s",
                attempt, max_retries, func.__name__, exc,
            )
            if attempt < max_retries:
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ── Core Fetching Functions ──────────────────────────────────


def fetch_ohlcv(symbol: str, period_days: int = config.DATA_LOOKBACK_DAYS) -> pd.DataFrame:
    """Download OHLCV data for the given NSE/BSE symbol.

    Parameters
    ----------
    symbol : str
        Plain symbol (e.g. ``"RELIANCE"``).  Exchange suffix is added
        automatically.
    period_days : int
        Number of calendar days of history to retrieve.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Open, High, Low, Close, Volume``
        indexed by date.
    """
    ticker_str = _build_ticker(symbol)
    logger.info("Fetching OHLCV for %s (last %d days)…", ticker_str, period_days)

    def _download() -> pd.DataFrame:
        ticker = yf.Ticker(ticker_str)
        df = ticker.history(period=f"{period_days}d")
        if df.empty:
            raise ValueError(f"No OHLCV data returned for {ticker_str}")
        return df

    df = _retry(_download)

    # Keep only essential columns and normalise names
    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in expected_cols:
        if col not in df.columns:
            logger.warning("Column %s missing for %s", col, ticker_str)
    df = df[[c for c in expected_cols if c in df.columns]].copy()
    logger.info(
        "Fetched %d rows of OHLCV for %s [%s → %s]",
        len(df), ticker_str,
        df.index.min().strftime("%Y-%m-%d") if len(df) else "N/A",
        df.index.max().strftime("%Y-%m-%d") if len(df) else "N/A",
    )
    return df


def fetch_fundamentals(symbol: str) -> StockFundamentals:
    """Retrieve key fundamental metrics for a stock.

    Parameters
    ----------
    symbol : str
        Plain symbol name (e.g. ``"TCS"``).

    Returns
    -------
    StockFundamentals
        Dataclass with PE ratio, market cap, 52-week range, etc.
    """
    ticker_str = _build_ticker(symbol)
    logger.info("Fetching fundamentals for %s…", ticker_str)

    def _get_info() -> dict:
        ticker = yf.Ticker(ticker_str)
        info = ticker.info
        if not info:
            raise ValueError(f"No info returned for {ticker_str}")
        return info

    info: dict = _retry(_get_info)

    fundamentals = StockFundamentals(
        symbol=symbol,
        current_price=info.get("currentPrice") or info.get("regularMarketPrice"),
        pe_ratio=info.get("trailingPE") or info.get("forwardPE"),
        market_cap=info.get("marketCap"),
        fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
        fifty_two_week_low=info.get("fiftyTwoWeekLow"),
        avg_volume=info.get("averageVolume"),
        sector=info.get("sector"),
        industry=info.get("industry"),
    )
    logger.info(
        "Fundamentals for %s: price=%.2f, PE=%.2f, mktcap=%s",
        symbol,
        fundamentals.current_price or 0,
        fundamentals.pe_ratio or 0,
        _format_market_cap(fundamentals.market_cap),
    )
    return fundamentals


def _format_market_cap(market_cap: Optional[float]) -> str:
    """Human-readable market cap string (in ₹ Crores)."""
    if market_cap is None:
        return "N/A"
    crores = market_cap / 1e7
    if crores >= 1e5:
        return f"₹{crores / 1e5:.2f} Lakh Cr"
    elif crores >= 1e3:
        return f"₹{crores / 1e3:.2f}K Cr"
    return f"₹{crores:.0f} Cr"


# ── Batch Fetching ───────────────────────────────────────────


def fetch_stock_data(symbol: str) -> StockData:
    """Fetch all data for a single stock (OHLCV + fundamentals).

    Errors are caught and recorded in :pyattr:`StockData.error_message`
    so one bad symbol doesn't crash the full pipeline.

    Parameters
    ----------
    symbol : str
        Plain NSE/BSE symbol.

    Returns
    -------
    StockData
        Complete data bundle.
    """
    fundamentals = StockFundamentals(symbol=symbol)
    ohlcv = pd.DataFrame()
    try:
        ohlcv = fetch_ohlcv(symbol)
        fundamentals = fetch_fundamentals(symbol)
        return StockData(
            symbol=symbol,
            fundamentals=fundamentals,
            ohlcv=ohlcv,
            fetch_success=True,
        )
    except Exception as exc:
        logger.error("Failed to fetch data for %s: %s", symbol, exc)
        return StockData(
            symbol=symbol,
            fundamentals=fundamentals,
            ohlcv=ohlcv,
            fetch_success=False,
            error_message=str(exc),
        )


def fetch_all_stocks(watchlist: Optional[list[str]] = None) -> list[StockData]:
    """Fetch data for every symbol in the watchlist.

    Parameters
    ----------
    watchlist : list[str] | None
        Symbols to fetch.  Defaults to :pydata:`config.WATCHLIST`.

    Returns
    -------
    list[StockData]
        One entry per symbol (successful or not).
    """
    if watchlist is None:
        watchlist = config.WATCHLIST

    logger.info("=== Starting batch fetch for %d stocks ===", len(watchlist))
    results: list[StockData] = []
    for idx, symbol in enumerate(watchlist, start=1):
        logger.info("[%d/%d] Processing %s …", idx, len(watchlist), symbol)
        stock_data = fetch_stock_data(symbol)
        results.append(stock_data)

    success = sum(1 for s in results if s.fetch_success)
    failed = len(results) - success
    logger.info(
        "=== Batch fetch complete: %d succeeded, %d failed ===",
        success, failed,
    )
    return results


# ── Quick Test ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    # Quick smoke test with a single stock
    data = fetch_stock_data("RELIANCE")
    if data.fetch_success:
        print(f"\n✅ {data.symbol}")
        print(f"   Price   : ₹{data.fundamentals.current_price}")
        print(f"   PE      : {data.fundamentals.pe_ratio}")
        print(f"   52W H/L : ₹{data.fundamentals.fifty_two_week_high} / ₹{data.fundamentals.fifty_two_week_low}")
        print(f"   Rows    : {len(data.ohlcv)}")
    else:
        print(f"\n❌ {data.symbol}: {data.error_message}")

"""
indicators.py — Calculate technical indicators for stock analysis.

Uses pure pandas and numpy to compute RSI, MACD, Bollinger Bands, EMAs,
volume analysis, and support/resistance levels.  No external TA library
required — everything is calculated from first principles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ── Data Model ───────────────────────────────────────────────


@dataclass
class TechnicalSummary:
    """Holds computed indicators in a convenient, JSON-serialisable form."""

    symbol: str = ""

    # RSI
    rsi: Optional[float] = None
    rsi_signal: str = "NEUTRAL"          # OVERBOUGHT / OVERSOLD / NEUTRAL

    # MACD
    macd_line: Optional[float] = None
    macd_signal_line: Optional[float] = None
    macd_histogram: Optional[float] = None
    macd_crossover: str = "NEUTRAL"      # BULLISH / BEARISH / NEUTRAL

    # Bollinger Bands
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_signal: str = "NEUTRAL"           # ABOVE_UPPER / BELOW_LOWER / WITHIN

    # EMAs
    ema_short: Optional[float] = None    # EMA 9
    ema_long: Optional[float] = None     # EMA 21
    ema_crossover: str = "NEUTRAL"       # BULLISH / BEARISH / NEUTRAL

    # Volume
    current_volume: Optional[int] = None
    avg_volume_20d: Optional[float] = None
    volume_ratio: Optional[float] = None  # current / avg
    volume_signal: str = "NORMAL"         # HIGH / LOW / NORMAL

    # Support & Resistance
    support: Optional[float] = None
    resistance: Optional[float] = None

    # VWAP (Volume Weighted Average Price)
    vwap: Optional[float] = None
    vwap_signal: str = "NEUTRAL"         # ABOVE_VWAP / BELOW_VWAP / AT_VWAP

    # Supertrend
    supertrend: Optional[float] = None
    supertrend_direction: str = "NEUTRAL"  # BULLISH / BEARISH / NEUTRAL

    # Fibonacci Retracement
    fib_levels: Optional[dict] = None    # {0.236: price, 0.382: price, ...}
    fib_signal: str = "NEUTRAL"          # AT_SUPPORT / AT_RESISTANCE / BETWEEN
    nearest_fib: Optional[str] = None    # e.g. "0.382 (₹1450.00)"

    # Trend Strength
    trend_strength: str = "NEUTRAL"      # STRONG_UP / UP / NEUTRAL / DOWN / STRONG_DOWN
    trend_score: int = 0                 # -100 to +100

    # Last close price (handy for context)
    last_close: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to a plain dict for JSON serialisation."""
        return {
            "symbol": self.symbol,
            "rsi": round(self.rsi, 2) if self.rsi is not None else None,
            "rsi_signal": self.rsi_signal,
            "macd_line": round(self.macd_line, 4) if self.macd_line is not None else None,
            "macd_signal_line": round(self.macd_signal_line, 4) if self.macd_signal_line is not None else None,
            "macd_histogram": round(self.macd_histogram, 4) if self.macd_histogram is not None else None,
            "macd_crossover": self.macd_crossover,
            "bb_upper": round(self.bb_upper, 2) if self.bb_upper is not None else None,
            "bb_middle": round(self.bb_middle, 2) if self.bb_middle is not None else None,
            "bb_lower": round(self.bb_lower, 2) if self.bb_lower is not None else None,
            "bb_signal": self.bb_signal,
            "ema_short": round(self.ema_short, 2) if self.ema_short is not None else None,
            "ema_long": round(self.ema_long, 2) if self.ema_long is not None else None,
            "ema_crossover": self.ema_crossover,
            "current_volume": self.current_volume,
            "avg_volume_20d": round(self.avg_volume_20d, 0) if self.avg_volume_20d is not None else None,
            "volume_ratio": round(self.volume_ratio, 2) if self.volume_ratio is not None else None,
            "volume_signal": self.volume_signal,
            "support": round(self.support, 2) if self.support is not None else None,
            "resistance": round(self.resistance, 2) if self.resistance is not None else None,
            "vwap": round(self.vwap, 2) if self.vwap is not None else None,
            "vwap_signal": self.vwap_signal,
            "supertrend": round(self.supertrend, 2) if self.supertrend is not None else None,
            "supertrend_direction": self.supertrend_direction,
            "fib_levels": self.fib_levels,
            "fib_signal": self.fib_signal,
            "nearest_fib": self.nearest_fib,
            "trend_strength": self.trend_strength,
            "trend_score": self.trend_score,
            "last_close": round(self.last_close, 2) if self.last_close is not None else None,
        }


# ── Pure Pandas/Numpy Indicator Helpers ──────────────────────


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


# ── Individual Indicator Calculations ────────────────────────


def calculate_rsi(df: pd.DataFrame, period: int = config.RSI_PERIOD) -> tuple[Optional[float], str]:
    """Compute the latest RSI value and classify it.

    RSI = 100 - (100 / (1 + RS))  where RS = avg_gain / avg_loss

    Returns
    -------
    tuple[float | None, str]
        (RSI value, signal string).
    """
    try:
        close = df["Close"]
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi_value = float(rsi.dropna().iloc[-1])

        if rsi_value >= config.RSI_OVERBOUGHT:
            signal = "OVERBOUGHT"
        elif rsi_value <= config.RSI_OVERSOLD:
            signal = "OVERSOLD"
        else:
            signal = "NEUTRAL"
        return rsi_value, signal
    except Exception as exc:
        logger.warning("RSI calculation failed: %s", exc)
        return None, "NEUTRAL"


def calculate_macd(
    df: pd.DataFrame,
    fast: int = config.MACD_FAST,
    slow: int = config.MACD_SLOW,
    signal: int = config.MACD_SIGNAL,
) -> tuple[Optional[float], Optional[float], Optional[float], str]:
    """Compute MACD line, signal line, histogram, and crossover status.

    MACD Line    = EMA(fast) - EMA(slow)
    Signal Line  = EMA(MACD Line, signal)
    Histogram    = MACD Line - Signal Line
    """
    try:
        close = df["Close"]
        ema_fast = _ema(close, fast)
        ema_slow = _ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = _ema(macd_line, signal)
        histogram = macd_line - signal_line

        macd_val = float(macd_line.dropna().iloc[-1])
        sig_val = float(signal_line.dropna().iloc[-1])
        hist_val = float(histogram.dropna().iloc[-1])

        # Detect crossover using last two rows
        if len(macd_line.dropna()) >= 2 and len(signal_line.dropna()) >= 2:
            prev_macd = float(macd_line.dropna().iloc[-2])
            prev_signal = float(signal_line.dropna().iloc[-2])
            if prev_macd <= prev_signal and macd_val > sig_val:
                crossover = "BULLISH"
            elif prev_macd >= prev_signal and macd_val < sig_val:
                crossover = "BEARISH"
            else:
                crossover = "BULLISH" if macd_val > sig_val else "BEARISH"
        else:
            crossover = "NEUTRAL"

        return macd_val, sig_val, hist_val, crossover
    except Exception as exc:
        logger.warning("MACD calculation failed: %s", exc)
        return None, None, None, "NEUTRAL"


def calculate_bollinger_bands(
    df: pd.DataFrame,
    period: int = config.BOLLINGER_PERIOD,
    std_dev: float = config.BOLLINGER_STD_DEV,
) -> tuple[Optional[float], Optional[float], Optional[float], str]:
    """Compute Bollinger Bands and determine if price is outside bands.

    Middle = SMA(period)
    Upper  = Middle + std_dev * StdDev(period)
    Lower  = Middle - std_dev * StdDev(period)
    """
    try:
        close = df["Close"]
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std

        upper_val = float(upper.dropna().iloc[-1])
        middle_val = float(middle.dropna().iloc[-1])
        lower_val = float(lower.dropna().iloc[-1])

        last_close = float(close.iloc[-1])
        if last_close > upper_val:
            bb_signal = "ABOVE_UPPER"
        elif last_close < lower_val:
            bb_signal = "BELOW_LOWER"
        else:
            bb_signal = "WITHIN"

        return upper_val, middle_val, lower_val, bb_signal
    except Exception as exc:
        logger.warning("Bollinger Bands calculation failed: %s", exc)
        return None, None, None, "NEUTRAL"


def calculate_ema(
    df: pd.DataFrame,
    short: int = config.EMA_SHORT,
    long: int = config.EMA_LONG,
) -> tuple[Optional[float], Optional[float], str]:
    """Compute EMA-short and EMA-long, and detect crossover."""
    try:
        close = df["Close"]
        ema_s = _ema(close, short)
        ema_l = _ema(close, long)

        ema_s_val = float(ema_s.dropna().iloc[-1])
        ema_l_val = float(ema_l.dropna().iloc[-1])

        # Crossover detection
        if len(ema_s.dropna()) >= 2 and len(ema_l.dropna()) >= 2:
            prev_s = float(ema_s.dropna().iloc[-2])
            prev_l = float(ema_l.dropna().iloc[-2])
            if prev_s <= prev_l and ema_s_val > ema_l_val:
                crossover = "BULLISH"
            elif prev_s >= prev_l and ema_s_val < ema_l_val:
                crossover = "BEARISH"
            else:
                crossover = "BULLISH" if ema_s_val > ema_l_val else "BEARISH"
        else:
            crossover = "NEUTRAL"

        return ema_s_val, ema_l_val, crossover
    except Exception as exc:
        logger.warning("EMA calculation failed: %s", exc)
        return None, None, "NEUTRAL"


def calculate_volume_analysis(
    df: pd.DataFrame,
    avg_period: int = config.VOLUME_AVG_PERIOD,
) -> tuple[Optional[int], Optional[float], Optional[float], str]:
    """Compare latest volume with the 20-day average."""
    try:
        if "Volume" not in df.columns or df["Volume"].dropna().empty:
            return None, None, None, "NORMAL"

        current_vol = int(df["Volume"].iloc[-1])
        avg_vol = float(df["Volume"].rolling(window=avg_period).mean().dropna().iloc[-1])
        ratio = current_vol / avg_vol if avg_vol > 0 else 0.0

        if ratio >= 1.5:
            signal = "HIGH"
        elif ratio <= 0.5:
            signal = "LOW"
        else:
            signal = "NORMAL"

        return current_vol, avg_vol, ratio, signal
    except Exception as exc:
        logger.warning("Volume analysis failed: %s", exc)
        return None, None, None, "NORMAL"


def calculate_support_resistance(
    df: pd.DataFrame,
    period: int = config.BOLLINGER_PERIOD,
) -> tuple[Optional[float], Optional[float]]:
    """Simple support/resistance as the low/high over the lookback window."""
    try:
        recent = df.tail(period)
        support = float(recent["Low"].min())
        resistance = float(recent["High"].max())
        return support, resistance
    except Exception as exc:
        logger.warning("Support/Resistance calculation failed: %s", exc)
        return None, None


def calculate_vwap(df: pd.DataFrame) -> tuple[Optional[float], str]:
    """Volume Weighted Average Price — institutional benchmark.

    VWAP = Σ(Typical Price × Volume) / Σ(Volume)
    Typical Price = (High + Low + Close) / 3

    Price above VWAP = bullish (institutions buying)
    Price below VWAP = bearish (institutions selling)
    """
    try:
        if "Volume" not in df.columns or df["Volume"].sum() == 0:
            return None, "NEUTRAL"

        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
        vwap_series = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()
        vwap_val = float(vwap_series.dropna().iloc[-1])
        last_close = float(df["Close"].iloc[-1])

        pct_diff = ((last_close - vwap_val) / vwap_val) * 100
        if pct_diff > 0.5:
            signal = "ABOVE_VWAP"
        elif pct_diff < -0.5:
            signal = "BELOW_VWAP"
        else:
            signal = "AT_VWAP"

        return vwap_val, signal
    except Exception as exc:
        logger.warning("VWAP calculation failed: %s", exc)
        return None, "NEUTRAL"


def calculate_supertrend(
    df: pd.DataFrame,
    period: int = config.SUPERTREND_PERIOD,
    multiplier: float = config.SUPERTREND_MULTIPLIER,
) -> tuple[Optional[float], str]:
    """Supertrend indicator — popular trend-following tool.

    Based on ATR (Average True Range):
    Upper Band = (High + Low)/2 + Multiplier × ATR
    Lower Band = (High + Low)/2 - Multiplier × ATR

    Returns
    -------
    tuple[float | None, str]
        (Supertrend value, direction: BULLISH/BEARISH/NEUTRAL)
    """
    try:
        if len(df) < period + 1:
            return None, "NEUTRAL"

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # Calculate ATR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        hl2 = (high + low) / 2.0
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=float)

        # Initialize
        supertrend.iloc[period] = upper_band.iloc[period]
        direction.iloc[period] = -1  # Start bearish

        for i in range(period + 1, len(df)):
            if close.iloc[i] > upper_band.iloc[i - 1]:
                direction.iloc[i] = 1  # Bullish
            elif close.iloc[i] < lower_band.iloc[i - 1]:
                direction.iloc[i] = -1  # Bearish
            else:
                direction.iloc[i] = direction.iloc[i - 1]

            if direction.iloc[i] == 1:
                supertrend.iloc[i] = max(lower_band.iloc[i],
                                          supertrend.iloc[i - 1] if direction.iloc[i - 1] == 1 else lower_band.iloc[i])
            else:
                supertrend.iloc[i] = min(upper_band.iloc[i],
                                          supertrend.iloc[i - 1] if direction.iloc[i - 1] == -1 else upper_band.iloc[i])

        st_val = float(supertrend.dropna().iloc[-1])
        dir_val = float(direction.dropna().iloc[-1])
        dir_str = "BULLISH" if dir_val == 1 else "BEARISH"

        return st_val, dir_str
    except Exception as exc:
        logger.warning("Supertrend calculation failed: %s", exc)
        return None, "NEUTRAL"


def calculate_fibonacci(
    df: pd.DataFrame,
    lookback: int = 50,
) -> tuple[Optional[dict], str, Optional[str]]:
    """Fibonacci retracement levels from recent swing high/low.

    Finds the highest and lowest points in the lookback period,
    then calculates standard Fibonacci retracement levels.

    Returns
    -------
    tuple[dict | None, str, str | None]
        (fib_levels dict, signal, nearest_fib description)
    """
    try:
        recent = df.tail(min(lookback, len(df)))
        swing_high = float(recent["High"].max())
        swing_low = float(recent["Low"].min())
        last_close = float(df["Close"].iloc[-1])

        diff = swing_high - swing_low
        if diff <= 0:
            return None, "NEUTRAL", None

        fib_levels = {}
        for level in config.FIBONACCI_LEVELS:
            # Retracement from high
            fib_price = swing_high - diff * level
            fib_levels[str(level)] = round(fib_price, 2)

        # Find nearest Fibonacci level
        nearest_dist = float("inf")
        nearest_label = ""
        nearest_price = 0.0
        for level_str, price in fib_levels.items():
            dist = abs(last_close - price)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_label = level_str
                nearest_price = price

        # Determine signal
        pct_from_nearest = (nearest_dist / last_close) * 100
        if pct_from_nearest < 1.5:
            # Close to a fib level
            if float(nearest_label) >= 0.618:
                signal = "AT_SUPPORT"
            elif float(nearest_label) <= 0.236:
                signal = "AT_RESISTANCE"
            else:
                signal = "BETWEEN"
        else:
            signal = "BETWEEN"

        nearest_desc = f"{nearest_label} (₹{nearest_price:.2f})"
        return fib_levels, signal, nearest_desc

    except Exception as exc:
        logger.warning("Fibonacci calculation failed: %s", exc)
        return None, "NEUTRAL", None


def calculate_trend_strength(summary: TechnicalSummary) -> tuple[str, int]:
    """Calculate overall trend strength score from all indicators.

    Combines RSI, MACD, EMA, Bollinger, VWAP, Supertrend signals into
    a single score from -100 (strong bearish) to +100 (strong bullish).

    Returns
    -------
    tuple[str, int]
        (trend label, score)
    """
    score = 0

    # RSI contribution (-20 to +20)
    if summary.rsi is not None:
        if summary.rsi >= 70:
            score += 15  # Overbought = still has momentum
        elif summary.rsi >= 55:
            score += 20
        elif summary.rsi >= 45:
            score += 0
        elif summary.rsi >= 30:
            score -= 15
        else:
            score -= 20  # Oversold

    # MACD contribution (-15 to +15)
    if summary.macd_crossover == "BULLISH":
        score += 15
    elif summary.macd_crossover == "BEARISH":
        score -= 15

    # EMA contribution (-15 to +15)
    if summary.ema_crossover == "BULLISH":
        score += 15
    elif summary.ema_crossover == "BEARISH":
        score -= 15

    # Bollinger contribution (-10 to +10)
    if summary.bb_signal == "BELOW_LOWER":
        score -= 10  # Oversold
    elif summary.bb_signal == "ABOVE_UPPER":
        score += 10  # Strong momentum

    # VWAP contribution (-10 to +10)
    if summary.vwap_signal == "ABOVE_VWAP":
        score += 10
    elif summary.vwap_signal == "BELOW_VWAP":
        score -= 10

    # Supertrend contribution (-15 to +15)
    if summary.supertrend_direction == "BULLISH":
        score += 15
    elif summary.supertrend_direction == "BEARISH":
        score -= 15

    # Volume confirmation (-5 to +10)
    if summary.volume_signal == "HIGH":
        # High volume confirms the direction
        score += 10 if score > 0 else -5
    elif summary.volume_signal == "LOW":
        score = int(score * 0.7)  # Low volume weakens the signal

    # Clamp to range
    score = max(-100, min(100, score))

    if score >= 50:
        label = "STRONG_UP"
    elif score >= 20:
        label = "UP"
    elif score >= -20:
        label = "NEUTRAL"
    elif score >= -50:
        label = "DOWN"
    else:
        label = "STRONG_DOWN"

    return label, score


# ── Master Function ──────────────────────────────────────────


def compute_all_indicators(symbol: str, df: pd.DataFrame) -> TechnicalSummary:
    """Run every indicator on the given OHLCV data and return a summary.

    Parameters
    ----------
    symbol : str
        Stock symbol (used only as a label).
    df : pd.DataFrame
        OHLCV DataFrame (must have columns: Open, High, Low, Close, Volume).

    Returns
    -------
    TechnicalSummary
        All computed technical indicators in a single object.
    """
    logger.info("Computing indicators for %s (%d rows)…", symbol, len(df))

    summary = TechnicalSummary(symbol=symbol)

    if df.empty or len(df) < 5:
        logger.warning("Insufficient data for %s — skipping indicators.", symbol)
        return summary

    summary.last_close = float(df["Close"].iloc[-1])

    # RSI
    summary.rsi, summary.rsi_signal = calculate_rsi(df)

    # MACD
    (summary.macd_line, summary.macd_signal_line,
     summary.macd_histogram, summary.macd_crossover) = calculate_macd(df)

    # Bollinger Bands
    (summary.bb_upper, summary.bb_middle,
     summary.bb_lower, summary.bb_signal) = calculate_bollinger_bands(df)

    # EMAs
    summary.ema_short, summary.ema_long, summary.ema_crossover = calculate_ema(df)

    # Volume
    (summary.current_volume, summary.avg_volume_20d,
     summary.volume_ratio, summary.volume_signal) = calculate_volume_analysis(df)

    # Support & Resistance
    summary.support, summary.resistance = calculate_support_resistance(df)

    # VWAP
    summary.vwap, summary.vwap_signal = calculate_vwap(df)

    # Supertrend
    summary.supertrend, summary.supertrend_direction = calculate_supertrend(df)

    # Fibonacci Retracement
    summary.fib_levels, summary.fib_signal, summary.nearest_fib = calculate_fibonacci(df)

    # Overall Trend Strength (computed from all indicators above)
    summary.trend_strength, summary.trend_score = calculate_trend_strength(summary)

    logger.info(
        "Indicators for %s → RSI=%.1f (%s) | MACD=%s | EMA=%s | Vol=%s | VWAP=%s | ST=%s | Trend=%s(%+d)",
        symbol,
        summary.rsi or 0,
        summary.rsi_signal,
        summary.macd_crossover,
        summary.ema_crossover,
        summary.volume_signal,
        summary.vwap_signal,
        summary.supertrend_direction,
        summary.trend_strength,
        summary.trend_score,
    )
    return summary


# ── Quick Test ───────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from data_fetcher import fetch_ohlcv

    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    test_symbol = "RELIANCE"
    ohlcv = fetch_ohlcv(test_symbol)
    indicators = compute_all_indicators(test_symbol, ohlcv)
    print(json.dumps(indicators.to_dict(), indent=2))

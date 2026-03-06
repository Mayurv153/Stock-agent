"""
analyzer.py — Use Groq AI to generate stock trading recommendations.

Sends stock fundamentals + technical indicator data to Groq's API
(OpenAI-compatible, powered by Llama 3.3 70B) and asks for a structured
JSON recommendation (BUY/SELL/HOLD, entry, target, stop-loss, risk level,
confidence score, reasons).

Includes retry logic with exponential back-off for transient API errors.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI, RateLimitError, APIError

import config
from data_fetcher import StockData, StockFundamentals
from indicators import TechnicalSummary

logger = logging.getLogger(__name__)


# ── Data Model for Analysis Result ───────────────────────────


@dataclass
class StockRecommendation:
    """Structured trading recommendation returned by Groq AI."""

    symbol: str = ""
    recommendation: str = "HOLD"            # BUY / SELL / HOLD
    trade_type: str = "AVOID"               # INTRADAY / SHORT_TERM / AVOID
    entry_price: float = 0.0
    target_price: float = 0.0
    stop_loss: float = 0.0
    risk_level: str = "MEDIUM"              # LOW / MEDIUM / HIGH
    confidence_score: int = 0               # 0-100
    key_reasons: list[str] = field(default_factory=list)
    risk_warning: str = ""
    analysis_success: bool = True
    error_message: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "recommendation": self.recommendation,
            "trade_type": self.trade_type,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "risk_level": self.risk_level,
            "confidence_score": self.confidence_score,
            "key_reasons": self.key_reasons,
            "risk_warning": self.risk_warning,
        }


# ── Prompt Construction ──────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Indian stock market analyst specialising in NSE/BSE equities.
You receive stock fundamentals, technical indicators (including VWAP, Supertrend,
Fibonacci retracement, and a composite trend score) and must return a structured
JSON trading recommendation.

IMPORTANT RULES:
1. Always return ONLY valid JSON — no markdown, no code fences, no commentary.
2. Be conservative: if the signal is unclear, recommend HOLD with a lower
   confidence score.
3. entry_price should be close to the current market price.
4. target_price must be realistic (within 2-8 % for intraday, 5-15 % for short-term).
5. stop_loss should protect against downside (1-3 % for intraday, 3-7 % for short-term).
6. confidence_score is 0-100 (100 = extremely confident).
7. Always include a clear risk_warning.
8. Place extra weight on VWAP (institutional activity), Supertrend (trend direction)
   and the composite Trend Score when forming your opinion.
9. Use Fibonacci levels to fine-tune entry, target and stop-loss prices.
10. If Trend Score is ≥ +50 consider bullish bias; if ≤ -50 consider bearish bias.

Return JSON in this exact schema:
{
  "symbol": "<SYMBOL>",
  "recommendation": "BUY" | "SELL" | "HOLD",
  "trade_type": "INTRADAY" | "SHORT_TERM" | "AVOID",
  "entry_price": <float>,
  "target_price": <float>,
  "stop_loss": <float>,
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "confidence_score": <int 0-100>,
  "key_reasons": ["reason1", "reason2", "reason3"],
  "risk_warning": "<string>"
}
"""


def _build_user_prompt(
    symbol: str,
    fundamentals: StockFundamentals,
    indicators: TechnicalSummary,
) -> str:
    """Build the user message sent to Grok with all data points."""
    return f"""Analyse the following Indian stock and provide a trading recommendation.

## Stock: {symbol} (NSE)

### Fundamentals
- Current Price : ₹{fundamentals.current_price}
- P/E Ratio     : {fundamentals.pe_ratio}
- Market Cap    : {fundamentals.market_cap}
- 52-Week High  : ₹{fundamentals.fifty_two_week_high}
- 52-Week Low   : ₹{fundamentals.fifty_two_week_low}
- Avg Volume    : {fundamentals.avg_volume}
- Sector        : {fundamentals.sector}
- Industry      : {fundamentals.industry}

### Technical Indicators (Last Trading Day)
- RSI (14)          : {indicators.rsi}  → Signal: {indicators.rsi_signal}
- MACD Line         : {indicators.macd_line}
- MACD Signal       : {indicators.macd_signal_line}
- MACD Histogram    : {indicators.macd_histogram}
- MACD Crossover    : {indicators.macd_crossover}
- Bollinger Upper   : {indicators.bb_upper}
- Bollinger Middle  : {indicators.bb_middle}
- Bollinger Lower   : {indicators.bb_lower}
- Bollinger Signal  : {indicators.bb_signal}
- EMA 9             : {indicators.ema_short}
- EMA 21            : {indicators.ema_long}
- EMA Crossover     : {indicators.ema_crossover}

### Volume Analysis
- Current Volume    : {indicators.current_volume}
- 20-Day Avg Volume : {indicators.avg_volume_20d}
- Volume Ratio      : {indicators.volume_ratio}
- Volume Signal     : {indicators.volume_signal}

### Advanced Indicators
- VWAP (Institutional) : {indicators.vwap}  → Signal: {indicators.vwap_signal}
- Supertrend           : {indicators.supertrend}  → Direction: {indicators.supertrend_direction}
- Nearest Fibonacci    : {indicators.nearest_fib}  → Signal: {indicators.fib_signal}

### Trend & Levels
- Support Level     : ₹{indicators.support}
- Resistance Level  : ₹{indicators.resistance}
- Trend Strength    : {indicators.trend_strength}
- Trend Score       : {indicators.trend_score}/100 (positive=bullish, negative=bearish)
- Last Close        : ₹{indicators.last_close}

Provide your analysis as a single JSON object (no code fences or extra text).
"""


# ── xAI Grok API Interaction ────────────────────────────────


def _call_groq(user_prompt: str) -> str:
    """Send a prompt to Groq AI and return the raw text response.

    Uses the OpenAI-compatible endpoint at https://api.groq.com/openai/v1.
    Implements retry with exponential back-off.

    Returns
    -------
    str
        Raw response text from Groq.

    Raises
    ------
    RuntimeError
        If all retry attempts fail.
    """
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Please add it to your .env file."
        )

    client = OpenAI(
        api_key=config.GROQ_API_KEY,
        base_url=config.GROQ_BASE_URL,
    )
    last_exc: Optional[Exception] = None

    for attempt in range(1, config.GROQ_MAX_RETRIES + 1):
        try:
            logger.info("Groq API call attempt %d/%d …", attempt, config.GROQ_MAX_RETRIES)
            response = client.chat.completions.create(
                model=config.GROQ_MODEL,
                max_tokens=config.GROQ_MAX_TOKENS,
                temperature=config.GROQ_TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            # Extract text from response
            response_text = response.choices[0].message.content
            logger.info("Groq response received (%d chars).", len(response_text))
            return response_text

        except RateLimitError as exc:
            last_exc = exc
            wait = config.GROQ_RETRY_DELAY * (2 ** (attempt - 1))
            logger.warning("Rate limited — waiting %.1fs before retry…", wait)
            time.sleep(wait)

        except APIError as exc:
            last_exc = exc
            wait = config.GROQ_RETRY_DELAY * (2 ** (attempt - 1))
            logger.warning("API error (%s) — retrying in %.1fs…", exc, wait)
            time.sleep(wait)

        except Exception as exc:
            last_exc = exc
            logger.error("Unexpected error calling Groq: %s", exc)
            break

    raise RuntimeError(f"Groq API failed after {config.GROQ_MAX_RETRIES} attempts: {last_exc}")


def _parse_recommendation(raw: str, symbol: str) -> StockRecommendation:
    """Parse Groq's JSON response into a StockRecommendation.

    Handles common issues like code-fenced JSON, trailing commas, etc.
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (optionally with 'json' label)
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data: dict = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse Groq response as JSON: %s\nRaw: %s", exc, raw[:500])
        return StockRecommendation(
            symbol=symbol,
            analysis_success=False,
            error_message=f"JSON parse error: {exc}",
        )

    return StockRecommendation(
        symbol=data.get("symbol", symbol),
        recommendation=data.get("recommendation", "HOLD").upper(),
        trade_type=data.get("trade_type", "AVOID").upper(),
        entry_price=float(data.get("entry_price", 0)),
        target_price=float(data.get("target_price", 0)),
        stop_loss=float(data.get("stop_loss", 0)),
        risk_level=data.get("risk_level", "MEDIUM").upper(),
        confidence_score=int(data.get("confidence_score", 0)),
        key_reasons=data.get("key_reasons", []),
        risk_warning=data.get("risk_warning", ""),
        analysis_success=True,
    )


# ── Public API ───────────────────────────────────────────────


def analyse_stock(
    stock_data: StockData,
    indicators: TechnicalSummary,
) -> StockRecommendation:
    """Run AI analysis on a single stock and return a recommendation.

    Parameters
    ----------
    stock_data : StockData
        Fetched fundamentals + OHLCV data.
    indicators : TechnicalSummary
        Pre-computed technical indicators.

    Returns
    -------
    StockRecommendation
        Structured recommendation (or a failed placeholder if the API call
        errors out).
    """
    symbol = stock_data.symbol
    logger.info("Requesting Groq analysis for %s …", symbol)

    try:
        prompt = _build_user_prompt(symbol, stock_data.fundamentals, indicators)
        raw_response = _call_groq(prompt)
        recommendation = _parse_recommendation(raw_response, symbol)
        logger.info(
            "AI recommendation for %s: %s (confidence %d%%)",
            symbol, recommendation.recommendation, recommendation.confidence_score,
        )
        return recommendation

    except Exception as exc:
        logger.error("Analysis failed for %s: %s", symbol, exc)
        return StockRecommendation(
            symbol=symbol,
            analysis_success=False,
            error_message=str(exc),
        )


def analyse_all(
    stocks: list[tuple[StockData, TechnicalSummary]],
) -> list[StockRecommendation]:
    """Analyse a batch of stocks sequentially (respects rate limits).

    Parameters
    ----------
    stocks : list[tuple[StockData, TechnicalSummary]]
        Pairs of (fetched data, computed indicators).

    Returns
    -------
    list[StockRecommendation]
        One recommendation per stock.
    """
    results: list[StockRecommendation] = []
    for idx, (stock_data, indicators) in enumerate(stocks, start=1):
        logger.info(
            "[%d/%d] Analysing %s …", idx, len(stocks), stock_data.symbol,
        )
        rec = analyse_stock(stock_data, indicators)
        results.append(rec)
        # Small delay to be polite to the API
        if idx < len(stocks):
            time.sleep(0.5)
    return results


# ── Quick Test ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    from data_fetcher import fetch_stock_data
    from indicators import compute_all_indicators

    test_symbol = "RELIANCE"
    sd = fetch_stock_data(test_symbol)
    if sd.fetch_success:
        ind = compute_all_indicators(test_symbol, sd.ohlcv)
        rec = analyse_stock(sd, ind)
        print(json.dumps(rec.to_dict(), indent=2))
    else:
        print(f"Data fetch failed: {sd.error_message}")

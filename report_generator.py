"""
report_generator.py — Generate beautifully formatted daily analysis reports.

Produces both a plain-text (.txt) and an HTML (.html) report from
the list of StockRecommendation objects.  Reports are saved to the
``reports/`` directory.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

import config
from analyzer import StockRecommendation

logger = logging.getLogger(__name__)


# ── Categorisation Helpers ───────────────────────────────────


def _categorise(
    recommendations: list[StockRecommendation],
) -> dict[str, list[StockRecommendation]]:
    """Split recommendations into BUY, INTRADAY, SELL/AVOID, HOLD buckets.

    Returns
    -------
    dict with keys: buy_picks, intraday_picks, avoid_picks, hold_picks
    """
    buy_picks: list[StockRecommendation] = []
    intraday_picks: list[StockRecommendation] = []
    avoid_picks: list[StockRecommendation] = []
    hold_picks: list[StockRecommendation] = []

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

    # Sort by confidence descending
    buy_picks.sort(key=lambda r: r.confidence_score, reverse=True)
    intraday_picks.sort(key=lambda r: r.confidence_score, reverse=True)

    # Limit top picks
    buy_picks = buy_picks[: config.MAX_BUY_PICKS]
    intraday_picks = intraday_picks[: config.MAX_INTRADAY_PICKS]

    return {
        "buy_picks": buy_picks,
        "intraday_picks": intraday_picks,
        "avoid_picks": avoid_picks,
        "hold_picks": hold_picks,
    }


# ── Plain-Text Report ───────────────────────────────────────

_DIVIDER = "=" * 60
_THIN = "-" * 60


def _recommendation_badge(rec: StockRecommendation) -> str:
    mapping = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "🟡 HOLD"}
    return mapping.get(rec.recommendation, rec.recommendation)


def generate_text_report(recommendations: list[StockRecommendation]) -> str:
    """Build a nicely formatted plain-text report.

    Parameters
    ----------
    recommendations : list[StockRecommendation]
        Full list of recommendations from the analyser.

    Returns
    -------
    str
        The complete text report.
    """
    cats = _categorise(recommendations)
    now = datetime.now()
    date_str = now.strftime(config.REPORT_FORMAT_DATE)

    lines: list[str] = []
    lines.append(_DIVIDER)
    lines.append("  📊  AI STOCK MARKET ANALYSIS REPORT")
    lines.append(f"  📅  {date_str}  |  Indian Markets (NSE)")
    lines.append(_DIVIDER)
    lines.append("")

    # Summary
    total = sum(1 for r in recommendations if r.analysis_success)
    buy_c = len(cats["buy_picks"]) + len(cats["intraday_picks"])
    sell_c = len(cats["avoid_picks"])
    hold_c = len(cats["hold_picks"])
    fail_c = sum(1 for r in recommendations if not r.analysis_success)
    lines.append(f"  Stocks Analysed : {total}")
    lines.append(f"  BUY picks       : {buy_c}")
    lines.append(f"  SELL / Avoid    : {sell_c}")
    lines.append(f"  HOLD            : {hold_c}")
    lines.append(f"  Failures        : {fail_c}")
    lines.append("")

    def _section(title: str, picks: list[StockRecommendation]) -> None:
        lines.append(_DIVIDER)
        lines.append(f"  {title}")
        lines.append(_DIVIDER)
        if not picks:
            lines.append("  (none)")
            lines.append("")
            return
        for rec in picks:
            lines.append("")
            lines.append(f"  {_recommendation_badge(rec)}  {rec.symbol}  ({rec.trade_type})")
            lines.append(_THIN)
            lines.append(f"    Entry Price  : ₹{rec.entry_price:.2f}")
            lines.append(f"    Target Price : ₹{rec.target_price:.2f}")
            lines.append(f"    Stop Loss    : ₹{rec.stop_loss:.2f}")
            lines.append(f"    Risk Level   : {rec.risk_level}")
            lines.append(f"    Confidence   : {rec.confidence_score}%")
            if rec.key_reasons:
                lines.append("    Key Reasons  :")
                for r in rec.key_reasons:
                    lines.append(f"      • {r}")
            if rec.risk_warning:
                lines.append(f"    ⚠️  {rec.risk_warning}")
        lines.append("")

    _section("🟢  TOP BUY PICKS", cats["buy_picks"])
    _section("⚡  INTRADAY PICKS", cats["intraday_picks"])
    _section("🚫  STOCKS TO AVOID", cats["avoid_picks"])
    _section("🟡  HOLD / NEUTRAL", cats["hold_picks"])

    # Disclaimer
    lines.append(_DIVIDER)
    lines.append("  ⚠️  DISCLAIMER")
    lines.append(_DIVIDER)
    lines.append(
        "  This report is AI-generated for educational and informational\n"
        "  purposes only. It does NOT constitute financial advice. Stock\n"
        "  markets are subject to market risk. Always do your own research\n"
        "  and consult a SEBI-registered financial advisor before investing.\n"
        "  The creators of this tool bear no responsibility for any losses."
    )
    lines.append(_DIVIDER)

    return "\n".join(lines)


# ── HTML Report ──────────────────────────────────────────────


def generate_html_report(recommendations: list[StockRecommendation]) -> str:
    """Render an HTML report using the Jinja2 template.

    Parameters
    ----------
    recommendations : list[StockRecommendation]
        Full list of recommendations.

    Returns
    -------
    str
        Complete HTML string.
    """
    cats = _categorise(recommendations)
    now = datetime.now()
    date_str = now.strftime(config.REPORT_FORMAT_DATE)

    total = sum(1 for r in recommendations if r.analysis_success)
    buy_c = len(cats["buy_picks"]) + len(cats["intraday_picks"])
    sell_c = len(cats["avoid_picks"])
    hold_c = len(cats["hold_picks"])
    fail_c = sum(1 for r in recommendations if not r.analysis_success)

    env = Environment(
        loader=FileSystemLoader(str(config.TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html")

    html = template.render(
        report_date=date_str,
        total_analysed=total,
        buy_count=buy_c,
        sell_count=sell_c,
        hold_count=hold_c,
        fail_count=fail_c,
        **cats,
    )
    return html


# ── Save to Disk ─────────────────────────────────────────────


def save_reports(
    recommendations: list[StockRecommendation],
    output_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Generate and save both TXT and HTML reports.

    Parameters
    ----------
    recommendations : list[StockRecommendation]
        The analysis results.
    output_dir : Path | None
        Directory to store reports. Defaults to ``config.REPORTS_DIR``.

    Returns
    -------
    tuple[Path, Path]
        Paths to the saved (txt, html) files.
    """
    if output_dir is None:
        output_dir = config.REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    date_slug = datetime.now().strftime("%Y-%m-%d")

    # Text report
    txt_path = output_dir / f"report_{date_slug}.txt"
    txt_content = generate_text_report(recommendations)
    txt_path.write_text(txt_content, encoding="utf-8")
    logger.info("Text report saved to %s", txt_path)

    # HTML report
    html_path = output_dir / f"report_{date_slug}.html"
    html_content = generate_html_report(recommendations)
    html_path.write_text(html_content, encoding="utf-8")
    logger.info("HTML report saved to %s", html_path)

    return txt_path, html_path


# ── Quick Test ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    # Create some dummy recommendations for testing
    dummy = [
        StockRecommendation(
            symbol="RELIANCE", recommendation="BUY", trade_type="SHORT_TERM",
            entry_price=2450.0, target_price=2600.0, stop_loss=2380.0,
            risk_level="MEDIUM", confidence_score=78,
            key_reasons=["Bullish MACD crossover", "RSI recovering from oversold", "Strong volume"],
            risk_warning="Market volatility is elevated due to global cues.",
        ),
        StockRecommendation(
            symbol="TCS", recommendation="HOLD", trade_type="SHORT_TERM",
            entry_price=3800.0, target_price=3950.0, stop_loss=3720.0,
            risk_level="LOW", confidence_score=55,
            key_reasons=["Flat MACD", "RSI neutral at 52"],
            risk_warning="IT sector under pressure.",
        ),
        StockRecommendation(
            symbol="ADANIENT", recommendation="SELL", trade_type="AVOID",
            entry_price=2200.0, target_price=0.0, stop_loss=0.0,
            risk_level="HIGH", confidence_score=70,
            key_reasons=["Bearish trend", "Below 200-DMA", "Weak fundamentals"],
            risk_warning="High risk due to governance concerns.",
        ),
    ]

    txt_path, html_path = save_reports(dummy)
    print(f"✅ Reports saved:\n   TXT  → {txt_path}\n   HTML → {html_path}")

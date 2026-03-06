"""
email_notifier.py — Send daily stock analysis reports via email (Gmail SMTP).

Generates a professional PDF report and sends it as an attachment,
along with an inline HTML summary.

Setup:
    1. Go to https://myaccount.google.com/apppasswords
    2. Generate an "App Password" (select Mail → Other → "Stock Agent")
    3. Add to .env:
         EMAIL_SENDER=your_email@gmail.com
         EMAIL_PASSWORD=your_16_char_app_password
         EMAIL_RECEIVER=your_email@gmail.com  (can be same or different)
"""

from __future__ import annotations

import logging
import os
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime
from typing import Optional

from fpdf import FPDF

import config
from analyzer import StockRecommendation

logger = logging.getLogger(__name__)

# Gmail SMTP settings
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587


# ── PDF Report Builder ────────────────────────────────────────


def _safe(text: str) -> str:
    """Strip non-ASCII chars so Helvetica won't crash."""
    return text.encode("ascii", "replace").decode("ascii").replace("?", " ")


def _build_pdf_report(recommendations: list[StockRecommendation]) -> str:
    """Generate a professional PDF report and return the file path.

    Returns
    -------
    str
        Absolute path to the generated PDF file.
    """
    date_str = datetime.now().strftime(config.REPORT_FORMAT_DATE)
    date_file = datetime.now().strftime("%Y-%m-%d")

    # Categorise
    buy_picks, sell_picks, hold_picks = [], [], []
    for rec in recommendations:
        if not rec.analysis_success:
            continue
        if rec.recommendation == "BUY":
            buy_picks.append(rec)
        elif rec.recommendation == "SELL":
            sell_picks.append(rec)
        else:
            hold_picks.append(rec)

    buy_picks.sort(key=lambda r: r.confidence_score, reverse=True)
    sell_picks.sort(key=lambda r: r.confidence_score, reverse=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Title Header ──
    pdf.set_fill_color(15, 23, 42)       # Dark blue background
    pdf.rect(0, 0, 210, 45, "F")
    pdf.set_text_color(56, 189, 248)      # Cyan title
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_y(10)
    pdf.cell(0, 12, "AI Stock Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 8, f"{date_str}  |  NSE India  |  Powered by Groq AI", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(50)

    # ── Summary Cards ──
    total = len(buy_picks) + len(sell_picks) + len(hold_picks)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 41, 59)

    # BUY card
    pdf.set_fill_color(34, 197, 94)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(40, 18, f"  BUY: {len(buy_picks)}", fill=True, new_x="RIGHT")
    pdf.cell(5, 18, "", new_x="RIGHT")
    # SELL card
    pdf.set_fill_color(239, 68, 68)
    pdf.cell(40, 18, f"  SELL: {len(sell_picks)}", fill=True, new_x="RIGHT")
    pdf.cell(5, 18, "", new_x="RIGHT")
    # HOLD card
    pdf.set_fill_color(234, 179, 8)
    pdf.cell(40, 18, f"  HOLD: {len(hold_picks)}", fill=True, new_x="RIGHT")
    pdf.cell(5, 18, "", new_x="RIGHT")
    # TOTAL card
    pdf.set_fill_color(56, 189, 248)
    pdf.cell(40, 18, f"  TOTAL: {total}", fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)

    # ── Stock Table ──
    def _draw_table_header():
        pdf.set_fill_color(51, 65, 85)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 9)
        col_widths = [25, 18, 22, 22, 22, 15, 66]
        headers = ["Stock", "Signal", "Entry(Rs)", "Target(Rs)", "SL(Rs)", "Conf%", "Key Reasons"]
        for w, h in zip(col_widths, headers):
            pdf.cell(w, 8, h, border=1, fill=True, align="C", new_x="RIGHT")
        pdf.ln()

    _draw_table_header()

    pdf.set_font("Helvetica", "", 8)
    all_recs = buy_picks + sell_picks + hold_picks
    row_idx = 0

    for rec in all_recs:
        # Alternate row color
        if row_idx % 2 == 0:
            pdf.set_fill_color(241, 245, 249)
        else:
            pdf.set_fill_color(255, 255, 255)

        col_widths = [25, 18, 22, 22, 22, 15, 66]
        pdf.set_text_color(30, 41, 59)

        # Stock name
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(col_widths[0], 7, rec.symbol, border=1, fill=True, align="L", new_x="RIGHT")

        # Signal with color
        if rec.recommendation == "BUY":
            pdf.set_text_color(34, 197, 94)
        elif rec.recommendation == "SELL":
            pdf.set_text_color(239, 68, 68)
        else:
            pdf.set_text_color(234, 179, 8)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(col_widths[1], 7, rec.recommendation, border=1, fill=True, align="C", new_x="RIGHT")

        # Numbers
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(col_widths[2], 7, f"{rec.entry_price:.1f}", border=1, fill=True, align="R", new_x="RIGHT")
        pdf.cell(col_widths[3], 7, f"{rec.target_price:.1f}", border=1, fill=True, align="R", new_x="RIGHT")
        pdf.cell(col_widths[4], 7, f"{rec.stop_loss:.1f}", border=1, fill=True, align="R", new_x="RIGHT")
        pdf.cell(col_widths[5], 7, f"{rec.confidence_score}%", border=1, fill=True, align="C", new_x="RIGHT")

        # Reasons (truncated to fit)
        reasons = ", ".join(rec.key_reasons[:2]) if rec.key_reasons else ""
        if len(reasons) > 55:
            reasons = reasons[:52] + "..."
        pdf.cell(col_widths[6], 7, _safe(reasons), border=1, fill=True, align="L", new_x="LMARGIN", new_y="NEXT")

        row_idx += 1

        # New page if running out of space
        if pdf.get_y() > 260:
            pdf.add_page()
            _draw_table_header()

    pdf.ln(6)

    # ── Detailed Analysis Section ──
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(56, 189, 248)
    pdf.cell(0, 10, "Detailed Analysis", new_x="LMARGIN", new_y="NEXT")

    for rec in all_recs:
        if pdf.get_y() > 240:
            pdf.add_page()

        # Stock header bar
        if rec.recommendation == "BUY":
            pdf.set_fill_color(34, 197, 94)
        elif rec.recommendation == "SELL":
            pdf.set_fill_color(239, 68, 68)
        else:
            pdf.set_fill_color(234, 179, 8)

        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, _safe(f"  {rec.symbol}  |  {rec.recommendation}  |  Confidence: {rec.confidence_score}%"), fill=True, new_x="LMARGIN", new_y="NEXT")

        # Metrics
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 9)
        # Calculate risk:reward
        risk = rec.entry_price - rec.stop_loss if rec.stop_loss > 0 else 0
        reward = rec.target_price - rec.entry_price if rec.target_price > 0 else 0
        rr = f"1:{reward/risk:.1f}" if risk > 0 else "N/A"
        pdf.cell(0, 6, _safe(f"Entry: Rs {rec.entry_price:.2f}  |  Target: Rs {rec.target_price:.2f}  |  Stop Loss: Rs {rec.stop_loss:.2f}  |  R:R {rr}"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, _safe(f"Type: {rec.trade_type}  |  Risk: {rec.risk_level}"), new_x="LMARGIN", new_y="NEXT")

        # Reasoning
        if rec.key_reasons:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 6, "Key Reasons:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 8)
            for reason in rec.key_reasons[:5]:
                pdf.cell(0, 5, _safe(f"  - {reason}"), new_x="LMARGIN", new_y="NEXT")

        if rec.risk_warning:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 6, "Risk Warning:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 8)
            pdf.multi_cell(0, 4.5, _safe(rec.risk_warning[:400]))

        pdf.ln(4)

    # ── Disclaimer ──
    pdf.ln(5)
    pdf.set_draw_color(100, 116, 139)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(0, 4,
        "DISCLAIMER: This AI-generated report is for educational and informational purposes only. "
        "It does not constitute financial advice. Past performance is not indicative of future results. "
        "Stock markets are subject to risk. Always do your own research (DYOR) before investing. "
        "The AI model may make errors in analysis. Consult a SEBI-registered financial advisor."
    )

    # Save
    config.REPORTS_DIR.mkdir(exist_ok=True)
    pdf_path = os.path.join(str(config.REPORTS_DIR), f"stock_report_{date_file}.pdf")
    pdf.output(pdf_path)
    logger.info("PDF report generated: %s", pdf_path)
    return pdf_path


# ── HTML Email Builder ───────────────────────────────────────


def _build_email_html(recommendations: list[StockRecommendation]) -> str:
    """Build a beautiful HTML email from recommendations."""
    date_str = datetime.now().strftime(config.REPORT_FORMAT_DATE)

    # Categorise
    buy_picks, sell_picks, hold_picks = [], [], []
    for rec in recommendations:
        if not rec.analysis_success:
            continue
        if rec.recommendation == "BUY":
            buy_picks.append(rec)
        elif rec.recommendation == "SELL":
            sell_picks.append(rec)
        else:
            hold_picks.append(rec)

    buy_picks.sort(key=lambda r: r.confidence_score, reverse=True)
    total = sum(1 for r in recommendations if r.analysis_success)

    def _color(rec_type):
        return {"BUY": "#22c55e", "SELL": "#ef4444", "HOLD": "#eab308"}.get(rec_type, "#999")

    def _stock_row(rec):
        c = _color(rec.recommendation)
        reasons = " &bull; ".join(rec.key_reasons[:3]) if rec.key_reasons else ""
        return f"""
        <tr>
          <td style="padding:10px;font-weight:bold;border-bottom:1px solid #333;">{rec.symbol}</td>
          <td style="padding:10px;border-bottom:1px solid #333;">
            <span style="background:{c};color:#fff;padding:3px 10px;border-radius:12px;font-size:13px;">
              {rec.recommendation}
            </span>
          </td>
          <td style="padding:10px;border-bottom:1px solid #333;">&#8377;{rec.entry_price:.2f}</td>
          <td style="padding:10px;border-bottom:1px solid #333;">&#8377;{rec.target_price:.2f}</td>
          <td style="padding:10px;border-bottom:1px solid #333;">&#8377;{rec.stop_loss:.2f}</td>
          <td style="padding:10px;border-bottom:1px solid #333;">{rec.confidence_score}%</td>
          <td style="padding:10px;border-bottom:1px solid #333;font-size:12px;">{reasons}</td>
        </tr>"""

    rows = ""
    for rec in buy_picks + sell_picks + hold_picks:
        rows += _stock_row(rec)

    return f"""
    <html>
    <body style="background:#0f172a;color:#e2e8f0;font-family:Arial,sans-serif;margin:0;padding:20px;">
      <div style="max-width:900px;margin:0 auto;background:#1e293b;border-radius:12px;padding:30px;">
        <h1 style="color:#38bdf8;margin:0;">&#128200; AI Stock Analysis Report</h1>
        <p style="color:#94a3b8;margin:5px 0 20px;">{date_str} &mdash; NSE India</p>

        <div style="display:flex;gap:15px;margin-bottom:25px;">
          <div style="background:#22c55e22;padding:15px 25px;border-radius:8px;text-align:center;">
            <div style="font-size:28px;font-weight:bold;color:#22c55e;">{len(buy_picks)}</div>
            <div style="color:#94a3b8;font-size:13px;">BUY</div>
          </div>
          <div style="background:#ef444422;padding:15px 25px;border-radius:8px;text-align:center;">
            <div style="font-size:28px;font-weight:bold;color:#ef4444;">{len(sell_picks)}</div>
            <div style="color:#94a3b8;font-size:13px;">SELL</div>
          </div>
          <div style="background:#eab30822;padding:15px 25px;border-radius:8px;text-align:center;">
            <div style="font-size:28px;font-weight:bold;color:#eab308;">{len(hold_picks)}</div>
            <div style="color:#94a3b8;font-size:13px;">HOLD</div>
          </div>
          <div style="background:#38bdf822;padding:15px 25px;border-radius:8px;text-align:center;">
            <div style="font-size:28px;font-weight:bold;color:#38bdf8;">{total}</div>
            <div style="color:#94a3b8;font-size:13px;">TOTAL</div>
          </div>
        </div>

        <table style="width:100%;border-collapse:collapse;color:#e2e8f0;font-size:14px;">
          <thead>
            <tr style="background:#334155;">
              <th style="padding:10px;text-align:left;">Stock</th>
              <th style="padding:10px;text-align:left;">Signal</th>
              <th style="padding:10px;text-align:left;">Entry</th>
              <th style="padding:10px;text-align:left;">Target</th>
              <th style="padding:10px;text-align:left;">Stop Loss</th>
              <th style="padding:10px;text-align:left;">Conf.</th>
              <th style="padding:10px;text-align:left;">Reasons</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>

        <p style="color:#64748b;font-size:12px;margin-top:25px;padding-top:15px;border-top:1px solid #334155;">
          &#9888;&#65039; AI-generated report for educational purposes only.
          Not financial advice. Markets are subject to risk. Always DYOR.
        </p>
      </div>
    </body>
    </html>"""


# ── Send Email ───────────────────────────────────────────────


def send_email_report(
    recommendations: list[StockRecommendation],
    sender: Optional[str] = None,
    password: Optional[str] = None,
    receiver: Optional[str] = None,
    extra_text: str = "",
) -> bool:
    """Send the stock analysis report via Gmail SMTP.

    Parameters
    ----------
    recommendations : list[StockRecommendation]
        Analysis results.
    sender : str | None
        Gmail address (defaults to config).
    password : str | None
        Gmail App Password (defaults to config).
    receiver : str | None
        Recipient email (defaults to config).
    extra_text : str
        Additional plain text to append.

    Returns
    -------
    bool
        True if email sent successfully.
    """
    sender = sender or config.EMAIL_SENDER
    password = password or config.EMAIL_PASSWORD
    receiver = receiver or config.EMAIL_RECEIVER

    if not sender or not password or not receiver:
        logger.error(
            "Email credentials missing — set EMAIL_SENDER, EMAIL_PASSWORD, "
            "and EMAIL_RECEIVER in your .env file."
        )
        return False

    date_str = datetime.now().strftime(config.REPORT_FORMAT_DATE)
    buy_count = sum(1 for r in recommendations if r.analysis_success and r.recommendation == "BUY")

    msg = MIMEMultipart("mixed")
    msg["Subject"] = f"Stock Report {date_str} — {buy_count} BUY picks"
    msg["From"] = sender
    msg["To"] = receiver

    # Plain text fallback
    plain = f"AI Stock Analysis Report — {date_str}\n\n"
    for rec in recommendations:
        if rec.analysis_success:
            plain += f"{rec.recommendation} {rec.symbol} Entry:{rec.entry_price} Target:{rec.target_price}\n"
    if extra_text:
        plain += f"\n{extra_text}"

    # HTML body (inline summary)
    html = _build_email_html(recommendations)

    # Build email body (text + HTML alternatives)
    body_part = MIMEMultipart("alternative")
    body_part.attach(MIMEText(plain, "plain"))
    body_part.attach(MIMEText(html, "html"))
    msg.attach(body_part)

    # Generate PDF and attach
    try:
        pdf_path = _build_pdf_report(recommendations)
        pdf_filename = os.path.basename(pdf_path)
        with open(pdf_path, "rb") as f:
            pdf_attachment = MIMEApplication(f.read(), _subtype="pdf")
            pdf_attachment.add_header(
                "Content-Disposition", "attachment", filename=pdf_filename
            )
            msg.attach(pdf_attachment)
        logger.info("PDF attached: %s", pdf_filename)
    except Exception as exc:
        logger.warning("PDF generation failed, sending without attachment: %s", exc)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())

        logger.info("Email report sent to %s", receiver)
        return True

    except smtplib.SMTPAuthenticationError:
        logger.error(
            "Gmail authentication failed. Make sure you're using an App Password "
            "(not your regular password). Get one at: https://myaccount.google.com/apppasswords"
        )
        return False
    except Exception as exc:
        logger.error("Email send failed: %s", exc)
        return False


# ── Quick Test ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    print("Email notifier module loaded. Set EMAIL_SENDER/EMAIL_PASSWORD/EMAIL_RECEIVER in .env to use.")

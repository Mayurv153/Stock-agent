# 📊 AI Stock Market Analysis Agent — Indian Markets (NSE/BSE)

An automated Python agent that fetches real-time stock data from **NSE/BSE**, computes **technical indicators**, asks **Claude AI** for **BUY / SELL / HOLD** recommendations, generates beautiful **daily reports**, and delivers them via **Telegram**.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Live Data** | Pulls OHLCV + fundamentals from Yahoo Finance for NSE stocks |
| **Technical Indicators** | RSI, MACD, Bollinger Bands, EMA 9/21, Volume analysis, Support/Resistance |
| **AI Analysis** | Claude AI (claude-sonnet-4-20250514) returns structured JSON recommendations |
| **Reports** | Beautiful TXT + HTML reports with categorised picks |
| **Telegram Bot** | Sends formatted report to your Telegram chat/channel |
| **Daily Scheduler** | Auto-runs at 08:30 IST on weekdays, skips holidays |
| **Error Handling** | Retry logic, graceful fallbacks, full logging |

---

## 📁 Project Structure

```
stock-agent/
├── main.py                 # Entry point (CLI)
├── config.py               # Settings, watchlist, API keys
├── data_fetcher.py         # Yahoo Finance data fetching
├── indicators.py           # Technical indicator calculations
├── analyzer.py             # Claude AI analysis
├── report_generator.py     # TXT + HTML report generation
├── notifier.py             # Telegram bot delivery
├── scheduler.py            # Daily automation
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── README.md               # This file
├── templates/
│   └── report.html         # Jinja2 HTML report template
├── reports/                # Generated reports (auto-created)
└── logs/                   # Log files (auto-created)
```

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.10+** installed ([download](https://www.python.org/downloads/))
- An **Anthropic API key** (for Claude AI)
- (Optional) A **Telegram bot** (for notifications)

### 2. Clone / Download the Project

```bash
cd "c:\stock agent"
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the template and fill in your keys:

```bash
copy .env.example .env
```

Then open `.env` in a text editor and replace the placeholder values:

```env
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=-100123456789
```

### 5. Run the Analysis

```bash
# Analyse the full 20-stock watchlist
python main.py

# Analyse specific stocks only
python main.py --stocks RELIANCE TCS INFY

# Skip Telegram notification
python main.py --no-telegram

# Verbose logging
python main.py -v

# Start daily scheduler (runs at 08:30 IST, Mon-Fri)
python main.py --schedule
```

---

## 🔑 How to Get API Keys

### Anthropic (Claude AI)

1. Go to [console.anthropic.com](https://console.anthropic.com/)
2. Sign up / log in
3. Navigate to **API Keys** → **Create Key**
4. Copy the key (starts with `sk-ant-...`)
5. Paste it as `ANTHROPIC_API_KEY` in your `.env`

### Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the prompts
3. BotFather will give you a **bot token** — copy it
4. Paste it as `TELEGRAM_BOT_TOKEN` in your `.env`

#### Finding your Chat ID

1. Start a chat with your new bot (send any message)
2. Open this URL in your browser (replace `<TOKEN>` with your bot token):
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
3. Look for `"chat":{"id": 123456789}` — that number is your Chat ID
4. For a **channel**, the ID is negative (e.g. `-1001234567890`)
5. Paste it as `TELEGRAM_CHAT_ID` in your `.env`

---

## ⚙️ Configuration

All settings are in **`config.py`**:

| Setting | Default | Description |
|---------|---------|-------------|
| `WATCHLIST` | 20 NSE stocks | Stocks to analyse daily |
| `EXCHANGE_SUFFIX` | `.NS` | `.NS` for NSE, `.BO` for BSE |
| `RSI_OVERBOUGHT` | `70` | RSI above this = overbought |
| `RSI_OVERSOLD` | `30` | RSI below this = oversold |
| `MIN_CONFIDENCE_SCORE` | `65` | Minimum AI confidence to recommend |
| `MAX_BUY_PICKS` | `5` | Max BUY picks shown in report |
| `SCHEDULE_TIME_IST` | `08:30` | Daily run time |

### Adding / Removing Stocks

Edit `WATCHLIST` in `config.py`:

```python
WATCHLIST = [
    "RELIANCE",
    "TCS",
    "INFY",
    # Add your stocks here...
    "ZOMATO",
    "PAYTM",
]
```

Use official NSE symbol names (as listed on [nseindia.com](https://www.nseindia.com/)).

---

## 📄 Sample Output

### Terminal Output
```
============================================================
🚀  AI Stock Market Analysis Agent — Pipeline Start
============================================================
Watchlist: RELIANCE, TCS, INFY, ... (20 stocks)

📥  STEP 1/5 — Fetching stock data from Yahoo Finance …
   ✅ Fetched: 20  |  ❌ Failed: 0

📊  STEP 2/5 — Computing technical indicators …
   Indicators computed for 20 stocks.

🤖  STEP 3/5 — Running Claude AI analysis …
   ✅ Analysed: 20  |  ❌ Failed: 0

📝  STEP 4/5 — Generating reports …
   TXT  → reports/report_2026-03-06.txt
   HTML → reports/report_2026-03-06.html

📨  STEP 5/5 — Sending Telegram notification …
   ✅ Telegram report sent!

============================================================
✅  Pipeline complete in 47.3 seconds.
============================================================
```

### Telegram Message Preview
```
📊 AI Stock Analysis Report
06 March 2026 — NSE

Analysed: 20 stocks
🟢 BUY: 4 | 🔴 SELL: 2 | 🟡 HOLD: 14

━━━ 🟢 TOP BUY PICKS ━━━

🟢 RELIANCE — BUY (SHORT_TERM)
   Entry: ₹2450.00 | Target: ₹2600.00 | SL: ₹2380.00
   Risk: 🟡 MEDIUM | Confidence: 78%
   📝 Bullish MACD • RSI rising • Volume spike
```

### HTML Report

The HTML report (saved in `reports/`) includes:
- Dark-themed modern UI
- Colour-coded recommendation badges
- Confidence score bars
- Sections for BUY, INTRADAY, HOLD, and AVOID picks
- Legal disclaimer

---

## 📅 Scheduler

The scheduler runs the pipeline automatically every weekday:

```bash
python main.py --schedule
```

- **Time**: 08:30 IST (before market opens at 09:15)
- **Days**: Monday to Friday
- **Skips**: Weekends + Indian market holidays (2025-2026)
- **Logs**: All activity logged to `logs/agent.log`

To run as a background service on Windows, you can use Task Scheduler:

1. Open **Task Scheduler** → **Create Basic Task**
2. Set trigger to **Daily** at **08:25 AM**
3. Action → **Start a Program** → `python.exe`
4. Arguments: `"c:\stock agent\main.py"`
5. Start in: `c:\stock agent`

---

## 🛠 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `ANTHROPIC_API_KEY not set` | Create `.env` file from `.env.example` and add your key |
| `No OHLCV data returned` | Check if the stock symbol is correct (e.g. `RELIANCE`, not `RELIANCE.NS`) |
| `Telegram send failed` | Verify bot token + chat ID; make sure bot is added to channel |
| `Rate limited by Claude` | The agent retries automatically; reduce watchlist size if persistent |
| `pandas-ta import error` | Run `pip install pandas-ta==0.3.14b1` specifically |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `anthropic` | Claude AI SDK |
| `yfinance` | Yahoo Finance data |
| `pandas` | Data manipulation |
| `pandas-ta` | Technical indicators |
| `python-dotenv` | .env file loading |
| `schedule` | Task scheduling |
| `python-telegram-bot` | Telegram delivery |
| `jinja2` | HTML templating |
| `requests` | HTTP utilities |
| `aiohttp` | Async HTTP for Telegram |

---

## ⚠️ Disclaimer

This tool is for **educational and informational purposes only**.
It does **NOT** constitute financial advice. Stock markets carry inherent
risk — past performance does not guarantee future results. Always perform
your own research (DYOR) and consult a **SEBI-registered financial
advisor** before making investment decisions. The creators of this tool
accept **no responsibility** for any financial losses.

---

## 📄 License

MIT License — free for personal and commercial use.

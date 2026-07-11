# Nifty Options MAS — Trading System
**Multi-Agent Autonomous Nifty Options Trading | Neo Kotak | macOS arm64 Python 3.11**

> **v31 update:** the Strategy Agent was rebuilt around a single thesis —
> trade only at a volume-confirmed Order Block, in the direction volume
> confirms. See `UPGRADE_NOTES_v31.md` for exactly what changed and why.

---

## Quick fix for your install errors

Copy-paste this single block — it replaces `pip install -r requirements.txt`:

```bash
source venv/bin/activate    # make sure venv is active

pip install --upgrade pip && pip install \
  "pandas==2.2.2" \
  "numpy==1.26.4" \
  "scipy==1.13.0" \
  "aiohttp==3.9.5" \
  "websocket-client==1.8.0" \
  "requests==2.31.0" \
  "dash==2.17.1" \
  "dash-bootstrap-components==1.6.0" \
  "plotly==5.22.0" \
  "sqlalchemy==2.0.30" \
  "python-dotenv==1.0.1" \
  "pydantic==2.7.1" \
  "loguru==0.7.2" \
  "schedule==1.2.2" \
  "pytz==2024.1" \
  "tabulate==0.9.0" \
  "matplotlib==3.9.0" \
  "seaborn==0.13.2" \
  "yfinance==0.2.40" \
  "pytest==8.2.0" \
  "pytest-asyncio==0.23.7"
```

Then for Neo Kotak (try in order):

```bash
pip install neo-api-client
# If that fails:
pip install git+https://github.com/Kotak-Neo/kotak-neo-api.git
# Paper mode works fine without it
```

Verify everything installed:

```bash
python3 -c "import pandas, numpy, scipy, dash, plotly, sqlalchemy, pydantic, loguru, yfinance; print('All OK')"
```

---

## Why each error happened

| Error | Root cause | Fix applied |
|-------|-----------|-------------|
| `pandas-ta==0.3.14b` not found | No Python 3.11 wheel exists for pandas-ta | **Removed entirely.** All indicators live in `core/indicators.py` (pure pandas + numpy — no external TA lib needed) |
| `neo-api-client==1.0.52` not found | Version `1.0.52` pinned too tightly; arm64 Mac wheel has a different version | Install without version pin: `pip install neo-api-client` |
| numpy version warnings | Old pins like `1.21.x` only support Python < 3.11 | Fixed to `numpy==1.26.4` which is the correct Python 3.11 build |

---

## Run order after install

```bash
# 1. Backtest first (no credentials needed, uses synthetic data)
python3 backtest/run_backtest_standalone.py
open backtest/results/backtest_report.png

# 2. Find best parameters
python3 backtest/param_sweep.py

# 3. Paper trade (live data, simulated orders)
python3 dashboard/app.py &
python3 main.py --mode paper
# Open http://localhost:8050

# 4. Go live (after 1-2 weeks paper validation)
python3 main.py --mode live
```

Or just run `bash start.sh` for an interactive menu.

---

## Architecture

```
8 Agents (asyncio concurrent):

  [1] Market Intelligence  → EMA 9/21 · VWAP · ATR · RSI · regime detection
  [6] Strike Selection     → Options chain ranking (OI · volume · spread · delta)
       ↓
  [2] Strategy Agent       → Order Block + Volume engine (score ≥ 7, R:R ≥ 1.5)
  [3] Risk Management      → Adaptive SL + position sizing (≤ ₹2,000 risk/trade)
  [4] Execution Agent      → Neo Kotak API orders + immediate broker SL-M
  [5] Trailing SL Agent    → Candle / VWAP / ATR trailing (activates at +0.5R)
  [7] Monitoring Agent     → Plotly Dash dashboard (port 8050)
  [8] Learning Agent       → Post-session weight updates to signal_weights.json
```

---

## Capital rules

| | |
|--|--|
| Capital | ₹2,00,000 |
| Max risk/trade | ₹2,000 (1%) |
| Lots | 2 × 65 = 130 qty |
| Daily target | ₹5,000 |
| Daily stop | ₹6,000 loss → auto-halt |
| Square-off | 3:20 PM IST (hard) |
| Overnight | Never |

---

## Indicators built-in (no pandas-ta needed)

All implemented in `core/indicators.py`:

```
ema()                  EMA 9, 21 (any period)
atr()                  ATR 14
rsi()                  RSI 14
vwap()                 Intraday VWAP
volume_sma()           Volume moving average
detect_engulfing()     Bullish/bearish engulfing patterns
detect_rejection_wick()  Pin bar / rejection candles
pivot_levels()         Support & resistance (N-bar high/low)
classify_regime()      TRENDING_BULL / TRENDING_BEAR / RANGING / VOLATILE
detect_order_blocks_obv()  v31: Volume-confirmed Order Blocks (WHERE + WHO)
relative_volume()      v31: RVOL = current volume / rolling avg volume
classify_volume_state()  v31: LOW / NORMAL / HIGH / CLIMAX bucket
```

---

## Neo Kotak credentials

Edit `config/settings.env`:

```env
NEO_CONSUMER_KEY=your_key
NEO_CONSUMER_SECRET=your_secret
NEO_PAN=ABCDE1234F
NEO_PASSWORD=your_password
NEO_MOBILE=9999999999
NEO_MPIN=123456
NEO_ENVIRONMENT=uat      # start with uat (sandbox), switch to prod when ready
TRADING_MODE=paper       # paper → live
```

Get API credentials: https://developers.kotaksecurities.com

---

## Files

```
main.py                      Orchestrator (paper / live / backtest)
start.sh                     macOS interactive launcher
requirements.txt             Python 3.11 arm64 compatible
config/
  settings.example.env       Template
  signal_weights.json        Agent 8 updates this nightly
agents/
  agent1_market.py           Market Intelligence
  agent2_strategy.py         Signal Scorer
  agent3_risk.py             Risk + Position Sizing
  agent4_execution.py        Neo Kotak orders
  agent5_trailing.py         Trailing SL
  agent6_strikes.py          Strike Selection
  agent7_monitor.py          Monitoring state
  agent8_learning.py         Nightly weight updates
core/
  models.py                  Pydantic data models
  indicators.py              All TA (EMA/ATR/RSI/VWAP/patterns)
  broker.py                  Neo Kotak + paper mode
  message_bus.py             Async pub-sub
  database.py                SQLite trade log
data/
  fetcher.py                 Live data (yfinance / WS)
  historical.py              Historical downloader + cache
backtest/
  run_backtest_standalone.py   Synthetic 60-day backtest
  param_sweep.py               Grid search
  engine.py                    Full backtest engine
dashboard/
  app.py                     Plotly Dash (port 8050)
tests/
  test_agents.py             35 unit tests
```

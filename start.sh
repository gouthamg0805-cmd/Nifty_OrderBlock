#!/bin/bash
# ============================================================
# Nifty Options MAS — macOS Quick Start
# Python 3.11 · arm64 (M1/M2/M3) · Neo Kotak
# ============================================================
set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║   Nifty Options MAS — macOS Launcher     ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

# ── Python check ─────────────────────────────────────────────
PYTHON=$(which python3.11 2>/dev/null || which python3 2>/dev/null)
[ -z "$PYTHON" ] && echo "  ✗ Python not found. brew install python@3.11" && exit 1
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  ✓ Python $PY_VER  ($PYTHON)"

# ── Virtual env ──────────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "  → Creating virtual environment..."
    $PYTHON -m venv venv
fi
source venv/bin/activate

# ── Install dependencies ─────────────────────────────────────
if ! python3 -c "import dash" 2>/dev/null; then
    echo "  → Installing dependencies (first run ~2 min)..."
    pip install -q --upgrade pip
    pip install -q \
        "pandas==2.2.2" "numpy==1.26.4" "scipy==1.13.0" \
        "aiohttp==3.9.5" "websocket-client==1.8.0" "requests==2.31.0" \
        "dash==2.17.1" "dash-bootstrap-components==1.6.0" "plotly==5.22.0" \
        "sqlalchemy==2.0.30" "python-dotenv==1.0.1" "pydantic==2.7.1" \
        "loguru==0.7.2" "schedule==1.2.2" "pytz==2024.1" "tabulate==0.9.0" \
        "matplotlib==3.9.0" "seaborn==0.13.2" "yfinance==0.2.40" \
        "pytest==8.2.0" "pytest-asyncio==0.23.7"

    # Neo Kotak
    pip install -q neo-api-client 2>/dev/null || \
    pip install -q "git+https://github.com/Kotak-Neo/kotak-neo-api.git" 2>/dev/null || \
    echo "  ⚠  neo-api-client skipped (paper mode still works)"
    echo "  ✓ Dependencies installed"
else
    echo "  ✓ Dependencies already installed"
fi

# ── Settings file ─────────────────────────────────────────────
if [ ! -f "config/settings.env" ]; then
    cp config/settings.example.env config/settings.env
    echo "  ⚠  config/settings.env created. Edit it with your Neo Kotak credentials."
fi

echo ""
echo "  ┌───────────────────────────────────────────────┐"
echo "  │  Select:                                      │"
echo "  │  1) Kotak Neo Login UI  (set up credentials)  │"
echo "  │  2) Backtest            (60-day simulation)   │"
echo "  │  3) Paper trading       (live data, sim orders)│"
echo "  │  4) Live trading        (real money ⚠)         │"
echo "  │  5) Dashboard only      (http://localhost:8050)│"
echo "  │  6) Parameter sweep     (optimise thresholds) │"
echo "  └───────────────────────────────────────────────┘"
echo ""
read -p "  Enter choice [1-6]: " CHOICE

case "$CHOICE" in
    1)
        echo ""
        echo "  Opening Kotak Neo Login UI at http://localhost:8051"
        python3 dashboard/login.py &
        LOGIN_PID=$!
        sleep 2
        open "http://localhost:8051" 2>/dev/null || \
            xdg-open "http://localhost:8051" 2>/dev/null || true
        echo "  Complete the login in your browser, then press Ctrl+C when done."
        wait $LOGIN_PID
        ;;
    2)
        echo "  Running backtest..."
        python3 backtest/run_backtest_standalone.py
        [ -f "backtest/results/backtest_report.png" ] && \
            open "backtest/results/backtest_report.png" 2>/dev/null || true
        ;;
    3)
        echo "  Starting paper trading + dashboard..."
        python3 dashboard/app.py &
        DASH_PID=$!
        sleep 2
        open "http://localhost:8050" 2>/dev/null || true
        echo "  Dashboard: http://localhost:8050  |  Ctrl+C to stop"
        python3 main.py --mode paper
        kill "$DASH_PID" 2>/dev/null || true
        ;;
    4)
        echo ""
        echo "  ⚠  Live mode — real money. Make sure you have logged in via option 1 first."
        echo "     This will also prompt for TOTP if no saved session is found."
        echo ""
        python3 dashboard/app.py &
        DASH_PID=$!
        sleep 2
        open "http://localhost:8050" 2>/dev/null || true
        python3 main.py --mode live
        kill "$DASH_PID" 2>/dev/null || true
        ;;
    5)
        python3 dashboard/app.py &
        sleep 2
        open "http://localhost:8050" 2>/dev/null || true
        echo "  Dashboard at http://localhost:8050  |  Ctrl+C to stop"
        wait
        ;;
    6)
        echo "  Running parameter sweep..."
        python3 backtest/param_sweep.py
        [ -f "backtest/results/param_sweep_heatmap.png" ] && \
            open "backtest/results/param_sweep_heatmap.png" 2>/dev/null || true
        ;;
    *)
        echo "  Invalid choice."; exit 1 ;;
esac

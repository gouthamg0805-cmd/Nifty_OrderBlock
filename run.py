"""
run.py — Unified launcher for the Nifty Options MAS Trading System

Starts everything in the correct order:
  1. Login UI        → http://localhost:8051  (complete Kotak Neo login)
  2. Trading Engine  → background process     (agents + strategy)
  3. Dashboard       → http://localhost:8050  (live monitoring)

Usage:
    python run.py                 # interactive menu
    python run.py --mode paper    # paper trading (no login needed)
    python run.py --mode live     # live trading (login required)
    python run.py --mode login    # just the login UI
    python run.py --mode backtest # run backtest
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ── load config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
load_dotenv(ROOT / "config" / "settings.env")

# ── loguru setup ─────────────────────────────────────────────────────────────
from loguru import logger
os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
logger.add("logs/trading.log", rotation="1 day", retention="30 days", level="DEBUG")


# ─── Session helpers ──────────────────────────────────────────────────────────

SESSION_FILE = ROOT / "config" / ".session"

def session_exists() -> bool:
    if not SESSION_FILE.exists():
        return False
    try:
        s = json.loads(SESSION_FILE.read_text())
        return bool(s.get("logged_in"))
    except Exception:
        return False

def session_info() -> dict:
    try:
        return json.loads(SESSION_FILE.read_text())
    except Exception:
        return {}


# ─── Service starters ─────────────────────────────────────────────────────────

def start_login_ui(wait_for_login: bool = True) -> subprocess.Popen | None:
    """Start the login UI on port 8051 and open the browser."""
    login_script = ROOT / "dashboard" / "login.py"
    proc = subprocess.Popen(
        [sys.executable, str(login_script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    webbrowser.open("http://localhost:8051")
    logger.info("Login UI → http://localhost:8051")

    if wait_for_login:
        print("\n" + "─"*55)
        print("  Complete the login in your browser:")
        print("  http://localhost:8051")
        print("─"*55)
        print("  Press Enter here once you see '✓ Logged in successfully'")
        print("─"*55 + "\n")
        input()
        proc.terminate()
        if session_exists():
            s = session_info()
            logger.info(f"Session confirmed — UCC: {s.get('ucc','?')} | {s.get('login_time','')[:19]}")
            return None
        else:
            logger.error("No session found after login. Please try again.")
            sys.exit(1)
    return proc


def start_dashboard() -> subprocess.Popen:
    """Start the Plotly Dash dashboard on port 8050."""
    dash_script = ROOT / "dashboard" / "app.py"
    proc = subprocess.Popen(
        [sys.executable, str(dash_script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)
    webbrowser.open("http://localhost:8050")
    logger.info("Dashboard → http://localhost:8050")
    return proc


def start_trading_engine(mode: str) -> subprocess.Popen:
    """Start the main trading engine."""
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "main.py"), "--mode", mode],
        # Keep stdout/stderr visible so logs appear in terminal
    )
    logger.info(f"Trading engine started (mode={mode}) | PID:{proc.pid}")
    return proc


# ─── Modes ────────────────────────────────────────────────────────────────────

def run_login_only():
    """Just open the login UI."""
    print("\n" + "═"*55)
    print("  Kotak Neo Login UI")
    print("═"*55)
    login_script = ROOT / "dashboard" / "login.py"
    proc = subprocess.Popen([sys.executable, str(login_script)])
    time.sleep(2)
    webbrowser.open("http://localhost:8051")
    logger.info("Login UI → http://localhost:8051  (Ctrl+C to stop)")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


def run_paper():
    """Paper trading — live market data, simulated orders. No login needed."""
    print("\n" + "═"*55)
    print("  Paper Trading Mode")
    print("  Live Nifty data · Simulated orders · No real money")
    print("═"*55 + "\n")

    dash_proc = start_dashboard()
    logger.info("Starting trading engine in paper mode...")
    time.sleep(1)

    try:
        engine_proc = start_trading_engine("paper")
        print("\n" + "─"*55)
        print("  ✓ System running:")
        print("    Dashboard  → http://localhost:8050")
        print("    Mode       → PAPER (simulated orders)")
        print("    Press Ctrl+C to stop everything")
        print("─"*55 + "\n")
        engine_proc.wait()
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        dash_proc.terminate()
        logger.info("All services stopped.")


def run_live():
    """Live trading — real orders through Neo Kotak."""
    print("\n" + "═"*55)
    print("  Live Trading Mode  ⚠  REAL MONEY")
    print("═"*55 + "\n")

    # Step 1: Check / get session
    if session_exists():
        s = session_info()
        logger.info(f"Using saved session — UCC:{s.get('ucc','?')} | {s.get('login_time','')[:19]}")
        print(f"  ✓ Session found for UCC: {s.get('ucc','?')}")
        print(f"    Logged in at: {s.get('login_time','')[:19]}")
        print(f"\n  Re-login to refresh session? (recommended before trading)")
        choice = input("  [y/N]: ").strip().lower()
        if choice == "y":
            SESSION_FILE.unlink(missing_ok=True)
            start_login_ui(wait_for_login=True)
    else:
        print("  No saved session found — please log in first.\n")
        start_login_ui(wait_for_login=True)

    if not session_exists():
        logger.error("Login required for live trading.")
        sys.exit(1)

    # Confirm before going live
    print("\n" + "─"*55)
    print("  ⚠  WARNING: This will place REAL orders with your Kotak account.")
    print(f"     Capital: ₹{os.getenv('TOTAL_CAPITAL','200000')} | "
          f"Max risk/trade: ₹{os.getenv('MAX_RISK_PER_TRADE','2000')}")
    print("─"*55)
    confirm = input("  Type  YES  to confirm live trading: ").strip()
    if confirm != "YES":
        print("  Cancelled.")
        sys.exit(0)

    # Step 2: Start dashboard + engine
    dash_proc = start_dashboard()
    time.sleep(1)

    try:
        engine_proc = start_trading_engine("live")
        print("\n" + "─"*55)
        print("  ✓ System running:")
        print("    Dashboard  → http://localhost:8050")
        print("    Mode       → LIVE (real orders)")
        print("    Square-off → 3:20 PM IST (automatic)")
        print("    Press Ctrl+C to stop and square off all positions")
        print("─"*55 + "\n")
        engine_proc.wait()
    except KeyboardInterrupt:
        logger.warning("Ctrl+C — stopping (positions will be squared off)")
    finally:
        dash_proc.terminate()
        logger.info("All services stopped.")


def run_backtest():
    """Run the standalone backtest."""
    print("\n" + "═"*55)
    print("  Backtest Mode — 60-day Synthetic Simulation")
    print("═"*55 + "\n")
    result = subprocess.run(
        [sys.executable, str(ROOT / "backtest" / "run_backtest_standalone.py")]
    )
    chart = ROOT / "backtest" / "results" / "backtest_report.png"
    if chart.exists():
        subprocess.run(["open", str(chart)], capture_output=True)
        logger.info(f"Chart opened: {chart}")


# ─── Interactive menu ─────────────────────────────────────────────────────────

MENU = """
  ╔══════════════════════════════════════════════════╗
  ║     Nifty Options MAS — Trading System           ║
  ║     Neo Kotak · Python 3.11 · macOS arm64        ║
  ╠══════════════════════════════════════════════════╣
  ║                                                  ║
  ║  1) Login to Kotak Neo          (do this first)  ║
  ║  2) Paper trading + Dashboard   (no real money)  ║
  ║  3) Live trading + Dashboard    (real money ⚠)   ║
  ║  4) Run backtest                                 ║
  ║  5) Open dashboard only                          ║
  ║                                                  ║
  ╚══════════════════════════════════════════════════╝
"""


def interactive_menu():
    print(MENU)

    # Show session status
    if session_exists():
        s = session_info()
        print(f"  ✓ Active session: UCC={s.get('ucc','?')} | {s.get('login_time','')[:19]}\n")
    else:
        print("  ✗ No session — login first (option 1) before live trading\n")

    choice = input("  Enter choice [1-5]: ").strip()

    if choice == "1":
        run_login_only()
    elif choice == "2":
        run_paper()
    elif choice == "3":
        run_live()
    elif choice == "4":
        run_backtest()
    elif choice == "5":
        dash_proc = start_dashboard()
        print("  Dashboard → http://localhost:8050  (Ctrl+C to stop)")
        try:
            dash_proc.wait()
        except KeyboardInterrupt:
            dash_proc.terminate()
    else:
        print("  Invalid choice.")
        sys.exit(1)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Nifty Options MAS — Unified Launcher"
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live", "login", "backtest", "dashboard"],
        default=None,
        help="Run mode (skip menu)",
    )
    args = parser.parse_args()

    if args.mode is None:
        interactive_menu()
    elif args.mode == "login":
        run_login_only()
    elif args.mode == "paper":
        run_paper()
    elif args.mode == "live":
        run_live()
    elif args.mode == "backtest":
        run_backtest()
    elif args.mode == "dashboard":
        dash_proc = start_dashboard()
        print("  Dashboard → http://localhost:8050  (Ctrl+C to stop)")
        try:
            dash_proc.wait()
        except KeyboardInterrupt:
            dash_proc.terminate()


if __name__ == "__main__":
    main()

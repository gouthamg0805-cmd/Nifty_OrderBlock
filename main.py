"""
main.py
Entry point for the Nifty Options MAS Trading System.

Usage:
    python main.py                    # paper mode (default)
    python main.py --mode paper       # paper trading
    python main.py --mode live        # live trading (real money)
    python main.py --mode backtest    # run backtest then exit
"""
from __future__ import annotations
import asyncio
import argparse
import os
import sys
import signal
from datetime import datetime, time as dtime
import pytz
import schedule
from dotenv import load_dotenv
from loguru import logger

# Load config
load_dotenv("config/settings.env")

from core.broker     import BrokerClient
from core.database   import Database
from core.message_bus import bus

from data.fetcher    import DataFetcher

from agents.agent1_market   import MarketIntelligenceAgent
from agents.agent2_strategy import StrategyAgent
from agents.agent3_risk     import RiskManagementAgent
from agents.agent4_execution import ExecutionAgent
from agents.agent5_trailing  import TrailingSlAgent
from agents.agent6_strikes   import StrikeSelectionAgent
from agents.agent7_monitor   import MonitoringAgent
from agents.agent8_learning  import LearningAgent

IST = pytz.timezone("Asia/Kolkata")

# ─── Setup logging ─────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add(sys.stdout, level=os.getenv("LOG_LEVEL", "INFO"),
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add(os.getenv("LOG_FILE", "logs/trading.log"),
           rotation="1 day", retention="30 days", level="DEBUG")


class TradingSystem:

    def __init__(self, mode: str = "paper"):
        self.mode     = mode
        self.running  = False
        self.sq_done  = False

        instrument = os.getenv("INSTRUMENT", "NIFTY").upper().strip()

        logger.info(f"{'='*55}")
        logger.info(f"  {instrument} OPTIONS MAS — Starting in {mode.upper()} mode")
        logger.info(f"  Capital: ₹{os.getenv('TOTAL_CAPITAL','200000')} | "
                    f"Max Risk/Trade: ₹{os.getenv('MAX_RISK_PER_TRADE','2000')}")
        logger.info(f"{'='*55}")

        # Core components
        self.db       = Database(os.getenv("DB_PATH", "logs/trades.db"))
        self.broker   = BrokerClient(mode=mode)
        self.fetcher  = DataFetcher(mode=mode, instrument=instrument)

        # Agents
        self.agent1   = MarketIntelligenceAgent(self.fetcher, poll_interval=30)
        self.agent2   = StrategyAgent()
        self.agent3   = RiskManagementAgent(self.db)
        self.agent6   = StrikeSelectionAgent(self.fetcher, instrument=instrument)
        self.agent4   = ExecutionAgent(self.broker, self.db, instrument=instrument)
        self.agent5   = TrailingSlAgent(self.broker, self.agent4)
        self.agent7   = MonitoringAgent(self.db)
        self.agent8   = LearningAgent(self.db)


    async def run(self):
        self.running = True

        # Login broker
        if self.mode == "live":
            self._handle_live_login()
        else:
            self.broker.login()

        # Give fetcher access to authenticated broker for live data
        if hasattr(self.fetcher, "set_broker"):
            self.fetcher.set_broker(self.broker)

        # Setup graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self._shutdown()))

        # Check if market is open
        now_ist = datetime.now(IST)
        market_open  = dtime(9, 15)
        trading_stop = dtime(15, 20)

        if not self._is_market_hours(now_ist):
            logger.warning(
                f"Market is {'not yet open' if now_ist.time() < market_open else 'closed'}. "
                f"Market hours: 09:15 – 15:20 IST. "
                f"System will start agent loops and wait."
            )

        # Schedule square-off
        sq_time = os.getenv("SQUARE_OFF_TIME", "15:20")
        schedule.every().day.at(sq_time).do(
            lambda: asyncio.create_task(self._square_off())
        )

        # Schedule end-of-day learning
        schedule.every().day.at("15:35").do(self.agent8.run_daily_update)

        # Start all agents concurrently
        logger.info("Starting all 8 agents...")
        await asyncio.gather(
            self.agent1.run(),
            self.agent2.run(),
            self.agent3.run(),
            self.agent4.run(),
            self.agent5.run(),
            self.agent6.run(),
            self.agent7.run(),
            self._schedule_loop(),
            self._update_dashboard_loop(),
            self._session_watcher(),
        )

    def _handle_live_login(self):
        """
        For live mode: check for a saved session from the login UI.

        IMPORTANT: The .session file only tells us that a login *was* completed
        previously.  It does NOT carry a reusable token — Kotak Neo sessions do
        not persist across process restarts.  We must always call _init_client()
        to create a fresh NeoAPI object, then do a full login (TOTP required).

        If credentials are already saved in settings.env we pre-fill them and
        only ask the user to enter a fresh TOTP via the login UI.
        """
        import json as _json
        session_file = os.path.join("config", ".session")

        # Always build a fresh NeoAPI client — never skip this
        self.broker._init_client()

        # Check whether we have credentials saved (so user only needs TOTP)
        creds_saved = bool(
            os.getenv("NEO_CONSUMER_KEY", "").strip()
            and os.getenv("NEO_MOBILE", "").strip()
            and os.getenv("NEO_MPIN", "").strip()
        )

        if creds_saved:
            logger.info(
                "Credentials found in settings.env. "
                "Open the login UI to enter your TOTP and authenticate."
            )
        else:
            logger.info("No saved credentials — opening login UI for full login.")

        self._launch_login_ui()
        # After UI login, broker.login() is called inside the UI flow;
        # but we still call it here to complete the SDK session handshake.
        self.broker.login()

    def _launch_login_ui(self):
        """Open the login UI in the browser and wait for user to log in."""
        import subprocess, webbrowser, time
        login_script = os.path.join(os.path.dirname(__file__), "dashboard", "login.py")
        proc = subprocess.Popen(
            ["python3", login_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        webbrowser.open("http://localhost:8051")
        logger.info("Login UI opened at http://localhost:8051")
        logger.info("Complete login in the browser, then press Enter here to continue...")
        input()
        proc.terminate()

    async def _session_watcher(self):
        """
        Listens for session_expired events.
        When detected: prompts for fresh TOTP and re-authenticates automatically.
        No system restart needed.
        """
        inbox = bus.subscribe("system_event")
        while self.running:
            try:
                event = await inbox.get()
                if event.get("event") == "session_expired":
                    logger.warning(
                        "\n" + "="*55 +
                        "\n  ⚠  KOTAK SESSION EXPIRED" +
                        "\n  All trading is HALTED until you re-login." +
                        "\n" + "="*55
                    )
                    if self.mode == "live":
                        await self._prompt_relogin()
            except Exception as e:
                logger.error(f"[SessionWatcher] Error: {e}")
                await asyncio.sleep(5)

    async def _prompt_relogin(self):
        """
        Prompts user for fresh TOTP and re-authenticates the broker session.
        After success: resets session flags and re-attaches broker to fetcher
        so live data and order execution resume immediately without restart.
        """
        def _do_relogin():
            print("\n" + "─"*55)
            print("  SESSION EXPIRED — Re-login required")
            print("  Stored credentials will be reused.")
            print("  You only need to enter a fresh TOTP.")
            print("─"*55)
            for attempt in range(3):
                try:
                    totp = input(f"  Enter TOTP (attempt {attempt+1}/3): ").strip()
                    if len(totp) != 6 or not totp.isdigit():
                        print("  Invalid TOTP — must be exactly 6 digits.")
                        continue
                    # relogin() calls _init_client() then login() internally
                    self.broker.relogin(totp=totp)

                    # Re-attach the freshly authenticated broker to the fetcher
                    # so _broker_ready() returns True again and live LTP resumes
                    if hasattr(self.fetcher, "set_broker"):
                        self.fetcher.set_broker(self.broker)

                    # Clear session_expired flag on monitoring agent
                    if hasattr(self.agent7, "session_expired"):
                        self.agent7.session_expired = False

                    logger.info("✓ Re-authentication successful — live data and trading resumed")
                    print("\n  ✓ Re-login successful. Trading resumed.")
                    return True
                except Exception as e:
                    logger.error(f"Re-login attempt {attempt+1} failed: {e}")
                    print(f"  ✗ Failed: {e}")
            logger.error("Re-login failed after 3 attempts. Stopping system.")
            self.running = False
            return False

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _do_relogin)

    async def _schedule_loop(self):
        """Run schedule jobs (square-off, learning) in async context."""
        while self.running:
            schedule.run_pending()
            await asyncio.sleep(30)

    async def _update_dashboard_loop(self):
        """Pushes state to dashboard module."""
        while self.running:
            try:
                from dashboard.app import update_state
                update_state(
                    self.agent7.current_state,
                    self.agent7.active_trades,
                    self.agent7.events,
                )
            except Exception:
                pass
            await asyncio.sleep(5)

    async def _square_off(self):
        if self.sq_done:
            return
        self.sq_done = True
        logger.warning("⚠️  SQUARE-OFF TIME: Closing all positions")
        await self.agent4.square_off_all()

    async def _shutdown(self):
        logger.info("Shutdown signal received. Closing positions...")
        await self.agent4.square_off_all()
        self.running = False
        self.agent1.stop()
        logger.info("System shut down cleanly.")
        asyncio.get_event_loop().stop()

    def _is_market_hours(self, now_ist) -> bool:
        from core.market_clock import market_clock
        return market_clock.is_open()


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Nifty Options MAS Trading System")
    parser.add_argument(
        "--mode", choices=["paper", "live", "backtest"],
        default=os.getenv("TRADING_MODE", "paper"),
        help="Trading mode"
    )
    args = parser.parse_args()

    if args.mode == "backtest":
        logger.info("Launching backtest mode...")
        os.system("python backtest/run_backtest.py")
        return

    if args.mode == "live":
        confirm = input(
            "\n⚠️  LIVE TRADING MODE — Real money will be used.\n"
            "   Capital: ₹2,00,000 | Max Risk/Trade: ₹2,000\n"
            "   Type 'YES I CONFIRM' to proceed: "
        )
        if confirm.strip() != "YES I CONFIRM":
            logger.info("Aborted. Run with --mode paper for simulation.")
            return

    system = TradingSystem(mode=args.mode)

    try:
        asyncio.run(system.run())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Exiting.")


if __name__ == "__main__":
    main()

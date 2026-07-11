"""
core/market_clock.py
Single source of truth for all NSE market hours logic.

Rules encoded:
  - NSE trades Monday–Friday only (Saturday/Sunday closed)
  - Market open:          09:15 IST  (retail order entry)
  - Pre-open session:     09:00–09:15 (no regular orders)
  - Trading allowed from: 09:20 IST  (skip first 5 chaotic candles)
  - No new trades after:  15:00 IST  (too close to close)
  - Hard square-off:      15:20 IST  (all positions closed)
  - Market close:         15:30 IST

Usage:
    from core.market_clock import market_clock

    if not market_clock.is_open():
        # market closed — do nothing
        ...

    if not market_clock.can_trade():
        # market open but trading window not yet started
        ...

    status = market_clock.status()   # human-readable string for logs
"""
from __future__ import annotations
from datetime import datetime, time, date
from typing import NamedTuple
import pytz

IST = pytz.timezone("Asia/Kolkata")

# ── Trading session times (IST) ──────────────────────────────────────────────
_MARKET_OPEN        = time(9, 15)    # Market opens (retail)
_TRADING_START      = time(9, 20)    # Start scanning for signals (skip first 5 candles)
_NEW_TRADE_CUTOFF   = time(15, 0)    # No new entry orders after this time
_SQUARE_OFF_TIME    = time(15, 20)   # Force-close all positions
_MARKET_CLOSE       = time(15, 30)   # Market closes


class MarketStatus(NamedTuple):
    is_open:       bool    # True if market is currently open (Mon–Fri, 9:15–15:30)
    can_trade:     bool    # True if new entry signals are allowed
    must_close:    bool    # True if square-off should trigger (≥ 15:20)
    reason:        str     # Human-readable reason (for logs)
    now_ist:       datetime


class MarketClock:
    """
    All market hours decisions go through this class.
    Agents import the singleton `market_clock` at the bottom of this file.
    """

    # NSE public holidays 2026 (full-day closures — no trading at all)
    # Note: expiry holidays (where only expiry shifts) are in agent6_strikes.py
    _NSE_HOLIDAYS_2026: set[date] = {
        date(2026, 1, 26),   # Republic Day
        date(2026, 2, 19),   # Chhatrapati Shivaji Maharaj Jayanti
        date(2026, 3, 20),   # Gudi Padwa / Ugadi
        date(2026, 3, 25),   # Holi
        date(2026, 4, 14),   # Dr. Ambedkar Jayanti / Good Friday
        date(2026, 5, 1),    # Maharashtra Day
        date(2026, 6, 15),   # Bakri Id (Eid ul-Adha) — tentative
        date(2026, 8, 15),   # Independence Day
        date(2026, 8, 27),   # Ganesh Chaturthi
        date(2026, 10, 2),   # Gandhi Jayanti / Mahatma Gandhi Jayanti
        date(2026, 10, 20),  # Dussehra
        date(2026, 11, 10),  # Diwali Laxmi Pujan (Muhurat trading only)
        date(2026, 11, 24),  # Diwali Balipratipada
        date(2026, 11, 25),  # Gurunanak Jayanti
        date(2026, 12, 25),  # Christmas
    }

    def now_ist(self) -> datetime:
        return datetime.now(IST)

    def is_weekday(self) -> bool:
        """True if today is Monday–Friday."""
        return self.now_ist().weekday() < 5   # 0=Mon … 4=Fri, 5=Sat, 6=Sun

    def is_nse_holiday(self) -> bool:
        """True if today is a declared NSE trading holiday."""
        return self.now_ist().date() in self._NSE_HOLIDAYS_2026

    def is_open(self) -> bool:
        """
        True if the NSE market is currently open.
        Conditions: weekday + not holiday + 09:15 ≤ time ≤ 15:30
        """
        now = self.now_ist()
        t   = now.time().replace(tzinfo=None)
        return (
            self.is_weekday()
            and not self.is_nse_holiday()
            and _MARKET_OPEN <= t <= _MARKET_CLOSE
        )

    def can_trade(self) -> bool:
        """
        True if the system is allowed to place NEW entry orders.
        Stricter than is_open():
          - Must be a trading day (weekday, not holiday)
          - Time must be 09:20–15:00 IST
          - Not in the square-off window (≥ 15:20)
        """
        now = self.now_ist()
        t   = now.time().replace(tzinfo=None)
        return (
            self.is_weekday()
            and not self.is_nse_holiday()
            and _TRADING_START <= t < _NEW_TRADE_CUTOFF
        )

    def must_square_off(self) -> bool:
        """
        True if all open positions must be closed immediately.
        Triggers at 15:20 IST on trading days.
        """
        now = self.now_ist()
        t   = now.time().replace(tzinfo=None)
        return (
            self.is_weekday()
            and not self.is_nse_holiday()
            and t >= _SQUARE_OFF_TIME
        )

    def minutes_to_open(self) -> int:
        """Minutes until market opens (09:15 IST). Returns 0 if already open."""
        now = self.now_ist()
        t   = now.time().replace(tzinfo=None)
        if t >= _MARKET_OPEN:
            return 0
        open_dt  = now.replace(hour=9, minute=15, second=0, microsecond=0)
        delta    = (open_dt - now).total_seconds()
        return max(0, int(delta / 60))

    def seconds_to_trading_start(self) -> int:
        """Seconds until 09:20 IST (signal scanning start). Returns 0 if past."""
        now = self.now_ist()
        t   = now.time().replace(tzinfo=None)
        if t >= _TRADING_START:
            return 0
        start_dt = now.replace(hour=9, minute=20, second=0, microsecond=0)
        return max(0, int((start_dt - now).total_seconds()))

    def status(self) -> str:
        """Human-readable market status string for log messages."""
        now  = self.now_ist()
        t    = now.time().replace(tzinfo=None)
        day  = now.strftime("%A")
        ts   = now.strftime("%H:%M IST")

        if not self.is_weekday():
            return f"MARKET CLOSED — {day} (weekend). Next open: Monday 09:15 IST"

        if self.is_nse_holiday():
            return f"MARKET CLOSED — NSE holiday today ({now.strftime('%d-%b-%Y')})"

        if t < _MARKET_OPEN:
            mins = self.minutes_to_open()
            return f"PRE-MARKET — Opens in {mins} min at 09:15 IST"

        if _MARKET_OPEN <= t < _TRADING_START:
            secs = self.seconds_to_trading_start()
            return f"MARKET OPEN — Signal scanning starts in {secs}s at 09:20 IST"

        if _TRADING_START <= t < _NEW_TRADE_CUTOFF:
            return f"TRADING ACTIVE — {ts}"

        if _NEW_TRADE_CUTOFF <= t < _SQUARE_OFF_TIME:
            return f"NO NEW TRADES — Approaching close. Square-off at 15:20 IST"

        if t >= _SQUARE_OFF_TIME:
            return f"SQUARE-OFF TIME — Closing all positions"

        return f"MARKET OPEN — {ts}"

    def wait_seconds(self) -> int:
        """
        How long agents should sleep before their next poll.
        Returns a short interval when close to market open,
        a longer one when market is clearly closed.
        """
        now = self.now_ist()
        t   = now.time().replace(tzinfo=None)

        if not self.is_weekday() or self.is_nse_holiday():
            return 3600   # 1 hour — check again next hour on weekends/holidays

        if t < time(9, 0):
            # More than 15 min before open — sleep until 09:00
            open_dt = now.replace(hour=9, minute=0, second=0, microsecond=0)
            return max(60, int((open_dt - now).total_seconds()))

        if t < _MARKET_OPEN:
            return 30   # Close to open — poll every 30s

        if t > _MARKET_CLOSE:
            return 3600  # After close — sleep 1 hour

        return 30   # During market hours — normal 30s poll


# ── Singleton used by all agents ─────────────────────────────────────────────
market_clock = MarketClock()

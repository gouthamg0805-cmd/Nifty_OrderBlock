"""
agents/agent6_strikes.py
Agent 6: Strike Price Selection — Nifty 50 OR Sensex Index Options

Instrument:  NIFTY  (NSE)  — strike step 50,  lot 65,  expiry TUESDAY
             SENSEX (BSE)  — strike step 100, lot 10,  expiry FRIDAY

Exchange:    nse_fo  (Nifty)  |  bse_fo  (Sensex)
Symbol format:
  NIFTY27JUN2422000CE  /  NIFTY27JUN2422000PE
  SENSEX27JUN2480000CE /  SENSEX27JUN2480000PE

Selection criteria:
  1. Liquidity (OI > threshold)
  2. Volume
  3. Tight bid-ask spread
  4. ATM or 1-strike ITM preferred
  5. Premium in ₹50–₹400 range (suitable for 2-lot trade)
"""
from __future__ import annotations
import asyncio
import os
import math
from datetime import datetime, timedelta, date
from typing import List, Optional
from loguru import logger

from core.models import TradeSignal, TradeBias, OptionStrike, OptionType
from core.message_bus import bus

# ── Instrument constants ────────────────────────────────────────────────────
_NIFTY_CFG = dict(
    symbol_prefix = "NIFTY",
    exchange      = "nse_fo",
    strike_step   = 50,
    lot_size      = 65,
    expiry_weekday = 1,          # Tuesday (Mon=0 … Sun=6)
)
_SENSEX_CFG = dict(
    symbol_prefix = "SENSEX",
    exchange      = "bse_fo",
    strike_step   = 100,
    lot_size      = 10,
    expiry_weekday = 4,          # Friday
)


class StrikeSelectionAgent:

    # ── Holiday tables ───────────────────────────────────────────────────────
    # Dates where the NORMAL weekly expiry day is an NSE/BSE holiday.
    # On these days expiry shifts to the PREVIOUS trading day (typically
    # the day before — Monday for Tuesday expiry, Thursday for Friday expiry).
    #
    # Source: NSE/BSE circulars + SEBI directive
    #
    # NIFTY 2026 — Tuesdays that are NSE holidays → expiry on Monday
    _NIFTY_EXPIRY_HOLIDAYS: set[str] = {
        "2026-04-14",  # Dr. Ambedkar Jayanti / Good Friday  → expiry Mon Apr 13
        "2026-10-20",  # Dussehra                            → expiry Mon Oct 19
        "2026-11-10",  # Diwali Laxmi Pujan                 → expiry Mon Nov  9
        "2026-11-24",  # Diwali Balipratipada               → expiry Mon Nov 23
    }
    # SENSEX 2026 — Fridays that are BSE holidays → expiry on Thursday
    _SENSEX_EXPIRY_HOLIDAYS: set[str] = {
        "2026-01-30",  # Tentative BSE holiday               → expiry Thu Jan 29
        "2026-08-15",  # Independence Day (Friday)           → expiry Thu Aug 13
        # Add additional BSE Friday holidays as declared
    }

    def __init__(self, data_fetcher, instrument: str = "NIFTY"):
        self.fetcher = data_fetcher
        self._inbox  = bus.subscribe("trade_signal_raw")

        instrument = instrument.upper().strip()
        if instrument == "SENSEX":
            cfg = _SENSEX_CFG
        else:
            cfg = _NIFTY_CFG

        self.symbol_prefix  = cfg["symbol_prefix"]
        self.exchange       = cfg["exchange"]
        self.strike_step    = cfg["strike_step"]
        self.lot_size       = cfg["lot_size"]
        self.expiry_weekday = cfg["expiry_weekday"]   # 1=Tue (Nifty), 4=Fri (Sensex)
        self._expiry_holidays = (
            self._SENSEX_EXPIRY_HOLIDAYS
            if instrument == "SENSEX"
            else self._NIFTY_EXPIRY_HOLIDAYS
        )

        logger.info(
            f"Agent 6 (Strike Selection) configured for {self.symbol_prefix} | "
            f"Strike step: {self.strike_step} | Lot: {self.lot_size} | "
            f"Expiry weekday: {self.expiry_weekday}"
        )

    async def run(self):
        logger.info(
            f"Agent 6 (Strike Selection) started — "
            f"{self.symbol_prefix} Options only"
        )
        while True:
            try:
                signal: TradeSignal = await self._inbox.get()
                enriched = await self._select_strike(signal)
                if enriched:
                    await bus.publish("trade_signal", enriched)
            except Exception as e:
                logger.error(f"[Agent6] Error: {e}")

    async def _select_strike(self, signal: TradeSignal) -> Optional[TradeSignal]:
        spot        = signal.market_state.spot_price
        option_type = OptionType.CALL if signal.bias == TradeBias.LONG_CALL else OptionType.PUT
        expiry      = self._nearest_expiry()

        logger.debug(
            f"[Agent6] Selecting {self.symbol_prefix} {option_type.value} option | "
            f"Spot:₹{spot:.2f} | Expiry:{expiry}"
        )

        # Fetch options chain (broker live or synthetic)
        strikes = await self.fetcher.get_options_chain(
            expiry      = expiry,
            option_type = option_type.value,
            spot        = spot,
        )

        # Filter to confirmed symbol prefix only
        strikes = [s for s in strikes if s.symbol.startswith(self.symbol_prefix)]

        # Determine actual data source: live strikes have a trading_symbol attached
        # by the fetcher after a successful search_scrip + quotes round-trip.
        # Synthetic strikes never have trading_symbol set.
        is_live_data = any(getattr(s, "trading_symbol", None) for s in strikes)

        if not strikes:
            best = self._build_atm_strike(spot, option_type, expiry)
            logger.info(
                f"[Agent6] Fallback ATM strike: {best.symbol} | "
                f"Premium:₹{best.ltp:.2f} [SYNTHETIC]"
            )
        else:
            ranked = self._rank_strikes(strikes, spot)
            best   = ranked[0]
            price_source = "LIVE" if is_live_data else "SYNTHETIC (paper/fallback)"
            logger.info(
                f"[Agent6] Selected: {best.symbol} | "
                f"Premium:₹{best.ltp:.2f} [{price_source}] | "
                f"OI:{best.oi:,.0f} | Score:{best.score:.1f}"
            )

        signal.selected_strike = best
        return signal

    def _rank_strikes(self, strikes: List[OptionStrike], spot: float) -> List[OptionStrike]:
        atm = round(spot / self.strike_step) * self.strike_step

        for s in strikes:
            score = 0.0
            dist  = abs(s.strike_price - atm)

            # ── Distance from ATM ──────────────────────────────────────────
            one_strike = self.strike_step
            if dist == 0:                     score += 40
            elif dist <= one_strike:          score += 30   # 1 strike away
            elif dist <= 2 * one_strike:      score += 15   # 2 strikes away
            elif dist <= 3 * one_strike:      score += 5    # 3 strikes away

            # ── Open Interest (liquidity) ──────────────────────────────────
            if s.oi > 1_000_000:   score += 25
            elif s.oi > 500_000:   score += 20
            elif s.oi > 100_000:   score += 10
            elif s.oi < 10_000:    score -= 15

            # ── Volume ────────────────────────────────────────────────────
            if s.volume > 200_000: score += 15
            elif s.volume > 100_000: score += 10
            elif s.volume > 50_000:  score += 5

            # ── Bid-ask spread ────────────────────────────────────────────
            if s.ltp > 0 and s.ask > 0 and s.bid > 0:
                spread_pct = (s.ask - s.bid) / s.ltp
                s.spread_pct = spread_pct
                if spread_pct < 0.01:   score += 15
                elif spread_pct < 0.02: score += 10
                elif spread_pct < 0.05: score += 5
                elif spread_pct > 0.10: score -= 10

            # ── Premium range ─────────────────────────────────────────────
            if 80 <= s.ltp <= 400:     score += 20
            elif 50 <= s.ltp < 80:     score += 10
            elif 400 < s.ltp <= 600:   score += 5
            elif s.ltp < 30:           score -= 20
            elif s.ltp > 800:          score -= 15

            s.score = score

        strikes.sort(key=lambda x: x.score, reverse=True)
        return strikes

    def _build_atm_strike(
        self, spot: float, option_type: OptionType, expiry: str
    ) -> OptionStrike:
        """Build an ATM strike using Black-Scholes when no chain data available."""
        K      = round(spot / self.strike_step) * self.strike_step
        S, r   = spot, 0.065
        T      = max(1, self._days_to_expiry(expiry)) / 365
        sigma  = 0.18

        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        from scipy.stats import norm
        import math as m
        if option_type == OptionType.CALL:
            ltp = S * norm.cdf(d1) - K * m.exp(-r*T) * norm.cdf(d2)
        else:
            ltp = K * m.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        ltp = max(round(float(ltp), 2), 5.0)
        sym = f"{self.symbol_prefix}{expiry}{int(K)}{option_type.value}"

        return OptionStrike(
            symbol       = sym,
            strike_price = float(K),
            option_type  = option_type,
            expiry       = expiry,
            ltp          = ltp,
            oi           = 500000.0,
            volume       = 100000.0,
            bid          = round(ltp * 0.99, 2),
            ask          = round(ltp * 1.01, 2),
            spread_pct   = 0.02,
            score        = 60.0,
        )

    def _nearest_expiry(self) -> str:
        """
        Returns the nearest weekly expiry date for the configured instrument.

        Rules:
          - NIFTY:  expiry is every TUESDAY  (weekday=1)
          - SENSEX: expiry is every FRIDAY   (weekday=4)
          - If the expiry day is an exchange holiday, expiry shifts to the
            PREVIOUS calendar day (usually Monday for Tuesday, Thursday for Friday).
          - If today IS the expiry day but past 2:30 PM, use the NEXT expiry.
          - Format: DDMMMYY  e.g. 15APR26
        """
        MONTHS = ["JAN","FEB","MAR","APR","MAY","JUN",
                  "JUL","AUG","SEP","OCT","NOV","DEC"]

        today     = datetime.now()
        today_d   = today.date()

        # Days ahead until next occurrence of the expiry weekday
        days_ahead = (self.expiry_weekday - today.weekday()) % 7

        # If today IS the expiry day and past 2:30 PM (870 min), roll to next week
        total_min = today.hour * 60 + today.minute
        if days_ahead == 0 and total_min >= 870:   # 14*60+30 = 870
            days_ahead = 7

        expiry_d = today_d + timedelta(days=days_ahead)

        # Holiday check: if the expiry day is an exchange holiday, shift to previous day
        # Keep stepping back until we land on a non-holiday weekday
        max_shift = 5  # guard against infinite loop
        for _ in range(max_shift):
            key = expiry_d.strftime("%Y-%m-%d")
            if key in self._expiry_holidays:
                expiry_d = expiry_d - timedelta(days=1)
                logger.info(
                    f"[Agent6] Expiry holiday detected — "
                    f"expiry shifted to {expiry_d} ({expiry_d.strftime('%A')})"
                )
            else:
                break

        return f"{expiry_d.day:02d}{MONTHS[expiry_d.month-1]}{str(expiry_d.year)[2:]}"

    def _days_to_expiry(self, expiry_str: str) -> int:
        """Parse expiry string (e.g. '15APR26') and return calendar days until expiry."""
        MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
                  "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
        try:
            day       = int(expiry_str[:2])
            month     = MONTHS[expiry_str[2:5]]
            year      = 2000 + int(expiry_str[5:])
            expiry_dt = datetime(year, month, day, 15, 30)
            return max(1, (expiry_dt - datetime.now()).days)
        except Exception:
            return 7

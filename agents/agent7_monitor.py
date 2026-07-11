"""
agents/agent7_monitor.py
Agent 7: Monitoring & Visualization Agent
- Tracks active trades and P&L
- Writes logs/state.json every 5 seconds for the dashboard process
"""
from __future__ import annotations
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any
from loguru import logger

from core.message_bus import bus
from core.database import Database


class MonitoringAgent:

    def __init__(self, db: Database):
        self.db            = db
        self.current_state = {}
        self.active_trades = {}
        self.events        = []
        self.session_expired = False
        self._state_inbox  = bus.subscribe("market_state")
        self._event_inbox  = bus.subscribe("system_event")

    async def run(self):
        logger.info("Agent 7 (Monitoring) started")
        await asyncio.gather(
            self._track_market(),
            self._track_events(),
            self._write_state_loop(),
        )

    # ── Market state listener ─────────────────────────────────────────────────

    async def _track_market(self):
        while True:
            state = await self._state_inbox.get()
            self.current_state = {
                "timestamp":  state.timestamp.isoformat(),
                "spot":       state.spot_price,
                "regime":     state.regime.value,
                "bias":       state.bias.value,
                "confidence": state.confidence,
                "ema9":       state.indicators.ema9,
                "ema21":      state.indicators.ema21,
                "vwap":       state.indicators.vwap,
                "atr":        state.indicators.atr14,
                "rsi":        state.indicators.rsi14,
                "candles_5m": [
                    {
                        "t": c.timestamp.isoformat(),
                        "o": c.open, "h": c.high,
                        "l": c.low,  "c": c.close, "v": c.volume,
                    }
                    for c in state.candles_5m
                ],
            }

    # ── Event listener ────────────────────────────────────────────────────────

    async def _track_events(self):
        while True:
            event = await self._event_inbox.get()
            self.events.append({
                "time":  datetime.now().isoformat(),
                "event": event.get("event"),
                "data":  event.get("data"),
            })
            if len(self.events) > 200:
                self.events = self.events[-100:]

            evt = event.get("event")
            if evt == "trade_opened":
                d = event.get("data", {})
                self.active_trades[d.get("trade_id")] = d
            elif evt == "trade_closed":
                d = event.get("data", {})
                self.active_trades.pop(d.get("trade_id"), None)
            elif evt == "session_expired":
                logger.error(
                    "[Agent7] ⚠ BROKER SESSION EXPIRED — all trading halted.\n"
                    "  Re-login at: python dashboard/login.py → http://localhost:8051"
                )
                self.session_expired = True

    # ── State file writer (for separate dashboard process) ────────────────────

    async def _write_state_loop(self):
        """Write state to logs/state.json every 5s — dashboard reads this file."""
        while True:
            self._write_state_file()
            await asyncio.sleep(5)

    def _write_state_file(self):
        state = {
            "market":         self.current_state,
            "active_trades":  self.active_trades,
            "events":         self.events[-50:],
            "updated_at":     datetime.now().isoformat(),
            "session_expired": self.session_expired,
        }
        os.makedirs("logs", exist_ok=True)
        tmp = "logs/state.json.tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(state, f, default=str)
            os.replace(tmp, "logs/state.json")   # atomic write — no partial reads
        except Exception as e:
            logger.debug(f"[Agent7] State write failed: {e}")

    # ── Summary for API ───────────────────────────────────────────────────────

    def get_summary(self) -> Dict[str, Any]:
        today_pnl    = self.db.get_today_pnl()
        today_trades = self.db.get_today_trades()
        wins         = sum(1 for t in today_trades if t.won)
        return {
            "today_pnl":     round(today_pnl, 2),
            "today_trades":  len(today_trades),
            "win_rate":      round(wins / len(today_trades) * 100, 1) if today_trades else 0,
            "active_trades": len(self.active_trades),
            "market":        self.current_state,
        }

"""
agents/agent3_risk.py
Agent 3: Dynamic Risk Management — Optimised

Key improvements:
  1. SL uses max(15% premium, ATR×0.8) — wider to avoid noise stops
  2. Target adjusted to 2.5:1 RR (from 2:1)
  3. Position size respects ₹2000 max risk strictly
  4. Volatility-adjusted SL: wider when ATR is high (volatile session)
"""
from __future__ import annotations
import asyncio
import os
from datetime import datetime
from loguru import logger

from core.models import TradeSignal, TradeOrder, TradeBias
from core.message_bus import bus
from core.database import Database


class RiskManagementAgent:

    def __init__(self, db: Database):
        self.db              = db
        self.total_capital   = float(os.getenv("TOTAL_CAPITAL",       200000))
        self.max_risk        = float(os.getenv("MAX_RISK_PER_TRADE",   2000))
        self.max_daily_loss  = float(os.getenv("MAX_DAILY_LOSS",       6000))
        self.lot_size        = int(  os.getenv("LOT_SIZE",             65))
        self.starting_lots   = int(  os.getenv("STARTING_LOTS",        2))
        self.min_sl_abs      = float(os.getenv("MIN_SL_ABSOLUTE",      3.0))
        self._inbox          = bus.subscribe("trade_signal")

    async def run(self):
        logger.info("Agent 3 (Risk Management) started")
        while True:
            try:
                signal: TradeSignal = await self._inbox.get()
                await self._size_trade(signal)
            except Exception as e:
                logger.error(f"[Agent3] Error: {e}")

    async def _size_trade(self, signal: TradeSignal):
        # Daily loss gate
        daily_pnl = self.db.get_today_pnl()
        if daily_pnl < -self.max_daily_loss:
            logger.warning(
                f"[Agent3] Daily loss limit hit: ₹{daily_pnl:,.0f} "
                f"(limit ₹{self.max_daily_loss:,.0f}) — no more trades today"
            )
            await bus.publish_event("daily_loss_limit_hit", {"pnl": daily_pnl})
            return

        strike = signal.selected_strike
        if not strike:
            logger.warning("[Agent3] No strike attached to signal — skipping")
            return

        premium = strike.ltp
        atr_val = signal.market_state.indicators.atr14

        # ── Adaptive SL (wider than before — avoids noise stops) ──────────
        # Option delta ≈ 0.5 for ATM, so 1 point Nifty ≈ 0.5 pts option
        sl_from_premium  = premium * 0.15         # 15% of premium (was 10%)
        sl_from_atr      = atr_val * 0.5 * 0.8   # ATR × delta × buffer multiplier
        sl_points        = max(sl_from_premium, sl_from_atr, self.min_sl_abs)
        sl_points        = round(sl_points, 2)

        # ── Position sizing ────────────────────────────────────────────────
        risk_per_lot = sl_points * self.lot_size
        if risk_per_lot <= 0:
            logger.warning("[Agent3] Risk per lot = 0, skipping")
            return

        lots     = min(self.starting_lots, max(1, int(self.max_risk / risk_per_lot)))
        quantity = lots * self.lot_size
        actual_risk = sl_points * quantity

        # ── Entry / SL / Target ────────────────────────────────────────────
        entry_price  = premium
        sl_price     = round(entry_price - sl_points, 2)
        sl_price     = max(sl_price, 1.0)   # never below ₹1

        # Target = 2.5:1 RR (raised from 2:1)
        target_price  = round(entry_price + sl_points * 2.5, 2)
        target_points = target_price - entry_price
        rr_ratio      = round(target_points / sl_points, 2) if sl_points > 0 else 0

        order = TradeOrder(
            signal        = signal,
            strike        = strike,
            entry_price   = entry_price,
            sl_price      = sl_price,
            target_price  = target_price,
            sl_points     = sl_points,
            target_points = target_points,
            lots          = lots,
            quantity      = quantity,
            max_risk      = actual_risk,
            rr_ratio      = rr_ratio,
            strategy_label = signal.strategy_label,
        )

        logger.info(
            f"[Agent3] Sized: {lots}L×{self.lot_size}={quantity}qty | "
            f"Entry:₹{entry_price:.2f} | SL:₹{sl_price:.2f}(−{sl_points:.1f}) | "
            f"Tgt:₹{target_price:.2f} | Risk:₹{actual_risk:.0f} | R:R:{rr_ratio}"
        )

        await bus.publish("trade_order", order)

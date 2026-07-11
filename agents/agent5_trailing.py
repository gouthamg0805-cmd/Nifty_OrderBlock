"""
agents/agent5_trailing.py
Agent 5: Trailing Stop-Loss Agent
- Polls active trades every 15 seconds
- Activates trailing after +0.5R
- Dynamically selects trail method based on market regime
- Modifies SL via broker API
"""
from __future__ import annotations
import asyncio
import os
from datetime import datetime
from loguru import logger

from core.models import (
    ExecutedTrade, TrailingState, TrailMethod,
    MarketRegime, TradeBias
)
from core.message_bus import bus
from core.broker import BrokerClient


class TrailingSlAgent:

    def __init__(self, broker: BrokerClient, execution_agent):
        self.broker           = broker
        self.execution_agent  = execution_agent
        self.trail_activation = float(os.getenv("TRAIL_ACTIVATION_R", 0.5))
        self.trail_atr_mult   = float(os.getenv("TRAIL_ATR_MULTIPLIER", 1.5))
        self.poll_interval    = 15  # seconds
        self._trailing: dict[str, TrailingState] = {}
        self._inbox           = bus.subscribe("executed_trade")
        self._market_inbox    = bus.subscribe("market_state")
        self._last_market_state = None

    async def run(self):
        logger.info("Agent 5 (Trailing SL) started")
        await asyncio.gather(
            self._listen_new_trades(),
            self._listen_market(),
            self._trail_loop(),
        )

    async def _listen_new_trades(self):
        while True:
            trade: ExecutedTrade = await self._inbox.get()
            state = TrailingState(
                trade         = trade,
                current_sl    = trade.order.sl_price,
                trail_method  = TrailMethod.ATR_BASED,
                activated     = False,
                highest_pnl   = 0.0,
            )
            self._trailing[trade.trade_id] = state
            logger.info(f"[Agent5] Tracking trade: {trade.trade_id}")

    async def _listen_market(self):
        while True:
            self._last_market_state = await self._market_inbox.get()

    async def _trail_loop(self):
        while True:
            await asyncio.sleep(self.poll_interval)
            for trade_id, state in list(self._trailing.items()):
                try:
                    await self._update_trail(state)
                except Exception as e:
                    logger.error(f"[Agent5] Trail error {trade_id}: {e}")

    async def _update_trail(self, state: TrailingState):
        trade    = state.trade
        order    = trade.order
        market   = self._last_market_state

        if not market:
            return

        # Get current option LTP
        current_ltp = await self._get_current_ltp(trade)
        if not current_ltp:
            return

        entry  = trade.entry_fill_price
        sl_pts = order.sl_points
        r      = sl_pts  # 1R = SL distance

        unrealized_pnl = (current_ltp - entry) * order.quantity
        unrealized_r   = (current_ltp - entry) / r if r > 0 else 0

        # Track peak
        if unrealized_pnl > state.highest_pnl:
            state.highest_pnl = unrealized_pnl

        # ── Check: has target been hit? ───────────────────────────────────
        if current_ltp >= order.target_price:
            logger.info(f"[Agent5] TARGET HIT: {trade.trade_id} @ ₹{current_ltp:.2f}")
            await self.execution_agent.close_trade(trade, current_ltp, "TARGET_HIT")
            self._trailing.pop(trade.trade_id, None)
            return

        # ── Check: SL hit (backup, broker handles primary) ───────────────
        if current_ltp <= state.current_sl:
            logger.info(f"[Agent5] SL HIT: {trade.trade_id} @ ₹{current_ltp:.2f}")
            await self.execution_agent.close_trade(trade, current_ltp, "SL_HIT")
            self._trailing.pop(trade.trade_id, None)
            return

        # ── Activation check ──────────────────────────────────────────────
        if not state.activated and unrealized_r >= self.trail_activation:
            state.activated = True
            state.trail_method = self._choose_trail_method(market)
            logger.info(
                f"[Agent5] Trail ACTIVATED: {trade.trade_id} | "
                f"Method:{state.trail_method.value} | +{unrealized_r:.2f}R"
            )

        if not state.activated:
            return

        # ── Compute new SL based on method ────────────────────────────────
        new_sl = self._compute_new_sl(state, current_ltp, market, order)

        # Only move SL UP (never down)
        if new_sl <= state.current_sl:
            return

        # ── Fixed RR locks ────────────────────────────────────────────────
        cost_price = entry  # Break-even
        if unrealized_r >= 1.0:
            new_sl = max(new_sl, cost_price)         # at 1R → move to cost
        if unrealized_r >= 1.5:
            new_sl = max(new_sl, entry + r * 0.5)    # at 1.5R → lock 0.5R profit

        if new_sl <= state.current_sl:
            return

        # ── Update broker SL ──────────────────────────────────────────────
        state.current_sl = round(new_sl, 2)
        state.sl_updates.append(state.current_sl)

        self.broker.modify_order(
            order_id      = trade.sl_order_id,
            trigger_price = state.current_sl,
        )

        logger.info(
            f"[Agent5] SL trailed → ₹{state.current_sl:.2f} | "
            f"Method:{state.trail_method.value} | LTP:₹{current_ltp:.2f} | "
            f"+{unrealized_r:.2f}R"
        )

    def _choose_trail_method(self, market) -> TrailMethod:
        regime = market.regime
        if regime in [MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR]:
            return TrailMethod.CANDLE_BASED
        elif regime == MarketRegime.VOLATILE:
            return TrailMethod.ATR_BASED
        else:
            return TrailMethod.VWAP

    def _compute_new_sl(self, state, current_ltp, market, order) -> float:
        atr     = market.indicators.atr14
        vwap    = market.indicators.vwap
        candles = market.candles_5m

        method = state.trail_method

        if method == TrailMethod.CANDLE_BASED and candles:
            # Trail below the low of the previous candle
            prev_low = candles[-2].low if len(candles) >= 2 else candles[-1].low
            return prev_low - 2  # small buffer

        elif method == TrailMethod.VWAP:
            # Use VWAP as dynamic support
            return vwap - atr * 0.3

        elif method == TrailMethod.ATR_BASED:
            # SL = current_ltp - ATR × multiplier
            return current_ltp - atr * self.trail_atr_mult

        else:  # FIXED_RR
            return state.current_sl

    async def _get_current_ltp(self, trade: ExecutedTrade) -> float | None:
        """
        In paper mode: simulate price movement based on market data.test
        In live mode: fetch from broker quote API.
        """
        if trade.mode == "paper" and self._last_market_state:
            # Simulate: option LTP moves roughly 0.6 × Nifty spot move (delta)
            spot_now   = self._last_market_state.spot_price
            entry_spot = trade.order.signal.market_state.spot_price
            spot_move  = spot_now - entry_spot
            delta      = 0.5  # ATM delta approximation
            ltp_change = spot_move * delta * 0.5
            return max(1.0, trade.entry_fill_price + ltp_change)
        return trade.entry_fill_price  # fallback

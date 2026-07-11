"""
agents/agent4_execution.py
Agent 4: Execution Agent — Neo Kotak Integration

Order flow for each trade:
  1. Entry  → place_order(MKT, B)         ← buy the option
  2. SL     → place_order(SL-M, S)        ← broker-side stop loss (MANDATORY)
  3. Target → place_order(L, S)           ← limit sell at target (optional)

All prices are strings to the broker API.
product = MIS (intraday — auto squares off at 3:20 PM by broker too)
validity = DAY
"""
from __future__ import annotations
import asyncio
import uuid
import os
import json
from datetime import datetime
from loguru import logger

from core.models import TradeOrder, ExecutedTrade, TradeBias, OrderStatus
from core.message_bus import bus
from core.broker import BrokerClient
from core.database import Database
from core.market_clock import market_clock
from core.broker import SessionExpiredError


class ExecutionAgent:

    def __init__(self, broker: BrokerClient, db: Database, instrument: str = "NIFTY"):
        self.broker        = broker
        self.db            = db
        self.mode          = os.getenv("TRADING_MODE", "paper")
        self.lot_size      = int(os.getenv("LOT_SIZE", 65))
        self._inbox        = bus.subscribe("trade_order")
        self.active_trades: dict[str, ExecutedTrade] = {}

        # Instrument-aware exchange and symbol prefix
        instrument = instrument.upper().strip()
        if instrument == "SENSEX":
            self._exchange       = "bse_fo"
            self._symbol_prefix  = "SENSEX"
        else:
            self._exchange       = "nse_fo"
            self._symbol_prefix  = "NIFTY"

    async def run(self):
        logger.info("Agent 4 (Execution) started")
        while True:
            try:
                order: TradeOrder = await self._inbox.get()
                await self._execute(order)
            except Exception as e:
                logger.error(f"[Agent4] Error: {e}")

    async def _execute(self, order: TradeOrder):
        # ── Guard 0: session valid? ────────────────────────────────────────────
        if not self.broker.is_authenticated():
            logger.error(
                "[Agent4] ORDER BLOCKED — broker session is not authenticated. "
                "Log in again: python dashboard/login.py → http://localhost:8051"
            )
            return

        # ── Guard 0b: market hours — NEVER place orders outside trading window ─
        if not market_clock.can_trade():
            logger.warning(
                f"[Agent4] ORDER BLOCKED — market not in trading window. "
                f"{market_clock.status()}"
            )
            return

        # ── Guard: only 1 active trade at a time ─────────────────────────────
        if self.active_trades:
            logger.warning("[Agent4] Active trade exists — skipping new signal")
            return

        strike = order.strike
        symbol = strike.symbol   # internal key e.g. NIFTY19MAY2623450CE
        # Use the Kotak trading symbol (pTrdSymbol) for actual order placement.
        # Kotak Neo v2 place_order requires pTrdSymbol format: NIFTY2651923450CE
        # This is attached by the fetcher after search_scrip; fall back to symbol if absent.
        trading_sym = getattr(strike, "trading_symbol", None) or symbol
        qty    = order.quantity

        # ── Safety: only configured index options ─────────────────────────────
        if not symbol.startswith(self._symbol_prefix):
            logger.error(
                f"[Agent4] BLOCKED — '{symbol}' does not match "
                f"configured instrument prefix '{self._symbol_prefix}'"
            )
            return

        if order.entry_price < 5 or order.entry_price > 1500:
            logger.error(
                f"[Agent4] BLOCKED — entry ₹{order.entry_price:.2f} "
                f"outside valid range ₹5–₹1500"
            )
            return

        logger.info(
            f"[Agent4] EXECUTING {order.signal.bias.value} | {symbol} | "
            f"Qty:{qty} | Entry:₹{order.entry_price:.2f} | "
            f"SL:₹{order.sl_price:.2f} | Tgt:₹{order.target_price:.2f}"
        )

        # ── Step 1: Entry order (Market Buy) ──────────────────────────────────
        try:
            entry_resp = self.broker.place_order(
                trading_symbol   = trading_sym,
                transaction_type = "B",
                quantity         = qty,
                order_type       = "MKT",
                price            = 0.0,
                trigger_price    = 0.0,
                product          = "MIS",
                exchange_segment = self._exchange,
                validity         = "DAY",
                amo              = "NO",
                tag              = "ENTRY",
            )
        except SessionExpiredError as e:
            logger.error(
                f"[Agent4] ENTRY BLOCKED — Session expired.\n"
                f"  Re-login at: http://localhost:8051\n  Details: {e}"
            )
            await bus.publish_event("session_expired", {"context": "entry_order"})
            return
        except Exception as e:
            logger.error(f"[Agent4] Entry order failed: {e}")
            return

        entry_oid   = entry_resp.get("order_id", str(uuid.uuid4())[:8].upper())
        entry_fill  = float(entry_resp.get("fill_price", order.entry_price))
        logger.info(f"[Agent4] Entry filled @ ₹{entry_fill:.2f} | OID:{entry_oid}")

        # ── Step 2: Broker-side SL-M (MANDATORY — placed immediately) ─────────
        try:
            sl_resp = self.broker.place_order(
                trading_symbol   = trading_sym,
                transaction_type = "S",
                quantity         = qty,
                order_type       = "SL-M",
                price            = 0.0,
                trigger_price    = order.sl_price,
                product          = "MIS",
                exchange_segment = self._exchange,
                validity         = "DAY",
                amo              = "NO",
                tag              = "SL",
            )
        except Exception as e:
            logger.error(
                f"[Agent4] SL order failed: {e}\n"
                f"⚠ Entry is live but SL not placed! Exit manually: {symbol}"
            )
            # Still register the trade so trailing agent can manage it
            sl_resp = {"order_id": "SL_FAILED"}

        sl_oid = sl_resp.get("order_id", "SL_FAILED")
        logger.info(f"[Agent4] SL-M placed @ ₹{order.sl_price:.2f} | OID:{sl_oid}")

        # ── Step 3: Target limit order (optional) ─────────────────────────────
        target_oid = None
        try:
            tgt_resp = self.broker.place_order(
                trading_symbol   = trading_sym,
                transaction_type = "S",
                quantity         = qty,
                order_type       = "L",
                price            = order.target_price,
                trigger_price    = 0.0,
                product          = "MIS",
                exchange_segment = self._exchange,
                validity         = "DAY",
                amo              = "NO",
                tag              = "TARGET",
            )
            target_oid = tgt_resp.get("order_id")
            logger.info(f"[Agent4] Target placed @ ₹{order.target_price:.2f} | OID:{target_oid}")
        except Exception as e:
            logger.warning(f"[Agent4] Target order skipped (non-fatal): {e}")

        # ── Register trade ────────────────────────────────────────────────────
        trade_id = (
            f"TRD-{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
            f"{entry_oid[:4] if entry_oid else 'XXXX'}"
        )

        executed = ExecutedTrade(
            trade_id          = trade_id,
            order             = order,
            entry_order_id    = entry_oid,
            sl_order_id       = sl_oid,
            target_order_id   = target_oid,
            entry_fill_price  = entry_fill,
            entry_time        = datetime.now(),
            status            = OrderStatus.FILLED,
            mode              = self.mode,
        )

        self.active_trades[trade_id] = executed

        # ── Log to DB ─────────────────────────────────────────────────────────
        self.db.save_trade({
            "trade_id":       trade_id,
            "entry_time":     executed.entry_time,
            "symbol":         symbol,
            "option_type":    strike.option_type.value,
            "strike_price":   strike.strike_price,
            "entry_price":    entry_fill,
            "sl_price":       order.sl_price,
            "target_price":   order.target_price,
            "lots":           order.lots,
            "quantity":       qty,
            "strategy_label": order.strategy_label,
            "signal_score":   order.signal.signal_score,
            "active_signals": json.dumps(order.signal.active_signals),
            "regime":         order.signal.market_state.regime.value,
            "mode":           self.mode,
        })

        await bus.publish("executed_trade", executed)
        await bus.publish_event("trade_opened", {
            "trade_id": trade_id,
            "symbol":   symbol,
            "entry":    entry_fill,
            "sl":       order.sl_price,
            "target":   order.target_price,
        })

        logger.info(
            f"[Agent4] ✓ Trade opened: {trade_id} | "
            f"{symbol} | Entry:₹{entry_fill:.2f} | "
            f"SL:₹{order.sl_price:.2f} | Target:₹{order.target_price:.2f}"
        )

    # ─── Close trade ──────────────────────────────────────────────────────────

    async def close_trade(
        self, trade: ExecutedTrade, exit_price: float, reason: str
    ):
        trade_id   = trade.trade_id
        qty        = trade.order.quantity
        symbol     = trade.order.strike.symbol
        trading_sym = getattr(trade.order.strike, "trading_symbol", None) or symbol

        logger.info(f"[Agent4] Closing {trade_id} | reason={reason} | exit=₹{exit_price:.2f}")

        # Cancel pending SL and target orders
        for oid in [trade.sl_order_id, trade.target_order_id]:
            if oid and oid != "SL_FAILED":
                try:
                    self.broker.cancel_order(oid)
                except Exception:
                    pass

        # Place exit market order
        try:
            self.broker.place_order(
                trading_symbol   = trading_sym,
                transaction_type = "S",
                quantity         = qty,
                order_type       = "MKT",
                price            = 0.0,
                trigger_price    = 0.0,
                product          = "MIS",
                exchange_segment = self._exchange,
                validity         = "DAY",
                amo              = "NO",
                tag              = f"EXIT_{reason}",
            )
        except Exception as e:
            logger.error(f"[Agent4] Exit order failed: {e}")

        pnl     = (exit_price - trade.entry_fill_price) * qty
        pnl_pct = (pnl / (trade.entry_fill_price * qty)) * 100 if trade.entry_fill_price else 0

        self.db.update_trade(
            trade_id,
            exit_time   = datetime.now(),
            exit_price  = exit_price,
            exit_reason = reason,
            pnl         = round(pnl, 2),
            pnl_pct     = round(pnl_pct, 2),
            won         = pnl > 0,
        )

        self.active_trades.pop(trade_id, None)

        await bus.publish_event("trade_closed", {
            "trade_id":   trade_id,
            "exit_price": exit_price,
            "pnl":        round(pnl, 2),
            "reason":     reason,
        })

        logger.info(
            f"[Agent4] ✓ Trade closed: {trade_id} | "
            f"Exit:₹{exit_price:.2f} | P&L:₹{pnl:+.0f} ({pnl_pct:+.1f}%) | {reason}"
        )

    # ─── Square off all (3:20 PM) ─────────────────────────────────────────────

    async def square_off_all(self):
        logger.warning("[Agent4] SQUARE-OFF: Closing all positions")
        for trade_id, trade in list(self.active_trades.items()):
            # Use last known entry price as fallback exit price
            await self.close_trade(trade, trade.entry_fill_price, "SQUAREOFF")

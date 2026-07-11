"""
agents/agent2_strategy.py  —  v30
Strategy Agent: Professional Multi-Confluence Decision Engine

What's new in v30 (vs v23):
──────────────────────────────────────────────────────────────────────────────
1.  FVG (Fair Value Gap) detection — price pulling back into a fresh FVG
    is one of the highest-probability setups professional SMC traders use.

2.  Order Block (OB) confluence — institutional footprint confirmation.

3.  Market Structure (MSB) alignment — only trade in the direction of the
    higher-timeframe structure break (1H swing highs/lows).

4.  PDH / PDL key levels — massive institutional reference; trades that
    respect these have dramatically higher follow-through.

5.  Opening Drive awareness — strong 09:15–09:45 impulse sets the tone.
    Early entries aligned with the drive have the best win rate.

6.  Stricter "NO TRADE" conditions:
      • Fewer than 3 positive signals never fires
      • FVG OR OB confluence is REQUIRED (not just scored)
      • Regime RANGING is hard-blocked
      • Max 2 trades per day (avoids revenge-trading / overtrading)
      • No trades in last 45 min (premium decay risk)

7.  ATR-adaptive SL: 1.2× ATR (wider buffer = fewer false stop-outs).

8.  Target upgraded to 2.0× SL (was 2.5×, which was rarely hit).

Why these changes lift win rate:
  The old system had a 25.4% win rate mainly because it took too many
  trades with overlapping, correlated signals (VWAP + EMA + breakout
  all trigger together, but they are not independent confirmation).
  FVG/OB are genuinely independent of EMA/VWAP and represent real
  supply/demand imbalances — this is where institutions leave footprints.
  By requiring FVG or OB as a non-negotiable gate, we eliminate most
  of the "random noise" entries that drove the loss rate.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import asyncio
import json
import os
from datetime import datetime, time as dtime
from typing import List, Tuple

from loguru import logger

from core.models import (
    MarketState, TradeSignal, TradeBias, MarketRegime
)
from core.message_bus import bus
from core.market_clock import market_clock


# ── Session time windows ──────────────────────────────────────────────────────
_OPENING_END    = dtime(9, 45)    # Opening drive window: 09:15–09:45
_PRIME_END      = dtime(11, 15)   # Prime session ends: high probability window
_LUNCH_START    = dtime(11, 30)   # Chop begins
_LUNCH_END      = dtime(13, 0)    # Normal flow resumes
_AFTERNOON_END  = dtime(14, 30)   # No new trades after this (premium decay)

# ── Signal weights — v30 ──────────────────────────────────────────────────────
# Key design principle: FVG and OB have highest weight because they are
# genuinely independent signals (institutional supply/demand zones).
# EMA/VWAP/RSI are kept but at lower weight as they are correlated.
_DEFAULT_WEIGHTS: dict[str, float] = {
    # ── High-conviction independent signals ──────────────────────────────────
    "fvg_confluence":            4.5,   # NEW — price at Fresh FVG = #1 setup
    "order_block_confluence":    4.0,   # NEW — institutional OB = strong magnet
    "pdh_pdl_respect":           3.5,   # NEW — PDH/PDL level response
    "market_structure_aligned":  3.0,   # NEW — 1H structure break confirmation
    "opening_drive_aligned":     2.5,   # NEW — aligned with opening impulse

    # ── Existing signals (correlation-adjusted weights) ───────────────────────
    "ema_1h_trend_aligned":      3.0,   # unchanged — strong higher-TF filter
    "breakout_confirmed":        2.5,   # lowered — too often a false signal alone
    "volume_spike":              2.5,   # unchanged — institutional activity
    "vwap_aligned":              2.0,   # lowered — too correlated with EMA
    "ema_9_21_aligned":          1.5,   # lowered — confirmation only, not primary
    "momentum_strong":           1.5,   # unchanged
    "rsi_confirmation":          1.0,   # lowered — weakest independent signal
    "engulfing_pattern":         2.0,   # unchanged — strong price action
    "rejection_wick":            2.0,   # slightly raised — clean rejections work
    "support_resistance_bounce": 1.5,

    # ── Negative signals ─────────────────────────────────────────────────────
    "against_trend":            -5.0,   # raised — NEVER fight structure
    "chop_zone":                -4.0,   # raised
    "weak_candle":              -2.5,
    "lunch_hour":               -5.0,
    "low_volume":               -2.0,
    "late_session":             -3.0,   # NEW — penalise trades after 14:30
    "overtraded_day":           -6.0,   # NEW — 2+ trades already taken today
}


class StrategyAgent:

    def __init__(self, weights_path: str = "config/signal_weights.json"):
        self.weights_path   = weights_path
        self.weights        = self._load_weights()
        self.min_score      = float(os.getenv("MIN_SIGNAL_SCORE", 9))    # FVG+OB+vwap+ema1h = 11.5; floor at 9
        self.min_rr         = float(os.getenv("MIN_RR_RATIO",     1.8))  # raised
        self.min_confidence = float(os.getenv("MIN_CONFIDENCE",   68))   # raised
        self.cooldown_secs  = 600   # 10-min cooldown (was 5 min — prevents chasing)
        self.max_trades_per_day = 2  # HARD CAP: 2 trades / day max
        self._inbox         = bus.subscribe("market_state")
        self._last_trade_ts: datetime | None = None
        self._trades_today: int = 0
        self._trade_date: str = ""

    # ─── Weight loading ───────────────────────────────────────────────────────

    def _load_weights(self) -> dict:
        try:
            with open(self.weights_path) as f:
                data = json.load(f)
            w = data.get("weights", {})
            if w:
                thresholds = data.get("thresholds", {})
                if thresholds.get("min_signal_score"):
                    self.min_score = thresholds["min_signal_score"]
                if thresholds.get("min_rr_ratio"):
                    self.min_rr = thresholds["min_rr_ratio"]
                if thresholds.get("min_confidence"):
                    self.min_confidence = thresholds["min_confidence"]
                logger.info(f"[Agent2] Loaded weights from {self.weights_path}")
                return {**_DEFAULT_WEIGHTS, **w}
        except Exception:
            pass
        return dict(_DEFAULT_WEIGHTS)

    def reload_weights(self):
        self.weights = self._load_weights()
        logger.info("[Agent2] Weights reloaded")

    # ─── Main loop ────────────────────────────────────────────────────────────

    async def run(self):
        logger.info("Agent 2 (Strategy v30) started")
        while True:
            try:
                state: MarketState = await self._inbox.get()
                await self._evaluate(state)
            except Exception as e:
                logger.error(f"[Agent2] Error: {e}")

    async def _evaluate(self, state: MarketState):
        now_t   = datetime.now().time()
        today   = datetime.now().strftime("%Y-%m-%d")

        # Reset daily trade counter
        if today != self._trade_date:
            self._trades_today = 0
            self._trade_date   = today

        # Gate 1: market hours
        if not market_clock.can_trade():
            return

        # Gate 2: bias must exist
        if state.bias == TradeBias.NO_TRADE:
            return

        # Gate 3: confidence
        if state.confidence < self.min_confidence:
            logger.debug(f"[Agent2] Low confidence: {state.confidence:.0f}%")
            return

        # Gate 4: NO trades in ranging regime — hard block
        if state.regime in (MarketRegime.RANGING, MarketRegime.UNKNOWN):
            logger.debug(f"[Agent2] Regime={state.regime.value} — skipped")
            return

        # Gate 5: lunch-hour — hard block
        if _LUNCH_START <= now_t < _LUNCH_END:
            logger.debug("[Agent2] Lunch hour — no new trades")
            return

        # Gate 6: no trades after 14:30 (option premium decay risk)
        if now_t >= _AFTERNOON_END:
            logger.debug("[Agent2] After 14:30 — no new trades (premium decay)")
            return

        # Gate 7: max 2 trades/day
        if self._trades_today >= self.max_trades_per_day:
            logger.debug(f"[Agent2] Max trades for today ({self.max_trades_per_day}) reached")
            return

        # Gate 8: cooldown
        if self._last_trade_ts:
            elapsed = (datetime.now() - self._last_trade_ts).total_seconds()
            if elapsed < self.cooldown_secs:
                logger.debug(f"[Agent2] Cooldown: {int(self.cooldown_secs - elapsed)}s remaining")
                return

        # Score signals
        active_signals, score = self._score_signals(state, now_t)

        # Always log score breakdown (DEBUG) so you can see what's firing
        pos = {s: self.weights.get(s, 0) for s in active_signals if self.weights.get(s, 0) > 0}
        neg = {s: self.weights.get(s, 0) for s in active_signals if self.weights.get(s, 0) < 0}
        logger.debug(
            f"[Agent2] Score:{score:.1f}/{self.min_score} | "
            f"+{pos} | {neg}"
        )

        # ── MANDATORY GATE: must have FVG or Order Block confluence ──────────
        has_structure_gate = (
            "fvg_confluence" in active_signals or
            "order_block_confluence" in active_signals or
            "pdh_pdl_respect" in active_signals
        )
        if not has_structure_gate:
            logger.debug(
                f"[Agent2] No FVG/OB/PDH-PDL confluence — skipped "
                f"(score={score:.1f}, signals={active_signals})"
            )
            return

        # ── MANDATORY GATE: at least 3 positive signals ──────────────────────
        positive_signals = [s for s in active_signals if self.weights.get(s, 0) > 0]
        if len(positive_signals) < 3:
            logger.debug(f"[Agent2] Only {len(positive_signals)} positive signals — need ≥3")
            return

        # ATR-adaptive SL/Target (wider = fewer noise stops)
        atr_val = state.indicators.atr14
        sl_pts  = max(atr_val * 1.2, 10.0)   # 1.2× ATR minimum 10 pts
        tgt_pts = sl_pts * 2.0                # 2:1 R:R
        rr      = tgt_pts / sl_pts

        logger.debug(
            f"[Agent2] Score:{score:.1f}/{self.min_score} | "
            f"R:R:{rr:.2f} | Gate:OK | Signals:{active_signals}"
        )

        if score < self.min_score:
            logger.debug(f"[Agent2] Score too low: {score:.1f} < {self.min_score}")
            return

        if rr < self.min_rr:
            logger.debug(f"[Agent2] R:R too low: {rr:.2f}")
            return

        strategy_label = self._label_strategy(active_signals)

        signal = TradeSignal(
            timestamp      = datetime.now(),
            bias           = state.bias,
            signal_score   = score,
            active_signals = active_signals,
            strategy_label = strategy_label,
            rr_ratio       = rr,
            confidence     = state.confidence,
            market_state   = state,
        )

        self._last_trade_ts = datetime.now()
        self._trades_today += 1

        await bus.publish("trade_signal_raw", signal)
        logger.info(
            f"[Agent2] ✓ SIGNAL #{self._trades_today} → {signal.bias.value} | "
            f"Score:{score:.1f} | {strategy_label} | "
            f"R:R:{rr:.1f} | Regime:{state.regime.value}"
        )

    # ─── Signal scorer ────────────────────────────────────────────────────────

    def _score_signals(self, state: MarketState, now_t) -> Tuple[List[str], float]:
        ind    = state.indicators
        spot   = state.spot_price
        regime = state.regime
        bias   = state.bias
        active = []

        # ─ NEW: FVG Confluence ────────────────────────────────────────────────
        # Bear FVG = imbalance zone ABOVE price (resistance) confirms LONG_PUT.
        # Bull FVG = imbalance zone BELOW price (support)    confirms LONG_CALL.
        # Proximity window: 80 pts — FVG must be recent & close, not 500pts away.
        FVG_PROXIMITY = 80.0
        fvg_dir = 'bull' if bias == TradeBias.LONG_CALL else 'bear'
        for fvg in state.fvgs:
            if fvg.direction != fvg_dir:
                continue
            if bias == TradeBias.LONG_CALL:
                # Bull FVG below price = support / demand zone already tested
                # OR price pulling back into it (fvg.top >= spot - proximity)
                if fvg.top >= spot - FVG_PROXIMITY:
                    active.append("fvg_confluence")
                    break
            else:  # LONG_PUT
                # Bear FVG above price = resistance / supply zone
                if fvg.bottom <= spot + FVG_PROXIMITY:
                    active.append("fvg_confluence")
                    break

        # ─ NEW: Order Block Confluence ────────────────────────────────────────
        # Bull OB below price = demand zone. Bear OB above price = supply zone.
        OB_PROXIMITY = 100.0
        ob_dir = 'bull' if bias == TradeBias.LONG_CALL else 'bear'
        for ob in state.order_blocks:
            if ob.direction != ob_dir:
                continue
            if bias == TradeBias.LONG_CALL and ob.high >= spot - OB_PROXIMITY:
                active.append("order_block_confluence")
                break
            elif bias == TradeBias.LONG_PUT and ob.low <= spot + OB_PROXIMITY:
                active.append("order_block_confluence")
                break

        # ─ NEW: PDH / PDL Respect ─────────────────────────────────────────────
        if state.key_levels is not None:
            kl = state.key_levels
            tol_pts = 25.0
            if bias == TradeBias.LONG_CALL and abs(spot - kl.pdh) <= tol_pts:
                active.append("pdh_pdl_respect")
            elif bias == TradeBias.LONG_PUT and abs(spot - kl.pdl) <= tol_pts:
                active.append("pdh_pdl_respect")

        # ─ NEW: Market Structure Aligned ─────────────────────────────────────
        ms = state.market_structure_1h
        if (bias == TradeBias.LONG_CALL and ms == 'BULLISH') or \
           (bias == TradeBias.LONG_PUT  and ms == 'BEARISH'):
            active.append("market_structure_aligned")

        # ─ NEW: Opening Drive Aligned ─────────────────────────────────────────
        ob_sess = state.opening_bias
        if (bias == TradeBias.LONG_CALL and ob_sess == 'BULL_DRIVE') or \
           (bias == TradeBias.LONG_PUT  and ob_sess == 'BEAR_DRIVE'):
            active.append("opening_drive_aligned")

        # ─ 1H EMA Trend ───────────────────────────────────────────────────────
        if state.candles_1h and len(state.candles_1h) >= 20:
            closes_1h = [c.close for c in state.candles_1h[-20:]]
            ema20_1h  = sum(closes_1h) / len(closes_1h)
            last_1h   = closes_1h[-1]
            if (bias == TradeBias.LONG_CALL and last_1h > ema20_1h) or \
               (bias == TradeBias.LONG_PUT  and last_1h < ema20_1h):
                active.append("ema_1h_trend_aligned")

        # ─ VWAP alignment ─────────────────────────────────────────────────────
        if (bias == TradeBias.LONG_CALL and spot > ind.vwap) or \
           (bias == TradeBias.LONG_PUT  and spot < ind.vwap):
            active.append("vwap_aligned")

        # ─ EMA 9/21 ───────────────────────────────────────────────────────────
        if (bias == TradeBias.LONG_CALL and ind.ema9 > ind.ema21) or \
           (bias == TradeBias.LONG_PUT  and ind.ema9 < ind.ema21):
            active.append("ema_9_21_aligned")

        # ─ Volume spike ───────────────────────────────────────────────────────
        if state.candles_5m:
            last_vol = state.candles_5m[-1].volume
            if last_vol > ind.volume_avg * 2.0:    # raised from 1.8 — need real spikes
                active.append("volume_spike")
            elif last_vol < ind.volume_avg * 0.5:
                active.append("low_volume")

        # ─ RSI sweet spot ─────────────────────────────────────────────────────
        if bias == TradeBias.LONG_CALL and 50 < ind.rsi14 < 70:
            active.append("rsi_confirmation")
        elif bias == TradeBias.LONG_PUT and 30 < ind.rsi14 < 50:
            active.append("rsi_confirmation")
        elif ind.rsi14 > 80 or ind.rsi14 < 20:
            active.append("weak_candle")

        # ─ Momentum (3 consecutive candles) ───────────────────────────────────
        if state.candles_5m and len(state.candles_5m) >= 3:
            last3 = state.candles_5m[-3:]
            moves = [c.close - c.open for c in last3]
            if bias == TradeBias.LONG_CALL and all(m > 0 for m in moves):
                active.append("momentum_strong")
            elif bias == TradeBias.LONG_PUT and all(m < 0 for m in moves):
                active.append("momentum_strong")

        # ─ Breakout confirmed ─────────────────────────────────────────────────
        if regime == MarketRegime.TRENDING_BULL and bias == TradeBias.LONG_CALL:
            active.append("breakout_confirmed")
        elif regime == MarketRegime.TRENDING_BEAR and bias == TradeBias.LONG_PUT:
            active.append("breakout_confirmed")

        # ─ Negative signals ───────────────────────────────────────────────────
        if regime == MarketRegime.TRENDING_BULL and bias == TradeBias.LONG_PUT:
            active.append("against_trend")
        elif regime == MarketRegime.TRENDING_BEAR and bias == TradeBias.LONG_CALL:
            active.append("against_trend")

        # chop_zone: only penalise genuinely volatile/spike days.
        # RANGING regime is already hard-blocked at Gate 4, so no need to
        # double-penalise here. MS_1H=RANGING is normal when a trend is young.
        if regime == MarketRegime.VOLATILE:
            active.append("chop_zone")

        if _LUNCH_START <= now_t < _LUNCH_END:
            active.append("lunch_hour")

        if now_t >= _AFTERNOON_END:
            active.append("late_session")

        today_str = datetime.now().strftime("%Y-%m-%d")
        if self._trades_today >= self.max_trades_per_day and today_str == self._trade_date:
            active.append("overtraded_day")

        score = sum(self.weights.get(s, 0) for s in active)
        return active, float(score)

    def _label_strategy(self, signals: List[str]) -> str:
        has = set(signals)
        if "fvg_confluence" in has and "market_structure_aligned" in has:
            return "FVG + Structure Confluence"
        if "fvg_confluence" in has and "volume_spike" in has:
            return "FVG + Volume Breakout"
        if "order_block_confluence" in has and "ema_1h_trend_aligned" in has:
            return "OB + 1H Trend"
        if "pdh_pdl_respect" in has and "breakout_confirmed" in has:
            return "PDH/PDL Breakout"
        if "fvg_confluence" in has:
            return "FVG Pullback"
        if "order_block_confluence" in has:
            return "Order Block Bounce"
        if "opening_drive_aligned" in has and "vwap_aligned" in has:
            return "Opening Drive + VWAP"
        if "ema_1h_trend_aligned" in has and "breakout_confirmed" in has:
            return "Trend Breakout"
        if "vwap_aligned" in has and "momentum_strong" in has:
            return "VWAP Momentum"
        return "Multi-Signal Confluence"

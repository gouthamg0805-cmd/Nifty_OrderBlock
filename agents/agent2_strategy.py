"""
agents/agent2_strategy.py  —  v31 "OBV" (Order Block + Volume)
Strategy Agent: Order-Block-and-Volume Decision Engine

This replaces the v30 multi-confluence scorer (FVG/PDH/EMA/VWAP/RSI/…) with a
much narrower, purpose-built engine: every trade must be justified by (1)
*where* price is — an institutional Order Block — and (2) *who* is behind it —
volume evidence of real participation. Everything else in this file exists
only to keep that core idea safe to run live (session windows, daily caps,
cooldown), not to generate alternate reasons to trade.

Core idea
──────────────────────────────────────────────────────────────────────────────
1. Order Block location (WHERE):
     • Bull OB = last down-candle before a strong bullish impulse → demand zone
     • Bear OB = last up-candle before a strong bearish impulse   → supply zone
     • "Fresh" (never retested) zones score higher than zones already tested
       once — a zone tested many times is increasingly likely to fail.

2. Volume confirmation (WHO):
     • The OB is only usable if its *impulse* candle shows a real volume
       spike (RVOL >= 1.5) — this is core/indicators.detect_order_blocks_obv's
       job, done at Agent 1. An OB with no volume behind it is just a shape.
     • On top of that, Agent 2 requires *live* volume confirmation at the
       moment of entry: either the current candle is itself showing a
       volume spike (renewed interest at the zone) or RVOL has been
       trending up over the last few candles (building participation).
     • Climax volume (RVOL >= 3.5) against the trade direction is treated as
       exhaustion risk and penalised — blow-off volume is a warning sign,
       not a confirmation, when it fires against you.
     • Dead/illiquid tape (RVOL persistently < 0.6) is penalised — there is
       no real order flow to trade against.

3. Stop-loss & target are themselves Order-Block-based: the stop sits just
   beyond the far edge of the Order Block being traded (invalidation = the
   zone actually failed), not an arbitrary ATR multiple. Target defaults to
   2R off that OB-based risk, or the next opposing Order Block if one exists
   closer than the 2R target (take profit where the next institutional zone
   is likely to react).

Mandatory gate (non-negotiable — this is what makes it an "OB + Volume"
strategy rather than a discretionary scorer with OB/Volume as one of many
inputs):
     • A valid Order Block in the trade direction within proximity of spot
       (fresh or first retest only — never a 2nd+ retest).
     • AND at least one live volume confirmation signal.
   No OB+Volume evidence -> no trade, regardless of any other signal.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import asyncio
import json
import os
from datetime import datetime, time as dtime
from typing import List, Tuple, Optional

from loguru import logger

from core.models import (
    MarketState, TradeSignal, TradeBias, MarketRegime, OrderBlockModel
)
from core.message_bus import bus
from core.market_clock import market_clock


# -- Session time windows ------------------------------------------------------
_LUNCH_START    = dtime(11, 30)   # Chop begins
_LUNCH_END      = dtime(13, 0)    # Normal flow resumes
_AFTERNOON_END  = dtime(14, 30)   # No new trades after this (premium decay)

# -- Tunables --------------------------------------------------------------------
OB_PROXIMITY_PTS   = 100.0   # how far price may be from an OB and still "at" it
LIVE_VOL_SPIKE     = 2.0     # current-candle RVOL needed for live confirmation
CLIMAX_RVOL        = 3.5     # RVOL at/above this = exhaustion territory
LOW_RVOL           = 0.6     # RVOL below this = dead tape
SL_BUFFER_PTS      = 5.0     # extra buffer beyond the OB edge for the stop
MIN_SL_PTS         = 12.0    # floor so tight/noisy OBs don't create razor stops
DEFAULT_RR         = 2.0     # fallback R:R if no further OB gives a better target

# -- Signal weights - v31 (OB + Volume only) -----------------------------------
_DEFAULT_WEIGHTS: dict[str, float] = {
    # -- Order Block location (WHERE) -----------------------------------------
    "ob_zone_fresh":            5.0,   # price at an untested OB in bias direction
    "ob_zone_retest":           3.0,   # price at a once-tested OB (still valid)
    "ob_multi_confluence":      2.0,   # 2+ OBs stacked in the same zone/direction
    "ob_high_strength":         1.5,   # OB composite strength score >= 0.7

    # -- Volume confirmation (WHO) --------------------------------------------
    "ob_impulse_volume":        3.5,   # the OB's own impulse candle had RVOL>=1.5
    "live_volume_spike":        3.0,   # current candle RVOL >= LIVE_VOL_SPIKE
    "rvol_rising":              1.5,   # RVOL trending up over last 3 candles

    # -- Risk/quality gates (not signals, but scored to keep visibility) ------
    "regime_aligned":           1.0,   # soft bonus when regime agrees with bias

    # -- Negative signals ------------------------------------------------------
    "volume_climax_exhaustion": -3.5,  # blow-off volume against bias direction
    "low_volume_grind":         -2.0,  # dead tape, no real participation
    "against_trend":            -4.0,  # fighting a strong regime trend
    "chop_zone":                -3.0,  # VOLATILE regime
    "lunch_hour":               -5.0,
    "late_session":             -3.0,
    "overtraded_day":           -6.0,
}


class StrategyAgent:

    def __init__(self, weights_path: str = "config/signal_weights.json"):
        self.weights_path   = weights_path
        self.weights        = self._load_weights()
        self.min_score      = float(os.getenv("MIN_SIGNAL_SCORE", 7))
        self.min_rr         = float(os.getenv("MIN_RR_RATIO",     1.5))
        self.min_confidence = float(os.getenv("MIN_CONFIDENCE",   60))
        self.cooldown_secs  = 600   # 10-min cooldown
        self.max_trades_per_day = 2  # HARD CAP: 2 trades / day max
        self._inbox         = bus.subscribe("market_state")
        self._last_trade_ts: datetime | None = None
        self._trades_today: int = 0
        self._trade_date: str = ""

    # --- Weight loading --------------------------------------------------------

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

    # --- Main loop ---------------------------------------------------------------

    async def run(self):
        logger.info("Agent 2 (Strategy v31 - Order Block + Volume) started")
        while True:
            try:
                state: MarketState = await self._inbox.get()
                await self._evaluate(state)
            except Exception as e:
                logger.error(f"[Agent2] Error: {e}")

    async def _evaluate(self, state: MarketState):
        now_t = datetime.now().time()
        today = datetime.now().strftime("%Y-%m-%d")

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

        # Gate 4: no trades in ranging / unknown regime - OBs need a directional
        # tape to work; in true chop even a "confirmed" OB is low-probability.
        if state.regime in (MarketRegime.RANGING, MarketRegime.UNKNOWN):
            logger.debug(f"[Agent2] Regime={state.regime.value} - skipped")
            return

        # Gate 5: lunch-hour - hard block
        if _LUNCH_START <= now_t < _LUNCH_END:
            logger.debug("[Agent2] Lunch hour - no new trades")
            return

        # Gate 6: no trades after 14:30 (option premium decay risk)
        if now_t >= _AFTERNOON_END:
            logger.debug("[Agent2] After 14:30 - no new trades (premium decay)")
            return

        # Gate 7: max trades/day
        if self._trades_today >= self.max_trades_per_day:
            logger.debug(f"[Agent2] Max trades for today ({self.max_trades_per_day}) reached")
            return

        # Gate 8: cooldown
        if self._last_trade_ts:
            elapsed = (datetime.now() - self._last_trade_ts).total_seconds()
            if elapsed < self.cooldown_secs:
                logger.debug(f"[Agent2] Cooldown: {int(self.cooldown_secs - elapsed)}s remaining")
                return

        # -- Locate the Order Block this trade would be based on ------------------
        ob_direction = 'bull' if state.bias == TradeBias.LONG_CALL else 'bear'
        trade_ob = self._nearest_tradeable_ob(state, ob_direction)

        # -- MANDATORY GATE: must have a fresh-or-1st-retest OB in direction ------
        if trade_ob is None:
            logger.debug(
                f"[Agent2] No valid Order Block ({ob_direction}) within "
                f"{OB_PROXIMITY_PTS}pts of spot {state.spot_price:.1f} - skipped"
            )
            return

        # Score signals (this also re-derives volume confirmation)
        active_signals, score = self._score_signals(state, now_t, trade_ob, ob_direction)

        pos = {s: self.weights.get(s, 0) for s in active_signals if self.weights.get(s, 0) > 0}
        neg = {s: self.weights.get(s, 0) for s in active_signals if self.weights.get(s, 0) < 0}
        logger.debug(f"[Agent2] Score:{score:.1f}/{self.min_score} | +{pos} | {neg}")

        # -- MANDATORY GATE: must have live volume confirmation --------------------
        has_volume_confirmation = (
            "ob_impulse_volume" in active_signals or
            "live_volume_spike" in active_signals
        )
        if not has_volume_confirmation:
            logger.debug(
                f"[Agent2] OB found but no volume confirmation - skipped "
                f"(signals={active_signals})"
            )
            return

        if score < self.min_score:
            logger.debug(f"[Agent2] Score too low: {score:.1f} < {self.min_score}")
            return

        # -- Order-Block-based SL / Target -----------------------------------------
        sl_pts, tgt_pts, rr = self._ob_risk_reward(state, trade_ob, ob_direction)

        if rr < self.min_rr:
            logger.debug(f"[Agent2] R:R too low: {rr:.2f}")
            return

        strategy_label = self._label_strategy(active_signals, trade_ob)

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
            f"[Agent2] SIGNAL #{self._trades_today} -> {signal.bias.value} | "
            f"Score:{score:.1f} | {strategy_label} | "
            f"R:R:{rr:.2f} | SL:{sl_pts:.1f}pts | OB:[{trade_ob.low:.1f}-{trade_ob.high:.1f}] "
            f"RVOL(impulse):{trade_ob.impulse_rvol:.2f}"
        )

    # --- Order Block selection -------------------------------------------------

    def _nearest_tradeable_ob(
        self, state: MarketState, direction: str
    ) -> Optional[OrderBlockModel]:
        """
        Returns the closest Order Block in `direction` that is:
          - volume-confirmed at formation
          - fresh (unmitigated) OR on its first retest only
          - within OB_PROXIMITY_PTS of spot, on the correct side (demand
            below/at spot for bull, supply above/at spot for bear)
        A repeatedly-tested zone is excluded - zones tested many times lose
        reliability and are not "fresh institutional footprint" anymore.
        """
        spot = state.spot_price
        candidates = []
        for ob in state.order_blocks:
            if ob.direction != direction:
                continue
            if not ob.volume_confirmed:
                continue
            if direction == 'bull':
                if ob.high >= spot - OB_PROXIMITY_PTS and ob.low <= spot + 5:
                    candidates.append(ob)
            else:
                if ob.low <= spot + OB_PROXIMITY_PTS and ob.high >= spot - 5:
                    candidates.append(ob)
        if not candidates:
            return None
        # Prefer the closest zone to spot (most immediately actionable)
        return min(candidates, key=lambda o: abs(spot - (o.high + o.low) / 2))

    # --- Signal scorer -----------------------------------------------------------

    def _score_signals(
        self, state: MarketState, now_t, trade_ob: OrderBlockModel, direction: str
    ) -> Tuple[List[str], float]:
        spot   = state.spot_price
        regime = state.regime
        bias   = state.bias
        active: List[str] = []

        # -- OB location signals ---------------------------------------------------
        if not trade_ob.mitigated:
            active.append("ob_zone_fresh")
        else:
            active.append("ob_zone_retest")

        same_zone_obs = [
            ob for ob in state.order_blocks
            if ob.direction == direction and
               abs(((ob.high + ob.low) / 2) - ((trade_ob.high + trade_ob.low) / 2)) < 40
        ]
        if len(same_zone_obs) >= 2:
            active.append("ob_multi_confluence")

        if trade_ob.strength >= 0.7:
            active.append("ob_high_strength")

        # -- Volume confirmation signals --------------------------------------------
        if trade_ob.volume_confirmed:
            active.append("ob_impulse_volume")

        current_rvol = state.indicators.rvol
        if current_rvol >= LIVE_VOL_SPIKE:
            active.append("live_volume_spike")

        hist = state.rvol_history
        if len(hist) >= 3 and hist[-1] > hist[-2] > hist[-3]:
            active.append("rvol_rising")

        # Climax volume against the trade direction = exhaustion risk.
        # A huge RVOL spike moving price *away* from our bias right as we'd
        # enter suggests the move is a blow-off, not fresh institutional
        # interest in our direction.
        if state.candles_5m:
            last = state.candles_5m[-1]
            candle_is_down = last.close < last.open
            candle_is_up   = last.close > last.open
            if current_rvol >= CLIMAX_RVOL:
                if (bias == TradeBias.LONG_CALL and candle_is_down) or \
                   (bias == TradeBias.LONG_PUT  and candle_is_up):
                    active.append("volume_climax_exhaustion")

        if current_rvol < LOW_RVOL:
            active.append("low_volume_grind")

        # -- Regime alignment (soft bonus, not a gate) -------------------------------
        if (bias == TradeBias.LONG_CALL and regime == MarketRegime.TRENDING_BULL) or \
           (bias == TradeBias.LONG_PUT  and regime == MarketRegime.TRENDING_BEAR):
            active.append("regime_aligned")

        # -- Negative / risk signals --------------------------------------------------
        if regime == MarketRegime.TRENDING_BULL and bias == TradeBias.LONG_PUT:
            active.append("against_trend")
        elif regime == MarketRegime.TRENDING_BEAR and bias == TradeBias.LONG_CALL:
            active.append("against_trend")

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

    # --- Order-Block-based risk/reward -------------------------------------------

    def _ob_risk_reward(
        self, state: MarketState, trade_ob: OrderBlockModel, direction: str
    ) -> Tuple[float, float, float]:
        """
        Stop-loss sits just beyond the far edge of the traded Order Block -
        if price trades through the whole zone, the "institutional footprint"
        thesis is invalidated. Target defaults to 2R, but if there is an
        opposing Order Block closer than the 2R target, that zone is used
        instead (take profit where the next institutional zone likely reacts).
        """
        spot = state.spot_price

        if direction == 'bull':
            sl_level = trade_ob.low - SL_BUFFER_PTS
            sl_pts   = max(spot - sl_level, MIN_SL_PTS)
        else:
            sl_level = trade_ob.high + SL_BUFFER_PTS
            sl_pts   = max(sl_level - spot, MIN_SL_PTS)

        default_target_pts = sl_pts * DEFAULT_RR

        # Look for a closer opposing OB to use as a realistic take-profit zone
        opposing_dir = 'bear' if direction == 'bull' else 'bull'
        best_target_pts = default_target_pts
        for ob in state.order_blocks:
            if ob.direction != opposing_dir:
                continue
            zone_mid = (ob.high + ob.low) / 2
            dist = (zone_mid - spot) if direction == 'bull' else (spot - zone_mid)
            if 0 < dist < best_target_pts:
                best_target_pts = dist

        rr = best_target_pts / sl_pts if sl_pts > 0 else 0.0
        return sl_pts, best_target_pts, float(rr)

    # --- Labeling ------------------------------------------------------------------

    def _label_strategy(self, signals: List[str], trade_ob: OrderBlockModel) -> str:
        has = set(signals)
        fresh = "ob_zone_fresh" in has
        zone  = "Fresh OB" if fresh else "OB Retest"

        if "live_volume_spike" in has and "ob_impulse_volume" in has:
            return f"{zone} + Double Volume Confirmation"
        if "live_volume_spike" in has:
            return f"{zone} + Live Volume Spike"
        if "ob_impulse_volume" in has and "rvol_rising" in has:
            return f"{zone} + Rising Volume"
        if "ob_impulse_volume" in has:
            return f"{zone} + Impulse Volume"
        if "ob_multi_confluence" in has:
            return f"{zone} + Multi-OB Confluence"
        return f"{zone} + Volume"

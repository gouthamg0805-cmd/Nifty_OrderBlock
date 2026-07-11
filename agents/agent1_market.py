"""
agents/agent1_market.py  —  v30
Market Intelligence Agent

New in v30:
  • Computes FVGs on 5m and 15m candles
  • Detects Order Blocks on 15m candles
  • Derives key levels (PDH/PDL/weekly) from 1H data
  • Classifies 1H market structure (bullish/bearish/ranging)
  • Detects opening-drive session bias
  • All new fields added to MarketState before publishing
"""
from __future__ import annotations
import asyncio
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from core.models import (
    MarketState, MarketRegime, TradeBias,
    Indicators, OHLCV,
    FVGModel, OrderBlockModel, KeyLevelsModel,
)
from core.message_bus import bus
from core.market_clock import market_clock
from core.indicators import (
    ema, atr, rsi, vwap, volume_sma, classify_regime,
    detect_engulfing, detect_fvg, detect_order_blocks,
    market_structure, key_levels, session_opening_bias,
    detect_order_blocks_obv, relative_volume,
)


class MarketIntelligenceAgent:

    def __init__(self, data_fetcher, poll_interval: int = 30):
        self.fetcher       = data_fetcher
        self.poll_interval = poll_interval
        self.running       = False
        self._last_status  = ""

    async def run(self):
        self.running = True
        logger.info("Agent 1 (Market Intelligence v30) started")

        while self.running:
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"[Agent1] Error: {e}")

            sleep_secs = market_clock.wait_seconds()
            await asyncio.sleep(sleep_secs)

    async def _tick(self):
        status = market_clock.status()
        if status != self._last_status:
            logger.info(f"[Agent1] {status}")
            self._last_status = status

        if not market_clock.is_open():
            return

        if not market_clock.can_trade() and not market_clock.must_square_off():
            secs = market_clock.seconds_to_trading_start()
            logger.debug(f"[Agent1] Trading starts in {secs}s. Standing by...")
            return

        state = await self._analyze()
        if state:
            await bus.publish("market_state", state)
            logger.info(
                f"[Agent1] {state.regime.value} | Bias:{state.bias.value} "
                f"| Conf:{state.confidence:.0f}% | Spot:₹{state.spot_price:.2f} "
                f"| FVGs:{len(getattr(state,'fvgs',[]))} "
                f"| OBs:{len(getattr(state,'order_blocks',[]))} "
                f"| MS:{getattr(state,'market_structure_1h','?')}"
            )

    async def _analyze(self) -> MarketState | None:
        spot = await self.fetcher.get_spot_price()
        if not spot:
            logger.warning("[Agent1] Nifty spot unavailable")
            return None

        df_5m  = await self.fetcher.get_candles("5m",  lookback=60)
        df_15m = await self.fetcher.get_candles("15m", lookback=60)
        df_1h  = await self.fetcher.get_candles("1h",  lookback=40)

        if df_5m is None or len(df_5m) < 21:
            return None

        # ── Standard indicators ───────────────────────────────────────────────
        e9      = ema(df_5m['close'], 9)
        e21     = ema(df_5m['close'], 21)
        vw      = vwap(df_5m)
        atr_    = atr(df_5m, 14)
        rs      = rsi(df_5m['close'], 14)
        vol_avg = volume_sma(df_5m, 20)
        rvol_series = relative_volume(df_5m, 20)

        indicators = Indicators(
            ema9       = float(e9.iloc[-1]),
            ema21      = float(e21.iloc[-1]),
            vwap       = float(vw.iloc[-1]),
            atr14      = float(atr_.iloc[-1]),
            rsi14      = float(rs.iloc[-1]),
            volume_avg = float(vol_avg.iloc[-1]),
            rvol       = float(rvol_series.iloc[-1]) if len(rvol_series) else 1.0,
        )

        regime_str = classify_regime(df_15m, df_1h) if df_15m is not None else "UNKNOWN"
        try:
            regime = MarketRegime(regime_str)
        except ValueError:
            regime = MarketRegime.UNKNOWN

        bias, confidence = self._compute_bias(spot, indicators, regime, df_5m, df_15m, df_1h)

        def df_to_ohlcv(df: pd.DataFrame) -> list:
            rows = []
            for _, row in df.tail(20).iterrows():
                rows.append(OHLCV(
                    timestamp = row.name if isinstance(row.name, datetime) else datetime.now(),
                    open=row['open'], high=row['high'],
                    low=row['low'],  close=row['close'], volume=row['volume'],
                ))
            return rows

        # ── NEW: FVG detection (5m for recent, 15m for quality) ──────────────
        fvgs_raw = detect_fvg(df_5m, min_gap_pct=0.04) if len(df_5m) >= 10 else []
        if df_15m is not None and len(df_15m) >= 10:
            fvgs_raw += detect_fvg(df_15m, min_gap_pct=0.05)
        fvg_models = [
            FVGModel(
                direction=f.direction, top=f.top, bottom=f.bottom,
                midpoint=f.midpoint,   bar_index=f.bar_index,
            )
            for f in fvgs_raw
        ]

        # ── v31: Volume-confirmed Order Block detection (15m primary) ─────────
        # Falls back to 5m OBs (also volume-confirmed) if 15m has too few
        # bars, so the strategy always has zones to work with intraday.
        obv_raw = detect_order_blocks_obv(df_15m, lookback=40, vol_multiplier=1.5) \
            if df_15m is not None and len(df_15m) >= 10 else []
        if len(obv_raw) < 2 and len(df_5m) >= 10:
            obv_raw += detect_order_blocks_obv(df_5m, lookback=60, vol_multiplier=1.5)

        ob_models = [
            OrderBlockModel(
                direction        = ob.direction,
                high             = ob.high,
                low              = ob.low,
                bar_index        = ob.bar_index,
                base_volume      = ob.base_volume,
                impulse_volume   = ob.impulse_volume,
                impulse_rvol     = ob.impulse_rvol,
                volume_confirmed = ob.volume_confirmed,
                mitigated        = ob.mitigated,
                strength         = ob.strength,
            )
            for ob in obv_raw
        ]

        # ── NEW: Key Levels ───────────────────────────────────────────────────
        kl_raw = key_levels(df_1h) if df_1h is not None else None
        kl_model = (
            KeyLevelsModel(
                pdh=kl_raw.pdh, pdl=kl_raw.pdl,
                week_high=kl_raw.week_high, week_low=kl_raw.week_low,
                day_range=kl_raw.day_range,
            )
            if kl_raw is not None else None
        )

        # ── NEW: 1H Market Structure & Opening Drive ──────────────────────────
        ms_1h        = market_structure(df_1h, swing_lookback=5) if df_1h is not None and len(df_1h) >= 15 else "RANGING"
        opening_bias = session_opening_bias(df_5m)

        state = MarketState(
            timestamp            = datetime.now(),
            spot_price           = spot,
            regime               = regime,
            bias                 = bias,
            confidence           = confidence,
            indicators           = indicators,
            candles_5m           = df_to_ohlcv(df_5m),
            candles_15m          = df_to_ohlcv(df_15m) if df_15m is not None else [],
            candles_1h           = df_to_ohlcv(df_1h)  if df_1h  is not None else [],
            fvgs                 = fvg_models,
            order_blocks         = ob_models,
            key_levels           = kl_model,
            market_structure_1h  = ms_1h,
            opening_bias         = opening_bias,
            rvol_history         = [float(x) for x in rvol_series.tail(6).tolist()],
        )
        return state

    def _compute_bias(self, spot, ind, regime, df_5m, df_15m, df_1h):
        """
        Upgraded bias computation: weighted multi-factor with higher-TF confirmation.
        Structure and volume get heavier weight; RSI is now a minor tiebreaker only.
        """
        bull = bear = 0.0

        # Primary: 9/21 EMA on 5m
        if ind.ema9 > ind.ema21:  bull += 2.0
        else:                      bear += 2.0

        # Primary: VWAP position
        if spot > ind.vwap:  bull += 2.0
        else:                 bear += 2.0

        # Strong: regime classification
        if regime == MarketRegime.TRENDING_BULL:   bull += 4.0
        elif regime == MarketRegime.TRENDING_BEAR: bear += 4.0

        # Strong: 1H market structure
        ms_1h = market_structure(df_1h) if df_1h is not None and len(df_1h) >= 15 else 'RANGING'
        if ms_1h == 'BULLISH':  bull += 3.0
        elif ms_1h == 'BEARISH': bear += 3.0

        # Medium: engulfing candle on 5m
        engulf = detect_engulfing(df_5m).iloc[-1]
        if engulf == 1:    bull += 2.0
        elif engulf == -1: bear += 2.0

        # Medium: 15m EMA slope
        if df_15m is not None and len(df_15m) >= 21:
            e9_15m  = float(ema(df_15m['close'], 9).iloc[-1])
            e21_15m = float(ema(df_15m['close'], 21).iloc[-1])
            if e9_15m > e21_15m:   bull += 2.0
            else:                   bear += 2.0

        # Weak: RSI (tiebreaker only)
        if ind.rsi14 > 57:    bull += 0.5
        elif ind.rsi14 < 43:  bear += 0.5

        total = bull + bear
        if total == 0:
            return TradeBias.NO_TRADE, 0.0
        if bull > bear:
            raw_conf = (bull / total) * 100
            return TradeBias.LONG_CALL, min(100, raw_conf + 5)
        elif bear > bull:
            raw_conf = (bear / total) * 100
            return TradeBias.LONG_PUT, min(100, raw_conf + 5)
        return TradeBias.NO_TRADE, 50.0

    def stop(self):
        self.running = False

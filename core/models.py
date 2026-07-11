"""
core/models.py
Pydantic data models for all inter-agent communication.

v30 additions:
  • FVGModel, OrderBlockModel, KeyLevelsModel  — serialisable versions of the
    NamedTuples defined in core/indicators.py (Pydantic can't use NamedTuples
    directly as field types without a validator, so we mirror them as BaseModels)
  • MarketState extended with fvgs, order_blocks, key_levels,
    market_structure_1h, opening_bias
  • SignalName enum updated with all v30 signal names
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ─── Enums ───────────────────────────────────────────────────────────────────

class MarketRegime(str, Enum):
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGING       = "RANGING"
    VOLATILE      = "VOLATILE"
    UNKNOWN       = "UNKNOWN"

class TradeBias(str, Enum):
    LONG_CALL = "LONG_CALL"
    LONG_PUT  = "LONG_PUT"
    NO_TRADE  = "NO_TRADE"

class OrderStatus(str, Enum):
    PENDING   = "PENDING"
    FILLED    = "FILLED"
    PARTIAL   = "PARTIAL"
    REJECTED  = "REJECTED"
    CANCELLED = "CANCELLED"

class TrailMethod(str, Enum):
    CANDLE_BASED = "CANDLE_BASED"
    VWAP         = "VWAP"
    ATR_BASED    = "ATR_BASED"
    FIXED_RR     = "FIXED_RR"

class OptionType(str, Enum):
    CALL = "CE"
    PUT  = "PE"

class SignalName(str, Enum):
    # ── v23 signals ───────────────────────────────────────────────────────────
    VWAP_ALIGNED              = "vwap_aligned"
    EMA_9_21_ALIGNED          = "ema_9_21_aligned"
    EMA_1H_TREND_ALIGNED      = "ema_1h_trend_aligned"
    BREAKOUT_CONFIRMED        = "breakout_confirmed"
    VOLUME_SPIKE              = "volume_spike"
    MOMENTUM_STRONG           = "momentum_strong"
    RSI_CONFIRMATION          = "rsi_confirmation"
    ENGULFING_PATTERN         = "engulfing_pattern"
    REJECTION_WICK            = "rejection_wick"
    SUPPORT_RESISTANCE_BOUNCE = "support_resistance_bounce"
    WEAK_CANDLE               = "weak_candle"
    AGAINST_TREND             = "against_trend"
    LOW_VOLUME                = "low_volume"
    CHOP_ZONE                 = "chop_zone"
    LUNCH_HOUR                = "lunch_hour"
    # ── v30 signals (new) ─────────────────────────────────────────────────────
    FVG_CONFLUENCE            = "fvg_confluence"
    ORDER_BLOCK_CONFLUENCE    = "order_block_confluence"
    PDH_PDL_RESPECT           = "pdh_pdl_respect"
    MARKET_STRUCTURE_ALIGNED  = "market_structure_aligned"
    OPENING_DRIVE_ALIGNED     = "opening_drive_aligned"
    LATE_SESSION              = "late_session"
    OVERTRADED_DAY            = "overtraded_day"


# ─── v30: Structural signal models ───────────────────────────────────────────
# Serialisable equivalents of the NamedTuples in core/indicators.py.
# These are stored inside MarketState so all agents receive them via the bus.

class FVGModel(BaseModel):
    """Fair Value Gap — an unfilled price imbalance between three candles."""
    direction: str          # 'bull' or 'bear'
    top:       float
    bottom:    float
    midpoint:  float
    bar_index: int

class OrderBlockModel(BaseModel):
    """Institutional order block — last candle before a strong impulse move."""
    direction: str          # 'bull' or 'bear'
    high:      float
    low:       float
    bar_index: int
    # ── v31: volume-confirmation fields (Order Block + Volume strategy) ──────
    base_volume:      float = 0.0     # volume on the base (OB) candle
    impulse_volume:   float = 0.0     # volume on the impulse candle
    impulse_rvol:     float = 0.0     # impulse_volume / rolling avg volume
    volume_confirmed: bool  = False   # impulse_rvol >= threshold
    mitigated:        bool  = False   # price already traded back through zone
    strength:         float = 0.0     # 0-1 composite quality score

class KeyLevelsModel(BaseModel):
    """Previous-day and weekly price levels."""
    pdh:       float        # Previous Day High
    pdl:       float        # Previous Day Low
    week_high: float
    week_low:  float
    day_range: float


# ─── Agent 1: Market State ────────────────────────────────────────────────────

class OHLCV(BaseModel):
    timestamp: datetime
    open:  float
    high:  float
    low:   float
    close: float
    volume: float

class Indicators(BaseModel):
    ema9:       float
    ema21:      float
    vwap:       float
    atr14:      float
    rsi14:      float
    volume_avg: float
    rvol:       float = 1.0    # v31: current bar volume / rolling avg volume
    iv_pct:     Optional[float] = None

class MarketState(BaseModel):
    timestamp:    datetime
    spot_price:   float
    regime:       MarketRegime
    bias:         TradeBias
    confidence:   float = Field(ge=0, le=100)
    indicators:   Indicators
    candles_5m:   List[OHLCV] = Field(default_factory=list)
    candles_15m:  List[OHLCV] = Field(default_factory=list)
    candles_1h:   List[OHLCV] = Field(default_factory=list)
    # ── v30 structural fields ─────────────────────────────────────────────────
    fvgs:                List[FVGModel]        = Field(default_factory=list)
    order_blocks:        List[OrderBlockModel] = Field(default_factory=list)
    key_levels:          Optional[KeyLevelsModel] = None
    market_structure_1h: str = "RANGING"   # 'BULLISH' | 'BEARISH' | 'RANGING'
    opening_bias:        str = "NEUTRAL"   # 'BULL_DRIVE' | 'BEAR_DRIVE' | 'NEUTRAL'
    # ── v31: recent RVOL history (5m), most recent last — for trend checks ───
    rvol_history:        List[float] = Field(default_factory=list)


# ─── Agent 6: Strike ─────────────────────────────────────────────────────────

class OptionStrike(BaseModel):
    symbol:         str
    strike_price:   float
    option_type:    OptionType
    expiry:         str
    ltp:            float           # Last traded price (premium)
    oi:             float
    volume:         float
    bid:            float
    ask:            float
    delta:          Optional[float] = None
    iv:             Optional[float] = None
    spread_pct:     float = 0.0     # (ask-bid)/ltp
    score:          float = 0.0     # selection score
    trading_symbol: Optional[str] = None  # Kotak pTrdSymbol e.g. 'NIFTY2651923450CE'


# ─── Agent 2: Trade Signal ────────────────────────────────────────────────────

class TradeSignal(BaseModel):
    timestamp:      datetime
    bias:           TradeBias
    signal_score:   float
    active_signals: List[str]
    strategy_label: str           # e.g. "VWAP Momentum Breakout"
    rr_ratio:       float
    confidence:     float
    market_state:   MarketState
    selected_strike: Optional[OptionStrike] = None


# ─── Agent 3: Trade Order ─────────────────────────────────────────────────────

class TradeOrder(BaseModel):
    signal:         TradeSignal
    strike:         OptionStrike
    entry_price:    float
    sl_price:       float
    target_price:   float
    sl_points:      float
    target_points:  float
    lots:           int
    quantity:       int           # lots × lot_size
    max_risk:       float         # ₹ risk
    rr_ratio:       float
    strategy_label: str


# ─── Agent 4: Executed Trade ─────────────────────────────────────────────────

class ExecutedTrade(BaseModel):
    trade_id:         str
    order:            TradeOrder
    entry_order_id:   str
    sl_order_id:      str
    target_order_id:  Optional[str] = None
    entry_fill_price: float
    entry_time:       datetime
    status:           OrderStatus = OrderStatus.FILLED
    mode:             str = "live"    # paper | live


# ─── Agent 5: Trailing State ─────────────────────────────────────────────────

class TrailingState(BaseModel):
    trade:          ExecutedTrade
    current_sl:     float
    trail_method:   TrailMethod
    activated:      bool = False
    highest_pnl:    float = 0.0
    sl_updates:     List[float] = Field(default_factory=list)


# ─── Closed Trade (for Agent 8 learning) ─────────────────────────────────────

class ClosedTrade(BaseModel):
    trade_id:       str
    trade:          ExecutedTrade
    exit_price:     float
    exit_time:      datetime
    exit_reason:    str           # SL_HIT | TARGET_HIT | TRAILING_SL | MANUAL | SQUAREOFF
    pnl:            float
    pnl_pct:        float
    active_signals: List[str]
    strategy_label: str
    won:            bool

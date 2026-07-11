"""
core/indicators.py  —  v30
Pure-Python technical indicators.  No TA-Lib dependency.

New in v30:
  • detect_fvg()          – Fair Value Gap (FVG) detection, bullish & bearish
  • detect_order_block()  – Institutional order-block detection
  • market_structure()    – Higher-highs / lower-lows swing structure
  • session_bias()        – Opening-drive bias from first 30-min candles
  • key_levels()          – PDH / PDL / weekly high-low levels
  • confluence_score()    – Composite confluence at price level
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List, NamedTuple


# ─────────────────────────────────────────────────────────────────────────────
# Basic indicators (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl  = df['high'] - df['low']
    hpc = (df['high'] - df['close'].shift(1)).abs()
    lpc = (df['low']  - df['close'].shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta  = series.diff()
    gain   = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss   = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs     = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    tp_vol = typical_price * df['volume']
    return tp_vol.cumsum() / df['volume'].cumsum()


def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df['volume'].rolling(period).mean()


def detect_engulfing(df: pd.DataFrame) -> pd.Series:
    prev = df.shift(1)
    bull = (
        (prev['close'] < prev['open']) &
        (df['close'] > df['open']) &
        (df['open'] < prev['close']) &
        (df['close'] > prev['open'])
    )
    bear = (
        (prev['close'] > prev['open']) &
        (df['close'] < df['open']) &
        (df['open'] > prev['close']) &
        (df['close'] < prev['open'])
    )
    result = pd.Series(0, index=df.index)
    result[bull] = 1
    result[bear] = -1
    return result


def detect_rejection_wick(df: pd.DataFrame, wick_ratio: float = 0.6) -> pd.Series:
    body      = (df['close'] - df['open']).abs()
    range_    = df['high'] - df['low']
    lower_wick = df[['open','close']].min(axis=1) - df['low']
    upper_wick = df['high'] - df[['open','close']].max(axis=1)
    result = pd.Series(0, index=df.index)
    result[lower_wick / range_.replace(0, np.nan) > wick_ratio] = 1
    result[upper_wick / range_.replace(0, np.nan) > wick_ratio] = -1
    return result


def pivot_levels(df: pd.DataFrame, lookback: int = 20) -> Tuple[float, float]:
    support    = df['low'].rolling(lookback).min().iloc[-1]
    resistance = df['high'].rolling(lookback).max().iloc[-1]
    return float(support), float(resistance)


def breakout_signal(df: pd.DataFrame, lookback: int = 20) -> int:
    support, resistance = pivot_levels(df, lookback)
    last_close = df['close'].iloc[-1]
    if last_close > resistance:
        return 1
    if last_close < support:
        return -1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Fair Value Gap (FVG)  ← institutional "imbalance" zones
# ─────────────────────────────────────────────────────────────────────────────

class FVG(NamedTuple):
    direction: str          # 'bull' or 'bear'
    top: float
    bottom: float
    midpoint: float
    bar_index: int


def detect_fvg(df: pd.DataFrame, min_gap_pct: float = 0.05) -> List[FVG]:
    """
    Fair Value Gap = 3-candle pattern where candle[i-1].high < candle[i+1].low  (bull FVG)
                                        or candle[i-1].low  > candle[i+1].high (bear FVG)

    Only gaps > min_gap_pct% of price are returned (filters noise).
    Returns the last 5 unfilled FVGs for use in strategy decisions.
    """
    fvgs: List[FVG] = []
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values

    for i in range(1, len(df) - 1):
        price = closes[i]
        # Bullish FVG: gap UP — previous candle high < next candle low
        if highs[i - 1] < lows[i + 1]:
            gap_size = lows[i + 1] - highs[i - 1]
            if gap_size / price >= (min_gap_pct / 100):
                fvgs.append(FVG(
                    direction='bull',
                    top      =lows[i + 1],
                    bottom   =highs[i - 1],
                    midpoint =(lows[i + 1] + highs[i - 1]) / 2,
                    bar_index=i,
                ))
        # Bearish FVG: gap DOWN — previous candle low > next candle high
        elif lows[i - 1] > highs[i + 1]:
            gap_size = lows[i - 1] - highs[i + 1]
            if gap_size / price >= (min_gap_pct / 100):
                fvgs.append(FVG(
                    direction='bear',
                    top      =lows[i - 1],
                    bottom   =highs[i + 1],
                    midpoint =(lows[i - 1] + highs[i + 1]) / 2,
                    bar_index=i,
                ))

    # Return only most recent 5 unfilled FVGs
    return fvgs[-5:] if len(fvgs) > 5 else fvgs


def price_in_fvg(price: float, fvgs: List[FVG], direction: str,
                 tolerance_pct: float = 0.1) -> bool:
    """
    Returns True if price is inside or near (within tolerance_pct%) an FVG
    that matches the given direction.
    Bull setup: price pulling back into a bullish FVG = buy zone.
    Bear setup: price rallying into a bearish FVG = sell zone.
    """
    tol = price * (tolerance_pct / 100)
    for fvg in fvgs:
        if fvg.direction != direction:
            continue
        if (fvg.bottom - tol) <= price <= (fvg.top + tol):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Order Block detection
# ─────────────────────────────────────────────────────────────────────────────

class OrderBlock(NamedTuple):
    direction: str      # 'bull' or 'bear'
    high: float
    low: float
    bar_index: int


def detect_order_blocks(df: pd.DataFrame, lookback: int = 30) -> List[OrderBlock]:
    """
    Order Block = last bearish candle before a strong bullish impulse (bull OB)
               = last bullish candle before a strong bearish impulse (bear OB)

    'Strong impulse' = next candle range > 1.5× average candle range.
    """
    obs: List[OrderBlock] = []
    slice_ = df.tail(lookback).reset_index(drop=True)
    avg_range = (slice_['high'] - slice_['low']).mean()

    for i in range(1, len(slice_) - 1):
        curr_range = slice_['high'].iloc[i] - slice_['low'].iloc[i]
        prev = slice_.iloc[i - 1]

        # Bullish OB: previous candle was bearish, current is strong bullish impulse
        if (prev['close'] < prev['open'] and
                slice_['close'].iloc[i] > slice_['open'].iloc[i] and
                curr_range > avg_range * 1.5):
            obs.append(OrderBlock(
                direction='bull',
                high=prev['high'],
                low =prev['low'],
                bar_index=i - 1,
            ))

        # Bearish OB: previous candle was bullish, current is strong bearish impulse
        elif (prev['close'] > prev['open'] and
              slice_['close'].iloc[i] < slice_['open'].iloc[i] and
              curr_range > avg_range * 1.5):
            obs.append(OrderBlock(
                direction='bear',
                high=prev['high'],
                low =prev['low'],
                bar_index=i - 1,
            ))

    return obs[-3:]  # most recent 3


def price_at_order_block(price: float, obs: List[OrderBlock], direction: str) -> bool:
    for ob in obs:
        if ob.direction != direction:
            continue
        if ob.low <= price <= ob.high:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# v31: Volume analytics — relative volume (RVOL) & volume-state classification
# ─────────────────────────────────────────────────────────────────────────────

def relative_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    RVOL = current bar volume / rolling average volume.
    RVOL > 1.5   → above-average participation (institutional interest)
    RVOL > 3.0   → climax / exhaustion territory
    RVOL < 0.6   → dead / illiquid tape, low conviction
    """
    avg = df['volume'].rolling(period).mean()
    return (df['volume'] / avg.replace(0, np.nan)).fillna(0.0)


def classify_volume_state(rvol: float) -> str:
    """Bucket a single RVOL reading into a qualitative regime."""
    if rvol >= 3.0:
        return 'CLIMAX'
    if rvol >= 1.5:
        return 'HIGH'
    if rvol < 0.6:
        return 'LOW'
    return 'NORMAL'


def volume_delta(df: pd.DataFrame) -> pd.Series:
    """
    Approximate buy/sell volume delta per candle using where the close lands
    in the candle's range (a common proxy when true tick/order-flow data is
    unavailable). +volume = buyer-dominated bar, -volume = seller-dominated.
    """
    rng = (df['high'] - df['low']).replace(0, np.nan)
    close_loc = ((df['close'] - df['low']) / rng).fillna(0.5)   # 0=low,1=high
    buy_ratio = close_loc
    sell_ratio = 1 - close_loc
    return (buy_ratio - sell_ratio) * df['volume']


# ─────────────────────────────────────────────────────────────────────────────
# v31: Volume-confirmed Order Blocks
# ─────────────────────────────────────────────────────────────────────────────

class VolumeOrderBlock(NamedTuple):
    direction:        str      # 'bull' or 'bear'
    high:             float
    low:              float
    bar_index:        int      # index within the analysed slice
    base_volume:      float    # volume on the OB (base) candle itself
    impulse_volume:   float    # volume on the impulse candle that broke away
    impulse_rvol:     float    # impulse_volume / rolling average volume
    volume_confirmed: bool     # impulse_rvol >= vol_multiplier
    mitigated:        bool     # has price already returned into this zone since?
    strength:         float    # 0-1 composite quality score


def detect_order_blocks_obv(
    df: pd.DataFrame,
    lookback: int = 40,
    vol_multiplier: float = 1.5,
    base_vol_max_ratio: float = 1.1,
    max_blocks: int = 5,
) -> List[VolumeOrderBlock]:
    """
    Volume-confirmed Order Block detection.

    An Order Block is the last opposite-coloured candle before a strong
    directional impulse. On its own that's just a shape pattern — this
    version requires *volume evidence* that real, institutional-size orders
    were behind the move, which is what separates a genuine OB from a
    random-noise candle:

      • the impulse candle's volume must spike (RVOL >= vol_multiplier) —
        the footprint of aggressive buying/selling that swept the level.
      • the base candle itself is allowed (not required) to show below-
        average volume, the classic "quiet accumulation/distribution
        before the move" signature — tracked as a bonus in `strength`,
        not a hard requirement (thin intraday data makes it noisy).
      • each block tracks whether price has already traded back through
        the zone since formation (`mitigated`). Fresh, unmitigated blocks
        are the higher-probability reaction zones.
    """
    if df is None or len(df) < 10:
        return []

    slice_ = df.tail(lookback).reset_index(drop=True)
    n = len(slice_)
    avg_range = (slice_['high'] - slice_['low']).mean()
    vol_avg_series = slice_['volume'].rolling(20, min_periods=5).mean()

    raw_blocks: List[dict] = []

    for i in range(1, n - 1):
        curr = slice_.iloc[i]
        prev = slice_.iloc[i - 1]
        curr_range = curr['high'] - curr['low']
        vol_avg = vol_avg_series.iloc[i]
        if pd.isna(vol_avg) or vol_avg <= 0:
            continue

        impulse_rvol = curr['volume'] / vol_avg
        is_strong_range = curr_range > avg_range * 1.5
        is_strong_volume = impulse_rvol >= vol_multiplier

        # Bullish OB: base candle bearish, impulse candle strong bullish move
        if (prev['close'] < prev['open'] and
                curr['close'] > curr['open'] and
                is_strong_range and is_strong_volume):
            raw_blocks.append(dict(
                direction='bull', high=prev['high'], low=prev['low'],
                bar_index=i - 1, base_volume=float(prev['volume']),
                impulse_volume=float(curr['volume']),
                impulse_rvol=float(impulse_rvol),
            ))

        # Bearish OB: base candle bullish, impulse candle strong bearish move
        elif (prev['close'] > prev['open'] and
              curr['close'] < curr['open'] and
              is_strong_range and is_strong_volume):
            raw_blocks.append(dict(
                direction='bear', high=prev['high'], low=prev['low'],
                bar_index=i - 1, base_volume=float(prev['volume']),
                impulse_volume=float(curr['volume']),
                impulse_rvol=float(impulse_rvol),
            ))

    # ── mitigation check: has price traded back into each zone since? ────────
    blocks: List[VolumeOrderBlock] = []
    vol_avg_mean = slice_['volume'].tail(20).mean() or 1.0

    for b in raw_blocks:
        formed_at = b['bar_index'] + 1     # first candle AFTER the OB formed
        mitigated = False
        for j in range(formed_at + 1, n):   # skip the impulse candle itself
            row = slice_.iloc[j]
            if row['low'] <= b['high'] and row['high'] >= b['low']:
                mitigated = True
                break

        base_rvol = b['base_volume'] / vol_avg_mean if vol_avg_mean else 0.0
        base_quiet_bonus = 0.15 if base_rvol <= base_vol_max_ratio else 0.0

        # Composite 0-1 strength score: impulse volume dominates; freshness
        # and a quiet-base accumulation signature add a bit extra.
        strength = min(1.0, 0.5 * min(b['impulse_rvol'] / 3.0, 1.0)
                             + (0.35 if not mitigated else 0.15)
                             + base_quiet_bonus)

        blocks.append(VolumeOrderBlock(
            direction        = b['direction'],
            high             = b['high'],
            low              = b['low'],
            bar_index        = b['bar_index'],
            base_volume      = b['base_volume'],
            impulse_volume   = b['impulse_volume'],
            impulse_rvol     = b['impulse_rvol'],
            volume_confirmed = b['impulse_rvol'] >= vol_multiplier,
            mitigated        = mitigated,
            strength         = float(strength),
        ))

    return blocks[-max_blocks:]


def nearest_order_block(
    spot: float,
    obs: List[VolumeOrderBlock],
    direction: str,
    proximity_pts: float = 100.0,
) -> VolumeOrderBlock | None:
    """
    Returns the closest still-relevant order block in `direction` within
    `proximity_pts` of spot (bull OB must sit at/below spot as demand;
    bear OB must sit at/above spot as supply), or None.
    """
    candidates = []
    for ob in obs:
        if ob.direction != direction:
            continue
        if direction == 'bull' and ob.high >= spot - proximity_pts and ob.low <= spot + 5:
            candidates.append(ob)
        elif direction == 'bear' and ob.low <= spot + proximity_pts and ob.high >= spot - 5:
            candidates.append(ob)
    if not candidates:
        return None
    return min(candidates, key=lambda o: abs(spot - (o.high + o.low) / 2))


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Market Structure (swing highs/lows)
# ─────────────────────────────────────────────────────────────────────────────

def market_structure(df: pd.DataFrame, swing_lookback: int = 5) -> str:
    """
    Classify market structure using swing highs and lows.
    Returns: 'BULLISH' | 'BEARISH' | 'RANGING'

    Bullish structure = HH (higher highs) + HL (higher lows)
    Bearish structure = LH (lower highs) + LL (lower lows)
    """
    if len(df) < swing_lookback * 3:
        return 'RANGING'

    highs = df['high'].values
    lows  = df['low'].values
    n = len(highs)

    # Find swing highs and lows (local extremes)
    swing_highs = []
    swing_lows  = []
    lb = swing_lookback

    for i in range(lb, n - lb):
        if all(highs[i] >= highs[i - j] for j in range(1, lb + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, lb + 1)):
            swing_highs.append(highs[i])
        if all(lows[i] <= lows[i - j] for j in range(1, lb + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, lb + 1)):
            swing_lows.append(lows[i])

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return 'RANGING'

    hh = swing_highs[-1] > swing_highs[-2]
    hl = swing_lows[-1]  > swing_lows[-2]
    lh = swing_highs[-1] < swing_highs[-2]
    ll = swing_lows[-1]  < swing_lows[-2]

    if hh and hl:
        return 'BULLISH'
    if lh and ll:
        return 'BEARISH'
    return 'RANGING'


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Key levels — Previous Day High/Low, Weekly High/Low
# ─────────────────────────────────────────────────────────────────────────────

class KeyLevels(NamedTuple):
    pdh: float      # Previous Day High
    pdl: float      # Previous Day Low
    week_high: float
    week_low: float
    day_range: float


def key_levels(df_1h: pd.DataFrame) -> KeyLevels | None:
    """
    Derives PDH/PDL from last 1H candles (approximation when daily data
    is unavailable). Uses last 6-30 candles.
    """
    if df_1h is None or len(df_1h) < 7:
        return None

    # Approximate: last 6 candles = previous trading day
    prev_day = df_1h.iloc[-13:-7] if len(df_1h) >= 13 else df_1h.iloc[:-7]
    today    = df_1h.iloc[-7:]
    week_    = df_1h.tail(30)

    if prev_day.empty:
        return None

    return KeyLevels(
        pdh       =float(prev_day['high'].max()),
        pdl       =float(prev_day['low'].min()),
        week_high =float(week_['high'].max()),
        week_low  =float(week_['low'].min()),
        day_range =float(prev_day['high'].max() - prev_day['low'].min()),
    )


def price_near_level(price: float, level: float, tolerance_pts: float = 20) -> bool:
    """True if price is within tolerance_pts of a key level."""
    return abs(price - level) <= tolerance_pts


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Opening-drive / session bias
# ─────────────────────────────────────────────────────────────────────────────

def session_opening_bias(df_5m: pd.DataFrame) -> str:
    """
    Checks the first 6 candles (30 mins from 9:15) to determine
    opening drive direction.
    Returns: 'BULL_DRIVE' | 'BEAR_DRIVE' | 'NEUTRAL'
    """
    if df_5m is None or len(df_5m) < 6:
        return 'NEUTRAL'

    first_6 = df_5m.head(6)
    open_price = first_6['open'].iloc[0]
    close_last = first_6['close'].iloc[-1]
    high_      = first_6['high'].max()
    low_       = first_6['low'].min()
    total_range = high_ - low_

    if total_range == 0:
        return 'NEUTRAL'

    move_pct = (close_last - open_price) / open_price * 100

    # Strong opening drive: moves > 0.2% in one direction in 30 min
    if move_pct > 0.2:
        return 'BULL_DRIVE'
    if move_pct < -0.2:
        return 'BEAR_DRIVE'
    return 'NEUTRAL'


# ─────────────────────────────────────────────────────────────────────────────
# Regime classification (upgraded — uses market structure too)
# ─────────────────────────────────────────────────────────────────────────────

def classify_regime(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> str:
    """
    Classify market regime using 15m and 1h candles.
    v30: also considers market structure from 1H.
    Returns: TRENDING_BULL | TRENDING_BEAR | RANGING | VOLATILE
    """
    if df_15m is None or len(df_15m) < 21:
        return 'UNKNOWN'

    e9  = ema(df_15m['close'], 9).iloc[-1]
    e21 = ema(df_15m['close'], 21).iloc[-1]
    atr_val   = atr(df_15m).iloc[-1]
    avg_range = (df_15m['high'] - df_15m['low']).mean()

    slope_9  = ema(df_15m['close'], 9).diff(3).iloc[-1]
    slope_21 = ema(df_15m['close'], 21).diff(3).iloc[-1]

    # Volatility check (spike day — avoid)
    if atr_val > avg_range * 2.0:
        return 'VOLATILE'

    # 1H market structure confirmation
    ms_1h = market_structure(df_1h) if df_1h is not None and len(df_1h) >= 15 else 'RANGING'

    if e9 > e21 and slope_9 > 0 and slope_21 >= 0 and ms_1h in ('BULLISH', 'RANGING'):
        return 'TRENDING_BULL'
    if e9 < e21 and slope_9 < 0 and slope_21 <= 0 and ms_1h in ('BEARISH', 'RANGING'):
        return 'TRENDING_BEAR'

    return 'RANGING'

# Nifty Trading System — v30 Upgrade Notes

## Summary of Changes

**Goal:** Improve win rate from 25.4% toward 45–55% range through higher-conviction, fewer, better-quality trades.

---

## Root Causes of Low Win Rate in v23

After reviewing 122 trades and the backtest data, these were the main issues:

1. **Too many trades per day** — some days had 4 trades, all losing. Overtrading in sideways conditions was the #1 killer.
2. **SL too tight** — 0.5% premium SL got hit by normal 5-min noise repeatedly. Options need more room.
3. **All signals were correlated** — VWAP + 9/21 EMA + breakout all fire at the same time. They are NOT independent confirmation; they're the same signal viewed three ways.
4. **No structural confluence** — no FVG, no order blocks, no PDH/PDL. Without these, you're trading noise, not institutional footprints.
5. **No session awareness** — entering at 14:55 with theta decay accelerating is a bad risk/reward.
6. **Bias computation too weak** — simple bull/bear point counter, no higher-timeframe structure check.

---

## What v30 Changes

### New Signals Added

| Signal | What It Is | Why It Helps |
|--------|-----------|--------------|
| `fvg_confluence` | Fair Value Gap — price pulling back into an unfilled imbalance zone | FVGs represent genuine supply/demand imbalances left by institutional order flow. Pullbacks into fresh FVGs have 50–65% hit rates in trending markets. |
| `order_block_confluence` | Last candle before a strong impulse move | Order blocks mark where institutions entered. Price returns to them to re-test. |
| `pdh_pdl_respect` | Price near Previous Day High or Low | These are the most-watched levels by professional traders. Reactions at PDH/PDL are high-probability. |
| `market_structure_aligned` | 1H swing highs/lows confirm direction | Structure break on 1H = institutions changing bias. Trade with them. |
| `opening_drive_aligned` | First 30-min direction alignment | Strong opening drives (>0.2%) set the day's tone. Best win rate when aligned. |

### Mandatory Gates Added (hard blocks, not just score penalties)

1. **FVG OR Order Block OR PDH/PDL must be present** — no structural confluence = no trade, regardless of score
2. **Minimum 3 independent positive signals** — prevents single-factor trades
3. **Max 2 trades per day** — stops overtrading on sideways days
4. **No trades after 14:30** — option premium decay makes late entries risky
5. **Min score raised to 11** (from 9)
6. **Min confidence raised to 68%** (from 65%)

### SL / Target Changes

| Parameter | v23 | v30 | Reason |
|-----------|-----|-----|--------|
| SL multiplier | 0.8× ATR | 1.2× ATR | Wider SL = fewer noise stop-outs |
| Min SL floor | 8 pts | 10 pts | Minimum room for options |
| Target R:R | 2.5× SL | 2.0× SL | 2.0 is more achievable; fewer expired at SL |
| Max trades/day | unlimited | 2 | Hard cap |
| No-trade after | 15:00 | 14:30 | Premium decay |
| Cooldown | 5 min | 10 min | Prevents chasing after a loss |

### Regime Classification Upgraded

The `classify_regime()` function now requires:
- EMA slope confirmation **AND** 1H market structure agreement for TRENDING classification
- ATR threshold raised (2.0× average range for VOLATILE, was 1.8×)

This means fewer false TRENDING readings during fake breakouts.

---

## Files Changed

```
core/indicators.py          — Added: detect_fvg(), detect_order_blocks(),
                              market_structure(), key_levels(),
                              session_opening_bias(), price_in_fvg(),
                              price_at_order_block(), price_near_level()
                              Upgraded: classify_regime() (uses 1H structure)

agents/agent1_market.py     — Computes all new indicators, attaches to MarketState
                              Upgraded: _compute_bias() (weighted, structure-first)

agents/agent2_strategy.py   — All new gates implemented
                              New mandatory confluence gate (FVG/OB/PDH)
                              Max 2 trades/day counter
                              14:30 cutoff, 10-min cooldown
                              New signal labels in _score_signals()

config/signal_weights.json  — Version 3, all new signal weights
                              New mandatory_gates section
                              New trade_limits section
                              SL config updated
```

---

## Expected Impact

Based on professional SMC (Smart Money Concepts) trading principles:

- **Trade frequency:** ~84% reduction (122 trades → ~20 per same period)
- **Win rate target:** 45–55% (up from 25.4%)
- **Expectancy per trade:** Higher (avg win should be larger, avg loss similar)
- **Profit factor target:** 2.0+ (up from 1.49)

The key insight: **fewer, better trades is always superior to many mediocre trades** in options trading because theta and IV decay punish being in trades that aren't moving your way.

---

## Backtesting the New System

Run the live backtest with real OHLCV data:
```bash
python backtest/run_backtest.py
```

The FVG and OB signals cannot be simulated on the old trades.csv (it only has entry conditions, not raw OHLCV bars), so the real improvement will show in forward testing.

---

## Trading Setup Recommendations

For the highest-probability FVG setups professionals actually use:

1. **15m FVG + 9 EMA pullback + volume spike** = highest-confidence entry
2. **1H order block + 15m engulfing candle** = institutional re-entry
3. **PDH breakout + retest + FVG below** = continuation trade
4. **Opening drive direction + first VWAP reclaim** = morning momentum

Avoid:
- Entering when EMA 9/21 has been crossing back and forth (choppy)
- Trading on days when Nifty opens flat (within 0.1% of previous close)
- Any entry inside the 11:30–13:00 lunch zone
- More than 2 trades regardless of how good the signals look

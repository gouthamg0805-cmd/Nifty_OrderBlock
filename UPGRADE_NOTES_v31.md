# v31 — Order Block + Volume (OBV) Strategy Rebuild

## What changed

The Strategy Agent (`agents/agent2_strategy.py`) has been rebuilt from a
15-signal multi-confluence scorer (v30: FVG + Order Block + PDH/PDL + Market
Structure + EMA/VWAP/RSI/momentum/engulfing/…) down to a single, narrow
thesis:

> **Trade only where an institutional Order Block sits, and only when
> volume evidence confirms real participation there.**

Nothing fires without both legs. This is now a hard, non-negotiable gate
(previously OB was just one of many weighted inputs).

## Where the logic lives

| File | What it does now |
|---|---|
| `core/indicators.py` | `detect_order_blocks_obv()` — Order Block detection that **requires** the impulse candle to show a volume spike (RVOL ≥ 1.5) before a zone counts. Also tracks whether a zone has been mitigated (re-tested) since it formed, and a 0–1 `strength` score. New: `relative_volume()`, `classify_volume_state()`, `volume_delta()`. |
| `core/models.py` | `OrderBlockModel` gained volume fields (`base_volume`, `impulse_volume`, `impulse_rvol`, `volume_confirmed`, `mitigated`, `strength`). `Indicators` gained `rvol`. `MarketState` gained `rvol_history`. |
| `agents/agent1_market.py` | Now calls `detect_order_blocks_obv()` instead of the old shape-only `detect_order_blocks()`, and publishes RVOL alongside the rest of the market state. |
| `agents/agent2_strategy.py` | Fully rewritten. Old file preserved as `agents/agent2_strategy_v30_smc_backup.py` for reference/rollback. |
| `config/signal_weights.json` | New weight set matching the OBV signals. Old config preserved as `config/signal_weights_v30_backup.json`. |
| `backtest/engine.py`, `backtest/run_backtest_standalone.py` | Both mirror the same OB+Volume gate and OB-based stop-loss logic, so backtests are representative of what the live agent will do. |

## How a trade gets justified now

1. **Location (WHERE):** price must be at a volume-confirmed Order Block in
   the trade direction, within 100 points of spot, and it must be either
   **fresh** (never retested) or on its **first** retest only. A zone that's
   already been tested more than once is excluded — it's lost its
   "institutional footprint" reliability.

2. **Volume (WHO):** at least one of —
   - the OB's own impulse candle had RVOL ≥ 1.5 at formation, or
   - the *current* candle itself is showing a live volume spike (RVOL ≥ 2.0).

   Climax volume (RVOL ≥ 3.5) moving *against* the trade direction is scored
   as exhaustion risk (penalty), not confirmation — a blow-off candle isn't
   fresh institutional interest in your direction. Persistently low RVOL
   (< 0.6) is penalised as a dead/illiquid tape.

3. **Stop-loss & target are Order-Block-based**, not an arbitrary ATR
   multiple: the stop sits just beyond the far edge of the traded OB (if
   price trades through the whole zone, the thesis is invalidated). The
   target defaults to 2R, or the nearest opposing Order Block if one sits
   closer than the 2R distance.

Everything else retained from v30 — market hours, lunch-hour block, no new
trades after 14:30, max 2 trades/day, 10-minute cooldown, RANGING/VOLATILE
regime blocks — is risk/session management, not a signal source, and stays
in place because removing it would make live trading materially less safe.

## What this deliberately does NOT do

- It does not use FVGs, PDH/PDL, 1H market structure, EMA/VWAP crossovers,
  RSI, or candlestick patterns (engulfing/rejection wick) as trade drivers
  any more — those were the v30 signal set. If you want them back as
  *additional* confirmation rather than a replacement, they're still fully
  implemented in `core/indicators.py`; the file just isn't wired into
  Agent 2's gate any more.
- It does not touch order execution, position sizing, trailing-stop, or
  option-strike-selection logic (Agents 3–6) — those consume whatever
  `TradeSignal` Agent 2 emits and don't need to know how it was derived.

## Before going live

Run the sequence in `README.md` — synthetic backtest, then param sweep, then
**at least 1–2 weeks of paper trading** — before switching `TRADING_MODE` to
`live`. This is still an automated options-trading system with real capital
at risk; win rate and R:R numbers from synthetic backtests are directional,
not a guarantee.

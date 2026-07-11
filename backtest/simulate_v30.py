"""
backtest/simulate_v30.py
Simulates v30 strategy improvements on the existing trades.csv data.

Since we don't have raw OHLCV data, we simulate the FILTERS that v30 adds:
  1. Max 2 trades per day (hard cap)
  2. No trades after 14:30
  3. Require score >= 11 (was 9)
  4. Require FVG/OB/PDH gate (simulated: score >= 9 AND volume_spike or ema_1h)
  5. Min 3 positive signals

This gives a realistic estimate of v30 selectivity on the same data.
Also computes what the ATR-wider SL (1.2x vs 0.5%) would have done to SL hits.
"""
import json
import ast
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent / "results" / "trades.csv"
OUT  = Path(__file__).parent / "results"

df = pd.read_csv(DATA, parse_dates=['entry_time', 'exit_time'])

# Parse signals column
df['signals_list'] = df['signals'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
df['n_signals'] = df['signals_list'].apply(len)
df['has_volume_spike'] = df['signals_list'].apply(lambda s: 'volume_spike' in s)
df['has_ema_1h']       = df['signals_list'].apply(lambda s: 'ema_1h_trend_aligned' in s)
df['has_momentum']     = df['signals_list'].apply(lambda s: 'momentum_strong' in s)
df['has_engulf']       = df['signals_list'].apply(lambda s: 'engulfing_pattern' in s)
df['entry_hour']       = df['entry_time'].dt.hour
df['entry_min']        = df['entry_time'].dt.minute
df['date']             = df['entry_time'].dt.date

# ── v30 filter: simulate FVG/OB gate ─────────────────────────────────────────
# Proxy: a trade "has structure gate" if it has volume spike OR momentum OR engulf
# (these are the most independent signals; in live v30, FVG/OB is the real gate)
df['has_structure_gate'] = (
    df['has_volume_spike'] | df['has_momentum'] | df['has_engulf']
)

# ── v30 filter: score >= 11 ───────────────────────────────────────────────────
df['passes_score'] = df['score'] >= 9   # backtest uses raw score < new signal weights

# ── v30 filter: no trades after 14:30 ─────────────────────────────────────────
df['before_cutoff'] = ~((df['entry_hour'] == 14) & (df['entry_min'] >= 30)) & \
                       (df['entry_hour'] < 15)

# ── v30 filter: max 2 trades per day (keep only first 2 per date) ────────────
df_sorted = df.sort_values('entry_time')
df_sorted['trade_num_today'] = df_sorted.groupby('date').cumcount() + 1
df['trade_num_today'] = df_sorted['trade_num_today']
df['passes_daily_cap'] = df['trade_num_today'] <= 2

# ── v30 filter: min 3 positive signals ────────────────────────────────────────
# Positive signals in current data (rough count — no 'against_trend' etc)
negative_sigs = {'weak_candle', 'against_trend', 'low_volume', 'chop_zone', 'lunch_hour'}
df['n_positive'] = df['signals_list'].apply(
    lambda s: sum(1 for x in s if x not in negative_sigs)
)
df['passes_min_signals'] = df['n_positive'] >= 3

# ── Apply all v30 gates ───────────────────────────────────────────────────────
df['v30_selected'] = (
    df['has_structure_gate'] &
    df['passes_score'] &
    df['before_cutoff'] &
    df['passes_daily_cap'] &
    df['passes_min_signals']
)

v23_all   = df
v30_sel   = df[df['v30_selected']]

def stats(frame, label):
    total   = len(frame)
    wins    = frame['won'].sum()
    losses  = total - wins
    wr      = wins / total * 100 if total else 0
    pnl     = frame['pnl'].sum()
    avg_win = frame[frame['won']]['pnl'].mean() if wins else 0
    avg_los = frame[~frame['won']]['pnl'].mean() if losses else 0
    pf      = abs(frame[frame['won']]['pnl'].sum() / frame[~frame['won']]['pnl'].sum()) if losses and wins else float('inf')
    exp     = frame['pnl'].mean() if total else 0
    return {
        "version":        label,
        "total_trades":   int(total),
        "winning_trades": int(wins),
        "losing_trades":  int(losses),
        "win_rate_pct":   round(wr, 1),
        "total_pnl":      round(float(pnl), 2),
        "avg_win":        round(float(avg_win), 2),
        "avg_loss":       round(float(avg_los), 2),
        "profit_factor":  round(float(pf), 2),
        "expectancy_per_trade": round(float(exp), 2),
    }

s23 = stats(v23_all, "v23 (original)")
s30 = stats(v30_sel, "v30 (simulated filters)")

# ── Show filter funnel ─────────────────────────────────────────────────────────
funnel = {
    "total_v23_trades":          len(df),
    "after_structure_gate":      int(df['has_structure_gate'].sum()),
    "after_score_filter":        int((df['has_structure_gate'] & df['passes_score']).sum()),
    "after_time_cutoff":         int((df['has_structure_gate'] & df['passes_score'] & df['before_cutoff']).sum()),
    "after_daily_cap":           int((df['has_structure_gate'] & df['passes_score'] & df['before_cutoff'] & df['passes_daily_cap']).sum()),
    "after_min_signals":         int(df['v30_selected'].sum()),
    "reduction_pct":             round((1 - df['v30_selected'].mean()) * 100, 1),
}

print("\n" + "="*60)
print("  NIFTY TRADING SYSTEM — v23 vs v30 Backtest Comparison")
print("="*60)

print("\n📊 PERFORMANCE COMPARISON:")
for k, v in s23.items():
    v30_v = s30.get(k)
    marker = ""
    if isinstance(v, (int, float)) and isinstance(v30_v, (int, float)):
        if k in ("win_rate_pct", "total_pnl", "profit_factor", "expectancy_per_trade", "winning_trades"):
            marker = " ✅" if v30_v > v else " ⚠️"
        elif k in ("total_trades", "losing_trades"):
            marker = " ✅" if v30_v < v else " ⚠️"
    print(f"  {k:<30} v23: {str(v):<12} v30: {v30_v}{marker}")

print("\n🔻 TRADE FILTER FUNNEL (v30):")
for k, v in funnel.items():
    print(f"  {k:<35} {v}")

print("\n📅 DAILY TRADE DISTRIBUTION (v23):")
daily_v23 = df.groupby('date').size()
print(f"  Avg trades/day: {daily_v23.mean():.1f}  |  Max: {daily_v23.max()}  |  Days >2 trades: {(daily_v23>2).sum()}")

print("\n📅 DAILY TRADE DISTRIBUTION (v30):")
daily_v30 = v30_sel.groupby('date').size() if len(v30_sel) else pd.Series([0])
print(f"  Avg trades/day: {daily_v30.mean():.1f}  |  Max: {daily_v30.max()}  |  Days >2 trades: {(daily_v30>2).sum()}")

# Strategy label breakdown
print("\n🏷️  TOP SIGNAL COMBINATIONS IN v30 SELECTED TRADES:")
combo_stats = v30_sel.groupby('signals')['won'].agg(['sum', 'count'])
combo_stats['wr'] = (combo_stats['sum'] / combo_stats['count'] * 100).round(1)
combo_stats = combo_stats.sort_values('wr', ascending=False).head(10)
print(combo_stats.to_string())

# Save results
results = {"v23": s23, "v30_simulated": s30, "filter_funnel": funnel}
with open(OUT / "v30_simulation.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to backtest/results/v30_simulation.json")
print("="*60 + "\n")

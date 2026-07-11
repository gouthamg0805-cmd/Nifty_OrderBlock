"""
backtest/param_sweep.py
Grid search over strategy parameters to find the optimal config.
Tests combinations of:
  - min_signal_score: [5, 6, 7, 8]
  - min_rr:           [1.2, 1.5, 2.0]
  - atr_sl_mult:      [0.4, 0.5, 0.7]
  - premium_sl_pct:   [0.08, 0.10, 0.12]

Output: backtest/results/param_sweep.csv
        backtest/results/param_sweep_heatmap.png

Usage: python backtest/param_sweep.py
"""
from __future__ import annotations
import os
import sys
import itertools
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.run_backtest_standalone import generate_nifty_data, run_backtest


def run_sweep(df: pd.DataFrame, param_grid: dict) -> pd.DataFrame:
    keys   = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        print(f"  [{i+1}/{len(combos)}] {params}", end="\r")

        trades, equity, init_cap = run_backtest(df, **params)

        if not trades:
            results.append({**params, "total_trades": 0, "win_rate": 0,
                             "total_pnl": 0, "sharpe": 0, "max_dd": 0, "pf": 0})
            continue

        df_t  = pd.DataFrame(trades)
        pnls  = df_t["pnl"].tolist()
        wins  = df_t[df_t["won"] == True]["pnl"]
        losses = df_t[df_t["won"] == False]["pnl"]

        tp   = sum(pnls)
        wr   = len(wins) / len(pnls) * 100 if pnls else 0
        eq_s = pd.Series([init_cap] + equity)
        rm   = eq_s.cummax()
        mdd  = ((eq_s - rm) / rm * 100).min()
        pf   = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else 99
        df_t["et2"] = pd.to_datetime(df_t["entry_time"])
        dr = df_t.groupby(df_t["et2"].dt.date)["pnl"].sum()
        sharpe = (dr.mean() / dr.std() * math.sqrt(252)) if dr.std() > 0 else 0

        results.append({
            **params,
            "total_trades": len(pnls),
            "win_rate":     round(wr, 1),
            "total_pnl":    round(tp, 0),
            "sharpe":       round(sharpe, 2),
            "max_dd":       round(mdd, 2),
            "pf":           round(pf, 2),
        })

    print()
    return pd.DataFrame(results)


def plot_heatmap(df_results: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#f8f9fa")
    fig.suptitle("Parameter Sweep — Nifty Options MAS",
                 fontsize=16, fontweight="bold", color="#1a237e")

    metrics = ["total_pnl", "win_rate", "sharpe"]
    titles  = ["Total P&L (₹)", "Win Rate (%)", "Sharpe Ratio"]

    for ax, metric, title in zip(axes, metrics, titles):
        pivot = df_results.pivot_table(
            index   = "min_signal_score",
            columns = "min_rr",
            values  = metric,
            aggfunc = "mean",
        )
        sns.heatmap(
            pivot,
            ax          = ax,
            annot       = True,
            fmt         = ".0f" if metric != "win_rate" else ".1f",
            cmap        = "RdYlGn",
            linewidths  = 0.5,
            cbar_kws    = {"shrink": 0.8},
        )
        ax.set_title(title, fontsize=12, fontweight="600", color="#1a237e", pad=8)
        ax.set_xlabel("Min R:R", fontsize=10)
        ax.set_ylabel("Min Signal Score", fontsize=10)
        ax.set_facecolor("#ffffff")

    plt.tight_layout()
    path = os.path.join(output_dir, "param_sweep_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
    plt.close()
    print(f"  Heatmap saved: {path}")


def main():
    print("\n  Generating synthetic data for parameter sweep...")
    df = generate_nifty_data(days=60, start_price=22000)

    param_grid = {
        "min_signal_score": [5, 6, 7, 8],
        "min_rr":           [1.2, 1.5, 2.0],
        "atr_sl_mult":      [0.4, 0.6],
        "premium_sl_pct":   [0.08, 0.12],
    }

    total = 1
    for v in param_grid.values():
        total *= len(v)
    print(f"  Testing {total} parameter combinations...")

    results_df = run_sweep(df, param_grid)

    output_dir = os.path.join(os.path.dirname(__file__), "results")
    csv_path   = os.path.join(output_dir, "param_sweep.csv")
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"  Results saved: {csv_path}")

    # Best config by Sharpe
    best = results_df.sort_values("sharpe", ascending=False).iloc[0]
    print(f"\n  Best config by Sharpe ({best['sharpe']:.2f}):")
    for k in param_grid:
        print(f"    {k}: {best[k]}")
    print(f"    Trades: {best['total_trades']} | WR: {best['win_rate']}% | P&L: ₹{best['total_pnl']:,.0f}")

    # Best config by P&L
    best_pnl = results_df.sort_values("total_pnl", ascending=False).iloc[0]
    print(f"\n  Best config by P&L (₹{best_pnl['total_pnl']:,.0f}):")
    for k in param_grid:
        print(f"    {k}: {best_pnl[k]}")

    try:
        plot_heatmap(results_df, output_dir)
    except Exception as e:
        print(f"  Heatmap skipped: {e}")

    # Save best params to signal_weights.json
    best_params = {
        "min_signal_score": int(best["min_signal_score"]),
        "min_rr_ratio":     float(best["min_rr"]),
        "atr_sl_mult":      float(best["atr_sl_mult"]),
        "premium_sl_pct":   float(best["premium_sl_pct"]),
    }
    best_path = os.path.join(output_dir, "best_params.json")
    with open(best_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\n  Best params saved: {best_path}")
    print("  Copy these into config/signal_weights.json → thresholds section\n")


if __name__ == "__main__":
    main()

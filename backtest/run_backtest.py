"""
backtest/run_backtest.py
Downloads historical Nifty data and runs the backtest engine.
Generates charts and a summary report.

Usage:
    python backtest/run_backtest.py
    python backtest/run_backtest.py --start 2024-01-01 --end 2024-12-31
"""
from __future__ import annotations
import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import yfinance as yf
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.engine import BacktestEngine

# ─── Style ────────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'bg':      '#0f1117',
    'panel':   '#1a1d27',
    'text':    '#e0e0e0',
    'green':   '#26a69a',
    'red':     '#ef5350',
    'blue':    '#42a5f5',
    'amber':   '#ffca28',
    'purple':  '#ab47bc',
    'gray':    '#546e7a',
}


def download_data(start: str, end: str, interval: str = "5m") -> pd.DataFrame:
    logger.info(f"Downloading Nifty data: {start} → {end} ({interval})")

    # yfinance limits 5m to last 60 days; use 1d for longer backtests
    ticker = yf.Ticker("^NSEI")

    if interval == "5m":
        chunks = []
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt   = datetime.strptime(end,   "%Y-%m-%d")
        curr = start_dt

        while curr < end_dt:
            chunk_end = min(curr + timedelta(days=55), end_dt)
            df_chunk  = ticker.history(
                start=curr.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval="5m"
            )
            if not df_chunk.empty:
                chunks.append(df_chunk)
            curr = chunk_end + timedelta(days=1)

        if not chunks:
            raise ValueError("No data downloaded. Try a shorter date range for 5m data.")

        df = pd.concat(chunks)
    else:
        df = ticker.history(start=start, end=end, interval=interval)

    df = df.rename(columns={
        'Open': 'open', 'High': 'high',
        'Low': 'low',   'Close': 'close', 'Volume': 'volume'
    })
    df = df[['open','high','low','close','volume']].dropna()
    df.index = pd.to_datetime(df.index)

    # Filter market hours (9:15 – 15:30 IST)
    if hasattr(df.index[0], 'hour'):
        df = df[
            (df.index.hour > 9) |
            ((df.index.hour == 9) & (df.index.minute >= 15))
        ]
        df = df[
            (df.index.hour < 15) |
            ((df.index.hour == 15) & (df.index.minute <= 30))
        ]

    # Fill zero volume
    if df['volume'].sum() == 0:
        df['volume'] = np.random.randint(100000, 500000, len(df)).astype(float)

    logger.info(f"Data loaded: {len(df)} bars | {df.index[0]} → {df.index[-1]}")
    return df


def plot_results(results: dict, output_dir: str = "backtest/results"):
    os.makedirs(output_dir, exist_ok=True)
    trades_df = results['trades_df']
    equity    = results['equity_curve']

    fig = plt.figure(figsize=(20, 24), facecolor='#f8f9fa')
    fig.suptitle(
        'Nifty Options MAS — Backtest Report',
        fontsize=22, fontweight='bold', color='#1a1d2e', y=0.98
    )

    gs = fig.add_gridspec(4, 3, hspace=0.42, wspace=0.35,
                           left=0.07, right=0.97, top=0.94, bottom=0.04)

    # ── 1. Equity Curve ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#ffffff')
    equity_series = pd.Series([200000] + equity)
    ax1.plot(equity_series.values, color='#1565c0', linewidth=1.8, label='Portfolio Value')
    ax1.fill_between(range(len(equity_series)), 200000, equity_series.values,
                     where=equity_series.values >= 200000, alpha=0.15, color='#1565c0')
    ax1.fill_between(range(len(equity_series)), 200000, equity_series.values,
                     where=equity_series.values < 200000, alpha=0.2, color='#c62828')
    ax1.axhline(200000, color='#546e7a', linewidth=0.8, linestyle='--', label='Initial Capital')
    ax1.set_title('Equity Curve', fontsize=13, fontweight='600', color='#1a1d2e', pad=10)
    ax1.set_ylabel('Portfolio Value (₹)', fontsize=11, color='#37474f')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
    ax1.legend(fontsize=10, loc='upper left')
    ax1.set_xlabel('Bars', fontsize=10, color='#37474f')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── 2. Drawdown ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_facecolor('#ffffff')
    eq_s  = pd.Series(equity_series.values)
    roll_max = eq_s.cummax()
    dd    = (eq_s - roll_max) / roll_max * 100
    ax2.fill_between(range(len(dd)), dd.values, 0, alpha=0.6, color='#c62828')
    ax2.plot(dd.values, color='#b71c1c', linewidth=0.8)
    ax2.set_title('Drawdown (%)', fontsize=13, fontweight='600', color='#1a1d2e', pad=10)
    ax2.set_ylabel('Drawdown %', fontsize=11, color='#37474f')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ── 3. Monthly P&L ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor('#ffffff')
    if not trades_df.empty:
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
        monthly = trades_df.groupby('month')['pnl'].sum()
        colors_bar = ['#1b5e20' if v >= 0 else '#b71c1c' for v in monthly.values]
        bars = ax3.bar([str(m) for m in monthly.index], monthly.values, color=colors_bar,
                       edgecolor='white', linewidth=0.5)
        ax3.axhline(0, color='#546e7a', linewidth=0.8)
        ax3.set_title('Monthly P&L (₹)', fontsize=12, fontweight='600', color='#1a1d2e', pad=8)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ── 4. Win/Loss Distribution ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_facecolor('#ffffff')
    if not trades_df.empty:
        wins_pnl   = trades_df[trades_df['won'] == True]['pnl']
        losses_pnl = trades_df[trades_df['won'] == False]['pnl']
        if len(wins_pnl) > 0:
            ax4.hist(wins_pnl, bins=20, color='#2e7d32', alpha=0.7, label=f'Wins ({len(wins_pnl)})')
        if len(losses_pnl) > 0:
            ax4.hist(losses_pnl, bins=20, color='#c62828', alpha=0.7, label=f'Losses ({len(losses_pnl)})')
        ax4.axvline(0, color='#546e7a', linewidth=1)
        ax4.set_title('P&L Distribution', fontsize=12, fontweight='600', color='#1a1d2e', pad=8)
        ax4.set_xlabel('P&L (₹)', fontsize=10)
        ax4.legend(fontsize=9)
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # ── 5. Exit Reason Breakdown ───────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.set_facecolor('#ffffff')
    if not trades_df.empty:
        reasons = trades_df.groupby('exit_reason')['pnl'].agg(['count','sum'])
        colors_pie = ['#2e7d32', '#c62828', '#1565c0', '#f57f17']
        wedges, texts, autotexts = ax5.pie(
            reasons['count'],
            labels=reasons.index,
            autopct='%1.0f%%',
            colors=colors_pie[:len(reasons)],
            startangle=90,
            textprops={'fontsize': 9},
        )
        ax5.set_title('Exit Reasons', fontsize=12, fontweight='600', color='#1a1d2e', pad=8)

    # ── 6. Key Metrics Card ────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, :])
    ax6.set_facecolor('#1a237e')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')

    metrics = [
        ("Total Trades",    str(results['total_trades'])),
        ("Win Rate",        f"{results['win_rate']}%"),
        ("Total P&L",       f"₹{results['total_pnl']:,.0f}"),
        ("Return",          f"{results['return_pct']}%"),
        ("Avg Win",         f"₹{results['avg_win']:,.0f}"),
        ("Avg Loss",        f"₹{results['avg_loss']:,.0f}"),
        ("Profit Factor",   str(results['profit_factor'])),
        ("Max Drawdown",    f"{results['max_drawdown_pct']:.1f}%"),
        ("Sharpe Ratio",    str(results['sharpe_ratio'])),
        ("Expectancy/tr",   f"₹{results['expectancy']:,.0f}"),
    ]

    for j, (label, value) in enumerate(metrics):
        x = (j % 5) * 0.21 + 0.02
        y = 0.72 if j < 5 else 0.22
        color = '#4fc3f7' if 'Loss' not in label and 'Drawdown' not in label else '#ef9a9a'
        ax6.text(x, y + 0.12, label, fontsize=9, color='#b0bec5',
                 transform=ax6.transAxes, ha='left')
        ax6.text(x, y, value, fontsize=15, fontweight='bold', color=color,
                 transform=ax6.transAxes, ha='left')

    ax6.set_title('Performance Summary', fontsize=13, fontweight='600',
                  color='white', pad=10, loc='left')

    chart_path = os.path.join(output_dir, "backtest_report.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    logger.info(f"Chart saved: {chart_path}")
    return chart_path


def save_csv(results: dict, output_dir: str = "backtest/results"):
    os.makedirs(output_dir, exist_ok=True)
    trades_df = results['trades_df']
    csv_path  = os.path.join(output_dir, "trades.csv")
    trades_df.to_csv(csv_path, index=False)
    logger.info(f"Trades CSV saved: {csv_path}")

    # Summary
    summary = {k: v for k, v in results.items()
               if k not in ('trades_df', 'equity_curve')}
    summary_path = os.path.join(output_dir, "summary.json")
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved: {summary_path}")


def print_report(results: dict):
    sep = "─" * 55
    print(f"\n{'═'*55}")
    print(f"  NIFTY OPTIONS MAS — BACKTEST RESULTS")
    print(f"{'═'*55}")
    print(f"  Total Trades     : {results['total_trades']}")
    print(f"  Winning Trades   : {results['winning_trades']}")
    print(f"  Losing Trades    : {results['losing_trades']}")
    print(f"  Win Rate         : {results['win_rate']}%")
    print(sep)
    print(f"  Total P&L        : ₹{results['total_pnl']:>12,.2f}")
    print(f"  Initial Capital  : ₹{200000:>12,.2f}")
    print(f"  Final Capital    : ₹{results['final_capital']:>12,.2f}")
    print(f"  Return           : {results['return_pct']}%")
    print(sep)
    print(f"  Avg Win          : ₹{results['avg_win']:>12,.2f}")
    print(f"  Avg Loss         : ₹{results['avg_loss']:>12,.2f}")
    print(f"  Profit Factor    : {results['profit_factor']}")
    print(f"  Max Drawdown     : {results['max_drawdown_pct']:.1f}%")
    print(f"  Sharpe Ratio     : {results['sharpe_ratio']}")
    print(f"  Expectancy/trade : ₹{results['expectancy']:>10,.2f}")
    print(f"{'═'*55}\n")


def main():
    parser = argparse.ArgumentParser(description="Nifty MAS Backtest")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--output", default="backtest/results", help="Output directory")
    args = parser.parse_args()

    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=55)   # yfinance 5m limit
    start    = args.start or start_dt.strftime("%Y-%m-%d")
    end      = args.end   or end_dt.strftime("%Y-%m-%d")

    # Download data
    df = download_data(start, end, interval="5m")

    # Run backtest
    config = {
        "lot_size":       65,
        "starting_lots":  2,
        "total_capital":  200000,
        "max_risk":       2000,
        "max_daily_loss": 6000,
        "daily_target":   5000,
        "min_score":      6,
        "min_rr":         1.2,
        "atr_sl_mult":    0.5,
        "premium_sl_pct": 0.10,
    }

    engine  = BacktestEngine(config)
    results = engine.run(df)

    if "error" in results:
        logger.error(f"Backtest failed: {results['error']}")
        return

    # Output
    print_report(results)
    save_csv(results, args.output)
    chart = plot_results(results, args.output)
    print(f"\n  Charts saved to : {args.output}/")
    print(f"  Open the report : open {chart}\n")


if __name__ == "__main__":
    main()

"""
backtest/engine.py
Vectorized backtest engine for Nifty Options strategy.
Uses historical Nifty spot data + synthetic option pricing.
"""
from __future__ import annotations
import math
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from scipy.stats import norm
from loguru import logger

from core.indicators import (
    ema, atr, rsi, vwap, volume_sma,
    detect_engulfing, classify_regime, breakout_signal,
    detect_order_blocks_obv, relative_volume,
)


# ─── Black-Scholes option pricer ─────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, option_type="CE") -> float:
    if T <= 0:
        if option_type == "CE":
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "CE":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(float(price), 0.5)


# ─── Signal scorer (mirrors Agent 2 v31 — Order Block + Volume) ──────────────

_OBV_WEIGHTS = {
    "ob_zone_fresh":            5,
    "ob_zone_retest":           3,
    "ob_impulse_volume":        3.5,
    "live_volume_spike":        3,
    "rvol_rising":              1.5,
    "regime_aligned":           1,
    "volume_climax_exhaustion": -3.5,
    "low_volume_grind":         -2,
    "against_trend":            -4,
    "chop_zone":                -3,
}

OB_PROXIMITY_PTS = 100.0
LIVE_VOL_SPIKE   = 2.0
CLIMAX_RVOL      = 3.5
LOW_RVOL         = 0.6
SL_BUFFER_PTS    = 5.0
MIN_SL_PTS       = 12.0


def compute_signals(df: pd.DataFrame, idx: int) -> Tuple[List[str], float, str, Optional[dict]]:
    """
    Compute Order-Block + Volume signals at bar index `idx`.
    Returns (active_signals, score, bias, trade_ob) where trade_ob is a dict
    with high/low/direction, or None if no valid OB is present.

    This mirrors agents/agent2_strategy.py's v31 logic so backtest results
    are representative of what the live OBV strategy would do.
    """
    if idx < 40:
        return [], 0, "NO_TRADE", None

    window = df.iloc[max(0, idx - 80): idx + 1]
    close  = window['close']
    e9     = ema(close, 9).iloc[-1]
    e21    = ema(close, 21).iloc[-1]
    vw     = vwap(window).iloc[-1]
    rs     = rsi(close, 14).iloc[-1]
    spot   = close.iloc[-1]
    rvol_series = relative_volume(window, 20)
    current_rvol = float(rvol_series.iloc[-1]) if len(rvol_series) else 1.0
    regime_str = classify_regime(window, window)  # single-timeframe approximation

    # Lightweight bias (Agent 1 also folds in 1H structure live; here we keep
    # it simple since this is a single-timeframe backtest approximation).
    bull = bear = 0
    if e9 > e21:   bull += 2
    else:          bear += 2
    if spot > vw:  bull += 2
    else:          bear += 2
    if rs > 55:    bull += 1
    elif rs < 45:  bear += 1

    if bull > bear:
        bias = "LONG_CALL"
    elif bear > bull:
        bias = "LONG_PUT"
    else:
        return [], 0, "NO_TRADE", None

    direction = 'bull' if bias == "LONG_CALL" else 'bear'

    obs = detect_order_blocks_obv(window, lookback=60, vol_multiplier=1.5)
    candidates = []
    for ob in obs:
        if ob.direction != direction or not ob.volume_confirmed:
            continue
        if direction == 'bull' and ob.high >= spot - OB_PROXIMITY_PTS and ob.low <= spot + 5:
            candidates.append(ob)
        elif direction == 'bear' and ob.low <= spot + OB_PROXIMITY_PTS and ob.high >= spot - 5:
            candidates.append(ob)

    if not candidates:
        return [], 0, "NO_TRADE", None

    trade_ob = min(candidates, key=lambda o: abs(spot - (o.high + o.low) / 2))

    active = []
    active.append("ob_zone_fresh" if not trade_ob.mitigated else "ob_zone_retest")
    if trade_ob.volume_confirmed:
        active.append("ob_impulse_volume")
    if current_rvol >= LIVE_VOL_SPIKE:
        active.append("live_volume_spike")

    hist = rvol_series.tail(3).tolist()
    if len(hist) == 3 and hist[-1] > hist[-2] > hist[-3]:
        active.append("rvol_rising")

    last_row = window.iloc[-1]
    candle_up   = last_row['close'] > last_row['open']
    candle_down = last_row['close'] < last_row['open']
    if current_rvol >= CLIMAX_RVOL:
        if (bias == "LONG_CALL" and candle_down) or (bias == "LONG_PUT" and candle_up):
            active.append("volume_climax_exhaustion")
    if current_rvol < LOW_RVOL:
        active.append("low_volume_grind")

    if (bias == "LONG_CALL" and regime_str == "TRENDING_BULL") or \
       (bias == "LONG_PUT" and regime_str == "TRENDING_BEAR"):
        active.append("regime_aligned")
    if (bias == "LONG_CALL" and regime_str == "TRENDING_BEAR") or \
       (bias == "LONG_PUT" and regime_str == "TRENDING_BULL"):
        active.append("against_trend")
    if regime_str == "VOLATILE":
        active.append("chop_zone")

    # Mandatory gate: must have volume confirmation, not just OB location
    has_volume = "ob_impulse_volume" in active or "live_volume_spike" in active
    if not has_volume:
        return [], 0, "NO_TRADE", None

    score = sum(_OBV_WEIGHTS.get(s, 0) for s in active)
    ob_info = dict(high=float(trade_ob.high), low=float(trade_ob.low), direction=direction)
    return active, float(score), bias, ob_info


# ─── Backtest Engine ──────────────────────────────────────────────────────────

class BacktestEngine:

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.lot_size       = self.config.get("lot_size", 65)
        self.starting_lots  = self.config.get("starting_lots", 2)
        self.total_capital  = self.config.get("total_capital", 200000)
        self.max_risk       = self.config.get("max_risk", 2000)
        self.max_daily_loss = self.config.get("max_daily_loss", 6000)
        self.daily_target   = self.config.get("daily_target", 5000)
        self.min_score      = self.config.get("min_score", 6)
        self.min_rr         = self.config.get("min_rr", 1.2)
        self.atr_sl_mult    = self.config.get("atr_sl_mult", 0.5)
        self.premium_sl_pct = self.config.get("premium_sl_pct", 0.10)
        self.r              = 0.065    # risk-free rate
        self.base_iv        = 0.18     # assumed IV
        self.trades: List[dict] = []
        self.equity_curve: List[float] = []

    def run(self, df: pd.DataFrame) -> dict:
        """
        Run full backtest on 5-minute OHLCV DataFrame.
        Returns results dictionary with metrics and trade log.
        """
        logger.info(f"Backtest starting: {len(df)} bars | {df.index[0]} → {df.index[-1]}")

        capital      = self.total_capital
        daily_pnl    = 0.0
        current_date = None
        in_trade     = False
        trade        = {}
        cooldown     = 0   # bars since last trade

        for i in range(30, len(df)):
            bar       = df.iloc[i]
            bar_time  = df.index[i]
            bar_date  = bar_time.date() if hasattr(bar_time, 'date') else bar_time

            # Daily reset
            if bar_date != current_date:
                current_date = bar_date
                daily_pnl    = 0.0

            # Skip early bars (before 9:20) and late bars (after 15:00)
            if hasattr(bar_time, 'hour'):
                if bar_time.hour == 9 and bar_time.minute < 20:
                    continue
                if (bar_time.hour == 15 and bar_time.minute >= 20) or bar_time.hour > 15:
                    # Square off if in trade
                    if in_trade:
                        exit_spot = bar['close']
                        exit_premium = self._option_price(
                            exit_spot, trade['strike'], trade['T_remaining'],
                            trade['option_type']
                        )
                        pnl = (exit_premium - trade['entry_premium']) * trade['quantity']
                        self._record_trade(trade, exit_premium, pnl, "SQUAREOFF", bar_time)
                        daily_pnl += pnl
                        capital   += pnl
                        in_trade   = False
                    continue

            # Daily loss limit
            if daily_pnl < -self.max_daily_loss:
                if in_trade:
                    exit_spot    = bar['close']
                    exit_premium = self._option_price(exit_spot, trade['strike'],
                                                      trade['T_remaining'], trade['option_type'])
                    pnl = (exit_premium - trade['entry_premium']) * trade['quantity']
                    self._record_trade(trade, exit_premium, pnl, "DAILY_LOSS_LIMIT", bar_time)
                    daily_pnl += pnl
                    capital   += pnl
                    in_trade   = False
                continue

            # Manage open trade
            if in_trade:
                spot         = bar['close']
                T_remaining  = max(0, trade['T_remaining'] - (5 / (252 * 375)))
                trade['T_remaining'] = T_remaining
                current_premium = self._option_price(
                    spot, trade['strike'], T_remaining, trade['option_type']
                )

                # Trailing SL update (ATR-based simplified)
                atr_now = atr(df.iloc[max(0, i-20):i+1]).iloc[-1]
                new_sl  = current_premium - atr_now * self.atr_sl_mult
                if new_sl > trade['current_sl'] and current_premium > trade['entry_premium']:
                    trade['current_sl'] = max(new_sl, trade['entry_premium'])  # at least break-even

                pnl = (current_premium - trade['entry_premium']) * trade['quantity']

                # SL hit
                if current_premium <= trade['current_sl']:
                    self._record_trade(trade, trade['current_sl'], 
                                       (trade['current_sl'] - trade['entry_premium']) * trade['quantity'],
                                       "SL_HIT", bar_time)
                    daily_pnl += (trade['current_sl'] - trade['entry_premium']) * trade['quantity']
                    capital   += (trade['current_sl'] - trade['entry_premium']) * trade['quantity']
                    in_trade   = False
                    cooldown   = 10  # 10-bar cooldown
                    continue

                # Target hit
                if current_premium >= trade['target_premium']:
                    self._record_trade(trade, trade['target_premium'],
                                       (trade['target_premium'] - trade['entry_premium']) * trade['quantity'],
                                       "TARGET_HIT", bar_time)
                    daily_pnl += (trade['target_premium'] - trade['entry_premium']) * trade['quantity']
                    capital   += (trade['target_premium'] - trade['entry_premium']) * trade['quantity']
                    in_trade   = False
                    cooldown   = 6
                    continue

            else:
                # Cooldown
                if cooldown > 0:
                    cooldown -= 1
                    self.equity_curve.append(capital)
                    continue

                # Signal evaluation
                signals, score, bias, trade_ob = compute_signals(df, i)

                if bias == "NO_TRADE" or score < self.min_score:
                    self.equity_curve.append(capital)
                    continue

                spot      = bar['close']
                atr_now   = atr(df.iloc[max(0, i-20):i+1]).iloc[-1]
                opt_type  = "CE" if bias == "LONG_CALL" else "PE"
                strike    = self._select_strike(spot, opt_type)
                T_entry   = 7 / 365  # ~1 week to expiry (weekly options)

                entry_premium = self._option_price(spot, strike, T_entry, opt_type)

                # ── OB-based stop distance on the underlying, converted to
                # an approximate premium-space risk via a ~0.5 ATM delta.
                # (A full options-Greeks backtest is out of scope here; this
                # keeps the backtest's SL genuinely derived from the Order
                # Block rather than an arbitrary ATR multiple, matching the
                # live Agent 2 logic.)
                if trade_ob is not None:
                    if trade_ob["direction"] == 'bull':
                        ob_sl_spot_pts = max(spot - (trade_ob["low"] - SL_BUFFER_PTS), MIN_SL_PTS)
                    else:
                        ob_sl_spot_pts = max((trade_ob["high"] + SL_BUFFER_PTS) - spot, MIN_SL_PTS)
                    approx_delta = 0.5
                    sl_pts = max(ob_sl_spot_pts * approx_delta, entry_premium * self.premium_sl_pct, 2.0)
                else:
                    sl_pts = max(entry_premium * self.premium_sl_pct, atr_now * self.atr_sl_mult, 2.0)

                sl_premium    = entry_premium - sl_pts
                tgt_premium   = entry_premium + sl_pts * 2
                rr            = (tgt_premium - entry_premium) / sl_pts

                if rr < self.min_rr:
                    self.equity_curve.append(capital)
                    continue

                # Position size
                risk_per_lot = sl_pts * self.lot_size
                lots         = min(self.starting_lots, max(1, int(self.max_risk / risk_per_lot)))
                quantity     = lots * self.lot_size

                in_trade = True
                trade = {
                    "entry_time":     bar_time,
                    "bias":           bias,
                    "strike":         strike,
                    "option_type":    opt_type,
                    "entry_spot":     spot,
                    "entry_premium":  entry_premium,
                    "current_sl":     sl_premium,
                    "initial_sl":     sl_premium,
                    "target_premium": tgt_premium,
                    "sl_pts":         sl_pts,
                    "lots":           lots,
                    "quantity":       quantity,
                    "T_remaining":    T_entry,
                    "signals":        signals,
                    "score":          score,
                    "rr":             rr,
                }

            self.equity_curve.append(capital)

        return self._compute_results(capital)

    def _option_price(self, spot, strike, T, opt_type) -> float:
        iv = self.base_iv * (1 + abs(spot - strike) / spot * 2)  # slight skew
        return bs_price(spot, strike, T, self.r, iv, opt_type)

    def _select_strike(self, spot: float, opt_type: str) -> float:
        atm = round(spot / 50) * 50
        if opt_type == "CE":
            return atm      # ATM call
        else:
            return atm      # ATM put

    def _record_trade(self, trade: dict, exit_premium: float, pnl: float,
                       reason: str, exit_time):
        self.trades.append({
            "entry_time":    trade["entry_time"],
            "exit_time":     exit_time,
            "bias":          trade["bias"],
            "strike":        trade["strike"],
            "option_type":   trade["option_type"],
            "entry_premium": round(trade["entry_premium"], 2),
            "exit_premium":  round(exit_premium, 2),
            "sl_pts":        round(trade["sl_pts"], 2),
            "lots":          trade["lots"],
            "quantity":      trade["quantity"],
            "pnl":           round(pnl, 2),
            "exit_reason":   reason,
            "won":           pnl > 0,
            "signal_score":  trade["score"],
            "signals":       ", ".join(trade["signals"]),
            "rr":            round(trade["rr"], 2),
        })

    def _compute_results(self, final_capital: float) -> dict:
        if not self.trades:
            return {"error": "No trades generated"}

        df_trades = pd.DataFrame(self.trades)
        pnls      = df_trades['pnl'].tolist()
        wins      = df_trades[df_trades['won'] == True]
        losses    = df_trades[df_trades['won'] == False]

        total_pnl    = sum(pnls)
        win_rate     = len(wins) / len(pnls) * 100
        avg_win      = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss     = losses['pnl'].mean() if len(losses) > 0 else 0
        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if losses['pnl'].sum() != 0 else float('inf')

        # Max drawdown
        equity = pd.Series([self.total_capital] + self.equity_curve)
        rolling_max = equity.cummax()
        drawdown    = (equity - rolling_max) / rolling_max * 100
        max_dd      = drawdown.min()

        # Sharpe (daily returns)
        df_trades['entry_date'] = pd.to_datetime(df_trades['entry_time']).dt.date
        daily_pnl = df_trades.groupby('entry_date')['pnl'].sum()
        sharpe    = (daily_pnl.mean() / daily_pnl.std() * math.sqrt(252)
                     if daily_pnl.std() > 0 else 0)

        # Expectancy
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

        return {
            "total_trades":    len(pnls),
            "winning_trades":  len(wins),
            "losing_trades":   len(losses),
            "win_rate":        round(win_rate, 1),
            "total_pnl":       round(total_pnl, 2),
            "final_capital":   round(final_capital, 2),
            "return_pct":      round((final_capital - self.total_capital) / self.total_capital * 100, 2),
            "avg_win":         round(avg_win, 2),
            "avg_loss":        round(avg_loss, 2),
            "profit_factor":   round(profit_factor, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio":    round(sharpe, 2),
            "expectancy":      round(expectancy, 2),
            "trades_df":       df_trades,
            "equity_curve":    self.equity_curve,
        }

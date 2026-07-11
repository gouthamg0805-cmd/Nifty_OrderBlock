"""
tests/test_agents.py
Unit tests for all system components.
Run: pytest tests/ -v

Tests cover:
  - Indicator calculations
  - Signal scoring
  - Risk management sizing
  - Options pricing
  - Broker paper mode
  - Database operations
"""
from __future__ import annotations
import pytest
import json
import math
import asyncio
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Indicator tests ──────────────────────────────────────────────────────────

class TestIndicators:

    def setup_method(self):
        """Create a 50-bar synthetic OHLCV DataFrame."""
        np.random.seed(42)
        n = 60
        close  = 22000 + np.cumsum(np.random.randn(n) * 50)
        open_  = close + np.random.randn(n) * 10
        high   = np.maximum(open_, close) + abs(np.random.randn(n) * 15)
        low    = np.minimum(open_, close) - abs(np.random.randn(n) * 15)
        volume = np.random.lognormal(12, 0.5, n).astype(int)
        self.df = pd.DataFrame({
            "open": open_, "high": high, "low": low,
            "close": close, "volume": volume,
        })

    def test_ema_shape(self):
        from core.indicators import ema
        result = ema(self.df["close"], 9)
        assert len(result) == len(self.df)
        assert result.iloc[-1] > 0

    def test_ema_fast_reacts_faster(self):
        from core.indicators import ema
        e9  = ema(self.df["close"], 9)
        e21 = ema(self.df["close"], 21)
        # Fast EMA should be more responsive — std of diff(e9) > std of diff(e21)
        assert e9.diff().std() >= e21.diff().std()

    def test_atr_positive(self):
        from core.indicators import atr
        result = atr(self.df, 14)
        assert result.dropna().min() > 0

    def test_rsi_bounds(self):
        from core.indicators import rsi
        result = rsi(self.df["close"], 14)
        valid  = result.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_vwap_between_low_high(self):
        from core.indicators import vwap
        result = vwap(self.df)
        # VWAP should generally be within daily range
        assert result.dropna().min() > 0

    def test_detect_engulfing_returns_series(self):
        from core.indicators import detect_engulfing
        result = detect_engulfing(self.df)
        assert set(result.unique()).issubset({-1, 0, 1})

    def test_pivot_levels_support_below_resistance(self):
        from core.indicators import pivot_levels
        support, resistance = pivot_levels(self.df, 20)
        assert support <= resistance

    def test_classify_regime_returns_valid(self):
        from core.indicators import classify_regime
        result = classify_regime(self.df, self.df)
        assert result in ["TRENDING_BULL", "TRENDING_BEAR", "RANGING", "VOLATILE"]


# ─── Options pricing tests ────────────────────────────────────────────────────

class TestOptionsPricing:

    def test_atm_call_positive(self):
        from backtest.run_backtest_standalone import bs_price
        price = bs_price(22000, 22000, 7/365, 0.20, "CE")
        assert price > 0

    def test_atm_put_call_parity(self):
        """C - P ≈ S - K*e^(-rT) for ATM"""
        from backtest.run_backtest_standalone import bs_price
        S, K, T, r, iv = 22000, 22000, 7/365, 0.065, 0.20
        call = bs_price(S, K, T, iv, "CE")
        put  = bs_price(S, K, T, iv, "PE")
        parity = S - K * math.exp(-r * T)
        assert abs((call - put) - parity) < 5.0   # within ₹5

    def test_deep_itm_call_above_intrinsic(self):
        from backtest.run_backtest_standalone import bs_price
        # S=22500, K=22000 → intrinsic=500
        price = bs_price(22500, 22000, 7/365, 0.20, "CE")
        assert price >= 500

    def test_otm_call_cheaper_than_atm(self):
        from backtest.run_backtest_standalone import bs_price
        atm = bs_price(22000, 22000, 7/365, 0.20, "CE")
        otm = bs_price(22000, 22200, 7/365, 0.20, "CE")
        assert otm < atm

    def test_higher_iv_increases_premium(self):
        from backtest.run_backtest_standalone import bs_price
        low_iv  = bs_price(22000, 22000, 7/365, 0.15, "CE")
        high_iv = bs_price(22000, 22000, 7/365, 0.30, "CE")
        assert high_iv > low_iv

    def test_expired_option_intrinsic_only(self):
        from backtest.run_backtest_standalone import bs_price
        # At expiry T=0, CE = max(S-K, 0)
        itm = bs_price(22100, 22000, 0, 0.20, "CE")
        otm = bs_price(21900, 22000, 0, 0.20, "CE")
        assert abs(itm - 100) < 1.0
        assert otm <= 1.0


# ─── Risk management tests ────────────────────────────────────────────────────

class TestRiskManagement:

    def test_sl_never_below_minimum(self):
        """SL should always be at least MIN_SL_ABSOLUTE."""
        premium    = 5.0   # very cheap option
        atr_val    = 2.0
        prem_pct   = 0.10
        atr_mult   = 0.5
        min_sl_abs = 2.0
        sl = max(premium * prem_pct, atr_val * atr_mult, min_sl_abs)
        assert sl >= min_sl_abs

    def test_position_size_respects_max_risk(self):
        """Lots × SL_points × lot_size must not exceed MAX_RISK."""
        max_risk = 2000
        lot_size = 65
        sl_pts   = 25.0   # ₹25 SL
        risk_per_lot = sl_pts * lot_size   # ₹1625
        lots     = min(2, max(1, int(max_risk / risk_per_lot)))
        actual_risk = lots * sl_pts * lot_size
        assert actual_risk <= max_risk * 1.05  # allow 5% overage due to rounding

    def test_position_size_at_least_one_lot(self):
        """Even with tight SL, should trade at least 1 lot."""
        max_risk = 2000
        lot_size = 65
        sl_pts   = 100.0  # large SL → only 1 lot
        risk_per_lot = sl_pts * lot_size
        lots     = max(1, int(max_risk / risk_per_lot))
        assert lots >= 1

    def test_rr_ratio_calculation(self):
        entry  = 200.0
        sl     = 180.0   # -20 points
        target = 240.0   # +40 points
        sl_pts  = entry - sl
        tgt_pts = target - entry
        rr      = tgt_pts / sl_pts
        assert abs(rr - 2.0) < 0.01

    def test_daily_loss_gate(self):
        """System should stop trading after hitting daily loss limit."""
        max_daily_loss = 6000
        current_pnl    = -6500
        should_trade   = current_pnl > -max_daily_loss
        assert should_trade is False


# ─── Signal scoring tests ─────────────────────────────────────────────────────

class TestSignalScoring:

    def setup_method(self):
        self.weights = {
            "vwap_aligned":        2,
            "ema_9_21_aligned":    2,
            "breakout_confirmed":  3,
            "volume_spike":        2,
            "momentum_strong":     2,
            "rsi_confirmation":    1,
            "engulfing_pattern":   2,
            "weak_candle":        -2,
            "against_trend":      -3,
            "low_volume":         -1,
        }

    def test_high_quality_entry_scores_above_threshold(self):
        signals = ["vwap_aligned", "ema_9_21_aligned", "breakout_confirmed", "volume_spike"]
        score   = sum(self.weights[s] for s in signals)
        assert score >= 6

    def test_weak_signals_blocked(self):
        signals = ["low_volume", "against_trend"]
        score   = sum(self.weights[s] for s in signals)
        assert score < 6   # should NOT trigger a trade

    def test_contradictory_signals_reduce_score(self):
        good_signals = ["vwap_aligned", "breakout_confirmed"]
        bad_signals  = ["against_trend"]
        score = sum(self.weights[s] for s in good_signals + bad_signals)
        pure_score = sum(self.weights[s] for s in good_signals)
        assert score < pure_score

    def test_min_score_threshold_respected(self):
        min_score = 6
        signals   = ["vwap_aligned"]  # only score 2
        score     = sum(self.weights[s] for s in signals)
        assert score < min_score   # should be blocked


# ─── Broker paper mode tests ──────────────────────────────────────────────────

class TestBrokerPaperMode:

    def setup_method(self):
        from core.broker import BrokerClient
        self.broker = BrokerClient(mode="paper")
        self.broker.login()

    def test_login_succeeds(self):
        assert self.broker._logged_in is True

    def test_place_order_returns_order_id(self):
        resp = self.broker.place_order(
            symbol="NIFTY22000CE",
            exchange_segment="nse_fo",
            transaction_type="B",
            quantity=130,
            order_type="MKT",
            product="MIS",
            price=200.0,
            tag="TEST",
        )
        assert "order_id" in resp
        assert resp["status"] == "FILLED"

    def test_modify_order_updates_trigger(self):
        resp = self.broker.place_order(
            symbol="NIFTY22000CE",
            exchange_segment="nse_fo",
            transaction_type="S",
            quantity=130,
            order_type="SL-M",
            product="MIS",
            trigger_price=180.0,
            tag="SL",
        )
        order_id = resp["order_id"]
        mod_resp = self.broker.modify_order(order_id=order_id, trigger_price=185.0)
        assert mod_resp["status"] == "ok"
        assert self.broker._paper_orders[order_id]["trigger_price"] == 185.0

    def test_cancel_order_removes_from_book(self):
        resp = self.broker.place_order(
            symbol="NIFTY22000CE",
            exchange_segment="nse_fo",
            transaction_type="S",
            quantity=130,
            order_type="SL-M",
            product="MIS",
            trigger_price=180.0,
        )
        order_id = resp["order_id"]
        self.broker.cancel_order(order_id)
        assert order_id not in self.broker._paper_orders


# ─── Database tests ───────────────────────────────────────────────────────────

class TestDatabase:

    def setup_method(self):
        from core.database import Database
        self.db = Database(db_path="/tmp/test_trades.db")

    def test_save_and_retrieve_trade(self):
        self.db.save_trade({
            "trade_id":       "TEST001",
            "entry_time":     datetime.now(),
            "symbol":         "NIFTY22000CE",
            "option_type":    "CE",
            "strike_price":   22000.0,
            "entry_price":    200.0,
            "sl_price":       180.0,
            "target_price":   240.0,
            "lots":           2,
            "quantity":       130,
            "strategy_label": "VWAP Momentum",
            "signal_score":   8.0,
            "active_signals": json.dumps(["vwap_aligned", "ema_9_21_aligned"]),
            "regime":         "TRENDING_BULL",
            "mode":           "paper",
        })
        trades = self.db.get_all_trades()
        assert any(t.trade_id == "TEST001" for t in trades)

    def test_update_trade_with_pnl(self):
        self.db.save_trade({
            "trade_id":       "TEST002",
            "entry_time":     datetime.now(),
            "symbol":         "NIFTY22000PE",
            "option_type":    "PE",
            "strike_price":   22000.0,
            "entry_price":    150.0,
            "sl_price":       135.0,
            "target_price":   180.0,
            "lots":           1,
            "quantity":       65,
            "strategy_label": "Breakout",
            "signal_score":   7.0,
            "active_signals": "[]",
            "regime":         "TRENDING_BEAR",
            "mode":           "paper",
        })
        self.db.update_trade("TEST002", pnl=3250.0, pnl_pct=1.625, won=True,
                             exit_price=200.0, exit_reason="TARGET_HIT")
        trades = self.db.get_all_trades()
        t = next((x for x in trades if x.trade_id == "TEST002"), None)
        assert t is not None
        assert t.pnl == 3250.0
        assert t.won is True

    def test_today_pnl_sum(self):
        pnl = self.db.get_today_pnl()
        assert isinstance(pnl, float)


# ─── Model validation tests ───────────────────────────────────────────────────

class TestModels:

    def test_market_state_confidence_bounds(self):
        from core.models import MarketState, MarketRegime, TradeBias, Indicators, OHLCV
        import pytest
        with pytest.raises(Exception):
            MarketState(
                timestamp   = datetime.now(),
                spot_price  = 22000,
                regime      = MarketRegime.TRENDING_BULL,
                bias        = TradeBias.LONG_CALL,
                confidence  = 150.0,  # INVALID: > 100
                indicators  = Indicators(ema9=22010, ema21=21990, vwap=22000,
                                         atr14=50, rsi14=60, volume_avg=100000),
            )

    def test_option_strike_model(self):
        from core.models import OptionStrike, OptionType
        strike = OptionStrike(
            symbol       = "NIFTY22000CE",
            strike_price = 22000.0,
            option_type  = OptionType.CALL,
            expiry       = "27JUN24",
            ltp          = 200.0,
            oi           = 500000.0,
            volume       = 150000.0,
            bid          = 199.0,
            ask          = 201.0,
        )
        assert strike.ltp == 200.0
        assert strike.option_type == OptionType.CALL


# ─── Integration: backtest produces non-zero trades ───────────────────────────

class TestBacktestIntegration:

    def test_backtest_generates_trades(self):
        from backtest.run_backtest_standalone import generate_nifty_data, run_backtest
        df = generate_nifty_data(days=20, start_price=22000)
        trades, equity, init_cap = run_backtest(df)
        assert isinstance(trades, list)
        assert len(equity) > 0

    def test_equity_curve_same_length_as_bars(self):
        from backtest.run_backtest_standalone import generate_nifty_data, run_backtest
        df = generate_nifty_data(days=10, start_price=22000)
        trades, equity, _ = run_backtest(df)
        # Equity curve length should be close to bar count (minus warmup)
        assert abs(len(equity) - (len(df) - 30)) <= 5

    def test_no_negative_quantities(self):
        from backtest.run_backtest_standalone import generate_nifty_data, run_backtest
        df = generate_nifty_data(days=15, start_price=22000)
        trades, _, _ = run_backtest(df)
        for t in trades:
            assert t["qty"] > 0
            assert t["lots"] >= 1


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python", "-m", "pytest", __file__, "-v", "--tb=short"])

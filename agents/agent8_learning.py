"""
agents/agent8_learning.py
Agent 8: Weekly Learning & Adaptation System

Tracks trade performance over a rolling 7-day window and:
  1. Analyses which signals genuinely predict wins
  2. Identifies what time-of-day / regime wins most
  3. Adjusts signal weights in config/signal_weights.json
  4. Writes a plain-English daily report: logs/learning_report.md
  5. Sends a summary to the dashboard event feed

Run automatically: after market close (15:35 IST) every trading day
Also callable manually: python -c "from agents.agent8_learning import run_learning; run_learning()"
"""
from __future__ import annotations
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, date
from pathlib import Path
from loguru import logger

from core.database import Database

WEIGHTS_FILE = Path("config/signal_weights.json")
REPORT_FILE  = Path("logs/learning_report.md")


class LearningAgent:

    MIN_TRADES_TO_ADAPT  = 8    # need at least 8 trades before changing weights
    MAX_WEIGHT_CHANGE    = 0.5  # max change per signal per day (avoid overfit)
    MAX_POSITIVE_WEIGHT  = 5.0
    MAX_NEGATIVE_WEIGHT  = -5.0
    LOOKBACK_DAYS        = 7    # rolling 7-day window

    def __init__(self, db: Database, strategy_agent=None):
        self.db             = db
        self.strategy_agent = strategy_agent   # reference to reload weights after update

    def run_daily_update(self):
        """Call this after market close. Analyses last 7 days and updates everything."""
        logger.info("[Agent8] Running daily learning update...")

        trades = self._get_recent_trades(self.LOOKBACK_DAYS)
        closed = [t for t in trades if t.exit_price is not None and t.pnl is not None]

        if len(closed) < self.MIN_TRADES_TO_ADAPT:
            logger.info(
                f"[Agent8] Only {len(closed)} closed trades in last {self.LOOKBACK_DAYS} days. "
                f"Need {self.MIN_TRADES_TO_ADAPT} to adapt. Generating report only."
            )
            self._write_report(closed, weights_changed=False)
            return

        # Analyse
        analysis = self._analyse_trades(closed)

        # Update weights
        weights_changed = self._update_weights(analysis)

        # Write human-readable report
        self._write_report(closed, analysis=analysis, weights_changed=weights_changed)

        # Tell strategy agent to reload
        if self.strategy_agent and hasattr(self.strategy_agent, "reload_weights"):
            self.strategy_agent.reload_weights()

        logger.info("[Agent8] Daily learning update complete. Report: logs/learning_report.md")

    # ─── Data retrieval ───────────────────────────────────────────────────────

    def _get_recent_trades(self, days: int) -> list:
        all_trades = self.db.get_all_trades()
        cutoff     = datetime.now() - timedelta(days=days)
        return [
            t for t in all_trades
            if t.entry_time and t.entry_time >= cutoff
        ]

    # ─── Analysis ─────────────────────────────────────────────────────────────

    def _analyse_trades(self, trades: list) -> dict:
        """Deep analysis of what made trades win or lose."""

        signal_wins   = defaultdict(int)
        signal_losses = defaultdict(int)
        signal_pnl    = defaultdict(float)

        time_wins     = defaultdict(int)    # hour → wins
        time_losses   = defaultdict(int)
        time_pnl      = defaultdict(float)

        regime_wins   = defaultdict(int)
        regime_losses = defaultdict(int)

        strategy_wins  = defaultdict(int)
        strategy_losses = defaultdict(int)
        strategy_pnl   = defaultdict(float)

        win_pnls  = []
        loss_pnls = []

        for trade in trades:
            won  = trade.won or False
            pnl  = trade.pnl or 0.0
            hour = trade.entry_time.hour if trade.entry_time else 0

            if won: win_pnls.append(pnl)
            else:   loss_pnls.append(pnl)

            # Signal breakdown
            try:
                signals = json.loads(trade.active_signals or "[]")
            except Exception:
                signals = []
            for sig in signals:
                if won:
                    signal_wins[sig]   += 1
                    signal_pnl[sig]    += pnl
                else:
                    signal_losses[sig] += 1
                    signal_pnl[sig]    += pnl

            # Time breakdown (by hour)
            if won:
                time_wins[hour]   += 1
                time_pnl[hour]    += pnl
            else:
                time_losses[hour] += 1
                time_pnl[hour]    += pnl

            # Regime breakdown
            regime = trade.regime or "UNKNOWN"
            if won: regime_wins[regime]   += 1
            else:   regime_losses[regime] += 1

            # Strategy breakdown
            strat = trade.strategy_label or "Unknown"
            if won:
                strategy_wins[strat]  += 1
                strategy_pnl[strat]   += pnl
            else:
                strategy_losses[strat] += 1
                strategy_pnl[strat]   += pnl

        # Signal win rates
        signal_stats = {}
        all_signals = set(list(signal_wins.keys()) + list(signal_losses.keys()))
        for sig in all_signals:
            w = signal_wins[sig]; l = signal_losses[sig]; total = w + l
            if total >= 3:   # minimum sample size
                signal_stats[sig] = {
                    "wins": w, "losses": l, "total": total,
                    "win_rate": w / total,
                    "avg_pnl": signal_pnl[sig] / total,
                }

        # Time win rates
        time_stats = {}
        for hour in set(list(time_wins.keys()) + list(time_losses.keys())):
            w = time_wins[hour]; l = time_losses[hour]; total = w + l
            time_stats[hour] = {
                "wins": w, "losses": l, "win_rate": w/total if total else 0,
                "avg_pnl": time_pnl[hour]/total if total else 0,
            }

        return {
            "total":           len(trades),
            "winning":         len(win_pnls),
            "losing":          len(loss_pnls),
            "win_rate":        len(win_pnls) / len(trades) if trades else 0,
            "total_pnl":       sum(t.pnl or 0 for t in trades),
            "avg_win":         sum(win_pnls) / len(win_pnls) if win_pnls else 0,
            "avg_loss":        sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0,
            "expectancy":      (
                (len(win_pnls)/len(trades)) * (sum(win_pnls)/len(win_pnls) if win_pnls else 0) +
                (len(loss_pnls)/len(trades)) * (sum(loss_pnls)/len(loss_pnls) if loss_pnls else 0)
            ) if trades else 0,
            "signal_stats":    signal_stats,
            "time_stats":      time_stats,
            "regime_wins":     dict(regime_wins),
            "regime_losses":   dict(regime_losses),
            "strategy_wins":   dict(strategy_wins),
            "strategy_losses": dict(strategy_losses),
            "strategy_pnl":    dict(strategy_pnl),
        }

    # ─── Weight updates ───────────────────────────────────────────────────────

    def _update_weights(self, analysis: dict) -> bool:
        """Adjust signal weights based on what actually worked."""
        try:
            with open(WEIGHTS_FILE) as f:
                config = json.load(f)
        except Exception:
            logger.error(f"[Agent8] Cannot read {WEIGHTS_FILE}")
            return False

        weights  = config.get("weights", {})
        changed  = []

        for sig, stats in analysis["signal_stats"].items():
            if sig not in weights:
                continue
            current_w = weights[sig]
            win_rate  = stats["win_rate"]
            total     = stats["total"]

            # Only adjust with enough data
            if total < 5:
                continue

            # Direction of change
            if win_rate >= 0.60:
                # Signal is working well → strengthen it
                delta = min(self.MAX_WEIGHT_CHANGE, (win_rate - 0.50) * 1.0)
                new_w = min(self.MAX_POSITIVE_WEIGHT, current_w + delta)
            elif win_rate <= 0.35:
                # Signal is hurting → weaken it
                delta = min(self.MAX_WEIGHT_CHANGE, (0.50 - win_rate) * 1.0)
                new_w = max(self.MAX_NEGATIVE_WEIGHT, current_w - delta)
            else:
                continue   # 35–60% → no change, not enough evidence

            if abs(new_w - current_w) > 0.05:
                weights[sig] = round(new_w, 2)
                changed.append(
                    f"  {sig}: {current_w:.1f} → {new_w:.1f} "
                    f"(WR:{win_rate*100:.0f}%, n={total})"
                )

        if changed:
            config["weights"]      = weights
            config["last_updated"] = datetime.now().isoformat()
            config["trade_stats"] = {
                "total_trades":   analysis["total"],
                "winning_trades": analysis["winning"],
                "losing_trades":  analysis["losing"],
                "win_rate":       round(analysis["win_rate"], 3),
            }
            with open(WEIGHTS_FILE, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"[Agent8] Updated {len(changed)} signal weights:\n" + "\n".join(changed))
            return True

        logger.info("[Agent8] No weight changes needed (all signals within acceptable range)")
        return False

    # ─── Report generation ────────────────────────────────────────────────────

    def _write_report(self, trades: list, analysis: dict = None, weights_changed: bool = False):
        """Write a plain-English markdown report that a human can read and act on."""
        now   = datetime.now()
        lines = []

        lines += [
            f"# Nifty Options MAS — Learning Report",
            f"**Generated:** {now.strftime('%d %b %Y, %H:%M IST')}",
            f"**Period:** Last {self.LOOKBACK_DAYS} days",
            "",
        ]

        if not trades:
            lines += ["No trades in this period.", ""]
            REPORT_FILE.parent.mkdir(exist_ok=True)
            REPORT_FILE.write_text("\n".join(lines))
            return

        total = len(trades)
        closed = [t for t in trades if t.pnl is not None]
        wins   = [t for t in closed if t.won]
        losses = [t for t in closed if not t.won]
        total_pnl = sum(t.pnl or 0 for t in closed)
        wr    = len(wins)/len(closed)*100 if closed else 0

        # ── Summary ───────────────────────────────────────────────────────
        lines += [
            "## 📊 Performance Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total trades | {total} |",
            f"| Winning | {len(wins)} ({wr:.1f}%) |",
            f"| Losing | {len(losses)} ({100-wr:.1f}%) |",
            f"| Total P&L | ₹{total_pnl:,.0f} |",
            f"| Avg win | ₹{(sum(t.pnl for t in wins)/len(wins)):,.0f} |" if wins else "| Avg win | — |",
            f"| Avg loss | ₹{(sum(t.pnl for t in losses)/len(losses)):,.0f} |" if losses else "| Avg loss | — |",
            "",
        ]

        if analysis:
            exp = analysis.get("expectancy", 0)
            lines += [
                f"**Expectancy per trade:** ₹{exp:,.0f}  ",
                f"*(Target: ₹{5000//2:,} = half daily goal per trade)*",
                "",
            ]

            # ── What worked ───────────────────────────────────────────────
            lines += ["## ✅ What Worked (High Win-Rate Signals)", ""]
            good = [(s, v) for s, v in analysis["signal_stats"].items()
                    if v["win_rate"] >= 0.55 and v["total"] >= 3]
            good.sort(key=lambda x: x[1]["win_rate"], reverse=True)
            if good:
                lines.append("| Signal | Win Rate | Trades | Avg P&L |")
                lines.append("|--------|----------|--------|---------|")
                for sig, v in good:
                    lines.append(
                        f"| `{sig}` | {v['win_rate']*100:.0f}% | {v['total']} | "
                        f"₹{v['avg_pnl']:,.0f} |"
                    )
            else:
                lines.append("*Not enough data yet. Keep trading!*")
            lines.append("")

            # ── What didn't work ──────────────────────────────────────────
            lines += ["## ❌ What Hurt (Low Win-Rate Signals)", ""]
            bad = [(s, v) for s, v in analysis["signal_stats"].items()
                   if v["win_rate"] < 0.40 and v["total"] >= 3]
            bad.sort(key=lambda x: x[1]["win_rate"])
            if bad:
                lines.append("| Signal | Win Rate | Trades | Avg P&L |")
                lines.append("|--------|----------|--------|---------|")
                for sig, v in bad:
                    lines.append(
                        f"| `{sig}` | {v['win_rate']*100:.0f}% | {v['total']} | "
                        f"₹{v['avg_pnl']:,.0f} |"
                    )
            else:
                lines.append("*No consistently bad signals detected.*")
            lines.append("")

            # ── Best trading hours ────────────────────────────────────────
            lines += ["## ⏰ Best Trading Hours", ""]
            time_stats = analysis.get("time_stats", {})
            if time_stats:
                sorted_hours = sorted(
                    [(h, v) for h, v in time_stats.items() if (v["wins"]+v["losses"]) >= 2],
                    key=lambda x: x[1]["win_rate"], reverse=True
                )
                lines.append("| Hour (IST) | Win Rate | Trades | Avg P&L |")
                lines.append("|------------|----------|--------|---------|")
                for h, v in sorted_hours:
                    total_h = v["wins"] + v["losses"]
                    lines.append(
                        f"| {h:02d}:00–{h+1:02d}:00 | "
                        f"{v['win_rate']*100:.0f}% | {total_h} | "
                        f"₹{v['avg_pnl']:,.0f} |"
                    )
            lines.append("")

            # ── Best strategies ───────────────────────────────────────────
            lines += ["## 🎯 Strategy Performance", ""]
            strat_wins  = analysis.get("strategy_wins", {})
            strat_loss  = analysis.get("strategy_losses", {})
            strat_pnl   = analysis.get("strategy_pnl", {})
            all_strats  = set(list(strat_wins.keys()) + list(strat_loss.keys()))
            if all_strats:
                lines.append("| Strategy | Win Rate | Trades | Total P&L |")
                lines.append("|----------|----------|--------|-----------|")
                for s in sorted(all_strats):
                    w = strat_wins.get(s, 0); l = strat_loss.get(s, 0); n = w + l
                    wr_s = w/n if n else 0
                    lines.append(
                        f"| {s} | {wr_s*100:.0f}% | {n} | "
                        f"₹{strat_pnl.get(s,0):,.0f} |"
                    )
            lines.append("")

            # ── Regime performance ────────────────────────────────────────
            lines += ["## 🌊 Performance by Market Regime", ""]
            rw = analysis.get("regime_wins", {})
            rl = analysis.get("regime_losses", {})
            all_regimes = set(list(rw.keys()) + list(rl.keys()))
            if all_regimes:
                lines.append("| Regime | Wins | Losses | Win Rate |")
                lines.append("|--------|------|--------|----------|")
                for r in sorted(all_regimes):
                    w = rw.get(r,0); l = rl.get(r,0); n = w + l
                    lines.append(f"| {r} | {w} | {l} | {w/n*100:.0f}% |" if n else f"| {r} | 0 | 0 | — |")
            lines.append("")

        # ── Action items ──────────────────────────────────────────────────
        lines += ["## 🔧 Automatic Adaptations Applied", ""]
        if weights_changed:
            lines += [
                "Signal weights have been automatically updated in `config/signal_weights.json`.",
                "Strategy agent has reloaded the new weights.",
                "Changes take effect on the next trading session.",
                "",
            ]
        else:
            lines += ["No weight changes applied (insufficient data or signals within range).", ""]

        # ── Actionable advice ─────────────────────────────────────────────
        lines += ["## 💡 Observations & Recommendations", ""]
        advice = self._generate_advice(trades, analysis)
        for item in advice:
            lines.append(f"- {item}")
        lines.append("")

        lines += [
            "---",
            f"*Report auto-generated by Agent 8. Next update: after market close tomorrow.*",
        ]

        REPORT_FILE.parent.mkdir(exist_ok=True)
        REPORT_FILE.write_text("\n".join(lines))
        logger.info(f"[Agent8] Report written: {REPORT_FILE}")

    def _generate_advice(self, trades: list, analysis: dict = None) -> list:
        advice = []
        if not analysis or not trades:
            advice.append("Not enough trade data yet. Run the bot for at least a week before expecting reliable insights.")
            return advice

        wr = analysis.get("win_rate", 0)
        avg_win  = analysis.get("avg_win", 0)
        avg_loss = analysis.get("avg_loss", 0)
        exp      = analysis.get("expectancy", 0)

        # Win rate advice
        if wr < 0.35:
            advice.append(
                f"Win rate {wr*100:.0f}% is below target. Primary cause is usually: "
                f"trading in ranging/volatile markets. Check that the RANGING regime filter "
                f"is active and the min_score threshold is at least 9."
            )
        elif wr >= 0.50:
            advice.append(f"Win rate {wr*100:.0f}% is good! Focus on increasing avg win size via trailing SL.")

        # Expectancy
        if exp < 0:
            advice.append(
                f"Negative expectancy (₹{exp:,.0f}/trade) means losses outweigh wins. "
                f"Either tighten entry criteria (raise min_score) or widen SL to avoid premature stops."
            )
        elif exp > 1000:
            advice.append(f"Strong expectancy ₹{exp:,.0f}/trade. System is working — don't over-optimise.")

        # RR advice
        if avg_win and avg_loss and abs(avg_loss) > 0:
            actual_rr = avg_win / abs(avg_loss)
            if actual_rr < 1.5:
                advice.append(
                    f"Actual R:R is {actual_rr:.2f}:1 — below the 1.5:1 target. "
                    f"Possible causes: (1) taking profits too early, or (2) targets set too tight. "
                    f"Let the trailing SL run longer."
                )

        # Time advice
        time_stats = analysis.get("time_stats", {})
        if time_stats:
            best_hour  = max(time_stats.items(), key=lambda x: x[1]["win_rate"], default=(None,{}))[0]
            worst_hour = min(time_stats.items(), key=lambda x: x[1]["win_rate"], default=(None,{}))[0]
            if best_hour:
                advice.append(f"Best trading hour: {best_hour:02d}:00–{best_hour+1:02d}:00 IST. Try to be more active here.")
            if worst_hour and worst_hour != best_hour:
                advice.append(f"Worst hour: {worst_hour:02d}:00 IST. Consider skipping this window.")

        # Regime advice
        rw = analysis.get("regime_wins", {})
        rl = analysis.get("regime_losses", {})
        for regime in ["RANGING", "VOLATILE"]:
            r_losses = rl.get(regime, 0)
            r_wins   = rw.get(regime, 0)
            if r_losses > r_wins and r_losses >= 3:
                advice.append(
                    f"{regime} regime trades: {r_wins}W / {r_losses}L — "
                    f"system should already filter these out. Check that regime gate is active."
                )

        if not advice:
            advice.append("System appears to be performing within expected parameters. Keep running.")

        return advice


# ── Convenience entry point ────────────────────────────────────────────────────

def run_learning():
    """Quick manual run: python -c 'from agents.agent8_learning import run_learning; run_learning()'"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.database import Database
    db = Database()
    agent = LearningAgent(db)
    agent.run_daily_update()
    print(f"\nReport written to: {REPORT_FILE}")
    print(REPORT_FILE.read_text())

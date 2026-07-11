"""
backtest/run_backtest_standalone.py
Standalone backtest — no external deps except pandas, numpy, scipy, matplotlib.
Generates synthetic Nifty 5m data and runs the full strategy simulation.
Also used by param_sweep.py via run_backtest(**kwargs).
"""
from __future__ import annotations
import os, sys, math, json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm


# ─── Indicators ───────────────────────────────────────────────────────────────

def ema(s, p): return s.ewm(span=p, adjust=False).mean()

def atr_calc(df, p=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def rsi_calc(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(span=p, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=p, adjust=False).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def vwap_calc(df):
    tp = (df['high'] + df['low'] + df['close']) / 3
    return (tp * df['volume']).cumsum() / df['volume'].cumsum()

def bs_price(S, K, T, sigma=0.20, opt="CE"):
    r = 0.065
    if T <= 0:
        return max(S - K, 0) if opt == "CE" else max(K - S, 0)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    p  = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2) if opt=="CE" \
         else K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return max(float(p), 1.0)


# ─── Order Block detection (mirrors core/indicators.detect_order_blocks_obv) ──

def detect_order_blocks(win, lookback=60, vol_mult=1.5):
    s = win.tail(lookback).reset_index(drop=True)
    n = len(s)
    if n < 10:
        return []
    avg_range = (s['high'] - s['low']).mean()
    vol_avg_series = s['volume'].rolling(20, min_periods=5).mean()
    blocks = []
    for i in range(1, n - 1):
        curr, prev = s.iloc[i], s.iloc[i - 1]
        va = vol_avg_series.iloc[i]
        if pd.isna(va) or va <= 0:
            continue
        rvol = curr['volume'] / va
        strong_range = (curr['high'] - curr['low']) > avg_range * 1.5
        strong_vol   = rvol >= vol_mult
        if prev['close'] < prev['open'] and curr['close'] > curr['open'] and strong_range and strong_vol:
            blocks.append(dict(direction='bull', high=prev['high'], low=prev['low'],
                                bar_index=i - 1, impulse_rvol=rvol, confirmed=True))
        elif prev['close'] > prev['open'] and curr['close'] < curr['open'] and strong_range and strong_vol:
            blocks.append(dict(direction='bear', high=prev['high'], low=prev['low'],
                                bar_index=i - 1, impulse_rvol=rvol, confirmed=True))
    # mitigation check
    for b in blocks:
        mitigated = False
        for j in range(b['bar_index'] + 2, n):
            row = s.iloc[j]
            if row['low'] <= b['high'] and row['high'] >= b['low']:
                mitigated = True
                break
        b['mitigated'] = mitigated
    return blocks[-5:]


# ─── Signal scorer — v31 Order Block + Volume (OBV) ──────────────────────────

WEIGHTS = {
    "ob_zone_fresh": 5, "ob_zone_retest": 3, "ob_impulse_volume": 3.5,
    "live_volume_spike": 3, "rvol_rising": 1.5, "regime_aligned": 1,
    "volume_climax_exhaustion": -3.5, "low_volume_grind": -2,
    "against_trend": -4, "chop_zone": -3,
}

OB_PROXIMITY_PTS = 100.0
LIVE_VOL_SPIKE   = 2.0
CLIMAX_RVOL      = 3.5
LOW_RVOL         = 0.6

def score_signals(df, i):
    """
    Order Block + Volume signal scorer — mirrors agents/agent2_strategy.py
    (v31). Returns (active_signals, score, bias, trade_ob | None).
    """
    if i < 40: return [], 0, "NO_TRADE", None
    win = df.iloc[max(0, i-80): i+1]; cl = win['close']
    e9  = ema(cl, 9).iloc[-1];  e21 = ema(cl, 21).iloc[-1]
    vw  = vwap_calc(win).iloc[-1]; rs = rsi_calc(cl, 14).iloc[-1]
    sp  = cl.iloc[-1]
    vol_avg_series = win['volume'].rolling(20).mean()
    rvol_series = (win['volume'] / vol_avg_series.replace(0, np.nan)).fillna(0.0)
    current_rvol = float(rvol_series.iloc[-1])

    bull = bear = 0
    if e9 > e21: bull += 2
    else:        bear += 2
    if sp > vw:  bull += 2
    else:        bear += 2
    if rs > 55:  bull += 1
    elif rs < 45: bear += 1
    bias = "LONG_CALL" if bull > bear else ("LONG_PUT" if bear > bull else None)
    if not bias: return [], 0, "NO_TRADE", None

    direction = 'bull' if bias == "LONG_CALL" else 'bear'
    obs = detect_order_blocks(win, lookback=60, vol_mult=1.5)
    candidates = [
        ob for ob in obs
        if ob['direction'] == direction and ob['confirmed'] and (
            (direction == 'bull' and ob['high'] >= sp - OB_PROXIMITY_PTS and ob['low'] <= sp + 5) or
            (direction == 'bear' and ob['low']  <= sp + OB_PROXIMITY_PTS and ob['high'] >= sp - 5)
        )
    ]
    if not candidates:
        return [], 0, "NO_TRADE", None
    trade_ob = min(candidates, key=lambda o: abs(sp - (o['high'] + o['low']) / 2))

    a = []
    a.append("ob_zone_fresh" if not trade_ob['mitigated'] else "ob_zone_retest")
    a.append("ob_impulse_volume")  # candidates are pre-filtered to confirmed==True
    if current_rvol >= LIVE_VOL_SPIKE:
        a.append("live_volume_spike")
    hist = rvol_series.tail(3).tolist()
    if len(hist) == 3 and hist[-1] > hist[-2] > hist[-3]:
        a.append("rvol_rising")

    last_up   = df['close'].iloc[i] > df['open'].iloc[i]
    last_down = df['close'].iloc[i] < df['open'].iloc[i]
    if current_rvol >= CLIMAX_RVOL:
        if (bias == "LONG_CALL" and last_down) or (bias == "LONG_PUT" and last_up):
            a.append("volume_climax_exhaustion")
    if current_rvol < LOW_RVOL:
        a.append("low_volume_grind")

    if (bias == "LONG_CALL" and bull >= bear + 4) or (bias == "LONG_PUT" and bear >= bull + 4):
        a.append("regime_aligned")

    has_volume = "ob_impulse_volume" in a or "live_volume_spike" in a
    if not has_volume:
        return [], 0, "NO_TRADE", None

    return a, float(sum(WEIGHTS.get(x, 0) for x in a)), bias, trade_ob


# ─── Synthetic data ───────────────────────────────────────────────────────────

def generate_nifty_data(days=60, start_price=22000):
    np.random.seed(13); bpd = 75; total = days * bpd
    rets = []; sv = 0.014
    for i in range(total):
        if i % 200 == 0: sv = float(np.random.choice([0.010, 0.013, 0.018, 0.024]))
        dt = 1/(252*bpd); r = (0.12/252-0.5*sv**2)*dt + sv*np.sqrt(dt)*float(np.random.randn())
        if i > 5 and abs(rets[-1]) > 0.0025: r *= -0.4
        rets.append(r)
    prices = start_price * np.exp(np.cumsum(rets))
    rows = []; base = datetime(2024, 10, 1, 9, 15); dc = 0
    for i in range(total):
        b = i % bpd
        if b == 0:
            dc += 1
            while (base + timedelta(days=dc-1)).weekday() >= 5: dc += 1
        d   = base + timedelta(days=dc-1)
        dt_ = datetime(d.year, d.month, d.day, 9+(b*5)//60, (b*5+15)%60)
        c = prices[i]; n = c*0.0008
        o = c+float(np.random.uniform(-n,n)); h = max(o,c)+abs(float(np.random.normal(0,n)))
        l = min(o,c)-abs(float(np.random.normal(0,n))); v = int(np.random.lognormal(12,0.5))
        rows.append({'datetime':dt_,'open':round(o,2),'high':round(h,2),'low':round(l,2),'close':round(c,2),'volume':v})
    return pd.DataFrame(rows).set_index('datetime')


# ─── Backtest engine ──────────────────────────────────────────────────────────

def run_backtest(
    df,
    min_signal_score=6, min_rr=1.2, atr_sl_mult=0.5, premium_sl_pct=0.10,
    lot_size=65, starting_lots=2, total_capital=200000,
    max_risk=2000, max_daily_loss=6000, delta=0.52,
):
    CAP=total_capital; LOT=lot_size; MR=max_risk; MDL=max_daily_loss; SLIP=0.005
    cap=CAP; trades=[]; equity=[]; dpnl={}; it=False; tr={}; cool=0
    for i in range(30, len(df)):
        bar=df.iloc[i]; bdt=df.index[i]; bd=bdt.date(); h,m=bdt.hour,bdt.minute
        if bd not in dpnl: dpnl[bd]=0.0
        if (h==9 and m<20) or h<9: equity.append(cap); continue
        if h>15 or (h==15 and m>=20):
            if it:
                spm=bar['close']-tr['esp']
                ep=(tr['ep']+spm*delta) if tr['opt']=="CE" else (tr['ep']-spm*delta)
                ep=max(ep,1.0)*(1-SLIP); pnl=(ep-tr['ep'])*tr['qty']
                trades.append({**tr,'exit_prem':round(ep,2),'pnl':round(pnl,2),'exit_reason':'SQUAREOFF','won':pnl>0,'exit_time':bdt})
                dpnl[bd]+=pnl; cap+=pnl; it=False
            equity.append(cap); continue
        if dpnl.get(bd,0)<-MDL: equity.append(cap); continue
        if it:
            spm=bar['close']-tr['esp']
            cp=(tr['ep']+spm*delta) if tr['opt']=="CE" else (tr['ep']-spm*delta)
            cp=max(cp,0.5); av=atr_calc(df.iloc[max(0,i-20):i+1]).iloc[-1]
            nsl=cp-av*atr_sl_mult
            if nsl>tr['sl'] and cp>tr['ep']: tr['sl']=max(nsl,tr['ep']*0.995)
            if cp<=tr['sl']:
                pnl=(tr['sl']-tr['ep'])*tr['qty']
                trades.append({**tr,'exit_prem':round(tr['sl'],2),'pnl':round(pnl,2),'exit_reason':'SL_HIT','won':pnl>0,'exit_time':bdt})
                dpnl[bd]+=pnl; cap+=pnl; it=False; cool=10
            elif cp>=tr['tgt']:
                pnl=(tr['tgt']-tr['ep'])*tr['qty']
                trades.append({**tr,'exit_prem':round(tr['tgt'],2),'pnl':round(pnl,2),'exit_reason':'TARGET_HIT','won':pnl>0,'exit_time':bdt})
                dpnl[bd]+=pnl; cap+=pnl; it=False; cool=8
        else:
            if cool>0: cool-=1; equity.append(cap); continue
            a,sc,bias,trade_ob=score_signals(df,i)
            if bias=="NO_TRADE" or sc<min_signal_score: equity.append(cap); continue
            sp=bar['close']; av=atr_calc(df.iloc[max(0,i-20):i+1]).iloc[-1]
            opt="CE" if bias=="LONG_CALL" else "PE"; K=round(sp/50)*50; T=7/365
            ep=bs_price(sp,K,T,sigma=0.20,opt=opt)*(1+SLIP)
            # OB-based SL: stop distance derives from the traded Order Block's
            # far edge (underlying points), converted to premium space via an
            # approximate ATM delta — matches the live Agent 2 v31 logic.
            if trade_ob is not None:
                if trade_ob['direction'] == 'bull':
                    ob_sl_spot_pts = max(sp - (trade_ob['low'] - 5.0), 12.0)
                else:
                    ob_sl_spot_pts = max((trade_ob['high'] + 5.0) - sp, 12.0)
                slp = max(ob_sl_spot_pts * delta, ep*premium_sl_pct, 2.0)
            else:
                slp=max(ep*premium_sl_pct,av*atr_sl_mult,2.0)
            sl=ep-slp; tgt=ep+slp*2
            rr=(tgt-ep)/slp if slp>0 else 0
            if rr<min_rr: equity.append(cap); continue
            rpl=slp*LOT; lots=min(starting_lots,max(1,int(MR/rpl))); qty=lots*LOT
            it=True
            tr={'entry_time':bdt,'bias':bias,'strike':K,'opt':opt,'entry_spot':sp,'ep':round(ep,2),
                'sl':sl,'tgt':tgt,'slp':slp,'lots':lots,'qty':qty,'T':T,'esp':sp,'signals':a,'score':sc,'rr':round(rr,2)}
        equity.append(cap)
    return trades, equity, CAP


# ─── Chart + report ───────────────────────────────────────────────────────────

def generate_report(trades, equity, init_cap, output_dir="backtest/results"):
    os.makedirs(output_dir, exist_ok=True)
    if not trades: print("  No trades generated."); return None, {}
    df_t=pd.DataFrame(trades); pnls=df_t['pnl'].tolist()
    wins=df_t[df_t['won']==True]; losses=df_t[df_t['won']==False]
    tp=sum(pnls); wr=len(wins)/len(pnls)*100 if pnls else 0
    aw=wins['pnl'].mean() if len(wins)>0 else 0
    al=losses['pnl'].mean() if len(losses)>0 else 0
    pf=abs(wins['pnl'].sum()/losses['pnl'].sum()) if losses['pnl'].sum()!=0 else 99
    eq_s=pd.Series([init_cap]+equity); rm=eq_s.cummax(); dd=(eq_s-rm)/rm*100; mdd=dd.min()
    fc=init_cap+tp; rp=(fc-init_cap)/init_cap*100
    df_t['et2']=pd.to_datetime(df_t['entry_time']); dr=df_t.groupby(df_t['et2'].dt.date)['pnl'].sum()
    sharpe=(dr.mean()/dr.std()*math.sqrt(252)) if dr.std()>0 else 0
    exp=(wr/100*aw)+((1-wr/100)*al)
    summary={"total_trades":len(pnls),"winning_trades":int(len(wins)),"losing_trades":int(len(losses)),
             "win_rate_pct":round(wr,1),"total_pnl":round(tp,2),"final_capital":round(fc,2),
             "return_pct":round(rp,2),"avg_win":round(aw,2),"avg_loss":round(al,2),
             "profit_factor":round(pf,2),"max_drawdown_pct":round(mdd,2),
             "sharpe_ratio":round(sharpe,2),"expectancy":round(exp,2)}

    fig=plt.figure(figsize=(22,26),facecolor='#f4f6f8')
    fig.suptitle('Nifty Options MAS — Backtest Report\n60-Day Simulation | Delta P&L | 0.5% Spread | ₹2,00,000 | 2 Lots×65',
                 fontsize=17,fontweight='bold',color='#1a237e',y=0.98)
    gs=gridspec.GridSpec(4,3,figure=fig,hspace=0.45,wspace=0.38,left=0.06,right=0.97,top=0.93,bottom=0.04)
    ax1=fig.add_subplot(gs[0,:]); ax1.set_facecolor('#ffffff'); x=range(len(eq_s))
    ax1.plot(x,eq_s.values,color='#1565c0',linewidth=2,zorder=3,label='Portfolio Value')
    ax1.fill_between(x,init_cap,eq_s.values,where=eq_s.values>=init_cap,alpha=0.12,color='#1565c0')
    ax1.fill_between(x,init_cap,eq_s.values,where=eq_s.values<init_cap,alpha=0.2,color='#c62828')
    ax1.axhline(init_cap,color='#78909c',linewidth=1,linestyle='--',label=f'Initial ₹{init_cap:,.0f}')
    ax1.axhline(fc,color='#2e7d32',linewidth=1.2,linestyle=':',alpha=0.9,label=f'Final ₹{fc:,.0f} ({rp:+.1f}%)')
    ax1.set_ylabel('Portfolio (₹)',fontsize=11); ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f'₹{v:,.0f}'))
    ax1.set_title('Equity Curve',fontsize=13,fontweight='600',color='#1a237e',pad=10)
    ax1.legend(fontsize=10,loc='upper left'); ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax2=fig.add_subplot(gs[1,:]); ax2.set_facecolor('#ffffff')
    ax2.fill_between(range(len(dd)),dd.values,0,alpha=0.65,color='#c62828')
    ax2.plot(dd.values,color='#b71c1c',linewidth=0.8)
    ax2.set_ylabel('Drawdown %',fontsize=11); ax2.set_title('Drawdown',fontsize=13,fontweight='600',color='#1a237e',pad=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f'{v:.1f}%'))
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax3=fig.add_subplot(gs[2,0]); ax3.set_facecolor('#ffffff')
    df_t['month']=df_t['et2'].dt.to_period('M'); monthly=df_t.groupby('month')['pnl'].sum()
    ax3.bar([str(m) for m in monthly.index],monthly.values,
            color=['#1b5e20' if v>=0 else '#b71c1c' for v in monthly.values],edgecolor='white',linewidth=0.5)
    ax3.axhline(0,color='#546e7a',linewidth=0.8)
    ax3.set_title('Monthly P&L (₹)',fontsize=12,fontweight='600',color='#1a237e',pad=8)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f'₹{v:,.0f}'))
    plt.setp(ax3.xaxis.get_majorticklabels(),rotation=40,ha='right',fontsize=9)
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
    ax4=fig.add_subplot(gs[2,1]); ax4.set_facecolor('#ffffff')
    if len(wins)>0: ax4.hist(wins['pnl'],bins=22,color='#2e7d32',alpha=0.75,label=f'Wins ({len(wins)})')
    if len(losses)>0: ax4.hist(losses['pnl'],bins=22,color='#c62828',alpha=0.75,label=f'Losses ({len(losses)})')
    ax4.axvline(0,color='#37474f',linewidth=1.2); ax4.set_title('P&L Distribution',fontsize=12,fontweight='600',color='#1a237e',pad=8)
    ax4.legend(fontsize=9); ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f'₹{v:,.0f}'))
    ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)
    ax5=fig.add_subplot(gs[2,2]); ax5.set_facecolor('#ffffff')
    r2=df_t['exit_reason'].value_counts()
    ax5.pie(r2.values,labels=r2.index,autopct='%1.0f%%',
            colors=['#1b5e20','#c62828','#1565c0','#f57f17'][:len(r2)],startangle=90,textprops={'fontsize':9})
    ax5.set_title('Exit Reasons',fontsize=12,fontweight='600',color='#1a237e',pad=8)
    ax6=fig.add_subplot(gs[3,:]); ax6.set_facecolor('#1a237e'); ax6.axis('off')
    mets=[("Total Trades",f"{len(pnls)}"),("Win Rate",f"{wr:.1f}%"),("Total P&L",f"₹{tp:,.0f}"),
          ("Return",f"{rp:.1f}%"),("Avg Win",f"₹{aw:,.0f}"),("Avg Loss",f"₹{al:,.0f}"),
          ("Profit Factor",f"{pf:.2f}"),("Max Drawdown",f"{mdd:.1f}%"),
          ("Sharpe Ratio",f"{sharpe:.2f}"),("Expectancy",f"₹{exp:,.0f}")]
    for j,(lb,vl) in enumerate(mets):
        x=(j%5)*0.205+0.01; y=0.68 if j<5 else 0.18
        ax6.text(x,y+0.15,lb,fontsize=9.5,color='#b0bec5',transform=ax6.transAxes,ha='left',va='bottom')
        ax6.text(x,y,vl,fontsize=17,fontweight='bold',color='#80deea',transform=ax6.transAxes,ha='left',va='bottom')
    ax6.set_title('Performance Summary',fontsize=13,fontweight='600',color='white',pad=10,loc='left')

    chart=os.path.join(output_dir,'backtest_report.png')
    plt.savefig(chart,dpi=150,bbox_inches='tight',facecolor='#f4f6f8'); plt.close()
    df_t.to_csv(os.path.join(output_dir,'trades.csv'),index=False)
    with open(os.path.join(output_dir,'summary.json'),'w') as f: json.dump(summary,f,indent=2)
    return chart, summary


def print_report(summary):
    sep="─"*58
    print(f"\n{'═'*58}\n  NIFTY OPTIONS MAS — BACKTEST RESULTS\n{'═'*58}")
    print(f"  Total Trades       : {summary['total_trades']}")
    print(f"  Winning Trades     : {summary['winning_trades']}")
    print(f"  Losing Trades      : {summary['losing_trades']}")
    print(f"  Win Rate           : {summary['win_rate_pct']}%")
    print(sep)
    print(f"  Total P&L          : ₹{summary['total_pnl']:>12,.0f}")
    print(f"  Final Capital      : ₹{summary['final_capital']:>12,.0f}")
    print(f"  Return             : {summary['return_pct']:.1f}%")
    print(sep)
    print(f"  Avg Win/Trade      : ₹{summary['avg_win']:>12,.0f}")
    print(f"  Avg Loss/Trade     : ₹{summary['avg_loss']:>12,.0f}")
    print(f"  Profit Factor      : {summary['profit_factor']}")
    print(f"  Max Drawdown       : {summary['max_drawdown_pct']:.1f}%")
    print(f"  Sharpe Ratio       : {summary['sharpe_ratio']}")
    print(f"  Expectancy/Trade   : ₹{summary['expectancy']:>10,.0f}")
    print(f"{'═'*58}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--output", type=str, default="backtest/results")
    args = parser.parse_args()
    print(f"\n  Generating {args.days}-day synthetic Nifty 5m data...")
    df = generate_nifty_data(days=args.days)
    print(f"  {len(df)} bars | {df.index[0].date()} → {df.index[-1].date()}")
    print("  Running backtest...")
    trades, equity, init_cap = run_backtest(df)
    chart, summary = generate_report(trades, equity, init_cap, args.output)
    if summary: print_report(summary)
    if chart:   print(f"  Chart: {chart}\n")

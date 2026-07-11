"""
dashboard/app.py
Plotly Dash live monitoring dashboard.
Run: python dashboard/app.py
Open: http://localhost:8050
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
import pandas as pd
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.database import Database

db  = Database()
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="Nifty MAS Dashboard",
    update_title=None,
)

# ─── Shared state ────────────────────────────────────────────────────────────
# State is exchanged via logs/state.json so dashboard and engine can run as
# separate processes (started by run.py).

STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "logs", "state.json")

_state = {
    "market":        {},
    "active_trades": {},
    "events":        [],
}

def _load_state():
    """Read live state from JSON file written by the trading engine."""
    global _state
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                data = json.load(f)
            _state["market"]        = data.get("market", {})
            _state["active_trades"] = data.get("active_trades", {})
            _state["events"]        = data.get("events", [])
    except Exception:
        pass   # keep last good state on file read error

def update_state(market_data: dict, active_trades: dict, events: list):
    """Called by trading engine when running in-process."""
    _state["market"]        = market_data
    _state["active_trades"] = active_trades
    _state["events"]        = events


# ─── Layout ──────────────────────────────────────────────────────────────────

def metric_card(title, value_id, value="—", color="white"):
    return dbc.Card([
        dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize": "12px"}),
            html.H4(value, id=value_id, style={"color": color, "fontWeight": "600"}),
        ])
    ], className="mb-2", style={"background": "#1a1d2e", "border": "1px solid #2d3561"})


app.layout = dbc.Container(fluid=True, style={"background": "#0d0f1a", "minHeight": "100vh"}, children=[

    dcc.Interval(id="refresh", interval=5000, n_intervals=0),

    # Header
    dbc.Row([
        dbc.Col(html.H2("🎯 Nifty Options MAS", style={"color": "#e0e0e0", "marginTop": "16px"}), width=6),
        dbc.Col([
            html.Div(id="market-status", style={"color": "#69f0ae", "fontSize": "13px", "marginTop": "24px", "textAlign": "right"})
        ], width=6),
    ], className="mb-3"),

    # Session expired alert
    dbc.Row([
        dbc.Col(html.Div(id="session-alert"), width=12),
    ], className="mb-2"),

    # Metric cards row
    dbc.Row([
        dbc.Col(metric_card("Today P&L",     "today-pnl",     color="#69f0ae"), width=2),
        dbc.Col(metric_card("Today Trades",  "today-trades",  color="#40c4ff"), width=2),
        dbc.Col(metric_card("Win Rate",      "win-rate",      color="#ffd740"), width=2),
        dbc.Col(metric_card("Active Trades", "active-trades", color="#ff6e40"), width=2),
        dbc.Col(metric_card("Nifty Spot",    "nifty-spot",    color="#e0e0e0"), width=2),
        dbc.Col(metric_card("Regime",        "regime",        color="#b388ff"), width=2),
    ], className="mb-3"),

    # Chart + trade table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Nifty 5m Chart + Indicators",
                               style={"background": "#1a1d2e", "color": "#90caf9", "fontSize": "13px"}),
                dbc.CardBody([
                    dcc.Graph(id="main-chart", style={"height": "450px"},
                              config={"displayModeBar": False}),
                ], style={"padding": "8px"}),
            ], style={"background": "#1a1d2e", "border": "1px solid #2d3561"}),
        ], width=8),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Active Position",
                               style={"background": "#1a1d2e", "color": "#90caf9", "fontSize": "13px"}),
                dbc.CardBody([html.Div(id="active-position")],
                             style={"padding": "8px"}),
            ], className="mb-3", style={"background": "#1a1d2e", "border": "1px solid #2d3561"}),

            dbc.Card([
                dbc.CardHeader("Signal Feed",
                               style={"background": "#1a1d2e", "color": "#90caf9", "fontSize": "13px"}),
                dbc.CardBody([html.Div(id="event-feed", style={"height": "300px", "overflowY": "auto"})],
                             style={"padding": "8px"}),
            ], style={"background": "#1a1d2e", "border": "1px solid #2d3561"}),
        ], width=4),
    ], className="mb-3"),

    # Trade history table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Trade History (Today)",
                               style={"background": "#1a1d2e", "color": "#90caf9", "fontSize": "13px"}),
                dbc.CardBody([html.Div(id="trade-table")], style={"padding": "8px"}),
            ], style={"background": "#1a1d2e", "border": "1px solid #2d3561"}),
        ])
    ]),
])


# ─── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output("session-alert", "children"),
    Input("refresh", "n_intervals"),
)
def check_session(_):
    _load_state()
    expired = _state.get("session_expired", False)
    if expired:
        return dbc.Alert([
            html.Strong("SESSION EXPIRED — Trading Halted"),
            html.Br(),
            html.Span("Your Kotak Neo session has expired. Re-login to resume trading."),
            html.Br(),
            html.A("Open Login UI at http://localhost:8051",
                   href="http://localhost:8051", target="_blank",
                   style={"color": "white", "fontWeight": "600"}),
        ], color="danger", dismissable=False, style={"fontSize": "13px"})
    return html.Div()


@app.callback(
    [
        Output("today-pnl",     "children"),
        Output("today-trades",  "children"),
        Output("win-rate",      "children"),
        Output("active-trades", "children"),
        Output("nifty-spot",    "children"),
        Output("regime",        "children"),
        Output("market-status", "children"),
        Output("main-chart",    "figure"),
        Output("trade-table",   "children"),
        Output("active-position", "children"),
        Output("event-feed",    "children"),
    ],
    Input("refresh", "n_intervals"),
)
def refresh_dashboard(_):
    _load_state()   # pull latest state from logs/state.json
    market = _state.get("market", {})
    active = _state.get("active_trades", {})
    events = _state.get("events", [])

    # DB stats
    today_pnl    = db.get_today_pnl()
    today_trades = db.get_today_trades()
    wins         = sum(1 for t in today_trades if t.won)
    win_rate     = round(wins / len(today_trades) * 100, 1) if today_trades else 0

    pnl_color    = "#69f0ae" if today_pnl >= 0 else "#ff5252"

    # Market status bar
    ts      = market.get("timestamp", "—")
    bias    = market.get("bias", "—")
    conf    = market.get("confidence", 0)
    status  = f"Last update: {ts[:19] if len(str(ts)) > 10 else ts} | Bias: {bias} | Conf: {conf:.0f}%"

    # Chart
    fig = _build_chart(market)

    # Trade table
    trade_table = _build_trade_table(today_trades)

    # Active position
    active_card = _build_active_position(active)

    # Event feed
    event_feed = _build_event_feed(events)

    return (
        f"₹{today_pnl:,.0f}",
        str(len(today_trades)),
        f"{win_rate}%",
        str(len(active)),
        f"₹{market.get('spot', 0):,.2f}",
        market.get("regime", "—"),
        status,
        fig,
        trade_table,
        active_card,
        event_feed,
    )


def _build_chart(market: dict) -> go.Figure:
    candles = market.get("candles_5m", [])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)

    if not candles:
        fig.update_layout(
            paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
            font_color="#90caf9",
            annotations=[dict(text="Waiting for market data...",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(color="#90caf9", size=14))]
        )
        return fig

    df   = pd.DataFrame(candles)
    df.index = pd.to_datetime(df['t'])

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['o'], high=df['h'],
        low=df['l'], close=df['c'],
        increasing_line_color='#69f0ae',
        decreasing_line_color='#ff5252',
        name="Nifty",
    ), row=1, col=1)

    # EMA 9 / 21
    if 'ema9' in market:
        fig.add_trace(go.Scatter(
            x=[df.index[-1]], y=[market['ema9']],
            mode='markers', marker=dict(color='#40c4ff', size=8),
            name=f"EMA9: {market['ema9']:.1f}",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[df.index[-1]], y=[market['ema21']],
            mode='markers', marker=dict(color='#ffd740', size=8),
            name=f"EMA21: {market['ema21']:.1f}",
        ), row=1, col=1)

    # VWAP line
    if 'vwap' in market:
        fig.add_hline(
            y=market['vwap'], line_color='#b388ff',
            line_dash='dash', line_width=1,
            annotation_text=f"VWAP {market['vwap']:.0f}",
            annotation_font_color='#b388ff', row=1, col=1,
        )

    # Volume bars
    vol_colors = ['#69f0ae' if c >= o else '#ff5252'
                  for c, o in zip(df['c'], df['o'])]
    fig.add_trace(go.Bar(
        x=df.index, y=df['v'],
        marker_color=vol_colors, opacity=0.7, name="Volume",
    ), row=2, col=1)

    fig.update_layout(
        paper_bgcolor="#1a1d2e", plot_bgcolor="#151824",
        font_color="#90caf9", font_size=11,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)", font_size=10),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis2=dict(title="", showgrid=True, gridcolor="#1e2440"),
        yaxis=dict(showgrid=True, gridcolor="#1e2440"),
        yaxis2=dict(showgrid=False),
    )
    return fig


def _build_trade_table(trades):
    if not trades:
        return html.P("No trades today.", style={"color": "#546e7a", "fontSize": "13px"})

    rows = []
    for t in trades[-20:]:
        pnl_color = "#69f0ae" if (t.pnl or 0) >= 0 else "#ff5252"
        rows.append(html.Tr([
            html.Td(str(t.entry_time)[:16] if t.entry_time else "—",
                    style={"color": "#90caf9", "fontSize": "12px"}),
            html.Td(t.symbol or "—",   style={"fontSize": "12px"}),
            html.Td(t.option_type or "—", style={"fontSize": "12px"}),
            html.Td(f"₹{t.entry_price:.2f}" if t.entry_price else "—", style={"fontSize": "12px"}),
            html.Td(f"₹{t.exit_price:.2f}" if t.exit_price else "Open",
                    style={"color": "#ffd740", "fontSize": "12px"}),
            html.Td(f"₹{t.pnl:,.0f}" if t.pnl is not None else "—",
                    style={"color": pnl_color, "fontWeight": "600", "fontSize": "12px"}),
            html.Td(t.exit_reason or "Open", style={"fontSize": "11px", "color": "#78909c"}),
        ]))

    return html.Table([
        html.Thead(html.Tr([
            html.Th(h, style={"color": "#78909c", "fontSize": "11px", "fontWeight": "500"})
            for h in ["Time", "Symbol", "Type", "Entry", "Exit", "P&L", "Reason"]
        ])),
        html.Tbody(rows),
    ], style={"width": "100%", "borderCollapse": "collapse"})


def _build_active_position(active: dict):
    if not active:
        return html.P("No active position.", style={"color": "#546e7a", "fontSize": "13px"})

    items = []
    for tid, data in active.items():
        items.append(dbc.Alert([
            html.Strong(data.get("symbol", tid), style={"color": "#ffd740"}),
            html.Br(),
            html.Small(f"Entry: ₹{data.get('entry', 0):.2f}", style={"color": "#90caf9"}),
            html.Span(" | ", style={"color": "#37474f"}),
            html.Small(f"SL: ₹{data.get('sl', 0):.2f}", style={"color": "#ff5252"}),
            html.Br(),
            html.Small(f"Target: ₹{data.get('target', 0):.2f}", style={"color": "#69f0ae"}),
        ], color="dark", style={"border": "1px solid #ffd740", "fontSize": "12px"}))

    return html.Div(items)


def _build_event_feed(events: list):
    if not events:
        return html.P("Waiting for events...", style={"color": "#546e7a", "fontSize": "12px"})

    items = []
    for ev in reversed(events[-30:]):
        color = {
            "trade_opened":       "#69f0ae",
            "trade_closed":       "#ffd740",
            "daily_loss_limit_hit": "#ff5252",
        }.get(ev.get("event"), "#90caf9")

        items.append(html.Div([
            html.Small(f"{ev['time'][:19]} ", style={"color": "#546e7a"}),
            html.Small(ev.get("event", ""), style={"color": color}),
        ], style={"borderBottom": "1px solid #1e2440", "padding": "4px 0"}))

    return html.Div(items)


if __name__ == "__main__":
    print("\n  Nifty MAS Dashboard starting...")
    print("  Open: http://localhost:8050\n")
    app.run(debug=False, host="0.0.0.0", port=8050)

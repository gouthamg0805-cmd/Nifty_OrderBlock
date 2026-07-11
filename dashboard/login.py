"""
dashboard/login.py
Kotak Neo Login UI — matches the confirmed working SDK exactly.

SDK: pip install git+https://github.com/Kotak-Neo/Kotak-neo-api-v2.git

Exact auth flow (from production code):
    client = NeoAPI(environment, access_token=None, neo_fin_key=None, consumer_key)
    r1 = client.totp_login(mobile_number, ucc, totp)
    r2 = client.totp_validate(mpin)

Fields on login screen → SDK params:
    MOBILE NUMBER  → mobile_number  (totp_login)
    UCC            → ucc            (totp_login)
    CONSUMER KEY   → consumer_key   (NeoAPI constructor)
    TOTP           → totp           (totp_login)
    MPIN           → mpin           (totp_validate)

Run:  python dashboard/login.py
Open: http://localhost:8051
"""
from __future__ import annotations
import os, sys, json
from datetime import datetime
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR   = PROJECT_ROOT / "config"
SESSION_FILE = CONFIG_DIR / ".session"
SETTINGS_ENV = CONFIG_DIR / "settings.env"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clean_mobile(m: str) -> str:
    """
    Return mobile in +91XXXXXXXXXX format.
    totp_login requires country code — "Invalid field MobileNumber" means wrong format.
    """
    m = m.strip()
    if m.startswith("+91"):  m = m[3:]
    elif m.startswith("91") and len(m) == 12: m = m[2:]
    return f"+91{m}"   # country code required by API

def load_saved() -> dict:
    src = SETTINGS_ENV if SETTINGS_ENV.exists() else CONFIG_DIR / "settings.example.env"
    creds = {}
    for line in src.read_text().splitlines():
        if not line.strip() or line.strip().startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        creds[k.strip()] = v.strip()
    return creds

def save_credentials(mobile, ucc, consumer_key, mpin, environment):
    """Persist credentials to settings.env (TOTP never saved)."""
    src   = SETTINGS_ENV if SETTINGS_ENV.exists() else CONFIG_DIR / "settings.example.env"
    lines = src.read_text().splitlines()
    updates = {
        "NEO_MOBILE":       _clean_mobile(mobile).replace("+91", ""),  # store 10 digits
        "NEO_UCC":          ucc.strip().upper(),
        "NEO_CONSUMER_KEY": consumer_key.strip(),
        "NEO_MPIN":         mpin.strip(),
        "NEO_ENVIRONMENT":  (environment or "prod").strip(),
    }
    updated, new_lines = set(), []
    for line in lines:
        if not line.strip() or line.strip().startswith("#") or "=" not in line:
            new_lines.append(line); continue
        k = line.split("=", 1)[0].strip()
        if k in updates:
            new_lines.append(f"{k}={updates[k]}"); updated.add(k)
        else:
            new_lines.append(line)
    for k, v in updates.items():
        if k not in updated:
            new_lines.append(f"{k}={v}")
    SETTINGS_ENV.write_text("\n".join(new_lines) + "\n")

def attempt_login(mobile, ucc, consumer_key, mpin, environment, totp) -> dict:
    """
    Perform the two-step TOTP login using the confirmed working SDK.
    """
    try:
        from neo_api_client import NeoAPI
    except ImportError:
        return {"success": False, "error":
                "neo-api-client not installed.\n"
                "Run: pip install git+https://github.com/Kotak-Neo/Kotak-neo-api-v2.git"}

    mobile = _clean_mobile(mobile)
    ucc    = ucc.strip().upper()
    totp   = (totp or "").replace(" ", "").strip()
    mpin   = (mpin or "").strip()
    env    = (environment or "prod").strip()

    # Step 0: construct client (no consumer_secret)
    try:
        client = NeoAPI(
            environment  = env,
            access_token = None,
            neo_fin_key  = None,
            consumer_key = consumer_key.strip(),
        )
    except Exception as e:
        return {"success": False, "error": f"NeoAPI init failed:\n{e}"}

    # Step 1: totp_login(mobile_number, ucc, totp)
    # mobile_number must include country code: +919880158227
    try:
        r1 = client.totp_login(
            mobile_number = mobile,
            ucc           = ucc,
            totp          = totp,
        )
        if isinstance(r1, dict) and r1.get("error"):
            return {"success": False,
                    "error": f"totp_login error:\n{r1['error']}"}
    except Exception as e:
        return {"success": False,
                "error": (
                    f"totp_login failed: {e}\n\n"
                    f"Mobile sent: {mobile}\n"
                    f"Make sure TOTP is entered within 30 seconds."
                )}

    # Step 2: totp_validate(mpin)
    try:
        r2 = client.totp_validate(mpin=mpin)
        if isinstance(r2, dict) and r2.get("error"):
            return {"success": False,
                    "error": f"totp_validate error:\n{r2['error']}"}
    except Exception as e:
        return {"success": False,
                "error": f"totp_validate (MPIN) failed:\n{e}\n\nCheck your MPIN."}

    # Save session token
    SESSION_FILE.write_text(json.dumps({
        "logged_in":  True,
        "login_time": datetime.now().isoformat(),
        "mobile":     mobile,
        "ucc":        ucc,
        "env":        env,
        "response":   str(r2),
    }, indent=2))

    return {"success": True, "message": f"✓ Logged in as {ucc}"}

# ─── Styles ───────────────────────────────────────────────────────────────────

PAGE = {
    "background": "#0d1117", "minHeight": "100vh",
    "display": "flex", "alignItems": "center", "justifyContent": "center",
    "fontFamily": "system-ui, -apple-system, sans-serif",
}
CARD = {
    "background": "#161b22", "border": "1px solid #30363d",
    "borderRadius": "14px", "padding": "32px 28px", "width": "460px",
}
LBL = {
    "color": "#7d8590", "fontSize": "11px", "fontWeight": "600",
    "letterSpacing": "1px", "display": "block", "marginBottom": "6px",
    "textTransform": "uppercase",
}
INP = {
    "width": "100%", "padding": "13px 14px", "background": "#0d1117",
    "border": "1px solid #30363d", "borderRadius": "8px",
    "color": "#e6edf3", "fontSize": "14px", "boxSizing": "border-box", "outline": "none",
}
BTN = {
    "width": "100%", "padding": "15px", "background": "#7c3aed",
    "color": "white", "border": "none", "borderRadius": "10px",
    "fontSize": "16px", "fontWeight": "600", "cursor": "pointer",
    "letterSpacing": "0.3px",
}

def field(label, component, hint=None):
    children = [html.Label(label, style=LBL), component]
    if hint:
        children.append(html.Small(
            hint, style={"color": "#484f58", "fontSize": "11px",
                         "display": "block", "marginTop": "4px"}))
    return html.Div(style={"marginBottom": "18px"}, children=children)

# ─── App ──────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="Kotak Neo Login",
    suppress_callback_exceptions=True,
)

def build_layout():
    saved     = load_saved()
    has_creds = bool(saved.get("NEO_CONSUMER_KEY") and saved.get("NEO_MOBILE"))

    return html.Div(style=PAGE, children=[html.Div(style=CARD, children=[

        # Header
        html.Div(style={"textAlign": "center", "marginBottom": "24px"}, children=[
            html.Div("N", style={
                "display": "inline-flex", "alignItems": "center",
                "justifyContent": "center", "width": "50px", "height": "50px",
                "background": "#7c3aed", "borderRadius": "12px",
                "fontSize": "24px", "fontWeight": "700", "color": "white",
                "marginBottom": "10px",
            }),
            html.P("Automated Options Intelligence",
                   style={"color": "#484f58", "fontSize": "12px", "margin": 0}),
        ]),

        html.Div(style={
            "display": "flex", "justifyContent": "space-between",
            "alignItems": "center", "marginBottom": "20px",
            "borderBottom": "1px solid #21262d", "paddingBottom": "16px",
        }, children=[
            html.Span("KOTAK NEO LOGIN",
                      style={"color": "#7d8590", "fontSize": "11px",
                             "fontWeight": "700", "letterSpacing": "1.5px"}),
            html.A("Clear saved", href="#", id="btn-clear",
                   style={"color": "#7c3aed", "fontSize": "12px", "textDecoration": "none"}),
        ]),

        # Saved-creds notice
        dbc.Alert(
            "✓ Credentials loaded — just enter TOTP and MPIN",
            color="success", is_open=has_creds,
            style={"fontSize": "12px", "padding": "10px 14px", "marginBottom": "18px"},
        ),

        # ── Fields matching the login screen exactly ───────────────────────────

        field("Mobile Number",
              dcc.Input(
                  id="inp-mobile", type="tel",
                  value=saved.get("NEO_MOBILE", ""),
                  placeholder="9880158227",
                  style=INP,
              ),
              hint="Enter 10 digits (e.g. 9880158227). +91 is added automatically."),

        field("UCC — Unique Client Code", dcc.Input(
            id="inp-ucc", type="text",
            value=saved.get("NEO_UCC", ""),
            placeholder="X3DNA",
            style=INP,
        )),

        field("Consumer Key", dcc.Input(
            id="inp-ckey", type="text",
            value=saved.get("NEO_CONSUMER_KEY", ""),
            placeholder="eeeeee-333-eee-sds-cccccc",
            style=INP,
        ), hint="From Kotak Neo app → Invest → Trade API → API Dashboard"),

        field(
            "TOTP  (changes every 30s — enter just before clicking Connect)",
            dcc.Input(
                id="inp-totp", type="tel",
                placeholder="1  2  3  4  5  6",
                maxLength=6,
                style={**INP, "fontSize": "26px", "letterSpacing": "14px",
                       "textAlign": "center", "fontWeight": "700"},
            ),
        ),

        field("MPIN", dcc.Input(
            id="inp-mpin", type="password",
            value=saved.get("NEO_MPIN", ""),
            placeholder="••••••",
            maxLength=6,
            style={**INP, "fontSize": "22px", "letterSpacing": "10px",
                   "textAlign": "center"},
        )),

        field("Environment", dcc.Dropdown(
            id="inp-env",
            options=[
                {"label": "Production  (prod)", "value": "prod"},
                {"label": "UAT / Sandbox  (uat)", "value": "uat"},
            ],
            value=saved.get("NEO_ENVIRONMENT", "prod"),
            clearable=False,
            style={"background": "#0d1117", "border": "1px solid #30363d",
                   "borderRadius": "8px", "color": "#e6edf3"},
        )),

        # Save checkbox
        dcc.Checklist(
            id="chk-save",
            options=[{"label": "  Save credentials (TOTP never saved)", "value": "save"}],
            value=["save"] if has_creds else [],
            style={"color": "#7d8590", "fontSize": "12px", "marginBottom": "20px"},
        ),

        # Connect button
        html.Button(
            "Connect to Kotak Neo →",
            id="btn-connect", n_clicks=0, style=BTN,
        ),

        html.Div(id="login-status",   style={"marginTop": "16px"}),
        html.Div(id="dashboard-link"),

        html.Hr(style={"borderColor": "#21262d", "margin": "24px 0 12px"}),
        html.P(
            "Paper mode needs no login — run: python main.py --mode paper",
            style={"color": "#30363d", "fontSize": "11px",
                   "textAlign": "center", "margin": 0},
        ),
    ])])

app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div(id="page-content", children=build_layout()),
])

# ─── Login callback ───────────────────────────────────────────────────────────

@app.callback(
    Output("login-status",   "children"),
    Output("dashboard-link", "children"),
    Input("btn-connect",     "n_clicks"),
    State("inp-mobile",  "value"),
    State("inp-ucc",     "value"),
    State("inp-ckey",    "value"),
    State("inp-env",     "value"),
    State("inp-totp",    "value"),
    State("inp-mpin",    "value"),
    State("chk-save",    "value"),
    prevent_initial_call=True,
)
def handle_login(n, mobile, ucc, ckey, env, totp, mpin, save_flag):
    if not n:
        return no_update, no_update

    # Validate all required fields
    missing = []
    if not (mobile or "").strip(): missing.append("Mobile Number")
    if not (ucc    or "").strip(): missing.append("UCC")
    if not (ckey   or "").strip(): missing.append("Consumer Key")
    if not (mpin   or "").strip(): missing.append("MPIN")
    totp_clean = (totp or "").replace(" ", "").strip()
    if len(totp_clean) != 6 or not totp_clean.isdigit():
        missing.append("TOTP (must be exactly 6 digits)")

    if missing:
        return (
            dbc.Alert(
                f"Please fill in: {', '.join(missing)}",
                color="warning",
                style={"fontSize": "12px", "padding": "10px 14px"},
            ),
            no_update,
        )

    # Save creds before attempting (so they're there even if TOTP fails)
    if save_flag and "save" in save_flag:
        try:
            save_credentials(mobile, ucc, ckey, mpin, env or "prod")
        except Exception:
            pass

    result = attempt_login(mobile, ucc, ckey, mpin, env or "prod", totp_clean)

    if result["success"]:
        status = dbc.Alert([
            html.Strong(result["message"]), html.Br(),
            html.Small("Session saved. Open the trading dashboard to start."),
        ], color="success", style={"fontSize": "12px", "padding": "10px 14px"})

        link = html.Div(style={"marginTop": "12px"}, children=[
            html.A(
                "→ Open Trading Dashboard",
                href="http://localhost:8050", target="_blank",
                style={
                    "display": "block", "padding": "13px",
                    "background": "#0d4f2f", "color": "#3fb950",
                    "borderRadius": "8px", "textAlign": "center",
                    "fontSize": "14px", "fontWeight": "600",
                    "textDecoration": "none",
                },
            )
        ])
        return status, link

    return (
        dbc.Alert([
            html.Strong("Login failed"), html.Br(),
            html.Code(
                result["error"],
                style={"fontSize": "11px", "whiteSpace": "pre-wrap",
                       "display": "block", "marginTop": "8px",
                       "background": "transparent", "color": "#f85149"},
            ),
        ], color="danger", style={"fontSize": "12px", "padding": "10px 14px"}),
        no_update,
    )


if __name__ == "__main__":
    port = int(os.getenv("LOGIN_PORT", 8051))
    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  Kotak Neo Login  →  localhost:{port}  ║")
    print(f"  ╚══════════════════════════════════════╝\n")
    app.run(debug=False, host="0.0.0.0", port=port)

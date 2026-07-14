"""
core/broker.py
Neo Kotak API wrapper — confirmed complete place_order signature.

place_order() full signature (from official docs):
    client.place_order(
        exchange_segment  = "nse_fo",        # nse_fo for F&O
        product           = "MIS",           # MIS for intraday
        price             = "0",             # "0" for MKT orders
        order_type        = "MKT",           # MKT | L | SL | SL-M
        quantity          = "65",            # string
        validity          = "DAY",           # DAY | IOC  ← was missing!
        trading_symbol    = "NIFTY..CE",     # from scrip master
        transaction_type  = "B",             # B | S
        amo               = "NO",            # After market order
        disclosed_quantity= "0",
        market_protection = "0",
        pf                = "N",
        trigger_price     = "0",             # for SL/SL-M orders
        tag               = None,
    )

modify_order() full signature:
    client.modify_order(
        order_id          = "",
        price             = "0",
        quantity          = "65",
        disclosed_quantity= "0",
        trigger_price     = "0",
        validity          = "DAY",
        order_type        = "SL-M",
        amo               = "NO",
    )
"""
from __future__ import annotations
import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from loguru import logger


class SessionExpiredError(Exception):
    """Raised when the Kotak broker session has expired and needs re-authentication."""
    pass


class BrokerClient:

    def __init__(self, mode: str = "paper"):
        self.mode          = mode
        self._client       = None
        self._logged_in    = False
        self._paper_orders: Dict[str, Dict] = {}

        self._session_invalid = False   # set True on 2FA/session error
        if mode == "live":
            self._init_client()

    # ─── Init ─────────────────────────────────────────────────────────────────

    def _init_client(self):
        try:
            from neo_api_client import NeoAPI
        except ImportError:
            raise ImportError(
                "neo-api-client (v2) not installed.\n"
                "Run: pip install git+https://github.com/Kotak-Neo/Kotak-neo-api-v2.git"
            )

        consumer_key = os.getenv("NEO_CONSUMER_KEY", "").strip()
        environment  = os.getenv("NEO_ENVIRONMENT", "prod").strip()

        if not consumer_key:
            raise ValueError("NEO_CONSUMER_KEY not set in config/settings.env")

        self._client = NeoAPI(
            environment  = environment,
            access_token = None,
            neo_fin_key  = None,
            consumer_key = consumer_key,
        )
        logger.info(f"NeoAPI client created (env={environment})")

    # ─── Login ────────────────────────────────────────────────────────────────

    def login(self, totp: Optional[str] = None) -> bool:
        if self.mode == "paper":
            logger.info("[PAPER] Login simulated")
            self._logged_in = True
            return True

        mobile = self._clean_mobile(os.getenv("NEO_MOBILE", "").strip())
        ucc    = os.getenv("NEO_UCC",  "").strip().upper()
        mpin   = os.getenv("NEO_MPIN", "").strip()

        if not mobile: raise ValueError("NEO_MOBILE not set")
        if not ucc:    raise ValueError("NEO_UCC not set")
        if not mpin:   raise ValueError("NEO_MPIN not set")

        if totp is None:
            totp = input(
                "\n  Enter TOTP from Kotak authenticator app (6 digits): "
            ).strip()

        # Step 1
        logger.info(f"totp_login | mobile=+91***{mobile[-4:]} | ucc={ucc}")
        try:
            r1 = self._client.totp_login(
                mobile_number = mobile,
                ucc           = ucc,
                totp          = totp,
            )
            logger.info(f"totp_login: {r1}")
            if isinstance(r1, dict) and r1.get("error"):
                raise Exception(f"totp_login error: {r1['error']}")
        except Exception as e:
            logger.error(f"totp_login failed: {e}")
            raise

        # Step 2
        logger.info("totp_validate (mpin)")
        try:
            r2 = self._client.totp_validate(mpin=mpin)
            logger.info(f"totp_validate: {r2}")
            if isinstance(r2, dict) and r2.get("error"):
                raise Exception(f"totp_validate error: {r2['error']}")
        except Exception as e:
            logger.error(f"totp_validate failed: {e}")
            raise

        self._logged_in       = True
        self._session_invalid = False   # reset on successful login
        logger.info("✓ Login successful")
        return True

    @staticmethod
    def _clean_mobile(m: str) -> str:
        if m.startswith("+91"): m = m[3:]
        elif m.startswith("91") and len(m) == 12: m = m[2:]
        return f"+91{m}"

    def relogin(self, totp: Optional[str] = None) -> bool:
        """
        Re-authenticate after session expiry — no system restart needed.
        Stored credentials are reused; only fresh TOTP is needed.
        """
        logger.info("[Broker] Re-authenticating after session expiry...")
        self._session_invalid = False
        self._logged_in       = False
        self._init_client()           # fresh client object
        return self.login(totp=totp)  # full login with fresh TOTP

    def is_authenticated(self) -> bool:
        return self._logged_in and not self._session_expired

    @property
    def _session_expired(self) -> bool:
        return getattr(self, '_session_invalid', False)

    def _check_response(self, resp: Any, context: str = "") -> Any:
        """
        Check every API response for session/2FA errors.
        If detected: mark session as invalid and raise immediately.
        Call this after EVERY broker API call.
        """
        if resp is None:
            return resp
        err_msg = None
        if isinstance(resp, dict):
            err_msg = (resp.get("Error Message") or resp.get("error_message")
                       or resp.get("error") or resp.get("message", ""))
        elif isinstance(resp, list) and resp and isinstance(resp[0], dict):
            err_msg = resp[0].get("Error Message") or resp[0].get("error")

        if err_msg and isinstance(err_msg, str):
            err_lower = err_msg.lower()
            if any(phrase in err_lower for phrase in [
                "complete the 2fa",
                "2fa process",
                "session expired",
                "session has been closed",
                "unauthorized",
                "invalid session",
                "token expired",
                "please login again",
            ]):
                self._session_invalid = True
                self._logged_in       = False
                logger.error(
                    f"[Broker] SESSION EXPIRED — {err_msg}\n"
                    f"  Context: {context}\n"
                    f"  ⚠ Trading HALTED. Run the login UI and log in again:\n"
                    f"  python dashboard/login.py  →  http://localhost:8051"
                )
                from core.message_bus import bus
                # NOTE: this can run on a worker thread (blocking broker calls
                # are typically invoked via loop.run_in_executor), which has
                # no event loop of its own. asyncio.get_event_loop() /
                # create_task() will raise "There is no current event loop in
                # thread '...'" there — use the thread-safe bus helper instead.
                bus.publish_event_threadsafe("session_expired", {
                    "error": err_msg, "context": context
                })
                raise SessionExpiredError(
                    f"Kotak session expired: {err_msg}\n"
                    f"Please log in again via: python dashboard/login.py"
                )
        return resp

    # ─── Place Order ──────────────────────────────────────────────────────────

    def place_order(
        self,
        trading_symbol:    str,
        transaction_type:  str,           # B | S
        quantity:          int,
        order_type:        str,           # MKT | L | SL | SL-M
        price:             float = 0.0,
        trigger_price:     float = 0.0,
        product:           str   = "MIS",
        exchange_segment:  str   = "nse_fo",
        validity:          str   = "DAY",
        amo:               str   = "NO",
        tag:               str   = "",
    ) -> Dict[str, Any]:

        if self.mode == "paper":
            return self._paper_order(
                trading_symbol, transaction_type, quantity,
                order_type, price, trigger_price, tag
            )

        try:
            resp = self._client.place_order(
                exchange_segment   = exchange_segment,
                product            = product,
                price              = str(price) if price else "0",
                order_type         = order_type,
                quantity           = str(quantity),
                validity           = validity,
                trading_symbol     = trading_symbol,
                transaction_type   = transaction_type,
                amo                = amo,
                disclosed_quantity = "0",
                market_protection  = "0",
                pf                 = "N",
                trigger_price      = str(trigger_price) if trigger_price else "0",
                tag                = tag or None,
            )
            self._check_response(resp, f"place_order {trading_symbol} {transaction_type}")
            logger.info(f"Order placed: {resp}")
            return resp
        except Exception as e:
            logger.error(f"place_order failed: {e}")
            raise

    # ─── Modify Order (trailing SL updates) ───────────────────────────────────

    def modify_order(
        self,
        order_id:       str,
        trigger_price:  float,
        price:          float  = 0.0,
        quantity:       int    = 0,
        order_type:     str    = "SL-M",
        validity:       str    = "DAY",
    ) -> Dict[str, Any]:

        if self.mode == "paper":
            if order_id in self._paper_orders:
                self._paper_orders[order_id]["trigger_price"] = trigger_price
            logger.debug(f"[PAPER] SL modified → ₹{trigger_price:.2f}")
            return {"status": "ok", "order_id": order_id}

        try:
            resp = self._client.modify_order(
                order_id           = order_id,
                price              = str(price) if price else "0",
                quantity           = str(quantity) if quantity else "0",
                disclosed_quantity = "0",
                trigger_price      = str(trigger_price),
                validity           = validity,
                order_type         = order_type,
                amo                = "NO",
            )
            self._check_response(resp, f"modify_order {order_id}")
            logger.info(f"Order modified: {resp}")
            return resp
        except Exception as e:
            logger.error(f"modify_order failed: {e}")
            raise

    # ─── Cancel Order ─────────────────────────────────────────────────────────

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        if self.mode == "paper":
            self._paper_orders.pop(order_id, None)
            return {"status": "ok"}
        try:
            return self._client.cancel_order(order_id=order_id)
        except Exception as e:
            logger.error(f"cancel_order failed: {e}")
            return {"status": "error", "error": str(e)}

    # ─── Positions / Balance ──────────────────────────────────────────────────

    def get_positions(self) -> list:
        if self.mode == "paper":
            return list(self._paper_orders.values())
        try:
            return self._client.positions() or []
        except Exception as e:
            logger.error(f"positions failed: {e}")
            return []

    def get_order_book(self) -> list:
        if self.mode == "paper":
            return list(self._paper_orders.values())
        try:
            return self._client.order_report() or []
        except Exception as e:
            logger.error(f"order_report failed: {e}")
            return []

    def get_balance(self) -> dict:
        if self.mode == "paper":
            return {"available": float(os.getenv("TOTAL_CAPITAL", 200000))}
        try:
            resp = self._client.limits()
            return {"available": float((resp or {}).get("Net", 0))}
        except Exception as e:
            logger.error(f"limits failed: {e}")
            return {"available": 0}

    def square_off_all(self):
        if self.mode == "paper":
            logger.info("[PAPER] Square-off all")
            self._paper_orders.clear()
            return
        try:
            self._client.positions_sq_off(
                pf="N", exchange_segment="nse_fo", product="MIS"
            )
        except Exception as e:
            logger.error(f"square_off_all failed: {e}")

    # ─── Paper helpers ────────────────────────────────────────────────────────

    def _paper_order(
        self, symbol, txn, qty, order_type, price, trigger_price, tag
    ) -> Dict[str, Any]:
        oid  = str(uuid.uuid4())[:8].upper()
        fill = price if price > 0 else trigger_price
        order = {
            "order_id":      oid,
            "symbol":        symbol,
            "transaction":   txn,
            "quantity":      qty,
            "order_type":    order_type,
            "price":         price,
            "trigger_price": trigger_price,
            "fill_price":    fill,
            "status":        "FILLED",
            "tag":           tag,
            "timestamp":     datetime.now().isoformat(),
        }
        self._paper_orders[oid] = order
        logger.info(
            f"[PAPER] {txn} {qty} {symbol} {order_type} "
            f"@ ₹{fill:.2f} | {oid} | {tag}"
        )
        return order

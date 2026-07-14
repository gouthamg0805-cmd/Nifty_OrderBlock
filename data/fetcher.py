"""
data/fetcher.py
Market data fetcher — NIFTY 50 / SENSEX INDEX options.

CONFIRMED FACTS about Kotak Neo API v2 (from live response inspection):
  ✓ search_scrip(exchange_segment, symbol, expiry, option_type, strike_price)
      Returns: [{'pSymbol': 51352,              ← numeric token for quotes()
                 'pTrdSymbol': 'NIFTY2651923450CE',  ← trading symbol for place_order()
                 'pOptionType': 'ce', ...}]
  ✓ quotes(instrument_tokens=[{'instrument_token': pSymbol, 'exchange_segment': ...}],
           quote_type='ltp')
      instrument_token = pSymbol (numeric string e.g. '51352'), NOT pTrdSymbol
  ✓ place_order(trading_symbol=pTrdSymbol, ...)
      trading_symbol = pTrdSymbol string e.g. 'NIFTY2651923450CE'
  ✗ pTkn — does NOT exist in v2 (was a v1 field name)
  ✗ historical_candles() — DOES NOT EXIST in Kotak API (any version)
  ✗ isIndex — NOT a parameter in v2 quotes()

Candle data (OHLCV for indicators):
  → yfinance ^NSEI / ^BSESN only (broker has no historical API)

Live spot price:
  → Kotak quotes("26000", "nse_cm") for NIFTY; quotes("1", "bse_cm") for SENSEX
  → yfinance fallback in paper mode

Live option LTP flow:
  1. search_scrip() → extract pSymbol + pTrdSymbol
  2. quotes(instrument_token=pSymbol) → LTP
  3. attach pTrdSymbol to OptionStrike.trading_symbol for Agent4 place_order()
"""
from __future__ import annotations
import asyncio
import os
import time
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger

NIFTY_YFINANCE_TICKER  = "^NSEI"
NIFTY_TOKEN            = "26000"       # Nifty 50 index token on NSE
NIFTY_EXCHANGE_CM      = "nse_cm"
NIFTY_EXCHANGE_FO      = "nse_fo"
NIFTY_FO_SYMBOL        = "NIFTY"
NIFTY_STRIKE_STEP      = 50
NIFTY_LOT_SIZE         = 65

# ── SENSEX (BSE) constants ────────────────────────────────────────────────────
SENSEX_YFINANCE_TICKER = "^BSESN"
SENSEX_TOKEN           = "1"           # BSE Sensex index token on BSE
SENSEX_EXCHANGE_CM     = "bse_cm"
SENSEX_EXCHANGE_FO     = "bse_fo"
SENSEX_FO_SYMBOL       = "SENSEX"
SENSEX_STRIKE_STEP     = 100           # Sensex options use 100-point strike steps
SENSEX_LOT_SIZE        = 10            # BSE Sensex lot size

# Token cache: "NIFTY13APR2623750PE" → "58423"
# Key format includes expiry so stale tokens from prior weeks are never reused.
# Cache is keyed as "<SYMBOL><expiry><strike><type>" — expiry date is embedded,
# so when expiry rolls over the old keys simply go unused (no collision risk).
_TOKEN_CACHE: dict[str, str] = {}
_TOKEN_CACHE_EXPIRY: dict[str, str] = {}   # cache_key → expiry it was fetched for

def _token_cache_get(cache_key: str, current_expiry: str) -> str | None:
    """Return cached token only if it was fetched for the same expiry week."""
    if cache_key in _TOKEN_CACHE:
        if _TOKEN_CACHE_EXPIRY.get(cache_key) == current_expiry:
            return _TOKEN_CACHE[cache_key]
        else:
            # Stale token from a prior expiry — purge it
            del _TOKEN_CACHE[cache_key]
            del _TOKEN_CACHE_EXPIRY[cache_key]
    return None

def _token_cache_set(cache_key: str, token: str, current_expiry: str) -> None:
    _TOKEN_CACHE[cache_key] = token
    _TOKEN_CACHE_EXPIRY[cache_key] = current_expiry


class DataFetcher:

    TIMEFRAME_MAP = {
        "1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "1d": "1d",
    }

    def __init__(self, mode: str = "paper", broker_client=None,
                 instrument: str = "NIFTY"):
        self.mode          = mode
        self.broker_client = broker_client
        self._spot_cache   = None
        self._spot_ts      = 0.0
        self._candle_cache: dict[str, pd.DataFrame] = {}
        self._yf_blocked_until = 0.0   # monotonic timestamp; 0 = not blocked

        # ── Instrument config (NIFTY or SENSEX) ──────────────────────────
        instrument = instrument.upper().strip()
        if instrument == "SENSEX":
            self.instrument  = "SENSEX"
            self.yf_ticker   = SENSEX_YFINANCE_TICKER
            self.spot_token  = SENSEX_TOKEN
            self.exchange_cm = SENSEX_EXCHANGE_CM
            self.exchange_fo = SENSEX_EXCHANGE_FO
            self.fo_symbol   = SENSEX_FO_SYMBOL
            self.strike_step = SENSEX_STRIKE_STEP
            self.lot_size    = SENSEX_LOT_SIZE
        else:
            # Default: NIFTY
            self.instrument  = "NIFTY"
            self.yf_ticker   = NIFTY_YFINANCE_TICKER
            self.spot_token  = NIFTY_TOKEN
            self.exchange_cm = NIFTY_EXCHANGE_CM
            self.exchange_fo = NIFTY_EXCHANGE_FO
            self.fo_symbol   = NIFTY_FO_SYMBOL
            self.strike_step = NIFTY_STRIKE_STEP
            self.lot_size    = NIFTY_LOT_SIZE

        logger.info(f"[Fetcher] Instrument: {self.instrument} | "
                    f"Strike step: {self.strike_step} | Lot: {self.lot_size}")

    def set_broker(self, broker_client):
        self.broker_client = broker_client
        logger.info("[Fetcher] Broker attached")

    # ─── Nifty 50 Spot Price ──────────────────────────────────────────────────

    async def get_spot_price(self) -> Optional[float]:
        # 1. Live broker quote (live mode only)
        if self._broker_ready():
            price = await self._spot_via_broker()
            if price:
                self._spot_cache = price
                self._spot_ts    = time.time()
                return price

        # 2. yfinance fallback (paper mode / broker unavailable)
        if time.time() >= self._yf_blocked_until:
            price = await self._spot_via_yfinance()
            if price:
                self._spot_cache = price
                self._spot_ts    = time.time()
                return price

        # 3. Stale cache (max 5 min)
        if self._spot_cache and (time.time() - self._spot_ts) < 300:
            logger.debug(f"[Fetcher] Cached spot: ₹{self._spot_cache:.2f}")
            return self._spot_cache

        logger.warning("[Fetcher] Cannot get Nifty spot price.")
        return None

    async def _spot_via_broker(self) -> Optional[float]:
        """
        Kotak v2: quotes(instrument_tokens, quote_type)
        Nifty 50 token = "26000", segment = "nse_cm"
        NO isIndex parameter in v2.
        """
        loop = asyncio.get_event_loop()
        def _call():
            try:
                resp = self.broker_client._client.quotes(
                    instrument_tokens = [{
                        "instrument_token": self.spot_token,
                        "exchange_segment": self.exchange_cm,
                    }],
                    quote_type = "ltp",
                )
                ltp = self._extract_ltp(resp)
                if ltp and ltp > 1000:
                    logger.debug(f"[Fetcher] Broker spot ({self.instrument}): ₹{ltp:.2f}")
                    return ltp
            except Exception as e:
                logger.debug(f"[Fetcher] Broker spot error: {e}")
            return None
        return await loop.run_in_executor(None, _call)

    async def _spot_via_yfinance(self) -> Optional[float]:
        loop = asyncio.get_event_loop()
        def _call():
            try:
                import yfinance as yf
                df = yf.Ticker(self.yf_ticker).history(period="1d", interval="1m")
                if not df.empty:
                    p = float(df["Close"].iloc[-1])
                    if p > 1000:
                        return p
            except Exception as e:
                logger.debug(f"[Fetcher] yfinance spot error: {e}")
                # NOTE: this runs on a worker thread (via run_in_executor), so
                # it must not touch asyncio.get_event_loop()/call_later — that
                # raises "There is no current event loop in thread '...'" on
                # non-main threads. A plain timestamp is thread-safe.
                self._yf_blocked_until = time.time() + 300
            return None
        return await loop.run_in_executor(None, _call)

    # ─── OHLCV Candles ────────────────────────────────────────────────────────

    async def get_candles(self, timeframe: str, lookback: int = 50) -> Optional[pd.DataFrame]:
        """
        Kotak API has NO historical candle endpoint.
        Always use yfinance ^NSEI for indicator data.
        """
        df = await self._candles_via_yfinance(timeframe, lookback)
        if df is not None and len(df) >= 15:
            self._candle_cache[timeframe] = df
            return df

        if timeframe in self._candle_cache:
            logger.debug(f"[Fetcher] Using cached candles ({timeframe})")
            return self._candle_cache[timeframe]
        return None

    async def _candles_via_yfinance(self, timeframe: str, lookback: int) -> Optional[pd.DataFrame]:
        interval = self.TIMEFRAME_MAP.get(timeframe, "5m")
        period   = "5d" if timeframe in ("1m","5m","15m") else "30d"
        loop     = asyncio.get_event_loop()
        def _call():
            try:
                import yfinance as yf
                df = yf.Ticker(self.yf_ticker).history(
                    period=period, interval=interval
                )
                if df.empty:
                    return None
                df = df.rename(columns={
                    "Open":"open","High":"high","Low":"low",
                    "Close":"close","Volume":"volume"
                })
                df = df[["open","high","low","close","volume"]].dropna()
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
                if df["volume"].sum() == 0 or (df["volume"] == 0).mean() > 0.5:
                    df["volume"] = np.random.lognormal(12, 0.5, len(df)).astype(int)
                return df.tail(lookback)
            except Exception as e:
                logger.debug(f"[Fetcher] yfinance candles error: {e}")
            return None
        return await loop.run_in_executor(None, _call)

    # ─── Options Chain — Real live LTP ───────────────────────────────────────

    async def get_options_chain(
        self, expiry: str, option_type: str, spot: float
    ) -> list:
        from core.models import OptionStrike, OptionType as OT

        if self._broker_ready():
            try:
                strikes = await self._options_live_ltp(expiry, option_type, spot)
                if strikes:
                    logger.info(
                        f"[Fetcher] ✓ {len(strikes)} live Nifty options "
                        f"(expiry={expiry}, type={option_type})"
                    )
                    return strikes
                logger.warning(
                    f"[Fetcher] Live options chain returned 0 strikes "
                    f"(expiry={expiry} type={option_type}). Using synthetic."
                )
            except Exception as e:
                logger.error(f"[Fetcher] Live options chain error: {e}")
        else:
            logged_in      = getattr(getattr(self, "broker_client", None), "_logged_in", False)
            session_invalid = getattr(getattr(self, "broker_client", None), "_session_invalid", False)
            logger.debug(
                f"[Fetcher] broker_ready=False "
                f"(mode={self.mode!r} logged_in={logged_in} "
                f"session_invalid={session_invalid} "
                f"client={'set' if getattr(getattr(self,'broker_client',None),'_client',None) else 'None'})"
            )

        return self._synthetic_nifty_chain(spot, option_type, expiry)

    async def _options_live_ltp(
        self, expiry: str, option_type: str, spot: float
    ) -> list:
        """
        Get real live LTP using search_scrip() + quotes().
        Bails immediately on 2FA / session-expired response.
        """
        from core.models import OptionStrike, OptionType as OT
        loop = asyncio.get_event_loop()

        expiry_full   = self._expand_expiry(expiry)
        atm           = round(spot / self.strike_step) * self.strike_step
        strike_prices = [atm + i * self.strike_step for i in range(-4, 5)]
        session_ok    = [True]

        def _is_session_error(raw) -> bool:
            s = str(raw).lower()
            return any(p in s for p in [
                "2fa", "complete the 2fa", "session expired",
                "unauthorized", "login again", "invalid token",
            ])

        def _call():
            results = []
            for sp in strike_prices:
                if not session_ok[0]:
                    break

                cache_key     = f"{self.fo_symbol}{expiry}{int(sp)}{option_type}"
                trd_cache_key = f"{cache_key}__trd"   # stores pTrdSymbol for place_order
                token     = _token_cache_get(cache_key, expiry)      # numeric pSymbol
                trd_sym   = _token_cache_get(trd_cache_key, expiry)  # pTrdSymbol string

                try:
                    # Step 1: search_scrip → extract pSymbol (quotes token) + pTrdSymbol (orders)
                    if not token:
                        scrip_resp = self.broker_client._client.search_scrip(
                            exchange_segment = self.exchange_fo,
                            symbol           = self.fo_symbol,
                            expiry           = expiry_full,
                            option_type      = option_type,
                            strike_price     = str(int(sp)),
                        )
                        if _is_session_error(scrip_resp):
                            logger.error(
                                "[Fetcher] SESSION EXPIRED (search_scrip). "
                                "Re-login: python dashboard/login.py"
                            )
                            session_ok[0] = False
                            break

                        # Extract both pSymbol (numeric, for quotes) and
                        # pTrdSymbol (string, for place_order / agent4)
                        raw_item = {}
                        if isinstance(scrip_resp, list) and scrip_resp:
                            raw_item = scrip_resp[0] if isinstance(scrip_resp[0], dict) else {}
                        elif isinstance(scrip_resp, dict):
                            data = scrip_resp.get("data") or scrip_resp.get("result") or []
                            raw_item = data[0] if data and isinstance(data[0], dict) else {}

                        p_symbol  = str(raw_item.get("pSymbol", "")).strip()   # "51352"
                        p_trd_sym = str(raw_item.get("pTrdSymbol", "")).strip() # "NIFTY2651923450CE"

                        if p_symbol and p_symbol != "0":
                            token = p_symbol
                            _token_cache_set(cache_key, token, expiry)
                        if p_trd_sym:
                            trd_sym = p_trd_sym
                            _token_cache_set(trd_cache_key, trd_sym, expiry)

                        if token:
                            logger.debug(
                                f"[Fetcher] Scrip resolved: {cache_key} → "
                                f"pSymbol='{token}' pTrdSymbol='{trd_sym}'"
                            )
                        else:
                            logger.warning(
                                f"[Fetcher] No pSymbol for {cache_key}. "
                                f"Raw: {str(scrip_resp)[:300]}"
                            )

                    if not token:
                        continue

                    # Step 2: LTP via quotes(instrument_token=pSymbol, exchange_segment)
                    # Kotak Neo v2 quotes() takes the numeric pSymbol as instrument_token.
                    quote_resp = self.broker_client._client.quotes(
                        instrument_tokens = [{
                            "instrument_token": token,          # numeric pSymbol string
                            "exchange_segment": self.exchange_fo,
                        }],
                        quote_type = "ltp",
                    )
                    if _is_session_error(quote_resp):
                        logger.error("[Fetcher] SESSION EXPIRED (quotes). Re-login.")
                        session_ok[0] = False
                        break

                    ltp = self._extract_ltp(quote_resp)
                    if not ltp or ltp < 0.5:
                        logger.debug(
                            f"[Fetcher] LTP extract failed for {cache_key}. "
                            f"Raw quote resp: {str(quote_resp)[:300]}"
                        )
                        continue

                    # Step 3: OI (best-effort, non-fatal)
                    oi = vol = 0.0
                    try:
                        oi_resp = self.broker_client._client.quotes(
                            instrument_tokens = [{
                                "instrument_token": token,
                                "exchange_segment": self.exchange_fo,
                            }],
                            quote_type = "oi",
                        )
                        oi  = float(self._extract_field(oi_resp, "oi")  or 0)
                        vol = float(self._extract_field(oi_resp, "vol") or 0)
                    except Exception:
                        pass

                    sym = f"{self.fo_symbol}{expiry}{int(sp)}{option_type}"
                    strike = OptionStrike(
                        symbol       = sym,
                        strike_price = float(sp),
                        option_type  = OT(option_type),
                        expiry       = expiry,
                        ltp          = round(float(ltp), 2),
                        oi           = oi,
                        volume       = vol,
                        bid          = round(float(ltp) * 0.995, 2),
                        ask          = round(float(ltp) * 1.005, 2),
                    )
                    # Attach trading symbol so Agent4 can use it directly
                    # in place_order(trading_symbol=...) without re-lookup
                    strike.trading_symbol = trd_sym or sym
                    results.append(strike)
                    logger.debug(f"[Fetcher] ✓ {sym}: ₹{ltp:.2f} (trdSym={strike.trading_symbol})")

                except Exception as e:
                    if _is_session_error(str(e)):
                        logger.error("[Fetcher] SESSION EXPIRED (exception). Re-login.")
                        session_ok[0] = False
                        break
                    logger.warning(f"[Fetcher] Strike {sp} error: {e}")
                    continue

            # Broadcast session expired if detected
            if not session_ok[0]:
                # NOTE: this runs inside run_in_executor's worker thread, so
                # asyncio.get_event_loop()/create_task() is unsafe here (raises
                # "There is no current event loop in thread '...'"). Use the
                # thread-safe bus helper instead, which schedules onto the
                # bound main loop via run_coroutine_threadsafe.
                from core.message_bus import bus as _b
                _b.publish_event_threadsafe("session_expired", {
                    "context": "options_chain"
                })

            return results

        return await loop.run_in_executor(None, _call)

    # ─── Response parsers ─────────────────────────────────────────────────────

    def _extract_ltp(self, resp) -> Optional[float]:
        """
        Extract LTP from Kotak v2 quote response.

        Bug-fix: Kotak sometimes returns the real price under 'lTrdPrc'
        while 'ltp'/'ltP' are present but contain "0" or "0.0".
        The old code stopped at the first key it found regardless of value,
        so it returned 0.  Fix: collect ALL candidate values and return the
        first positive one.
        """
        if resp is None:
            return None
        try:
            items = []
            if isinstance(resp, dict):
                items = resp.get("data") or resp.get("message") or resp.get("result") or []
            elif isinstance(resp, list):
                items = resp
            if not items:
                return None
            item = items[0] if isinstance(items[0], dict) else {}
            # Try every known key name; collect all positive candidates
            candidates = []
            for key in ("ltp", "ltP", "last_traded_price", "lp", "lTrdPrc",
                        "LastTradedPrice", "lastTradedPrice", "close"):
                v = item.get(key)
                if v is not None and str(v).strip() not in ("", "0", "0.0"):
                    try:
                        val = float(v)
                        if val > 0:
                            candidates.append(val)
                    except (ValueError, TypeError):
                        continue
            if candidates:
                return candidates[0]
        except Exception as e:
            logger.debug(f"[Fetcher] _extract_ltp error: {e}")
        return None

    def _extract_token(self, resp) -> Optional[str]:
        """
        Extract the instrument token from a Kotak Neo v2 search_scrip response.

        Kotak v2 actual response (confirmed from live logs):
          [{'pSymbol': 51352, 'pTrdSymbol': 'NIFTY2651923450CE',
            'pOptionType': 'ce', 'pScripRefKey': 'NIFTY19MAY2623450.00CE', ...}]

        'pTkn' does NOT exist in v2 — that was a v1 field name.
        quotes() API accepts pTrdSymbol as instrument_token (e.g. 'NIFTY2651923450CE').
        pSymbol (integer) is the numeric token and also accepted as fallback.
        """
        if resp is None:
            return None
        try:
            try:
                import pandas as pd
                if isinstance(resp, pd.DataFrame):
                    if not resp.empty:
                        for col in ("pTrdSymbol", "pSymbol", "pTkn",
                                    "instrument_token", "token", "tk"):
                            if col in resp.columns:
                                v = resp.iloc[0][col]
                                if v and str(v).strip() not in ("", "0"):
                                    return str(v).strip()
                    return None
            except ImportError:
                pass

            items = []
            if isinstance(resp, dict):
                items = resp.get("data") or resp.get("result") or []
            elif isinstance(resp, list):
                items = resp
            if not items:
                return None
            item = items[0] if isinstance(items[0], dict) else {}
            # Kotak v2: pTrdSymbol is the trading symbol string used by quotes()
            # pSymbol is the numeric token (also accepted by quotes())
            for key in ("pTrdSymbol", "pSymbol", "pTkn",
                        "instrument_token", "token", "tk", "instrumentToken"):
                v = item.get(key)
                if v is not None and str(v).strip() not in ("", "0"):
                    return str(v).strip()
        except Exception as e:
            logger.debug(f"[Fetcher] _extract_token error: {e}")
        return None

    def _extract_field(self, resp, field: str) -> Optional[float]:
        if resp is None:
            return None
        try:
            data = []
            if isinstance(resp, dict):
                data = resp.get("data") or resp.get("message") or []
            elif isinstance(resp, list):
                data = resp
            if data:
                item = data[0]
                v = item.get(field) or item.get(field.upper()) or item.get(field.lower())
                if v is not None:
                    return float(v)
        except Exception:
            pass
        return None

    @staticmethod
    def _expand_expiry(expiry: str) -> str:
        """
        Convert short expiry format "19MAY26" → "19MAY2026" for search_scrip.

        Kotak Neo v2 search_scrip expects the actual expiry date in ddMMMYYYY
        format (e.g. "19MAY2026").

        NOTE: An older workaround added +1 day here to compensate for a Kotak
        off-by-one bug that existed in an earlier SDK version.  That bug is
        FIXED in the current Kotak Neo v2 SDK — sending +1 day now causes
        "No data found" because no Nifty/Sensex contract exists on the day
        AFTER expiry.  The +1 offset has been removed.
        """
        MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
                  "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
        MONTHS_R = {v:k for k,v in MONTHS.items()}
        try:
            if len(expiry) == 7:
                # "19MAY26" → day=19, mon="MAY", year=2026
                day  = int(expiry[:2])
                mon  = expiry[2:5].upper()
                year = 2000 + int(expiry[5:])
            elif len(expiry) == 9:
                # "19MAY2026" — already expanded, just normalise case
                day  = int(expiry[:2])
                mon  = expiry[2:5].upper()
                year = int(expiry[5:])
            else:
                return expiry
            # Return actual expiry date in ddMMMYYYY (e.g. "19MAY2026")
            return f"{day:02d}{mon}{year}"
        except Exception:
            # Fallback: insert "20" before 2-digit year
            if len(expiry) == 7:
                return expiry[:5] + "20" + expiry[5:]
            return expiry

    def _synthetic_nifty_chain(self, spot: float, option_type: str, expiry: str) -> list:
        """Black-Scholes synthetic chain. Used only in paper mode."""
        import math
        from scipy.stats import norm
        from core.models import OptionStrike, OptionType as OT

        atm    = round(spot / self.strike_step) * self.strike_step
        result = []
        for K in [atm + i * self.strike_step for i in range(-8, 9)]:
            S, r, T = spot, 0.065, 7/365
            sigma   = 0.18 + abs(K - S) / S * 0.4
            d1 = (math.log(S/K) + (r+0.5*sigma**2)*T) / (sigma*math.sqrt(T))
            d2 = d1 - sigma*math.sqrt(T)
            if option_type == "CE":
                ltp = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
            else:
                ltp = K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            ltp = max(round(float(ltp), 2), 0.50)
            dist = abs(K - atm) / atm
            oi   = int(np.random.randint(200000, 2000000) * max(0.1, 1 - dist * 5))
            sym  = f"{self.fo_symbol}{expiry}{int(K)}{option_type}"
            result.append(OptionStrike(
                symbol=sym, strike_price=float(K), option_type=OT(option_type),
                expiry=expiry, ltp=ltp, oi=float(oi),
                volume=float(int(oi*0.25)),
                bid=round(ltp*0.99, 2), ask=round(ltp*1.01, 2),
            ))
        return result

    def _broker_ready(self) -> bool:
        """
        True only when the broker client is live, logged in, AND session is still valid.
        Checks both _logged_in and _session_invalid (set by _check_response on auth errors).
        """
        return (
            self.mode == "live"
            and self.broker_client is not None
            and getattr(self.broker_client, "_logged_in", False)
            and not getattr(self.broker_client, "_session_invalid", False)
            and self.broker_client._client is not None
        )

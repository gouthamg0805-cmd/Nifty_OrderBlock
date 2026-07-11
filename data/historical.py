"""
data/historical.py
Downloads real Nifty 50 historical OHLCV data for backtesting.
Uses yfinance (^NSEI) for spot data.

Usage:
    python data/historical.py --days 60 --interval 5m
    python data/historical.py --days 365 --interval 15m

Output: data/cache/nifty_{interval}_{start}_{end}.csv
"""
from __future__ import annotations
import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger


CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def download_nifty(
    days: int = 60,
    interval: str = "5m",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download Nifty 50 OHLCV data.
    yfinance 5m data is limited to last ~60 days.
    For longer periods use 15m or 1h intervals.

    Returns cleaned DataFrame with columns: open, high, low, close, volume
    Index: DatetimeIndex (IST naive)
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days)

    cache_file = os.path.join(
        CACHE_DIR,
        f"nifty_{interval}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
    )

    if os.path.exists(cache_file) and not force_refresh:
        logger.info(f"Loading cached data: {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(df)} bars from cache")
        return df

    logger.info(f"Downloading Nifty {interval} data ({days} days)...")

    # yfinance 5m is limited to 60-day chunks
    if interval in ("1m", "2m", "5m", "15m", "30m", "60m", "90m"):
        chunks = []
        cur = start_dt
        chunk_size = 50 if interval == "5m" else 59
        while cur < end_dt:
            chunk_end = min(cur + timedelta(days=chunk_size), end_dt)
            logger.debug(f"  Fetching {cur.date()} → {chunk_end.date()}")
            try:
                ticker = yf.Ticker("^NSEI")
                chunk  = ticker.history(
                    start    = cur.strftime("%Y-%m-%d"),
                    end      = chunk_end.strftime("%Y-%m-%d"),
                    interval = interval,
                    auto_adjust = True,
                )
                if not chunk.empty:
                    chunks.append(chunk)
            except Exception as e:
                logger.warning(f"  Chunk failed: {e}")
            cur = chunk_end + timedelta(days=1)

        if not chunks:
            raise RuntimeError(
                f"No data downloaded for {interval}. "
                "yfinance 5m is limited to the last 60 days."
            )
        df = pd.concat(chunks)
        df = df[~df.index.duplicated(keep="first")]
    else:
        ticker = yf.Ticker("^NSEI")
        df = ticker.history(
            start    = start_dt.strftime("%Y-%m-%d"),
            end      = end_dt.strftime("%Y-%m-%d"),
            interval = interval,
            auto_adjust = True,
        )

    # Normalise columns
    df = df.rename(columns={
        "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index)

    # Remove timezone info (convert IST → naive)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)

    # Filter to market hours only
    df = _filter_market_hours(df)

    # Fill zero volume with estimated values (^NSEI sometimes has 0)
    if df["volume"].sum() == 0 or (df["volume"] == 0).mean() > 0.5:
        logger.warning("Volume data missing — filling with synthetic estimates")
        df["volume"] = np.random.lognormal(12, 0.5, len(df)).astype(int)

    df.to_csv(cache_file)
    logger.info(
        f"Downloaded {len(df)} bars | "
        f"{df.index[0]} → {df.index[-1]} | "
        f"Saved: {cache_file}"
    )
    return df


def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only NSE trading hours: 09:15 – 15:30."""
    if df.empty or not hasattr(df.index[0], "hour"):
        return df
    mask = (
        (df.index.hour > 9)
        | ((df.index.hour == 9) & (df.index.minute >= 15))
    ) & (
        (df.index.hour < 15)
        | ((df.index.hour == 15) & (df.index.minute <= 30))
    )
    return df[mask]


def load_or_generate(days: int = 60, interval: str = "5m") -> pd.DataFrame:
    """
    Try to download real data; fall back to synthetic if offline.
    """
    try:
        return download_nifty(days=days, interval=interval)
    except Exception as e:
        logger.warning(f"Live data unavailable ({e}). Using synthetic data.")
        from backtest.run_backtest_standalone import generate_nifty_data
        return generate_nifty_data(days=days)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Nifty historical data")
    parser.add_argument("--days",     type=int, default=60,   help="Lookback days")
    parser.add_argument("--interval", type=str, default="5m", help="Bar interval")
    parser.add_argument("--refresh",  action="store_true",    help="Force re-download")
    args = parser.parse_args()

    df = download_nifty(days=args.days, interval=args.interval, force_refresh=args.refresh)
    print(f"\nData summary:")
    print(f"  Bars     : {len(df)}")
    print(f"  From     : {df.index[0]}")
    print(f"  To       : {df.index[-1]}")
    print(f"  Interval : {args.interval}")
    print(f"  Close range: ₹{df['close'].min():,.0f} – ₹{df['close'].max():,.0f}")
    print(df.tail())

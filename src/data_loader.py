# src/data_loader.py
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from src.config import cfg

logger = logging.getLogger(__name__)

def _get_session():
    """Browser-like session – fixes empty/NaN data on GitHub Actions."""
    try:
        from curl_cffi import requests as cffi_requests
        session = cffi_requests.Session(impersonate="chrome")
        return session
    except Exception as e:
        logger.warning(f"curl_cffi not available, falling back to default session: {e}")
        return None


def get_universe_symbols():
    cache = Path(cfg.paths.nifty_cache)
    if cache.exists():
        try:
            df = pd.read_csv(cache)
            symbols = [
                s if str(s).endswith(".NS") else f"{s}.NS"
                for s in df["Symbol"].tolist()
            ]
            logger.info(f"Loaded {len(symbols)} symbols from cache")
            return symbols
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")

    fallback = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
        "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
        "POWERGRID.NS", "NTPC.NS", "TECHM.NS", "HCLTECH.NS", "M&M.NS",
        "TATAMOTORS.NS", "ADANIENT.NS", "JSWSTEEL.NS", "INDUSINDBK.NS", "BAJAJFINSV.NS"
    ]
    logger.warning(f"Using fallback universe ({len(fallback)} symbols)")
    return fallback


def download_history(symbol: str, period: str = "5d", retries: int = 5):
    """
    Reliable history fetch for GitHub Actions.
    Uses curl_cffi Chrome impersonation when available.
    """
    session = None
    try:
        from curl_cffi import requests as cffi_requests
        session = cffi_requests.Session(impersonate="chrome")
    except Exception as e:
        logger.warning(f"curl_cffi session not available: {e}")

    for attempt in range(1, retries + 1):
        try:
            ticker = yf.Ticker(symbol, session=session) if session else yf.Ticker(symbol)
            df = ticker.history(period=period, auto_adjust=True)

            if df is None or df.empty:
                raise ValueError("Empty dataframe")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if "Close" not in df.columns:
                raise ValueError("Close column missing")

            # Drop rows where Close is NaN
            df = df.dropna(subset=["Close"])
            if df.empty:
                raise ValueError("All Close values are NaN")

            return df

        except Exception as e:
            logger.warning(f"{symbol} attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 * attempt)

    return None


def get_actual_ohlc(symbol: str, retries: int = 5):
    """
    Returns dict with actual Open, High, Low, Close + previous close
    or None if failed.
    """
    df = download_history(symbol, period="5d", retries=retries)
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    for col in required:
        if col not in df.columns:
            return None

    last = df.iloc[-1]
    prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else float(last["Close"])

    if pd.isna(last["Close"]) or pd.isna(last["Open"]):
        return None

    return {
        "Open": float(last["Open"]),
        "High": float(last["High"]),
        "Low": float(last["Low"]),
        "Close": float(last["Close"]),
        "prev_close": prev_close
    }

# src/data_loader.py
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from src.config import cfg

logger = logging.getLogger(__name__)

# Better headers to reduce blocking
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def get_universe_symbols():
    cache = Path(cfg.paths.nifty_cache)
    if cache.exists():
        try:
            df = pd.read_csv(cache)
            symbols = [s if str(s).endswith(".NS") else f"{s}.NS" for s in df["Symbol"].tolist()]
            logger.info(f"Loaded {len(symbols)} symbols from cache")
            return symbols
        except Exception as e:
            logger.warning(f"Failed reading cache: {e}")

    # Solid liquid fallback (Nifty 50 style)
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
    More reliable way to get data on GitHub Actions.
    Uses Ticker.history() instead of yf.download() + better retries.
    """
    for attempt in range(1, retries + 1):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, auto_adjust=True)

            if df is None or df.empty:
                raise ValueError("Empty dataframe returned")

            # Fix MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Ensure we have Close
            if "Close" not in df.columns:
                raise ValueError("Close column missing")

            return df

        except Exception as e:
            logger.warning(f"{symbol} attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 * attempt)  # longer backoff

    return None


def get_actual_close(symbol: str, retries: int = 5):
    """Get the latest actual close + previous close."""
    df = download_history(symbol, period="5d", retries=retries)
    if df is None or df.empty:
        return None, None

    actual_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else actual_close
    return actual_close, prev_close

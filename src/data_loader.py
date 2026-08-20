# src/data_loader.py
"""
Multi-source market data loader for Indian stocks.
Primary : yfinance (+ curl_cffi Chrome impersonation)
Fallback: nsepython history → nse_eq live quote
"""

import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from src.config import cfg

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Session helper (reduces Yahoo blocks on GitHub Actions)
# --------------------------------------------------
def _session():
    try:
        from curl_cffi import requests as cffi_requests
        return cffi_requests.Session(impersonate="chrome")
    except Exception:
        return None


# --------------------------------------------------
# yfinance history
# --------------------------------------------------
def download_history(symbol: str, period: str = "6mo", retries: int = 3):
    """
    Download OHLCV history via yfinance.
    Returns a clean DataFrame or None.
    """
    session = _session()

    for attempt in range(1, retries + 1):
        try:
            ticker = yf.Ticker(symbol, session=session) if session else yf.Ticker(symbol)
            df = ticker.history(period=period, auto_adjust=True)

            if df is None or df.empty:
                raise ValueError("empty dataframe")

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if "Close" not in df.columns:
                raise ValueError("Close column missing")

            df = df.dropna(subset=["Close"])
            if len(df) < 15:
                raise ValueError("too few valid rows")

            return df

        except Exception as e:
            logger.debug(f"{symbol} yfinance attempt {attempt}/{retries}: {e}")
            time.sleep(1.2 * attempt)

    return None


# --------------------------------------------------
# nsepython history fallback
# --------------------------------------------------
def _from_nsepython_history(symbol: str, days: int = 12) -> pd.DataFrame | None:
    """
    Fallback historical data from NSE via nsepython.
    Handles missing 'data' key safely.
    """
    try:
        from nsepython import equity_history

        clean = symbol.replace(".NS", "").replace(".BO", "")
        end = datetime.now()
        start = end - timedelta(days=days)

        raw = equity_history(
            clean,
            "EQ",
            start.strftime("%d-%m-%Y"),
            end.strftime("%d-%m-%Y"),
        )

        if raw is None:
            return None

        # Sometimes API returns a dict (error / unexpected payload)
        if isinstance(raw, dict):
            if "data" not in raw or not raw["data"]:
                logger.warning(f"{symbol} nse history response has no 'data'")
                return None
            raw = pd.DataFrame(raw["data"])

        if not isinstance(raw, pd.DataFrame) or raw.empty:
            return None

        # Normalize column names
        colmap = {}
        for c in raw.columns:
            cl = str(c).lower()
            if "open" in cl and "Open" not in colmap.values():
                colmap[c] = "Open"
            elif "high" in cl and "High" not in colmap.values():
                colmap[c] = "High"
            elif "low" in cl and "Low" not in colmap.values():
                colmap[c] = "Low"
            elif "close" in cl and "prev" not in cl and "Close" not in colmap.values():
                colmap[c] = "Close"
            elif "vol" in cl and "Volume" not in colmap.values():
                colmap[c] = "Volume"

        df = raw.rename(columns=colmap)

        for need in ["Open", "High", "Low", "Close"]:
            if need not in df.columns:
                return None

        if "Volume" not in df.columns:
            df["Volume"] = 0

        df = df.dropna(subset=["Close", "Open"])
        return df if not df.empty else None

    except Exception as e:
        logger.warning(f"{symbol} nsepython history failed: {e}")
        return None


# --------------------------------------------------
# nsepython live quote (last resort for same-day OHLC)
# --------------------------------------------------
def _nse_quote_ohlc(symbol: str) -> dict | None:
    """
    Live quote via nse_eq – useful when history APIs fail.
    """
    try:
        from nsepython import nse_eq

        clean = symbol.replace(".NS", "").replace(".BO", "")
        q = nse_eq(clean)

        if not q or not isinstance(q, dict) or "priceInfo" not in q:
            return None

        info = q["priceInfo"]
        open_p = info.get("open") or info.get("openingPrice")
        close_p = info.get("lastPrice") or info.get("close")
        prev = info.get("previousClose") or close_p

        high_p, low_p = None, None
        ih = info.get("intraDayHighLow")
        if isinstance(ih, dict):
            high_p = ih.get("max")
            low_p = ih.get("min")

        if open_p is None or close_p is None:
            return None

        return {
            "Open": float(open_p),
            "High": float(high_p) if high_p is not None else float(close_p),
            "Low": float(low_p) if low_p is not None else float(open_p),
            "Close": float(close_p),
            "prev_close": float(prev) if prev is not None else float(close_p),
            "source": "nse_eq",
        }

    except Exception as e:
        logger.warning(f"{symbol} nse_eq failed: {e}")
        return None


# --------------------------------------------------
# Main function used by evening job
# --------------------------------------------------
def get_actual_ohlc(symbol: str, retries: int = 3) -> dict | None:
    """
    Get actual OHLC with fallback chain:
      1) yfinance
      2) nsepython history
      3) nse_eq live quote
    """
    # 1) yfinance
    df = download_history(symbol, period="5d", retries=retries)
    if df is not None and not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
            last = df.iloc[-1]
            if not pd.isna(last["Close"]) and not pd.isna(last["Open"]):
                prev = (
                    float(df["Close"].iloc[-2])
                    if len(df) >= 2
                    else float(last["Close"])
                )
                return {
                    "Open": float(last["Open"]),
                    "High": float(last["High"]),
                    "Low": float(last["Low"]),
                    "Close": float(last["Close"]),
                    "prev_close": prev,
                    "source": "yfinance",
                }

    # 2) nsepython history
    nse_df = _from_nsepython_history(symbol, days=12)
    if nse_df is not None and not nse_df.empty:
        last = nse_df.iloc[-1]
        prev = (
            float(nse_df["Close"].iloc[-2])
            if len(nse_df) >= 2
            else float(last["Close"])
        )
        logger.info(f"{symbol}: actual via nsepython history")
        return {
            "Open": float(last["Open"]),
            "High": float(last["High"]),
            "Low": float(last["Low"]),
            "Close": float(last["Close"]),
            "prev_close": prev,
            "source": "nsepython",
        }

    # 3) live quote
    quote = _nse_quote_ohlc(symbol)
    if quote is not None:
        logger.info(f"{symbol}: actual via nse_eq quote")
        return quote

    return None


# --------------------------------------------------
# Optional: simple universe helper (prefer src/universe.py)
# --------------------------------------------------
def get_basic_fallback_symbols() -> list[str]:
    """Small liquid fallback list if universe module is unavailable."""
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
        "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "HCLTECH.NS", "M&M.NS",
        "TATAMOTORS.NS", "JSWSTEEL.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS",
    ]

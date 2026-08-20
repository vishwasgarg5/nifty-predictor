import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from src.config import cfg

logger = logging.getLogger(__name__)


def _session():
    try:
        from curl_cffi import requests as cffi_requests
        return cffi_requests.Session(impersonate="chrome")
    except Exception:
        return None


def download_history(symbol: str, period: str = "6mo", retries: int = 3):
    """Primary: yfinance with browser impersonation."""
    session = _session()
    for attempt in range(1, retries + 1):
        try:
            ticker = yf.Ticker(symbol, session=session) if session else yf.Ticker(symbol)
            df = ticker.history(period=period, auto_adjust=True)
            if df is None or df.empty:
                raise ValueError("empty")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close"])
            if len(df) < 20:
                raise ValueError("too few rows")
            return df
        except Exception as e:
            logger.debug(f"{symbol} yfinance {attempt}/{retries}: {e}")
            time.sleep(1.2 * attempt)
    return None


def _from_nsepython(symbol: str, days: int = 15) -> pd.DataFrame | None:
    """Fallback: NSE historical via nsepython."""
    try:
        from nsepython import equity_history
        clean = symbol.replace(".NS", "").replace(".BO", "")
        end = datetime.now()
        start = end - timedelta(days=days)
        raw = equity_history(
            clean, "EQ",
            start.strftime("%d-%m-%Y"),
            end.strftime("%d-%m-%Y")
        )
        if raw is None or raw.empty:
            return None

        # Normalize column names
        colmap = {}
        for c in raw.columns:
            cl = str(c).lower()
            if "open" in cl and "Open" not in colmap:
                colmap[c] = "Open"
            elif "high" in cl and "High" not in colmap:
                colmap[c] = "High"
            elif "low" in cl and "Low" not in colmap:
                colmap[c] = "Low"
            elif "close" in cl and "prev" not in cl and "Close" not in colmap:
                colmap[c] = "Close"
            elif "vol" in cl and "Volume" not in colmap:
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
        logger.warning(f"{symbol} nsepython failed: {e}")
        return None


def get_actual_ohlc(symbol: str, retries: int = 4) -> dict | None:
    """
    Actual OHLC with fallback chain:
    1) yfinance
    2) nsepython
    """
    # --- yfinance ---
    df = download_history(symbol, period="5d", retries=retries)
    if df is not None and not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
            last = df.iloc[-1]
            if not pd.isna(last["Close"]) and not pd.isna(last["Open"]):
                prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else float(last["Close"])
                return {
                    "Open": float(last["Open"]),
                    "High": float(last["High"]),
                    "Low": float(last["Low"]),
                    "Close": float(last["Close"]),
                    "prev_close": prev,
                    "source": "yfinance"
                }

    # --- nsepython fallback ---
    nse_df = _from_nsepython(symbol, days=12)
    if nse_df is not None and not nse_df.empty:
        last = nse_df.iloc[-1]
        prev = float(nse_df["Close"].iloc[-2]) if len(nse_df) >= 2 else float(last["Close"])
        logger.info(f"{symbol}: actual via nsepython")
        return {
            "Open": float(last["Open"]),
            "High": float(last["High"]),
            "Low": float(last["Low"]),
            "Close": float(last["Close"]),
            "prev_close": prev,
            "source": "nsepython"
        }

    return None

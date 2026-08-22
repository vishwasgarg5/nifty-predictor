import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _session():
    try:
        from curl_cffi import requests as cffi_requests
        return cffi_requests.Session(impersonate="chrome")
    except Exception:
        return None


def download_history(symbol: str, period: str = "6mo", retries: int = 3):
    session = _session()
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            t = yf.Ticker(symbol, session=session) if session else yf.Ticker(symbol)
            df = t.history(period=period, auto_adjust=True, timeout=20)
            if df is None or df.empty:
                # try date range as backup
                end = datetime.utcnow()
                start = end - timedelta(days=14)
                df = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), auto_adjust=True)
            if df is None or df.empty:
                raise ValueError("empty history")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close"])
            if len(df) < 1:
                raise ValueError("no rows")
            return df
        except Exception as e:
            last_err = e
            time.sleep(1.0 * attempt)
    logger.debug(f"{symbol} yfinance failed: {last_err}")
    return None


def _nse_quote(symbol: str):
    """Live/last quote via nsepython — good for same-day actuals."""
    if symbol.startswith("^"):
        return None
    try:
        from nsepython import nse_eq
        clean = symbol.replace(".NS", "").replace(".BO", "")
        q = nse_eq(clean)
        if not q or "priceInfo" not in q:
            return None
        info = q["priceInfo"]
        o = info.get("open") or info.get("openingPrice")
        c = info.get("lastPrice") or info.get("close")
        prev = info.get("previousClose") or c
        ih = info.get("intraDayHighLow") or {}
        h = ih.get("max") if isinstance(ih, dict) else None
        l = ih.get("min") if isinstance(ih, dict) else None
        if o is None or c is None:
            return None
        return {
            "Open": float(o),
            "High": float(h if h is not None else c),
            "Low": float(l if l is not None else o),
            "Close": float(c),
            "prev_close": float(prev if prev is not None else c),
            "source": "nse_eq",
        }
    except Exception as e:
        logger.debug(f"{symbol} nse_eq: {e}")
        return None


def _nse_history_quiet(symbol: str, days: int = 10):
    """equity_history often breaks on cloud; keep quiet and short."""
    if symbol.startswith("^"):
        return None
    try:
        import logging as _logging
        # nsepython is very chatty
        _logging.getLogger("nsepython").setLevel(_logging.ERROR)
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
        if isinstance(raw, dict):
            if "data" not in raw or not raw["data"]:
                return None
            raw = pd.DataFrame(raw["data"])
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            return None

        colmap = {}
        for c in raw.columns:
            cl = str(c).lower()
            if "open" in cl:
                colmap[c] = "Open"
            elif "high" in cl:
                colmap[c] = "High"
            elif "low" in cl:
                colmap[c] = "Low"
            elif "close" in cl and "prev" not in cl:
                colmap[c] = "Close"
            elif "vol" in cl:
                colmap[c] = "Volume"
        df = raw.rename(columns=colmap)
        if not all(k in df.columns for k in ("Open", "High", "Low", "Close")):
            return None
        return df.dropna(subset=["Close", "Open"])
    except Exception as e:
        logger.debug(f"{symbol} nse history: {e}")
        return None


def get_actual_ohlc(symbol: str, retries: int = 3):
    """
    Order:
      1) yfinance recent history (most reliable for evening compare)
      2) nse_eq live quote
      3) nsepython equity_history (often fails on GH Actions)
    """
    # 1) Yahoo
    df = download_history(symbol, period="5d", retries=retries)
    if df is not None and not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if all(c in df.columns for c in ("Open", "High", "Low", "Close")):
            last = df.iloc[-1]
            if pd.notna(last["Close"]) and pd.notna(last["Open"]):
                prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else float(last["Close"])
                return {
                    "Open": float(last["Open"]),
                    "High": float(last["High"]),
                    "Low": float(last["Low"]),
                    "Close": float(last["Close"]),
                    "prev_close": prev,
                    "source": "yfinance",
                }

    # 2) NSE quote (same-day)
    quote = _nse_quote(symbol)
    if quote:
        return quote

    # 3) NSE history (quiet)
    nse = _nse_history_quiet(symbol)
    if nse is not None and not nse.empty:
        last = nse.iloc[-1]
        prev = float(nse["Close"].iloc[-2]) if len(nse) >= 2 else float(last["Close"])
        return {
            "Open": float(last["Open"]),
            "High": float(last["High"]),
            "Low": float(last["Low"]),
            "Close": float(last["Close"]),
            "prev_close": prev,
            "source": "nsepython",
        }

    logger.error(f"No actual data for {symbol}")
    return None

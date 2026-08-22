import time, logging
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

def download_history(symbol: str, period="6mo", retries=3):
    session = _session()
    for attempt in range(1, retries + 1):
        try:
            t = yf.Ticker(symbol, session=session) if session else yf.Ticker(symbol)
            df = t.history(period=period, auto_adjust=True)
            if df is None or df.empty:
                raise ValueError("empty")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close"])
            if len(df) < 15:
                raise ValueError("few rows")
            return df
        except Exception as e:
            logger.debug(f"{symbol} yf {attempt}: {e}")
            time.sleep(1.2 * attempt)
    return None

def _nse_history(symbol: str, days=12):
    if symbol.startswith("^"):
        return None
    try:
        from nsepython import equity_history
        clean = symbol.replace(".NS", "").replace(".BO", "")
        end, start = datetime.now(), datetime.now() - timedelta(days=days)
        raw = equity_history(clean, "EQ", start.strftime("%d-%m-%Y"), end.strftime("%d-%m-%Y"))
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
            if "open" in cl: colmap[c] = "Open"
            elif "high" in cl: colmap[c] = "High"
            elif "low" in cl: colmap[c] = "Low"
            elif "close" in cl and "prev" not in cl: colmap[c] = "Close"
            elif "vol" in cl: colmap[c] = "Volume"
        df = raw.rename(columns=colmap)
        if not all(k in df.columns for k in ["Open", "High", "Low", "Close"]):
            return None
        if "Volume" not in df.columns:
            df["Volume"] = 0
        return df.dropna(subset=["Close", "Open"])
    except Exception as e:
        logger.warning(f"{symbol} nse hist: {e}")
        return None

def _nse_quote(symbol: str):
    if symbol.startswith("^"):
        return None
    try:
        from nsepython import nse_eq
        q = nse_eq(symbol.replace(".NS", "").replace(".BO", ""))
        if not q or "priceInfo" not in q:
            return None
        info = q["priceInfo"]
        o = info.get("open") or info.get("openingPrice")
        c = info.get("lastPrice") or info.get("close")
        prev = info.get("previousClose") or c
        ih = info.get("intraDayHighLow") or {}
        h, l = ih.get("max"), ih.get("min")
        if o is None or c is None:
            return None
        return {
            "Open": float(o),
            "High": float(h or c),
            "Low": float(l or o),
            "Close": float(c),
            "prev_close": float(prev or c),
            "source": "nse_eq",
        }
    except Exception as e:
        logger.warning(f"{symbol} nse_eq: {e}")
        return None

def get_actual_ohlc(symbol: str, retries=3):
    df = download_history(symbol, period="5d", retries=retries)
    if df is not None and not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
            last = df.iloc[-1]
            if not pd.isna(last["Close"]) and not pd.isna(last["Open"]):
                prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else float(last["Close"])
                return {
                    "Open": float(last["Open"]), "High": float(last["High"]),
                    "Low": float(last["Low"]), "Close": float(last["Close"]),
                    "prev_close": prev, "source": "yfinance",
                }
    nse = _nse_history(symbol)
    if nse is not None and not nse.empty:
        last = nse.iloc[-1]
        prev = float(nse["Close"].iloc[-2]) if len(nse) >= 2 else float(last["Close"])
        return {
            "Open": float(last["Open"]), "High": float(last["High"]),
            "Low": float(last["Low"]), "Close": float(last["Close"]),
            "prev_close": prev, "source": "nsepython",
        }
    return _nse_quote(symbol)

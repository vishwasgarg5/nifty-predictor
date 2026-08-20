import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from src.config import cfg

logger = logging.getLogger(__name__)

NIFTY500_FALLBACK = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","HINDUNILVR.NS",
    "ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS","LT.NS","AXISBANK.NS",
    "BAJFINANCE.NS","ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS","TITAN.NS","WIPRO.NS",
    "ULTRACEMCO.NS","NESTLEIND.NS","POWERGRID.NS","NTPC.NS","TECHM.NS","HCLTECH.NS",
    "M&M.NS","TATAMOTORS.NS","ADANIENT.NS","JSWSTEEL.NS","INDUSINDBK.NS","BAJAJFINSV.NS",
    "ONGC.NS","COALINDIA.NS","BPCL.NS","HINDALCO.NS","GRASIM.NS","DIVISLAB.NS",
    "CIPLA.NS","DRREDDY.NS","EICHERMOT.NS","HEROMOTOCO.NS","BRITANNIA.NS","APOLLOHOSP.NS",
    "ADANIPORTS.NS","TATASTEEL.NS","SBILIFE.NS","HDFCLIFE.NS","BAJAJ-AUTO.NS","PIDILITIND.NS"
]

def _session():
    try:
        from curl_cffi import requests as cffi_requests
        return cffi_requests.Session(impersonate="chrome")
    except Exception:
        return None

def get_universe_symbols():
    cache = Path(cfg.paths.nifty_cache)
    if cache.exists():
        try:
            df = pd.read_csv(cache)
            syms = [s if str(s).endswith(".NS") else f"{s}.NS" for s in df["Symbol"].tolist()]
            if len(syms) > 50:
                logger.info(f"Loaded {len(syms)} symbols from Nifty 500 cache")
                return syms
        except Exception as e:
            logger.warning(f"Cache failed: {e}")

    # Try official CSV
    try:
        url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
        df = pd.read_csv(url)
        syms = [s + ".NS" for s in df["Symbol"].tolist()]
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        logger.info(f"Fetched live Nifty 500 list: {len(syms)}")
        return syms
    except Exception as e:
        logger.warning(f"Live Nifty 500 fetch failed: {e}")

    logger.warning(f"Using fallback list ({len(NIFTY500_FALLBACK)})")
    return NIFTY500_FALLBACK

def download_history(symbol: str, period: str = "6mo", retries: int = 3):
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
            if len(df) < 30:
                raise ValueError("too few rows")
            return df
        except Exception as e:
            logger.debug(f"{symbol} attempt {attempt}: {e}")
            time.sleep(1.2 * attempt)
    return None

def get_actual_ohlc(symbol: str, retries: int = 4):
    df = download_history(symbol, period="5d", retries=retries)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            return None
    last = df.iloc[-1]
    if pd.isna(last["Close"]) or pd.isna(last["Open"]):
        return None
    prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else float(last["Close"])
    return {
        "Open": float(last["Open"]),
        "High": float(last["High"]),
        "Low": float(last["Low"]),
        "Close": float(last["Close"]),
        "prev_close": prev
    }

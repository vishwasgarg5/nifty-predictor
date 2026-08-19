import time
import logging
import pandas as pd
import yfinance as yf
from pathlib import Path
from src.config import cfg

logger = logging.getLogger(__name__)

def get_universe_symbols():
    cache = Path(cfg.paths.nifty_cache)
    if cache.exists():
        df = pd.read_csv(cache)
        return [s if s.endswith(".NS") else s + ".NS" for s in df["Symbol"].tolist()]
    
    # Fallback static list (you should replace with real Nifty 100 list)
    logger.warning("Using fallback universe list")
    return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS"]

def download_history(symbol: str, period="1y", retries=3):
    for attempt in range(retries):
        try:
            df = yf.download(symbol, period=period, auto_adjust=True, progress=False, threads=False)
            if not df.empty and len(df) > 30:
                df = df.dropna()
                return df
        except Exception as e:
            logger.warning(f"{symbol} attempt {attempt+1} failed: {e}")
            time.sleep(1.5 * (attempt + 1))
    return None

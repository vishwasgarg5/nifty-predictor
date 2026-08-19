# src/data_loader.py
import time
import logging
import pandas as pd
import yfinance as yf
from pathlib import Path
from src.config import cfg

logger = logging.getLogger(__name__)

def get_universe_symbols():
    """Return list of symbols. Prefer cache, otherwise use a solid Nifty-100 style list."""
    cache = Path(cfg.paths.nifty_cache)
    
    if cache.exists():
        try:
            df = pd.read_csv(cache)
            symbols = [s if str(s).endswith(".NS") else str(s) + ".NS" for s in df["Symbol"].tolist()]
            logger.info(f"Loaded {len(symbols)} symbols from cache")
            return symbols
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")

    # Strong fallback list (liquid Nifty stocks)
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


def download_history(symbol: str, period: str = "1y", retries: int = 4):
    """Robust download with retries and longer sleep."""
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                symbol,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=20
            )
            
            if df is not None and not df.empty and len(df) > 40:
                # Flatten multi-index columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna(how="all")
                return df
                
        except Exception as e:
            logger.warning(f"{symbol} download attempt {attempt}/{retries} failed: {e}")
        
        time.sleep(1.5 * attempt)  # increasing delay
    
    logger.error(f"{symbol} → All download attempts failed")
    return None

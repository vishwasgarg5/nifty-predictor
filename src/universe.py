# src/universe.py
import logging
from pathlib import Path
import pandas as pd
from src.config import cfg
from src.ipo import filter_eligible_ipos

logger = logging.getLogger(__name__)

_FALLBACKS = {
    "nifty50": [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
        "HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
        "LT.NS","AXISBANK.NS","BAJFINANCE.NS","ASIANPAINT.NS","MARUTI.NS"
    ],
    "nifty100": [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
        "HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
        "LT.NS","AXISBANK.NS","BAJFINANCE.NS","ASIANPAINT.NS","MARUTI.NS",
        "SUNPHARMA.NS","TITAN.NS","WIPRO.NS","ULTRACEMCO.NS","NESTLEIND.NS",
        "POWERGRID.NS","NTPC.NS","TECHM.NS","HCLTECH.NS","M&M.NS",
        "TATAMOTORS.NS","ADANIENT.NS","JSWSTEEL.NS","INDUSINDBK.NS","BAJAJFINSV.NS",
        "ONGC.NS","COALINDIA.NS","GRASIM.NS","BRITANNIA.NS","CIPLA.NS"
    ],
    "nifty500": [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
        "HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
        "LT.NS","AXISBANK.NS","BAJFINANCE.NS","ASIANPAINT.NS","MARUTI.NS",
        "SUNPHARMA.NS","TITAN.NS","WIPRO.NS","ULTRACEMCO.NS","NESTLEIND.NS",
        "POWERGRID.NS","NTPC.NS","TECHM.NS","HCLTECH.NS","M&M.NS",
        "TATAMOTORS.NS","ADANIENT.NS","JSWSTEEL.NS","INDUSINDBK.NS","BAJAJFINSV.NS",
        "ONGC.NS","COALINDIA.NS","GRASIM.NS","BRITANNIA.NS","CIPLA.NS",
        "DRREDDY.NS","EICHERMOT.NS","HEROMOTOCO.NS","APOLLOHOSP.NS","ADANIPORTS.NS",
        "TATASTEEL.NS","SBILIFE.NS","HDFCLIFE.NS","BAJAJ-AUTO.NS","PIDILITIND.NS"
    ],
}


def _load_universe_file(name: str) -> list[str]:
    path = Path(cfg.paths.universes_dir) / f"{name}.csv"
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        return [s if str(s).endswith(".NS") else f"{s}.NS" for s in df[col].tolist()]
    except Exception as e:
        logger.warning(f"Failed reading {path}: {e}")
        return []


def get_universe_symbols(names: list[str] | None = None) -> list[str]:
    """Merge multiple universes + optional eligible IPOs."""
    if names is None:
        primary = list(getattr(cfg.universes, "primary", ["nifty100"]))
        secondary = list(getattr(cfg.universes, "secondary", []))
        names = primary + secondary

    symbols = set()

    for name in names:
        name = str(name).lower().strip()
        file_syms = _load_universe_file(name)
        if file_syms:
            symbols.update(file_syms)
            logger.info(f"Universe {name}: {len(file_syms)} from file")
        elif name in _FALLBACKS:
            fb = _FALLBACKS[name]
            symbols.update(fb)
            logger.info(f"Universe {name}: {len(fb)} fallback")
        else:
            logger.warning(f"Universe {name}: no file and no fallback")

    # IPO
    if getattr(cfg.universes, "include_ipo", False):
        try:
            ipos = filter_eligible_ipos()
            if ipos:
                symbols.update(ipos)
                logger.info(f"Added {len(ipos)} eligible IPO stocks")
        except Exception as e:
            logger.warning(f"IPO filter failed: {e}")

    result = sorted(symbols)
    logger.info(f"Total unique symbols: {len(result)}")
    return result

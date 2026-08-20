import logging
from pathlib import Path
import pandas as pd
from src.config import cfg
from src.ipo import load_ipo_watchlist, filter_eligible_ipos

logger = logging.getLogger(__name__)

# Minimal built-in fallbacks
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
        "POWERGRID.NS","NTPC.NS","TECHM.NS","HCLTECH.NS","M&M.NS"
    ],
    "nifty500": None,  # use file or extended fallback
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
    """
    Merge multiple universes + optional IPO watchlist.
    """
    names = names or list(cfg.universes.primary) + list(getattr(cfg.universes, "secondary", []))
    symbols = set()

    for name in names:
        name = str(name).lower()
        file_syms = _load_universe_file(name)
        if file_syms:
            symbols.update(file_syms)
            logger.info(f"Universe {name}: {len(file_syms)} from file")
        elif name in _FALLBACKS and _FALLBACKS[name]:
            symbols.update(_FALLBACKS[name])
            logger.info(f"Universe {name}: {len(_FALLBACKS[name])} fallback")
        else:
            # try nifty500 style extended fallback via data_loader if needed
            from src.data_loader import get_universe_symbols as legacy
            # only once
            pass

    # IPO eligible
    if getattr(cfg.universes, "include_ipo", False):
        ipos = filter_eligible_ipos()
        if ipos:
            symbols.update(ipos)
            logger.info(f"Added {len(ipos)} eligible IPO stocks")

    result = sorted(symbols)
    logger.info(f"Total unique symbols: {len(result)}")
    return result

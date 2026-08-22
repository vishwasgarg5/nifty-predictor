import logging
from pathlib import Path
import pandas as pd
from src.config import cfg
from src.ipo import filter_eligible_ipos

logger = logging.getLogger(__name__)
_FALLBACKS = {
    "nifty100": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"],
    "nifty500": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"],
}

def _load_file(name):
    path = Path(cfg.paths.universes_dir) / f"{name}.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    return [s if str(s).endswith(".NS") else f"{s}.NS" for s in df[col].tolist()]

def get_universe_symbols(names=None):
    if names is None:
        if hasattr(cfg, "universes"):
            names = list(getattr(cfg.universes, "primary", [])) + list(getattr(cfg.universes, "secondary", []))
        else:
            names = ["nifty100"]
    symbols = set()
    for name in names:
        name = str(name).lower()
        syms = _load_file(name)
        if syms:
            symbols.update(syms)
            logger.info(f"Universe {name}: {len(syms)} from file")
        elif name in _FALLBACKS:
            symbols.update(_FALLBACKS[name])
            logger.info(f"Universe {name}: fallback")
    if hasattr(cfg, "universes") and getattr(cfg.universes, "include_ipo", False):
        try:
            symbols.update(filter_eligible_ipos())
        except Exception as e:
            logger.warning(e)
    result = sorted(symbols)
    logger.info(f"Total unique symbols: {len(result)}")
    return result

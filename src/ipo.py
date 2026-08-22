import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from src.config import cfg
from src.data_loader import download_history

logger = logging.getLogger(__name__)

def load_ipo_watchlist() -> pd.DataFrame:
    path = Path(cfg.paths.ipo_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        df = pd.DataFrame(columns=["Symbol", "ListingDate", "Name", "Status"])
        df.to_csv(path, index=False)
        return df
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["Symbol", "ListingDate", "Name", "Status"])

def filter_eligible_ipos() -> list:
    df = load_ipo_watchlist()
    if df.empty:
        return []
    min_days = int(getattr(cfg.universes, "ipo_min_trading_days", 30))
    min_vol = int(getattr(cfg.universes, "ipo_min_avg_volume", 100000))
    out = []
    for _, row in df.iterrows():
        sym = str(row["Symbol"])
        if not sym.endswith(".NS"):
            sym += ".NS"
        try:
            age = (datetime.now() - pd.to_datetime(row["ListingDate"]).to_pydatetime()).days
            if age < min_days:
                continue
        except Exception:
            pass
        hist = download_history(sym, period="3mo", retries=2)
        if hist is None or len(hist) < min_days:
            continue
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        avg_vol = hist["Volume"].tail(20).mean() if "Volume" in hist.columns else 0
        if avg_vol >= min_vol:
            out.append(sym)
    return out

def ipo_watchlist_telegram_lines(max_rows: int = 8) -> list:
    df = load_ipo_watchlist()
    if df.empty:
        return []
    eligible = set(filter_eligible_ipos())
    lines = ["*LISTED IPO WATCHLIST*"]
    for _, row in df.head(max_rows).iterrows():
        sym = str(row.get("Symbol", ""))
        key = sym if sym.endswith(".NS") else (sym + ".NS" if sym else "")
        st = "ACTIVE" if key in eligible else "WATCH"
        listing = str(row.get("ListingDate", "") or "")[:10]
        clean = sym.replace(".NS", "")
        lines.append(f"• `{clean}` | {st} | {listing}")
    return lines

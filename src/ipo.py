import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from src.config import cfg
from src.data_loader import download_history

logger = logging.getLogger(__name__)


def load_ipo_watchlist() -> pd.DataFrame:
    path = Path(cfg.paths.ipo_file)
    if not path.exists():
        # create empty template
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(columns=["Symbol", "ListingDate", "Name", "Status"])
        df.to_csv(path, index=False)
        return df
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.warning(f"IPO file read failed: {e}")
        return pd.DataFrame(columns=["Symbol", "ListingDate", "Name", "Status"])


def save_ipo_watchlist(df: pd.DataFrame):
    path = Path(cfg.paths.ipo_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def add_ipo(symbol: str, listing_date: str, name: str = ""):
    """Manually or automatically add an IPO."""
    df = load_ipo_watchlist()
    sym = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    if sym in df["Symbol"].astype(str).tolist():
        return
    new = pd.DataFrame([{
        "Symbol": sym,
        "ListingDate": listing_date,  # YYYY-MM-DD
        "Name": name,
        "Status": "watch"
    }])
    df = pd.concat([df, new], ignore_index=True)
    save_ipo_watchlist(df)
    logger.info(f"IPO added: {sym}")


def filter_eligible_ipos() -> list[str]:
    """
    Return IPO symbols that have enough trading history + liquidity.
    """
    df = load_ipo_watchlist()
    if df.empty:
        return []

    min_days = int(getattr(cfg.universes, "ipo_min_trading_days", 30))
    min_vol = int(getattr(cfg.universes, "ipo_min_avg_volume", 100000))
    eligible = []

    for _, row in df.iterrows():
        sym = str(row["Symbol"])
        if not sym.endswith(".NS"):
            sym = sym + ".NS"
        try:
            listing = pd.to_datetime(row.get("ListingDate"))
            age = (datetime.now() - listing.to_pydatetime()).days
            if age < min_days:
                continue
        except Exception:
            # if no listing date, still check history length
            pass

        hist = download_history(sym, period="3mo", retries=2)
        if hist is None or len(hist) < min_days:
            continue
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        avg_vol = hist["Volume"].tail(20).mean() if "Volume" in hist.columns else 0
        if avg_vol < min_vol:
            continue
        eligible.append(sym)

    logger.info(f"Eligible IPOs: {len(eligible)}")
    return eligible


def refresh_ipo_status():
    """Mark IPOs as active/expired based on age."""
    df = load_ipo_watchlist()
    if df.empty:
        return
    min_days = int(getattr(cfg.universes, "ipo_min_trading_days", 30))
    statuses = []
    for _, row in df.iterrows():
        try:
            age = (datetime.now() - pd.to_datetime(row["ListingDate"]).to_pydatetime()).days
            statuses.append("active" if age >= min_days else "watch")
        except Exception:
            statuses.append(row.get("Status", "watch"))
    df["Status"] = statuses
    save_ipo_watchlist(df)

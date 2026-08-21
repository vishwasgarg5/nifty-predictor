#!/usr/bin/env python3
"""
Build / refresh Nifty universe CSV files.
Sources (in order):
  1) Official Nifty Indices CSV (if reachable)
  2) Keep existing file if download fails
  3) Write a clear log
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import pandas as pd
import requests
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

UNIVERSE_DIR = Path("data/universes")
UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)

# Official / public sources (may change; script handles failure safely)
SOURCES = {
    "nifty50": [
        "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
        "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    ],
    "nifty100": [
        "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
        "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    ],
    "nifty500": [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    ],
    "midcap150": [
        "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
        "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
    ],
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/csv,text/plain,*/*",
}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize to Symbol, Company Name, Industry."""
    cols = {c.lower().strip(): c for c in df.columns}

    symbol_col = None
    for key in ["symbol", "symbols"]:
        if key in cols:
            symbol_col = cols[key]
            break
    if symbol_col is None:
        symbol_col = df.columns[0]

    name_col = None
    for key in ["company name", "company", "name"]:
        if key in cols:
            name_col = cols[key]
            break

    industry_col = None
    for key in ["industry", "sector"]:
        if key in cols:
            industry_col = cols[key]
            break

    out = pd.DataFrame()
    out["Symbol"] = (
        df[symbol_col]
        .astype(str)
        .str.strip()
        .str.replace(".NS", "", regex=False)
        .str.replace(".BO", "", regex=False)
        .str.upper()
    )
    out["Company Name"] = df[name_col].astype(str).str.strip() if name_col else ""
    out["Industry"] = df[industry_col].astype(str).str.strip() if industry_col else ""

    out = out[out["Symbol"].str.len() > 0]
    out = out.drop_duplicates(subset=["Symbol"]).reset_index(drop=True)
    return out


def download_universe(name: str) -> pd.DataFrame | None:
    urls = SOURCES.get(name, [])
    for url in urls:
        try:
            logger.info(f"Trying {name}: {url}")
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code != 200:
                logger.warning(f"HTTP {r.status_code} for {url}")
                continue
            # Parse CSV from text
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            if df.empty:
                continue
            norm = _normalize(df)
            if len(norm) < 10:
                logger.warning(f"Too few rows ({len(norm)}) from {url}")
                continue
            logger.info(f"{name}: downloaded {len(norm)} symbols")
            return norm
        except Exception as e:
            logger.warning(f"Failed {url}: {e}")
    return None


def save_universe(name: str, df: pd.DataFrame):
    path = UNIVERSE_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved {path} ({len(df)} rows)")


def refresh_all(names: list[str] | None = None):
    names = names or list(SOURCES.keys())
    summary = []

    for name in names:
        path = UNIVERSE_DIR / f"{name}.csv"
        df = download_universe(name)

        if df is not None:
            save_universe(name, df)
            summary.append(f"{name}: {len(df)} symbols (updated)")
        elif path.exists():
            old = pd.read_csv(path)
            summary.append(f"{name}: download failed, kept existing ({len(old)} symbols)")
            logger.warning(f"{name}: kept existing file")
        else:
            summary.append(f"{name}: FAILED (no file, no download)")
            logger.error(f"{name}: no data available")

    return summary


def main():
    logger.info("=" * 50)
    logger.info(f"Universe refresh started | {datetime.now():%Y-%m-%d %H:%M}")
    summary = refresh_all(["nifty50", "nifty100", "nifty500", "midcap150"])
    logger.info("Summary:")
    for line in summary:
        logger.info(f"  {line}")
    logger.info("Universe refresh finished")
    logger.info("=" * 50)

    # Optional Telegram notify
    try:
        from src.telegram_utils import send_telegram
        msg = "*Universe lists refreshed*\n\n" + "\n".join(f"• {s}" for s in summary)
        send_telegram(msg)
    except Exception:
        pass


if __name__ == "__main__":
    main()

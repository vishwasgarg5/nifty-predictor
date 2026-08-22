"""Auto IPO + GMP via InvestorGain tables (best-effort)."""
from __future__ import annotations
import logging
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests
from src.config import cfg

logger = logging.getLogger(__name__)
PIPELINE = Path(getattr(getattr(cfg, "paths", None), "ipo_pipeline", "data/ipo_pipeline.csv"))
COLS = ["Name", "Symbol", "OpenDate", "CloseDate", "PriceLow", "PriceHigh", "GMP", "LotSize", "Status", "UpdatedAt"]

def _empty():
    return pd.DataFrame(columns=COLS)

def load_ipo_pipeline() -> pd.DataFrame:
    if not PIPELINE.exists():
        PIPELINE.parent.mkdir(parents=True, exist_ok=True)
        df = _empty()
        df.to_csv(PIPELINE, index=False)
        return df
    try:
        return pd.read_csv(PIPELINE)
    except Exception:
        return _empty()

def _verdict(gmp_pct: float) -> str:
    if gmp_pct >= 20:
        return "Positive"
    if gmp_pct >= 0:
        return "Neutral"
    return "Weak"

def _to_float(x, default=0.0):
    try:
        s = str(x).replace(",", "").replace("₹", "").replace("%", "").strip()
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group()) if m else default
    except Exception:
        return default

def _scrape_investorgain() -> pd.DataFrame:
    """
    InvestorGain live GMP page often has HTML tables.
    GMP is unofficial — for decision support only.
    """
    url = "https://www.investorgain.com/report/live-ipo-gmp/331/ipo/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        # IMPORTANT: pass text to read_html, never a wrong path
        tables = pd.read_html(r.text)
    except Exception as e:
        logger.warning(f"InvestorGain fetch/parse failed: {e}")
        return _empty()

    rows = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        # flexible column find
        name_col = next((c for c in t.columns if any(k in str(c).lower() for k in ("name", "ipo", "company"))), None)
        gmp_col = next((c for c in t.columns if "gmp" in str(c).lower()), None)
        price_col = next((c for c in t.columns if "price" in str(c).lower()), None)
        open_col = next((c for c in t.columns if str(c).lower().startswith("open")), None)
        close_col = next((c for c in t.columns if "close" in str(c).lower()), None)
        lot_col = next((c for c in t.columns if "lot" in str(c).lower()), None)
        if name_col is None:
            continue

        for _, row in t.iterrows():
            name = str(row[name_col]).strip()
            if not name or name.lower() in ("nan", "name", "ipo"):
                continue
            gmp = _to_float(row[gmp_col], 0) if gmp_col is not None else 0
            price = _to_float(row[price_col], 0) if price_col is not None else 0
            # if price band like 140-148, take high
            if price_col is not None and "-" in str(row[price_col]):
                parts = re.findall(r"\d+(?:\.\d+)?", str(row[price_col]).replace(",", ""))
                high = float(parts[-1]) if parts else price
                low = float(parts[0]) if parts else ""
            else:
                high, low = price, ""

            status = "open"
            rows.append({
                "Name": name[:60],
                "Symbol": "",
                "OpenDate": str(row[open_col])[:12] if open_col is not None else "",
                "CloseDate": str(row[close_col])[:12] if close_col is not None else "",
                "PriceLow": low,
                "PriceHigh": high,
                "GMP": gmp,
                "LotSize": _to_float(row[lot_col], 0) if lot_col is not None else "",
                "Status": status,
                "UpdatedAt": now,
            })

    if not rows:
        logger.warning("InvestorGain: no rows parsed from tables")
        return _empty()
    return pd.DataFrame(rows)

def refresh_ipo_gmp() -> dict:
    old = load_ipo_pipeline()
    scraped = _scrape_investorgain()

    if scraped.empty:
        logger.warning("IPO scrape empty — keeping previous pipeline")
        return {"updated": 0, "kept": len(old), "source": "previous"}

    PIPELINE.parent.mkdir(parents=True, exist_ok=True)
    scraped.to_csv(PIPELINE, index=False)
    logger.info(f"IPO pipeline updated: {len(scraped)} rows (investorgain)")
    return {"updated": len(scraped), "kept": 0, "source": "investorgain"}

def build_ipo_desk_message(max_rows: int = 10) -> str:
    df = load_ipo_pipeline()
    if df.empty:
        return "*IPO DESK*\nNo IPO data yet (auto-scrape pending)."

    lines = [
        "*IPO DESK – Auto GMP*",
        f"Updated: `{datetime.now():%Y-%m-%d %H:%M}`",
        "_GMP is unofficial grey-market data – not advice_",
        "",
        "```",
        f"{'IPO':<20} {'Band':>10} {'GMP':>6} {'%':>6} {'Verdict':>8}",
        "-" * 56,
    ]
    shown = 0
    for _, r in df.iterrows():
        if shown >= max_rows:
            break
        name = str(r.get("Name", ""))[:20]
        try:
            high = float(r.get("PriceHigh") or 0)
        except Exception:
            high = 0.0
        try:
            gmp = float(r.get("GMP") or 0)
        except Exception:
            gmp = 0.0
        pct = (gmp / high * 100) if high else 0.0
        band = f"{r.get('PriceLow','')}-{r.get('PriceHigh','')}"
        lines.append(f"{name:<20} {str(band)[:10]:>10} {gmp:>6.0f} {pct:>5.1f}% {_verdict(pct):>8}")
        shown += 1
    lines.append("```")
    return "\n".join(lines)

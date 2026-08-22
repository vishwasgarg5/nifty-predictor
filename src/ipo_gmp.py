"""Auto-update upcoming IPOs + GMP (best-effort scrapers)."""
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

def _scrape_chittorgarh() -> pd.DataFrame:
    """
    Best-effort: Chittorgarh IPO calendar / GMP pages change often.
    If this breaks, refresh_ipo_gmp keeps last CSV.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    rows = []
    # Main IPO list page (HTML table). May need adjustment if site changes.
    urls = [
        "https://www.chittorgarh.com/report/main-board-ipo-list-in-india-bse-nse/82/",
        "https://www.chittorgarh.com/report/ipo-gmp-grey-market-premium/448/",
    ]
    session = requests.Session()
    session.headers.update(headers)

    # GMP map from GMP page
    gmp_map = {}
    try:
        r = session.get(urls[1], timeout=25)
        if r.status_code == 200:
            tables = pd.read_html(r.text)
            for t in tables:
                cols = [str(c).lower() for c in t.columns]
                # try find name + gmp columns
                name_col = next((c for c in t.columns if "ipo" in str(c).lower() or "name" in str(c).lower()), None)
                gmp_col = next((c for c in t.columns if "gmp" in str(c).lower()), None)
                if name_col is None or gmp_col is None:
                    continue
                for _, row in t.iterrows():
                    name = str(row[name_col]).strip()
                    gmp_raw = str(row[gmp_col])
                    m = re.search(r"-?\d+", gmp_raw.replace(",", ""))
                    if name and m:
                        gmp_map[name.lower()[:40]] = float(m.group())
    except Exception as e:
        logger.warning(f"GMP page scrape failed: {e}")

    try:
        r = session.get(urls[0], timeout=25)
        if r.status_code != 200:
            return _empty()
        tables = pd.read_html(r.text)
        for t in tables:
            # flexible column detection
            lower = {str(c).lower(): c for c in t.columns}
            name_c = lower.get("issuer company") or lower.get("company name") or lower.get("ipo name")
            if not name_c:
                # pick first object-like column
                name_c = t.columns[0]
            for _, row in t.iterrows():
                name = str(row[name_c]).strip()
                if not name or name.lower() == "nan":
                    continue
                # try extract band / dates if columns exist
                band = ""
                for k, c in lower.items():
                    if "price" in k and "band" in k:
                        band = str(row[c])
                low = high = None
                m = re.findall(r"\d+", band.replace(",", ""))
                if len(m) >= 2:
                    low, high = float(m[0]), float(m[1])
                gmp = 0.0
                for key, val in gmp_map.items():
                    if key[:15] in name.lower() or name.lower()[:15] in key:
                        gmp = val
                        break
                rows.append({
                    "Name": name[:60],
                    "Symbol": "",
                    "OpenDate": "",
                    "CloseDate": "",
                    "PriceLow": low if low is not None else "",
                    "PriceHigh": high if high is not None else "",
                    "GMP": gmp,
                    "LotSize": "",
                    "Status": "upcoming",
                    "UpdatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
                })
        return pd.DataFrame(rows) if rows else _empty()
    except Exception as e:
        logger.warning(f"IPO list scrape failed: {e}")
        return _empty()

def refresh_ipo_gmp() -> dict:
    """
    Auto-update IPO pipeline.
    Returns stats dict.
    """
    old = load_ipo_pipeline()
    scraped = _scrape_chittorgarh()

    if scraped.empty:
        logger.warning("IPO scrape empty — keeping previous pipeline")
        return {"updated": 0, "kept": len(old), "source": "previous"}

    # merge: prefer new names, keep old GMP if new GMP is 0 and old had value
    if not old.empty and "Name" in old.columns:
        old_map = {str(n).lower(): o for n, o in zip(old.get("Name", []), old.get("GMP", []))}
        gmp_vals = []
        for _, r in scraped.iterrows():
            g = float(r.get("GMP") or 0)
            if g == 0:
                g = float(old_map.get(str(r["Name"]).lower(), 0) or 0)
            gmp_vals.append(g)
        scraped["GMP"] = gmp_vals

    PIPELINE.parent.mkdir(parents=True, exist_ok=True)
    scraped.to_csv(PIPELINE, index=False)
    logger.info(f"IPO pipeline updated: {len(scraped)} rows")
    return {"updated": len(scraped), "kept": 0, "source": "scrape"}

def build_ipo_desk_message(max_rows: int = 10) -> str:
    df = load_ipo_pipeline()
    if df.empty:
        return "*IPO DESK*\nNo IPO data yet (auto-scrape pending)."

    lines = [
        "*IPO DESK – Auto GMP*",
        f"Updated: `{datetime.now():%Y-%m-%d %H:%M}`",
        "_GMP is unofficial; not investment advice_",
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
            high = 0
        try:
            gmp = float(r.get("GMP") or 0)
        except Exception:
            gmp = 0
        pct = (gmp / high * 100) if high else 0
        band = f"{r.get('PriceLow','')}-{r.get('PriceHigh','')}"
        lines.append(f"{name:<20} {band:>10} {gmp:>6.0f} {pct:>5.1f}% {_verdict(pct):>8}")
        shown += 1
    lines.append("```")
    return "\n".join(lines)

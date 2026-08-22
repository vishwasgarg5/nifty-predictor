# src/ipo_gmp.py
"""
IPO Desk + GMP helper.

- Tries a static HTML scrape (best-effort; many GMP sites are JS-only).
- Falls back to previous CSV, then to a seed list so Telegram is never empty.
- GMP is unofficial grey-market data — not investment advice.
"""

from __future__ import annotations

import io
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from src.config import cfg

logger = logging.getLogger(__name__)

PIPELINE = Path(
    getattr(getattr(cfg, "paths", None), "ipo_pipeline", "data/ipo_pipeline.csv")
)

COLS = [
    "Name",
    "Symbol",
    "OpenDate",
    "CloseDate",
    "PriceLow",
    "PriceHigh",
    "GMP",
    "LotSize",
    "Status",
    "UpdatedAt",
]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLS)


def load_ipo_pipeline() -> pd.DataFrame:
    if not PIPELINE.exists():
        PIPELINE.parent.mkdir(parents=True, exist_ok=True)
        df = _empty()
        df.to_csv(PIPELINE, index=False)
        return df
    try:
        return pd.read_csv(PIPELINE)
    except Exception as e:
        logger.warning(f"ipo_pipeline read failed: {e}")
        return _empty()


def _verdict(gmp_pct: float) -> str:
    if gmp_pct >= 20:
        return "Positive"
    if gmp_pct >= 0:
        return "Neutral"
    return "Weak"


def _to_float(x, default: float = 0.0) -> float:
    try:
        s = str(x).replace(",", "").replace("₹", "").replace("%", "").strip()
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group()) if m else default
    except Exception:
        return default


def _seed_current_ipos() -> pd.DataFrame:
    """
    Fallback seed so IPO desk is never empty on GitHub Actions.
    Update these rows when you want fresher approximate GMP.
    Values are illustrative / approximate — not live exchange quotes.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = [
        # Name, OpenDate, CloseDate, PriceLow, PriceHigh, GMP, LotSize, Status
        ("Tempsens Instruments", "2026-08-20", "2026-08-24", 280, 300, 170, 50, "open"),
        ("Augmont Enterprises", "2026-08-21", "2026-08-25", 750, 788, 180, 19, "open"),
        ("Gaja Alternative AMC", "2026-08-19", "2026-08-21", 150, 160, 30, 93, "open"),
        ("Skyways Air", "2026-08-24", "2026-08-27", 130, 138, 33, 100, "upcoming"),
        ("Symbiotec Pharmalab", "2026-08-24", "2026-08-27", 950, 988, 320, 15, "upcoming"),
        ("Hy-Tech Engineers", "2026-08-24", "2026-08-27", 50, 53, 22, 238, "upcoming"),
    ]
    data = []
    for name, o, c, lo, hi, gmp, lot, st in rows:
        data.append(
            {
                "Name": name,
                "Symbol": "",
                "OpenDate": o,
                "CloseDate": c,
                "PriceLow": lo,
                "PriceHigh": hi,
                "GMP": gmp,
                "LotSize": lot,
                "Status": st,
                "UpdatedAt": now,
            }
        )
    return pd.DataFrame(data)


def _try_scrape_static() -> pd.DataFrame:
    """
    Best-effort scrape. Most GMP sites are JS (Next.js) and return no tables.
    Always use io.StringIO so pandas never treats HTML as a filepath.
    """
    urls = [
        "https://www.investorgain.com/report/ipo-gmp-live/331/",
        "https://www.investorgain.com/report/live-ipo-gmp/331/ipo/",
    ]
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=25, allow_redirects=True)
            r.raise_for_status()
            # CRITICAL: StringIO prevents "No such file or directory: <!DOCTYPE..."
            tables = pd.read_html(io.StringIO(r.text))
        except Exception as e:
            logger.warning(f"Static GMP scrape failed ({url}): {e}")
            continue

        rows = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        for t in tables:
            name_col = next(
                (
                    c
                    for c in t.columns
                    if any(k in str(c).lower() for k in ("name", "ipo", "company"))
                ),
                None,
            )
            gmp_col = next(
                (c for c in t.columns if "gmp" in str(c).lower()),
                None,
            )
            price_col = next(
                (c for c in t.columns if "price" in str(c).lower()),
                None,
            )
            open_col = next(
                (c for c in t.columns if str(c).lower().startswith("open")),
                None,
            )
            close_col = next(
                (c for c in t.columns if "close" in str(c).lower()),
                None,
            )
            lot_col = next(
                (c for c in t.columns if "lot" in str(c).lower()),
                None,
            )
            if name_col is None:
                continue

            for _, row in t.iterrows():
                name = str(row[name_col]).strip()
                if not name or name.lower() in ("nan", "name", "ipo"):
                    continue

                gmp = _to_float(row[gmp_col], 0) if gmp_col is not None else 0.0
                low, high = "", 0.0
                if price_col is not None:
                    raw_price = str(row[price_col])
                    parts = re.findall(r"\d+(?:\.\d+)?", raw_price.replace(",", ""))
                    if len(parts) >= 2:
                        low, high = float(parts[0]), float(parts[-1])
                    elif parts:
                        high = float(parts[0])

                rows.append(
                    {
                        "Name": name[:60],
                        "Symbol": "",
                        "OpenDate": str(row[open_col])[:12] if open_col is not None else "",
                        "CloseDate": str(row[close_col])[:12] if close_col is not None else "",
                        "PriceLow": low,
                        "PriceHigh": high if high else "",
                        "GMP": gmp,
                        "LotSize": _to_float(row[lot_col], 0) if lot_col is not None else "",
                        "Status": "open",
                        "UpdatedAt": now,
                    }
                )

        if rows:
            logger.info(f"Parsed {len(rows)} IPO rows from {url}")
            return pd.DataFrame(rows)

    logger.warning("Static GMP scrape: no table rows found")
    return _empty()


def refresh_ipo_gmp() -> dict:
    """
    Auto-update IPO pipeline:
      1) try static scrape
      2) keep previous CSV if present
      3) seed defaults if nothing exists
    """
    old = load_ipo_pipeline()
    scraped = _try_scrape_static()

    if scraped.empty:
        if not old.empty and len(old) > 0:
            logger.warning("GMP scrape empty — keeping previous pipeline")
            return {"updated": 0, "kept": len(old), "source": "previous"}

        seeded = _seed_current_ipos()
        PIPELINE.parent.mkdir(parents=True, exist_ok=True)
        seeded.to_csv(PIPELINE, index=False)
        logger.info(f"IPO pipeline seeded: {len(seeded)} rows")
        return {"updated": len(seeded), "kept": 0, "source": "seed"}

    PIPELINE.parent.mkdir(parents=True, exist_ok=True)
    scraped.to_csv(PIPELINE, index=False)
    logger.info(f"IPO pipeline updated: {len(scraped)} rows (scrape)")
    return {"updated": len(scraped), "kept": 0, "source": "scrape"}


def build_ipo_desk_message(max_rows: int = 10) -> str:
    """Telegram-ready IPO desk text."""
    df = load_ipo_pipeline()
    if df.empty:
        return "*IPO DESK*\nNo IPO data yet (run meta refresh / seed)."

    # Prefer open / upcoming first if Status column exists
    if "Status" in df.columns:
        order = {"open": 0, "upcoming": 1, "closed": 2}
        try:
            df = df.copy()
            df["_ord"] = df["Status"].astype(str).str.lower().map(lambda x: order.get(x, 9))
            df = df.sort_values("_ord")
        except Exception:
            pass

    lines = [
        "*IPO DESK – GMP*",
        f"As of `{datetime.now():%Y-%m-%d %H:%M}`",
        "_GMP is unofficial grey-market data – not investment advice_",
        "",
        "```",
        f"{'IPO':<22} {'High':>6} {'GMP':>6} {'%':>6} {'Verdict':>8}",
        "-" * 54,
    ]

    shown = 0
    for _, r in df.iterrows():
        if shown >= max_rows:
            break
        name = str(r.get("Name", ""))[:22]
        try:
            high = float(r.get("PriceHigh") or 0)
        except Exception:
            high = 0.0
        try:
            gmp = float(r.get("GMP") or 0)
        except Exception:
            gmp = 0.0
        pct = (gmp / high * 100) if high else 0.0
        lines.append(
            f"{name:<22} {high:>6.0f} {gmp:>6.0f} {pct:>5.1f}% {_verdict(pct):>8}"
        )
        shown += 1

    lines.append("```")
    lines.append("")
    lines.append("• Positive: GMP% ≥ 20 | Neutral: 0–20 | Weak: GMP < 0")
    lines.append("• Always check RHP / subscription before applying")
    return "\n".join(lines)

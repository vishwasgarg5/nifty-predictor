# src/ipo_gmp.py
"""
Dynamic IPO Desk: GMP + subscription status (best-effort).

Sources (in order):
  1) IPOGuru API  (if IPOGURU_API_KEY set) — clean JSON
  2) Static HTML tables (ipowatch / others) via pd.read_html(StringIO)
  3) Previous CSV
  4) Seed list

GMP is unofficial grey-market data — not investment advice.
"""

from __future__ import annotations

import io
import logging
import os
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
    "SubTotal",
    "SubQIB",
    "SubNII",
    "SubRetail",
    "Source",
    "UpdatedAt",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLS)


def load_ipo_pipeline() -> pd.DataFrame:
    if not PIPELINE.exists():
        PIPELINE.parent.mkdir(parents=True, exist_ok=True)
        df = _empty()
        df.to_csv(PIPELINE, index=False)
        return df
    try:
        df = pd.read_csv(PIPELINE)
        for c in COLS:
            if c not in df.columns:
                df[c] = ""
        return df
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
        s = str(x).replace(",", "").replace("₹", "").replace("%", "").replace("x", "").strip()
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group()) if m else default
    except Exception:
        return default


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().replace("ipo", "")).strip()[:40]


# ---------------------------------------------------------------------------
# Source 1: IPOGuru API (optional key)
# ---------------------------------------------------------------------------
def _from_ipoguru() -> pd.DataFrame:
    key = os.getenv("IPOGURU_API_KEY", "").strip()
    if not key:
        return _empty()

    base = "https://www.ipoguru.in/api/v1"
    session = requests.Session()
    session.headers.update({**HEADERS, "X-API-Key": key, "Authorization": f"Bearer {key}"})

    rows = []
    for status in ("open", "upcoming"):
        try:
            r = session.get(
                f"{base}/ipos",
                params={"status": status},
                timeout=20,
            )
            if r.status_code != 200:
                logger.warning(f"IPOGuru {status}: HTTP {r.status_code}")
                continue
            payload = r.json()
            items = payload.get("data") or payload.get("ipos") or []
            if isinstance(payload, list):
                items = payload
            for it in items:
                if not isinstance(it, dict):
                    continue
                name = str(it.get("name") or it.get("company") or "").strip()
                if not name:
                    continue
                band = str(it.get("price_band") or it.get("priceBand") or "")
                parts = re.findall(r"\d+(?:\.\d+)?", band.replace(",", ""))
                low = float(parts[0]) if parts else _to_float(it.get("issue_price"), 0)
                high = float(parts[-1]) if parts else low
                gmp = _to_float(it.get("gmp") or it.get("gmp_price") or it.get("grey_market_premium"), 0)
                sub = it.get("subscription") or it.get("sub") or {}
                if not isinstance(sub, dict):
                    sub = {}
                rows.append(
                    {
                        "Name": name[:60],
                        "Symbol": str(it.get("symbol") or ""),
                        "OpenDate": str(it.get("open_date") or it.get("startDate") or "")[:12],
                        "CloseDate": str(it.get("close_date") or it.get("endDate") or "")[:12],
                        "PriceLow": low,
                        "PriceHigh": high,
                        "GMP": gmp,
                        "LotSize": _to_float(it.get("lot_size") or it.get("minQty"), 0),
                        "Status": status,
                        "SubTotal": _to_float(sub.get("total") or it.get("subscription_total"), 0),
                        "SubQIB": _to_float(sub.get("qib") or it.get("qib"), 0),
                        "SubNII": _to_float(sub.get("nii") or it.get("nii"), 0),
                        "SubRetail": _to_float(sub.get("retail") or it.get("rii") or it.get("retail"), 0),
                        "Source": "ipoguru",
                        "UpdatedAt": _now(),
                    }
                )
        except Exception as e:
            logger.warning(f"IPOGuru failed ({status}): {e}")

    if rows:
        logger.info(f"IPOGuru: {len(rows)} rows")
        return pd.DataFrame(rows)
    return _empty()


# ---------------------------------------------------------------------------
# Source 2: static HTML tables
# ---------------------------------------------------------------------------
def _parse_tables(html: str, source: str) -> list[dict]:
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception as e:
        logger.debug(f"read_html failed ({source}): {e}")
        return []

    rows = []
    for t in tables:
        cols_l = [str(c).lower() for c in t.columns]
        name_col = next(
            (c for c in t.columns if any(k in str(c).lower() for k in ("name", "company", "ipo"))),
            None,
        )
        if name_col is None:
            continue

        gmp_col = next((c for c in t.columns if "gmp" in str(c).lower()), None)
        price_col = next(
            (c for c in t.columns if "price" in str(c).lower() or "band" in str(c).lower()),
            None,
        )
        open_col = next((c for c in t.columns if str(c).lower().startswith("open")), None)
        close_col = next((c for c in t.columns if "close" in str(c).lower()), None)
        lot_col = next((c for c in t.columns if "lot" in str(c).lower()), None)
        sub_col = next(
            (c for c in t.columns if "sub" in str(c).lower() or "subscription" in str(c).lower()),
            None,
        )
        qib_col = next((c for c in t.columns if "qib" in str(c).lower()), None)
        nii_col = next((c for c in t.columns if "nii" in str(c).lower() or "hni" in str(c).lower()), None)
        rii_col = next(
            (c for c in t.columns if "retail" in str(c).lower() or "rii" in str(c).lower()),
            None,
        )
        status_col = next((c for c in t.columns if "status" in str(c).lower()), None)

        for _, row in t.iterrows():
            name = str(row[name_col]).strip()
            if not name or name.lower() in ("nan", "name", "ipo", "company"):
                continue

            low, high = "", 0.0
            if price_col is not None:
                parts = re.findall(r"\d+(?:\.\d+)?", str(row[price_col]).replace(",", ""))
                if len(parts) >= 2:
                    low, high = float(parts[0]), float(parts[-1])
                elif parts:
                    high = float(parts[0])

            st = "open"
            if status_col is not None:
                raw = str(row[status_col]).lower()
                if "upcom" in raw or "soon" in raw:
                    st = "upcoming"
                elif "close" in raw or "list" in raw:
                    st = "closed"

            rows.append(
                {
                    "Name": name[:60],
                    "Symbol": "",
                    "OpenDate": str(row[open_col])[:12] if open_col is not None else "",
                    "CloseDate": str(row[close_col])[:12] if close_col is not None else "",
                    "PriceLow": low,
                    "PriceHigh": high if high else "",
                    "GMP": _to_float(row[gmp_col], 0) if gmp_col is not None else 0.0,
                    "LotSize": _to_float(row[lot_col], 0) if lot_col is not None else "",
                    "Status": st,
                    "SubTotal": _to_float(row[sub_col], 0) if sub_col is not None else 0.0,
                    "SubQIB": _to_float(row[qib_col], 0) if qib_col is not None else 0.0,
                    "SubNII": _to_float(row[nii_col], 0) if nii_col is not None else 0.0,
                    "SubRetail": _to_float(row[rii_col], 0) if rii_col is not None else 0.0,
                    "Source": source,
                    "UpdatedAt": _now(),
                }
            )
    return rows


def _from_html_sources() -> pd.DataFrame:
    urls = [
        ("ipowatch", "https://ipowatch.in/ipo-grey-market-premium-latest-ipo-gmp/"),
        ("ipowatch_home", "https://ipowatch.in/"),
        ("chanakya", "https://chanakyanipothi.com/ipo-gmp-today/"),
    ]
    all_rows = []
    for name, url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=25, allow_redirects=True)
            if r.status_code != 200:
                logger.warning(f"{name}: HTTP {r.status_code}")
                continue
            parsed = _parse_tables(r.text, name)
            if parsed:
                logger.info(f"{name}: {len(parsed)} rows")
                all_rows.extend(parsed)
                break  # first good source wins
            else:
                logger.warning(f"{name}: no tables")
        except Exception as e:
            logger.warning(f"{name} scrape failed: {e}")

    if not all_rows:
        return _empty()

    # de-dupe by name
    df = pd.DataFrame(all_rows)
    df["_k"] = df["Name"].map(_norm_name)
    df = df.drop_duplicates(subset=["_k"], keep="first").drop(columns=["_k"])
    return df


# ---------------------------------------------------------------------------
# Seed fallback
# ---------------------------------------------------------------------------
def _seed_current_ipos() -> pd.DataFrame:
    now = _now()
    rows = [
        ("Tempsens Instruments", "2026-08-20", "2026-08-24", 280, 300, 170, 50, "open", 6.0, 10.0, 5.0, 3.0),
        ("Augmont Enterprises", "2026-08-21", "2026-08-25", 750, 788, 180, 19, "open", 3.0, 5.0, 2.0, 1.5),
        ("Gaja Alternative AMC", "2026-08-19", "2026-08-21", 150, 160, 30, 93, "open", 30.0, 40.0, 20.0, 10.0),
        ("Skyways Air", "2026-08-24", "2026-08-27", 130, 138, 33, 100, "upcoming", 0, 0, 0, 0),
        ("Symbiotec Pharmalab", "2026-08-24", "2026-08-27", 950, 988, 320, 15, "upcoming", 0, 0, 0, 0),
        ("Hy-Tech Engineers", "2026-08-24", "2026-08-27", 50, 53, 22, 238, "upcoming", 0, 0, 0, 0),
    ]
    data = []
    for name, o, c, lo, hi, gmp, lot, st, tot, qib, nii, rii in rows:
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
                "SubTotal": tot,
                "SubQIB": qib,
                "SubNII": nii,
                "SubRetail": rii,
                "Source": "seed",
                "UpdatedAt": now,
            }
        )
    return pd.DataFrame(data)


def _merge_prefer_new(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old.empty:
        return new
    if new.empty:
        return old
    old = old.copy()
    new = new.copy()
    old["_k"] = old["Name"].map(_norm_name)
    new["_k"] = new["Name"].map(_norm_name)
    # new overwrites same name; keep old-only names
    keys_new = set(new["_k"])
    keep_old = old[~old["_k"].isin(keys_new)]
    out = pd.concat([new, keep_old], ignore_index=True)
    return out.drop(columns=["_k"], errors="ignore")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def refresh_ipo_gmp() -> dict:
    old = load_ipo_pipeline()

    # 1) API
    df = _from_ipoguru()
    source = "ipoguru"

    # 2) HTML
    if df.empty:
        df = _from_html_sources()
        source = "html"

    if df.empty:
        if not old.empty:
            logger.warning("Dynamic GMP empty — keeping previous pipeline")
            return {"updated": 0, "kept": len(old), "source": "previous"}
        df = _seed_current_ipos()
        source = "seed"
        logger.info(f"IPO pipeline seeded: {len(df)} rows")
    else:
        df = _merge_prefer_new(old, df)
        logger.info(f"IPO pipeline updated: {len(df)} rows ({source})")

    PIPELINE.parent.mkdir(parents=True, exist_ok=True)
    # ensure columns
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    df[COLS].to_csv(PIPELINE, index=False)
    return {"updated": len(df), "kept": 0, "source": source}


def build_ipo_desk_message(max_rows: int = 10) -> str:
    df = load_ipo_pipeline()
    if df.empty:
        return "*IPO DESK*\nNo IPO data yet."

    df = df.copy()
    for c in ("GMP", "PriceHigh", "PriceLow", "SubTotal", "SubQIB", "SubNII", "SubRetail"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Keep rows that have some signal
    mask = (df["GMP"].abs() > 0) | (df["PriceHigh"] > 0) | (df.get("SubTotal", 0) > 0)
    useful = df.loc[mask].copy()
    if useful.empty:
        # show a short notice instead of 40 empty names
        return (
            "*IPO DESK*\n"
            f"As of `{datetime.now():%Y-%m-%d %H:%M}`\n"
            "No usable GMP/price rows in pipeline.\n"
            "_Scrape got names only — update source or seed._"
        )

    if "Status" in useful.columns:
        order = {"open": 0, "upcoming": 1, "closed": 2}
        useful["_ord"] = useful["Status"].astype(str).str.lower().map(lambda x: order.get(x, 9))
        useful = useful.sort_values(["_ord", "GMP"], ascending=[True, False])
    else:
        useful = useful.sort_values("GMP", ascending=False)

    lines = [
        "*IPO DESK – GMP + Subscription*",
        f"As of `{datetime.now():%Y-%m-%d %H:%M}`",
        "_GMP unofficial · Sub in x · not advice_",
        "",
        "```",
        f"{'IPO':<18} {'GMP':>5} {'%':>5} {'Sub':>6} {'Verdict':>8}",
        "-" * 48,
    ]

    shown = 0
    for _, r in useful.iterrows():
        if shown >= max_rows:
            break
        name = str(r.get("Name", ""))[:18]
        high = float(r.get("PriceHigh") or 0)
        gmp = float(r.get("GMP") or 0)
        sub = float(r.get("SubTotal") or 0)
        pct = (gmp / high * 100) if high > 0 else 0.0
        sub_s = f"{sub:.1f}x" if sub > 0 else "-"
        lines.append(
            f"{name:<18} {gmp:>5.0f} {pct:>4.0f}% {sub_s:>6} {_verdict(pct):>8}"
        )
        shown += 1

    lines.append("```")

    detail = []
    for _, r in useful.iterrows():
        if str(r.get("Status", "")).lower() != "open":
            continue
        qib = float(r.get("SubQIB") or 0)
        nii = float(r.get("SubNII") or 0)
        rii = float(r.get("SubRetail") or 0)
        if qib or nii or rii:
            detail.append(
                f"• `{str(r.get('Name', ''))[:20]}` "
                f"QIB `{qib:.1f}x` NII `{nii:.1f}x` RII `{rii:.1f}x`"
            )
    if detail:
        lines += ["", "*Subscription (open)*"] + detail[:6]

    lines += ["", "• Positive: GMP% ≥ 20 · check RHP before applying"]
    return "\n".join(lines)

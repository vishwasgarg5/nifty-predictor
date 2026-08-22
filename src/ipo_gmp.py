"""Upcoming IPO desk with GMP helper."""
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from src.config import cfg

logger = logging.getLogger(__name__)

def load_ipo_pipeline() -> pd.DataFrame:
    path = Path(getattr(cfg.paths, "ipo_pipeline", "data/ipo_pipeline.csv"))
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(columns=[
            "Name", "Symbol", "OpenDate", "CloseDate",
            "PriceLow", "PriceHigh", "GMP", "LotSize", "Status"
        ])
        df.to_csv(path, index=False)
        return df
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.warning(f"ipo_pipeline read failed: {e}")
        return pd.DataFrame()

def _verdict(gmp_pct: float) -> str:
    if gmp_pct >= 20:
        return "Positive"
    if gmp_pct >= 0:
        return "Neutral"
    return "Weak"

def build_ipo_desk_message(max_rows: int = 10) -> str:
    df = load_ipo_pipeline()
    if df.empty:
        return "*IPO DESK*\nNo upcoming IPO data. Update `data/ipo_pipeline.csv`."

    rows = []
    for _, r in df.iterrows():
        try:
            high = float(r.get("PriceHigh") or 0)
            gmp = float(r.get("GMP") or 0)
            gmp_pct = (gmp / high * 100) if high else 0
            est = high + gmp
            rows.append({
                "Name": str(r.get("Name", ""))[:18],
                "Open": str(r.get("OpenDate", ""))[:10],
                "Close": str(r.get("CloseDate", ""))[:10],
                "Band": f"{r.get('PriceLow', '')}-{r.get('PriceHigh', '')}",
                "GMP": gmp,
                "GMP%": gmp_pct,
                "Est": est,
                "Status": str(r.get("Status", "")),
                "Verdict": _verdict(gmp_pct),
            })
        except Exception:
            continue

    if not rows:
        return "*IPO DESK*\nNo valid IPO rows."

    # prefer open/upcoming
    order = {"open": 0, "upcoming": 1, "closed": 2}
    rows.sort(key=lambda x: order.get(str(x["Status"]).lower(), 9))

    lines = [
        "*IPO DESK – Upcoming / Open + GMP*",
        f"As of `{datetime.now():%Y-%m-%d}`",
        "_GMP is unofficial grey-market data – not a guarantee_",
        "",
        "```",
        f"{'IPO':<18} {'Band':>9} {'GMP':>6} {'%':>6} {'Verdict':>8}",
        "-" * 52,
    ]
    for x in rows[:max_rows]:
        lines.append(
            f"{x['Name']:<18} {x['Band']:>9} {x['GMP']:>6.0f} {x['GMP%']:>5.1f}% {x['Verdict']:>8}"
        )
    lines.append("```")
    lines.append("")
    lines.append("• Positive: GMP% > 20 | Neutral: 0–20 | Weak: GMP < 0")
    lines.append("• Check RHP, subscription & risks before applying")
    return "\n".join(lines)

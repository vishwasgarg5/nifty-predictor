"""Trading day check with auto-refreshed NSE holiday file."""
from __future__ import annotations
import logging
from datetime import date, datetime
from pathlib import Path
import json
import re
import requests

logger = logging.getLogger(__name__)
HOLIDAY_FILE = Path("data/holidays.json")

# minimal built-in fallback if file missing
_FALLBACK = {
    date(2026, 1, 26), date(2026, 3, 3), date(2026, 3, 26),
    date(2026, 4, 3), date(2026, 4, 14), date(2026, 5, 1),
    date(2026, 8, 15), date(2026, 10, 2), date(2026, 10, 20),
    date(2026, 11, 8), date(2026, 12, 25),
}

def _load_holidays() -> set[date]:
    if HOLIDAY_FILE.exists():
        try:
            raw = json.loads(HOLIDAY_FILE.read_text())
            return {datetime.strptime(x, "%Y-%m-%d").date() for x in raw}
        except Exception as e:
            logger.warning(f"holiday file read failed: {e}")
    return set(_FALLBACK)

def refresh_holidays() -> int:
    """
    Best-effort refresh from NSE.
    Tries known endpoints; on failure keeps existing file.
    """
    urls = [
        "https://www.nseindia.com/api/holiday-master?type=trading",
        "https://www.nseindia.com/api/holiday-master?type=clearing",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/",
    }
    session = requests.Session()
    session.headers.update(headers)
    found: set[date] = set()

    try:
        # NSE often needs a cookie warm-up
        session.get("https://www.nseindia.com", timeout=15)
        for url in urls:
            try:
                r = session.get(url, timeout=20)
                if r.status_code != 200:
                    continue
                data = r.json()
                # structure varies: try common patterns
                items = []
                if isinstance(data, dict):
                    for v in data.values():
                        if isinstance(v, list):
                            items.extend(v)
                elif isinstance(data, list):
                    items = data
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    for key in ("tradingDate", "date", "holidayDate", "Date"):
                        if key in it and it[key]:
                            val = str(it[key])[:10]
                            for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y"):
                                try:
                                    found.add(datetime.strptime(val, fmt).date())
                                    break
                                except Exception:
                                    pass
            except Exception as e:
                logger.debug(f"holiday url fail {url}: {e}")
    except Exception as e:
        logger.warning(f"holiday refresh failed: {e}")

    if not found:
        logger.warning("No holidays downloaded; keeping previous")
        return 0

    HOLIDAY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HOLIDAY_FILE.write_text(json.dumps(sorted(x.isoformat() for x in found), indent=2))
    logger.info(f"Holidays refreshed: {len(found)}")
    return len(found)

def is_trading_day(d: date | None = None) -> bool:
    d = d or date.today()
    if d.weekday() >= 5:
        return False
    return d not in _load_holidays()

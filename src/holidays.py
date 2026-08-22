"""Trading day check. Holidays auto-written from official NSE 2026 list."""
from __future__ import annotations
import json
import logging
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)
HOLIDAY_FILE = Path("data/holidays.json")

# Official NSE CM holidays 2026 (circular NSE/CMTR/71775)
# Weekends already excluded by is_trading_day()
NSE_2026 = [
    "2026-01-26",  # Republic Day
    "2026-03-03",  # Holi
    "2026-03-26",  # Ram Navami
    "2026-03-31",  # Mahavir Jayanti
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-05-28",  # Bakri Id
    "2026-06-26",  # Muharram
    "2026-09-14",  # Ganesh Chaturthi
    "2026-10-02",  # Gandhi Jayanti
    "2026-10-20",  # Dussehra
    "2026-11-10",  # Diwali-Balipratipada
    "2026-11-24",  # Guru Nanak Jayanti
    "2026-12-25",  # Christmas
]

def _parse_dates(items) -> set[date]:
    out = set()
    for x in items:
        try:
            out.add(datetime.strptime(str(x)[:10], "%Y-%m-%d").date())
        except Exception:
            pass
    return out

def _load_holidays() -> set[date]:
    if HOLIDAY_FILE.exists():
        try:
            raw = json.loads(HOLIDAY_FILE.read_text())
            return _parse_dates(raw)
        except Exception as e:
            logger.warning(f"holiday file read failed: {e}")
    return _parse_dates(NSE_2026)

def refresh_holidays() -> int:
    """
    Always ensure holidays.json exists with official list.
    Optionally try NSE API; on failure still write bundled list.
    """
    found = set()

    # Optional live try (often blocked on cloud)
    try:
        import requests
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/",
        })
        s.get("https://www.nseindia.com", timeout=12)
        r = s.get("https://www.nseindia.com/api/holiday-master?type=trading", timeout=15)
        if r.status_code == 200:
            data = r.json()
            items = []
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        items.extend(v)
            for it in items:
                if not isinstance(it, dict):
                    continue
                for key in ("tradingDate", "date", "holidayDate"):
                    if key in it and it[key]:
                        val = str(it[key])[:15]
                        for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y"):
                            try:
                                found.add(datetime.strptime(val, fmt).date())
                                break
                            except Exception:
                                pass
    except Exception as e:
        logger.warning(f"NSE holiday API failed (using official list): {e}")

    if not found:
        found = _parse_dates(NSE_2026)
        source = "official_2026_circular"
    else:
        # merge with official so we never miss known days
        found |= _parse_dates(NSE_2026)
        source = "nse_api+official"

    HOLIDAY_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = sorted(d.isoformat() for d in found)
    HOLIDAY_FILE.write_text(json.dumps(payload, indent=2))
    logger.info(f"Holidays written: {len(payload)} ({source})")
    return len(payload)

def is_trading_day(d: date | None = None) -> bool:
    d = d or date.today()
    if d.weekday() >= 5:
        return False
    return d not in _load_holidays()

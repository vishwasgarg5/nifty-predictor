from datetime import date
import pandas as pd

# Simplified NSE holiday list for 2025-2026 (expand as needed)
NSE_HOLIDAYS = {
    date(2025, 1, 26), date(2025, 3, 14), date(2025, 3, 31),
    date(2025, 4, 10), date(2025, 4, 14), date(2025, 4, 18),
    date(2025, 5, 1),  date(2025, 8, 15), date(2025, 8, 27),
    date(2025, 10, 2), date(2025, 10, 21), date(2025, 10, 22),
    date(2025, 11, 5), date(2025, 12, 25),
    date(2026, 1, 26), date(2026, 3, 3), date(2026, 3, 26),
    date(2026, 4, 3),  date(2026, 4, 14), date(2026, 5, 1),
    date(2026, 8, 15), date(2026, 10, 2), date(2026, 10, 20),
    date(2026, 11, 8), date(2026, 12, 25),
}

def is_trading_day(d: date = None) -> bool:
    d = d or date.today()
    if d.weekday() >= 5:          # Saturday / Sunday
        return False
    return d not in NSE_HOLIDAYS

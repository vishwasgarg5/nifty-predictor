from datetime import date

NSE_HOLIDAYS = {
    date(2026, 1, 26), date(2026, 3, 3), date(2026, 3, 26),
    date(2026, 4, 3), date(2026, 4, 14), date(2026, 5, 1),
    date(2026, 8, 15), date(2026, 10, 2), date(2026, 10, 20),
    date(2026, 11, 8), date(2026, 12, 25),
}

def is_trading_day(d=None):
    d = d or date.today()
    return d.weekday() < 5 and d not in NSE_HOLIDAYS

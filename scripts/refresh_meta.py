#!/usr/bin/env python3
"""Auto refresh: holidays + IPO/GMP (+ optional universes)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from src.holidays import refresh_holidays
from src.ipo_gmp import refresh_ipo_gmp, build_ipo_desk_message
from src.telegram_utils import send_telegram

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Meta refresh start")
    n_hol = refresh_holidays()
    ipo_stats = refresh_ipo_gmp()

    msg = (
        f"*Meta refresh*\n"
        f"• Holidays loaded: `{n_hol}`\n"
        f"• IPO rows: `{ipo_stats.get('updated') or ipo_stats.get('kept')}` "
        f"({ipo_stats.get('source')})\n"
    )
    try:
        send_telegram(msg)
        # optional: also push desk snapshot
        # send_telegram(build_ipo_desk_message(8))
    except Exception:
        pass
    logger.info("Meta refresh done")

if __name__ == "__main__":
    main()

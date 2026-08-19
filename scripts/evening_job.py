#!/usr/bin/env python3
"""
Evening Job - Actual vs Predicted + Error Logging + Light Retrain
Runs every trading day ~16:15 IST
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import traceback

from src.config import cfg
from src.holidays import is_trading_day
from src.data_loader import download_history
from src.model import OHLCPredictor
from src.telegram_utils import send_telegram

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    start_time = datetime.now()
    today_str = start_time.strftime("%Y-%m-%d")
    logger.info("=" * 60)
    logger.info(f"Evening Job started | {today_str}")

    # --------------------------------------------------
    # 1. Holiday / Weekend check
    # --------------------------------------------------
    if not is_trading_day():
        logger.info("Today is not a trading day. Exiting cleanly.")
        return

    try:
        # --------------------------------------------------
        # 2. Load today's predictions
        # --------------------------------------------------
        pred_dir = Path(cfg.paths.predictions)
        pred_file = pred_dir / f"{today_str}.csv"

        # Fallback to yesterday if today's file is missing
        if not pred_file.exists():
            yesterday = (start_time - timedelta(days=1)).strftime("%Y-%m-%d")
            pred_file = pred_dir / f"{yesterday}.csv"
            logger.warning(f"Today's prediction file not found. Trying {yesterday}")

        if not pred_file.exists():
            msg = f"⚠️ *Evening Job*: No prediction file found for `{today_str}`"
            send_telegram(msg)
            logger.error("No prediction file found")
            return

        preds = pd.read_csv(pred_file)
        logger.info(f"Loaded predictions from {pred_file.name} | {len(preds)} stocks")

        # --------------------------------------------------
        # 3. Compare Actual vs Predicted
        # --------------------------------------------------
        lines = [
            f"*Actual vs Predicted Report*",
            f"Date: `{today_str}`",
            ""
        ]

        error_rows = []
        success_count = 0

        for _, row in preds.iterrows():
            symbol = row["symbol"]
            clean_symbol = symbol.replace(".NS", "")

            try:
                hist = download_history(symbol, period="5d")
                if hist is None or len(hist) < 1:
                    logger.warning(f"No actual data for {symbol}")
                    continue

                actual = hist.iloc[-1]
                pred_close = float(row["Close"])
                actual_close = float(actual["Close"])

                diff = actual_close - pred_close
                pct_error = (diff / pred_close) * 100 if pred_close != 0 else 0
                abs_pct = abs(pct_error)

                # Direction accuracy (simple)
                prev_close = float(hist.iloc[-2]["Close"]) if len(hist) >= 2 else pred_close
                pred_direction = 1 if pred_close > prev_close else 0
                actual_direction = 1 if actual_close > prev_close else 0
                direction_correct = int(pred_direction == actual_direction)

                lines.append(f"*{clean_symbol}*")
                lines.append(
                    f"Pred C: `{pred_close:.2f}` → Actual C: `{actual_close:.2f}`"
                )
                lines.append(
                    f

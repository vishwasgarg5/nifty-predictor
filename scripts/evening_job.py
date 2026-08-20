#!/usr/bin/env python3
"""
Evening Job - Actual vs Predicted + Error Logging + Light Retrain
Clean table format + better error handling
"""

import sys
from pathlib import Path

# Fix Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import pandas as pd
from datetime import datetime, timedelta
import traceback

from src.config import cfg
from src.holidays import is_trading_day
from src.data_loader import download_history, get_actual_close
from src.model import OHLCPredictor
from src.telegram_utils import send_telegram

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

    if not is_trading_day():
        logger.info("Today is not a trading day. Exiting.")
        return

    try:
        pred_dir = Path(cfg.paths.predictions)
        pred_file = pred_dir / f"{today_str}.csv"

        # Try previous days if today's file is missing
        if not pred_file.exists():
            for days_back in range(1, 4):
                prev_date = (start_time - timedelta(days=days_back)).strftime("%Y-%m-%d")
                candidate = pred_dir / f"{prev_date}.csv"
                if candidate.exists():
                    pred_file = candidate
                    logger.warning(f"Using prediction file from {prev_date}")
                    break

        if not pred_file.exists():
            send_telegram(f"⚠️ *Evening Job*: No prediction file found for `{today_str}`")
            logger.error("No prediction file found")
            return

        preds = pd.read_csv(pred_file)
        logger.info(f"Loaded predictions: {len(preds)} stocks from {pred_file.name}")

        results = []
        error_rows = []

        for _, row in preds.iterrows():
            symbol = row["symbol"]
            clean_symbol = symbol.replace(".NS", "")
            pred_close = float(row["Close"])

            actual_close, prev_close = get_actual_close(symbol)

            if actual_close is None:
                logger.error(f"Could not fetch actual data for {symbol}")
                continue

            diff = actual_close - pred_close
            pct_error = (diff / pred_close) * 100 if pred_close != 0 else 0
            abs_pct = abs(pct_error)

            # Direction accuracy
            pred_direction = 1 if pred_close > prev_close else 0
            actual_direction = 1 if actual_close > prev_close else 0
            direction_correct = int(pred_direction == actual_direction)

            results.append({
                "symbol": clean_symbol,
                "pred": pred_close,
                "actual": actual_close,
                "diff": diff,
                "pct": pct_error
            })

            error_rows.append({
                "date": today_str,
                "symbol": symbol,
                "pred_close": pred_close,
                "actual_close": actual_close,
                "abs_error": abs(diff),
                "abs_error_pct": abs_pct,
                "direction_correct": direction_correct
            })

            logger.info(f"{clean_symbol}: Pred {pred_close:.2f} → Actual {actual_close:.2f} ({pct_error:+.2f}%)")

            # Light retrain
            try:
                predictor = OHLCPredictor(symbol)
                predictor.train()
                logger.info(f"Model updated for {symbol}")
            except Exception as e:
                logger.warning(f"Retrain failed for {symbol}: {e}")

        # -------------------------------------------------
        # Check if we got any data
        # -------------------------------------------------
        if not results:
            send_telegram(f"⚠️ *Evening Job*: Could not fetch actual data for any stock on `{today_str}`")
            logger.error("No actual data fetched for any stock")
            return

        # -------------------------------------------------
        # Save error history
        # -------------------------------------------------
        err_path = Path(cfg.paths.errors_file)
        err_path.parent.mkdir(parents=True, exist_ok=True)

        err_df = pd.DataFrame(error_rows)
        if err_path.exists():
            err_df.to_csv(err_path, mode="a", header=False, index=False)
        else:
            err_df.to_csv(err_path, index=False)

        logger.info(f"Error history updated → {err_path}")

        # --------------------------------------------------
        # 6. Build clean table message
        # --------------------------------------------------
        lines = [
            f"*ACTUAL vs PREDICTED*",
            f"Date: `{today_str}`",
            "",
            "```",
            f"{'Stock':<12} {'Pred C':>9} {'Actual C':>9} {'Diff':>8} {'Error%':>8}",
            "-" * 50
        ]

        for r in results:
            lines.append(
                f"{r['symbol']:<12} {r['pred']:>9.2f} {r['actual']:>9.2f} "
                f"{r['diff']:>+8.2f} {r['pct']:>+7.2f}%"
            )

        lines.append("```")
        lines.append("")

        # Summary
        avg_error = err_df["abs_error_pct"].mean()
        dir_acc = err_df["direction_correct"].mean() * 100

        lines.append(f"*Summary*")
        lines.append(f"• Stocks processed: `{len(results)}`")
        lines.append(f"• Average Absolute Error: `{avg_error:.2f}%`")
        lines.append(f"• Directional Accuracy: `{dir_acc:.1f}%`")
        lines.append(f"_Job finished in {(datetime.now() - start_time).seconds}s_")

        message = "\n".join(lines)
        success = send_telegram(message)

        if success:
            logger.info("Evening report sent successfully")
        else:
            logger.error("Failed to send Telegram message")

    except Exception as e:
        error_msg = f"❌ *Evening Job Failed*\n`{today_str}`\n\n```{str(e)[:800]}```"
        logger.error(traceback.format_exc())
        send_telegram(error_msg)

    logger.info("Evening Job finished")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

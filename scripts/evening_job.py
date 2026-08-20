#!/usr/bin/env python3
"""
Evening Job - Actual vs Predicted (Open + Close)
Clean table format + Error logging + Light retrain
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
from src.data_loader import get_actual_ohlc
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
        # --------------------------------------------------
        # 1. Load predictions
        # --------------------------------------------------
        pred_dir = Path(cfg.paths.predictions)
        pred_file = pred_dir / f"{today_str}.csv"

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

        # --------------------------------------------------
        # 2. Compare Actual vs Predicted
        # --------------------------------------------------
        results = []
        error_rows = []

        for _, row in preds.iterrows():
            symbol = row["symbol"]
            clean = symbol.replace(".NS", "")

            pred_o = float(row.get("Open", 0))
            pred_h = float(row.get("High", 0))
            pred_l = float(row.get("Low", 0))
            pred_c = float(row["Close"])

            actual = get_actual_ohlc(symbol)

            if actual is None:
                logger.error(f"Could not fetch valid actual data for {symbol}")
                continue

            # Open metrics
            diff_o = actual["Open"] - pred_o
            pct_o = (diff_o / pred_o * 100) if pred_o else 0

            # Close metrics
            diff_c = actual["Close"] - pred_c
            pct_c = (diff_c / pred_c * 100) if pred_c else 0

            results.append({
                "symbol": clean,
                "pred_o": pred_o,
                "actual_o": actual["Open"],
                "diff_o": diff_o,
                "pct_o": pct_o,
                "pred_c": pred_c,
                "actual_c": actual["Close"],
                "diff_c": diff_c,
                "pct_c": pct_c,
            })

            error_rows.append({
                "date": today_str,
                "symbol": symbol,
                "pred_open": pred_o,
                "actual_open": actual["Open"],
                "pred_high": pred_h,
                "actual_high": actual["High"],
                "pred_low": pred_l,
                "actual_low": actual["Low"],
                "pred_close": pred_c,
                "actual_close": actual["Close"],
                "abs_error_open": abs(diff_o),
                "abs_error_pct_open": abs(pct_o),
                "abs_error": abs(diff_c),
                "abs_error_pct": abs(pct_c),
                "direction_correct": int(
                    (pred_c > actual["prev_close"]) == (actual["Close"] > actual["prev_close"])
                )
            })

            logger.info(
                f"{clean}: O {pred_o:.2f}→{actual['Open']:.2f} | "
                f"C {pred_c:.2f}→{actual['Close']:.2f}"
            )

            # Light retrain
            try:
                predictor = OHLCPredictor(symbol)
                predictor.train()
                logger.info(f"Model updated for {symbol}")
            except Exception as e:
                logger.warning(f"Retrain failed for {symbol}: {e}")

        if not results:
            send_telegram(f"⚠️ *Evening Job*: Could not fetch actual data for any stock on `{today_str}`")
            logger.error("No actual data fetched for any stock")
            return

        # --------------------------------------------------
        # 3. Save error history
        # --------------------------------------------------
        err_path = Path(cfg.paths.errors_file)
        err_path.parent.mkdir(parents=True, exist_ok=True)

        err_df = pd.DataFrame(error_rows)
        if err_path.exists():
            err_df.to_csv(err_path, mode="a", header=False, index=False)
        else:
            err_df.to_csv(err_path, index=False)

        logger.info(f"Error history updated → {err_path}")

        # --------------------------------------------------
        # 4. Build clean Telegram message (Open + Close)
        # --------------------------------------------------
        lines = [
            f"*ACTUAL vs PREDICTED*",
            f"Date: `{today_str}`",
            "",
            "*OPEN*",
            "```",
            f"{'Stock':<12} {'Pred O':>9} {'Actual O':>9} {'Diff':>8} {'Err%':>7}",
            "-" * 50
        ]

        for r in results:
            lines.append(
                f"{r['symbol']:<12} {r['pred_o']:>9.2f} {r['actual_o']:>9.2f} "
                f"{r['diff_o']:>+8.2f} {r['pct_o']:>+6.2f}%"
            )

        lines.append("```")
        lines.append("")
        lines.append("*CLOSE*")
        lines.append("```")
        lines.append(f"{'Stock':<12} {'Pred C':>9} {'Actual C':>9} {'Diff':>8} {'Err%':>7}")
        lines.append("-" * 50)

        for r in results:
            lines.append(
                f"{r['symbol']:<12} {r['pred_c']:>9.2f} {r['actual_c']:>9.2f} "
                f"{r['diff_c']:>+8.2f} {r['pct_c']:>+6.2f}%"
            )

        lines.append("```")
        lines.append("")

        # Summary
        avg_open_error = err_df["abs_error_pct_open"].mean()
        avg_close_error = err_df["abs_error_pct"].mean()
        dir_acc = err_df["direction_correct"].mean() * 100

        lines.append("*Summary*")
        lines.append(f"• Stocks processed: `{len(results)}`")
        lines.append(f"• Avg Error (Open): `{avg_open_error:.2f}%`")
        lines.append(f"• Avg Error (Close): `{avg_close_error:.2f}%`")
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

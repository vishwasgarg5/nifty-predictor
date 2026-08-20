#!/usr/bin/env python3
"""
Evening Job
- Actual vs Predicted (Open + Close tables)
- Error logging
- Light retrain with error weights
- Drift detection
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import pandas as pd
from datetime import datetime, timedelta
import traceback

from src.config import cfg
from src.holidays import is_trading_day
from src.data_loader import get_actual_ohlc
from src.model import OHLCPredictor
from src.drift import detect_drift
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
        logger.info("Not a trading day. Exit.")
        return

    try:
        pred_dir = Path(cfg.paths.predictions)
        pred_file = pred_dir / f"{today_str}.csv"

        if not pred_file.exists():
            for days_back in range(1, 4):
                prev = (start_time - timedelta(days=days_back)).strftime("%Y-%m-%d")
                candidate = pred_dir / f"{prev}.csv"
                if candidate.exists():
                    pred_file = candidate
                    logger.warning(f"Using predictions from {prev}")
                    break

        if not pred_file.exists():
            send_telegram(f"⚠️ Evening Job: No prediction file for `{today_str}`")
            return

        preds = pd.read_csv(pred_file)
        logger.info(f"Loaded {len(preds)} predictions")

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
                logger.error(f"No actual data for {symbol}")
                continue

            diff_o = actual["Open"] - pred_o
            pct_o = (diff_o / pred_o * 100) if pred_o else 0
            diff_c = actual["Close"] - pred_c
            pct_c = (diff_c / pred_c * 100) if pred_c else 0

            results.append({
                "symbol": clean,
                "pred_o": pred_o, "actual_o": actual["Open"],
                "diff_o": diff_o, "pct_o": pct_o,
                "pred_c": pred_c, "actual_c": actual["Close"],
                "diff_c": diff_c, "pct_c": pct_c
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
                predictor.train(use_error_weights=True, do_tune=False)
            except Exception as e:
                logger.warning(f"Retrain failed {symbol}: {e}")

        if not results:
            send_telegram(f"⚠️ Evening Job: Could not fetch actual data on `{today_str}`")
            return

        # Save errors
        err_path = Path(cfg.paths.errors_file)
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_df = pd.DataFrame(error_rows)
        if err_path.exists():
            err_df.to_csv(err_path, mode="a", header=False, index=False)
        else:
            err_df.to_csv(err_path, index=False)

        # Telegram tables
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

        avg_o = err_df["abs_error_pct_open"].mean()
        avg_c = err_df["abs_error_pct"].mean()
        dir_acc = err_df["direction_correct"].mean() * 100

        lines.append("*Summary*")
        lines.append(f"• Stocks: `{len(results)}`")
        lines.append(f"• Avg Error Open: `{avg_o:.2f}%`")
        lines.append(f"• Avg Error Close: `{avg_c:.2f}%`")
        lines.append(f"• Directional Accuracy: `{dir_acc:.1f}%`")
        lines.append(f"_Finished in {(datetime.now() - start_time).seconds}s_")

        send_telegram("\n".join(lines))

        # Drift detection
        drift = detect_drift()
        logger.info(f"Drift status: {drift}")

    except Exception as e:
        logger.error(traceback.format_exc())
        send_telegram(f"❌ *Evening Job Failed*\n`{today_str}`\n```{str(e)[:700]}```")

    logger.info("Evening Job finished")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

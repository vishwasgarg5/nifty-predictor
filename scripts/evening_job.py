#!/usr/bin/env python3
"""
Evening Job – Actual vs Predicted (Open + Close)
Uses multi-source data loader + light retrain + drift
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
    start = datetime.now()
    today = start.strftime("%Y-%m-%d")
    logger.info("=" * 60)
    logger.info(f"Evening Job started | {today}")

    if not is_trading_day():
        logger.info("Not a trading day.")
        return

    try:
        pred_dir = Path(cfg.paths.predictions)
        pred_file = pred_dir / f"{today}.csv"
        if not pred_file.exists():
            for i in range(1, 4):
                d = (start - timedelta(days=i)).strftime("%Y-%m-%d")
                c = pred_dir / f"{d}.csv"
                if c.exists():
                    pred_file = c
                    logger.warning(f"Using predictions from {d}")
                    break

        if not pred_file.exists():
            send_telegram(f"⚠️ Evening: No prediction file for `{today}`")
            return

        preds = pd.read_csv(pred_file)
        results, error_rows = [], []

        for _, row in preds.iterrows():
            symbol = row["symbol"]
            clean = symbol.replace(".NS", "")
            pred_o = float(row.get("Open", 0))
            pred_c = float(row["Close"])
            pred_h = float(row.get("High", 0))
            pred_l = float(row.get("Low", 0))

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
                "diff_c": diff_c, "pct_c": pct_c,
                "source": actual.get("source", "?")
            })

            error_rows.append({
                "date": today,
                "symbol": symbol,
                "pred_open": pred_o, "actual_open": actual["Open"],
                "pred_high": pred_h, "actual_high": actual["High"],
                "pred_low": pred_l, "actual_low": actual["Low"],
                "pred_close": pred_c, "actual_close": actual["Close"],
                "abs_error_open": abs(diff_o),
                "abs_error_pct_open": abs(pct_o),
                "abs_error": abs(diff_c),
                "abs_error_pct": abs(pct_c),
                "direction_correct": int(
                    (pred_c > actual["prev_close"]) == (actual["Close"] > actual["prev_close"])
                )
            })

            logger.info(
                f"{clean} [{actual.get('source')}]: "
                f"O {pred_o:.2f}→{actual['Open']:.2f} | "
                f"C {pred_c:.2f}→{actual['Close']:.2f}"
            )

            try:
                OHLCPredictor(symbol).train(use_error_weights=True, do_tune=False)
            except Exception as e:
                logger.warning(f"Retrain {symbol}: {e}")

        if not results:
            send_telegram(f"⚠️ Evening: Could not fetch actual data on `{today}`")
            return

        err_path = Path(cfg.paths.errors_file)
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_df = pd.DataFrame(error_rows)
        err_df.to_csv(err_path, mode="a", header=not err_path.exists(), index=False)

        # Telegram
        lines = [
            f"*ACTUAL vs PREDICTED*",
            f"Date: `{today}`",
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
        lines += ["```", "", "*CLOSE*", "```",
                  f"{'Stock':<12} {'Pred C':>9} {'Actual C':>9} {'Diff':>8} {'Err%':>7}",
                  "-" * 50]
        for r in results:
            lines.append(
                f"{r['symbol']:<12} {r['pred_c']:>9.2f} {r['actual_c']:>9.2f} "
                f"{r['diff_c']:>+8.2f} {r['pct_c']:>+6.2f}%"
            )
        lines.append("```")
        lines.append("")
        lines.append("*Summary*")
        lines.append(f"• Stocks: `{len(results)}`")
        lines.append(f"• Avg Error Open: `{err_df['abs_error_pct_open'].mean():.2f}%`")
        lines.append(f"• Avg Error Close: `{err_df['abs_error_pct'].mean():.2f}%`")
        lines.append(f"• Directional Acc: `{err_df['direction_correct'].mean()*100:.1f}%`")
        lines.append(f"_Done in {(datetime.now()-start).seconds}s_")

        send_telegram("\n".join(lines))
        detect_drift()

    except Exception as e:
        logger.error(traceback.format_exc())
        send_telegram(f"❌ Evening Failed\n`{today}`\n```{str(e)[:700]}```")

    logger.info("Evening Job finished")


if __name__ == "__main__":
    main()

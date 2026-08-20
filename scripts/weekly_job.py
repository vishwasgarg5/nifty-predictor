#!/usr/bin/env python3
"""
Weekly Job (Saturday)
- Precision report
- Drift summary
- Full retrain + hyperparameter tuning
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import pandas as pd
from datetime import datetime, timedelta
import traceback

from src.config import cfg
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
    logger.info(f"Weekly Job started | {today_str}")

    try:
        errors_file = Path(cfg.paths.errors_file)
        if not errors_file.exists():
            send_telegram("⚠️ Weekly Report: No error history yet.")
            return

        df = pd.read_csv(errors_file)
        df["date"] = pd.to_datetime(df["date"])

        cutoff = start_time - timedelta(days=10)
        week = df[df["date"] >= cutoff].copy()

        if week.empty or len(week) < 3:
            send_telegram("⚠️ Weekly Report: Insufficient data.")
            return

        # Metrics
        mape = week["abs_error_pct"].mean()
        mae = week["abs_error"].mean()
        dir_acc = week["direction_correct"].mean() * 100 if "direction_correct" in week.columns else 0

        stock_perf = week.groupby("symbol")["abs_error_pct"].mean().sort_values()
        best = stock_perf.head(3)
        worst = stock_perf.tail(3)

        start_d = week["date"].min().strftime("%d-%b")
        end_d = week["date"].max().strftime("%d-%b-%Y")

        # Previous week trend
        prev_cut = cutoff - timedelta(days=7)
        prev = df[(df["date"] >= prev_cut) & (df["date"] < cutoff)]
        trend = ""
        if not prev.empty:
            prev_mape = prev["abs_error_pct"].mean()
            delta = prev_mape - mape
            if delta > 0.05:
                trend = f"• MAPE improved by `{delta:.2f}%` ✅"
            elif delta < -0.05:
                trend = f"• MAPE worsened by `{abs(delta):.2f}%` ⚠️"
            else:
                trend = "• MAPE almost unchanged"

        # Drift
        drift = detect_drift()

        # ---------- Full retrain + hyperparameter tuning ----------
        logger.info("Starting weekly full retrain + tuning...")
        symbols = week["symbol"].unique().tolist()
        retrained = 0
        for symbol in symbols:
            try:
                predictor = OHLCPredictor(symbol)
                # do_tune=True only on Saturday
                if predictor.train(use_error_weights=True, do_tune=cfg.tuning.enabled):
                    retrained += 1
            except Exception as e:
                logger.warning(f"Weekly retrain failed {symbol}: {e}")

        logger.info(f"Retrained {retrained}/{len(symbols)} models")

        # Telegram report
        lines = [
            f"📊 *Weekly Model Precision Report*",
            f"Period: `{start_d}` → `{end_d}`",
            f"Records: `{len(week)}`",
            "",
            "*Overall (Close)*",
            f"• MAPE: `{mape:.2f}%`",
            f"• MAE: `₹{mae:.2f}`",
            f"• Directional Accuracy: `{dir_acc:.1f}%`",
            ""
        ]

        lines.append("*Best Stocks*")
        for sym, err in best.items():
            lines.append(f"• {sym.replace('.NS','')}: `{err:.2f}%`")

        lines.append("")
        lines.append("*Needs Improvement*")
        for sym, err in worst.items():
            lines.append(f"• {sym.replace('.NS','')}: `{err:.2f}%`")

        if trend:
            lines.append("")
            lines.append("*Trend vs Previous Week*")
            lines.append(trend)

        lines.append("")
        lines.append("*Drift Status*")
        lines.append(f"• Status: `{drift.get('status', 'n/a')}`")
        if drift.get("mape") is not None:
            lines.append(f"• Recent MAPE: `{drift['mape']}%`")
            lines.append(f"• Recent DirAcc: `{drift.get('dir_acc', 0)}%`")

        lines.append("")
        lines.append(f"*Models retrained this week:* `{retrained}`")
        if mape < 1.5:
            status = "Excellent ✅"
        elif mape < 2.5:
            status = "Good 👍"
        elif mape < 4.0:
            status = "Average 😐"
        else:
            status = "Needs attention ⚠️"
        lines.append(f"*Model Status:* {status}")
        lines.append(f"_Finished in {(datetime.now() - start_time).seconds}s_")

        send_telegram("\n".join(lines))

        # Save weekly summary
        weekly_path = Path(cfg.paths.weekly_file)
        weekly_path.parent.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame([{
            "report_date": today_str,
            "period_start": week["date"].min().strftime("%Y-%m-%d"),
            "period_end": week["date"].max().strftime("%Y-%m-%d"),
            "records": len(week),
            "mape": round(mape, 3),
            "mae": round(mae, 2),
            "directional_accuracy": round(dir_acc, 1),
            "models_retrained": retrained,
            "drift_status": drift.get("status", "n/a")
        }])
        if weekly_path.exists():
            summary.to_csv(weekly_path, mode="a", header=False, index=False)
        else:
            summary.to_csv(weekly_path, index=False)

    except Exception as e:
        logger.error(traceback.format_exc())
        send_telegram(f"❌ *Weekly Job Failed*\n`{today_str}`\n```{str(e)[:700]}```")

    logger.info("Weekly Job finished")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

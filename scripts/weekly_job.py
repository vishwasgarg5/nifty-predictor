#!/usr/bin/env python3
"""
Weekly Precision Report
Runs every Saturday ~10:00 IST
Shows how accurate the model has been over the last trading week.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import traceback

from src.config import cfg
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
    logger.info(f"Weekly Report Job started | {today_str}")

    try:
        errors_file = Path(cfg.paths.errors_file)

        if not errors_file.exists():
            send_telegram("⚠️ Weekly Report: No error history file found yet.")
            logger.warning("No error history file found")
            return

        df = pd.read_csv(errors_file)
        df["date"] = pd.to_datetime(df["date"])

        # --------------------------------------------------
        # Filter last 7-10 calendar days (covers one trading week)
        # --------------------------------------------------
        cutoff = start_time - timedelta(days=10)
        week_df = df[df["date"] >= cutoff].copy()

        if week_df.empty or len(week_df) < 3:
            send_telegram(
                f"⚠️ Weekly Report: Insufficient data "
                f"(only {len(week_df)} records found)."
            )
            logger.warning("Not enough data for weekly report")
            return

        # --------------------------------------------------
        # Calculate Metrics
        # --------------------------------------------------
        mape = week_df["abs_error_pct"].mean()
        mae = week_df["abs_error"].mean()
        directional_acc = week_df["direction_correct"].mean() * 100

        # Best & Worst stocks
        stock_performance = (
            week_df.groupby("symbol")["abs_error_pct"]
            .mean()
            .sort_values()
        )

        best_stocks = stock_performance.head(3)
        worst_stocks = stock_performance.tail(3)

        # Date range
        start_date = week_df["date"].min().strftime("%d-%b")
        end_date = week_df["date"].max().strftime("%d-%b-%Y")

        # --------------------------------------------------
        # Previous week comparison (optional)
        # --------------------------------------------------
        prev_cutoff = cutoff - timedelta(days=7)
        prev_week = df[(df["date"] >= prev_cutoff) & (df["date"] < cutoff)]

        trend_text = ""
        if not prev_week.empty:
            prev_mape = prev_week["abs_error_pct"].mean()
            mape_change = prev_mape - mape
            if mape_change > 0.05:
                trend_text = f"• MAPE improved by `{mape_change:.2f}%` ✅"
            elif mape_change < -0.05:
                trend_text = f"• MAPE worsened by `{abs(mape_change):.2f}%` ⚠️"
            else:
                trend_text = "• MAPE almost unchanged"

        # --------------------------------------------------
        # Build Telegram Message
        # --------------------------------------------------
        lines = [
            f"📊 *Weekly Model Precision Report*",
            f"Period: `{start_date}` → `{end_date}`",
            f"Records analysed: `{len(week_df)}`",
            "",
            "*Overall Performance (Close Price)*",
            f"• MAPE: `{mape:.2f}%`",
            f"• MAE: `₹{mae:.2f}`",
            f"• Directional Accuracy: `{directional_acc:.1f}%`",
            ""
        ]

        # Best stocks
        lines.append("*Best Performing Stocks*")
        for sym, err in best_stocks.items():
            clean = sym.replace(".NS", "")
            lines.append(f"• {clean}: `{err:.2f}%` error")

        lines.append("")

        # Worst stocks
        lines.append("*Needs Improvement*")
        for sym, err in worst_stocks.items():
            clean = sym.replace(".NS", "")
            lines.append(f"• {clean}: `{err:.2f}%` error")

        if trend_text:
            lines.append("")
            lines.append("*Trend vs Previous Week*")
            lines.append(trend_text)

        # Status
        lines.append("")
        if mape < 1.5:
            status = "Excellent ✅"
        elif mape < 2.5:
            status = "Good 👍"
        elif mape < 4.0:
            status = "Average 😐"
        else:
            status = "Needs attention ⚠️"

        lines.append(f"*Model Status:* {status}")
        lines.append(f"_Report generated in {(datetime.now() - start_time).seconds}s_")

        message = "\n".join(lines)

        # --------------------------------------------------
        # Send + optionally save weekly summary
        # --------------------------------------------------
        success = send_telegram(message)

        if success:
            logger.info("Weekly report sent successfully")
        else:
            logger.error("Failed to send weekly report")

        # Save summary for long-term tracking
        weekly_path = Path(cfg.paths.weekly_file)
        weekly_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "report_date": today_str,
            "period_start": week_df["date"].min().strftime("%Y-%m-%d"),
            "period_end": week_df["date"].max().strftime("%Y-%m-%d"),
            "records": len(week_df),
            "mape": round(mape, 3),
            "mae": round(mae, 2),
            "directional_accuracy": round(directional_acc, 1)
        }

        summary_df = pd.DataFrame([summary])
        if weekly_path.exists():
            summary_df.to_csv(weekly_path, mode="a", header=False, index=False)
        else:
            summary_df.to_csv(weekly_path, index=False)

        logger.info(f"Weekly summary saved → {weekly_path}")

    except Exception as e:
        error_msg = f"❌ *Weekly Report Failed*\n`{today_str}`\n\n```{str(e)[:700]}```"
        logger.error(traceback.format_exc())
        send_telegram(error_msg)

    logger.info("Weekly Report Job finished")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

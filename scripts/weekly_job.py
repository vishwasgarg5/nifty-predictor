import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from src.config import cfg
from src.telegram_utils import send_telegram
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    errors_file = Path(cfg.paths.errors_file)
    if not errors_file.exists():
        send_telegram("No error history found for weekly report.")
        return

    df = pd.read_csv(errors_file)
    df["date"] = pd.to_datetime(df["date"])
    
    # Last 5-7 trading days
    cutoff = datetime.now() - timedelta(days=10)
    week = df[df["date"] >= cutoff].copy()
    
    if week.empty:
        send_telegram("Insufficient data for weekly report.")
        return

    # Metrics
    mape_close = (week["abs_error_pct"].mean())
    mae_close = week["abs_error"].mean()
    directional = (week["direction_correct"].mean() * 100) if "direction_correct" in week.columns else 0

    best = week.groupby("symbol")["abs_error_pct"].mean().nsmallest(3)
    worst = week.groupby("symbol")["abs_error_pct"].mean().nlargest(3)

    msg = f"""📊 *Weekly Model Precision Report*
Period: {week['date'].min().date()} → {week['date'].max().date()}

*Overall (Close)*
• MAPE: `{mape_close:.2f}%`
• MAE: `₹{mae_close:.2f}`
• Directional Accuracy: `{directional:.1f}%`

*Best Stocks*
"""
    for sym, err in best.items():
        msg += f"• {sym.replace('.NS','')}: {err:.2f}%\n"

    msg += "\n*Needs Improvement*\n"
    for sym, err in worst.items():
        msg += f"• {sym.replace('.NS','')}: {err:.2f}%\n"

    msg += "\nModel Status: Tracking active ✅"
    send_telegram(msg)
    logger.info("Weekly report sent")

if __name__ == "__main__":
    main()

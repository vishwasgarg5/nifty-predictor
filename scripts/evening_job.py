import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from src.config import cfg
from src.holidays import is_trading_day
from src.data_loader import download_history
from src.telegram_utils import send_telegram
from src.model import OHLCPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    if not is_trading_day():
        logger.info("Not a trading day. Skipping evening job.")
        return

    logger.info("=== Evening Job Started ===")
    today = datetime.now().strftime("%Y-%m-%d")

    pred_file = Path(cfg.paths.predictions) / f"{today}.csv"
    if not pred_file.exists():
        # Try yesterday in case of delay
        from datetime import timedelta
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        pred_file = Path(cfg.paths.predictions) / f"{yesterday}.csv"

    if not pred_file.exists():
        send_telegram(f"⚠️ No prediction file found for {today}")
        return

    preds = pd.read_csv(pred_file)
    if "symbol" not in preds.columns:
        preds = preds.reset_index().rename(columns={"index": "symbol"})

    lines = [f"*Actual vs Predicted* ({today})\n"]
    error_rows = []

    for _, row in preds.iterrows():
        symbol = row["symbol"]
        try:
            hist = download_history(symbol, period="5d")
            if hist is None or len(hist) < 1:
                continue

            actual = hist.iloc[-1]
            pred_close = row.get("Close") or row.get("pred_close")
            actual_close = actual["Close"]

            diff = actual_close - pred_close
            pct = (diff / pred_close) * 100 if pred_close else 0

            lines.append(
                f"*{symbol.replace('.NS', '')}*\n"
                f"Pred C: `{pred_close:.2f}` → Actual C: `{actual_close:.2f}`\n"
                f"Diff: `{diff:+.2f}` (`{pct:+.2f}%`)\n"
            )

            error_rows.append({
                "date": today,
                "symbol": symbol,
                "pred_close": pred_close,
                "actual_close": actual_close,
                "abs_error": abs(diff),
                "abs_error_pct": abs(pct),
                "direction_correct": int((pred_close > row.get("prev_close", pred_close)) == 
                                         (actual_close > row.get("prev_close", actual_close)))
            })

            # Light retrain with new data
            predictor = OHLCPredictor(symbol)
            predictor.train()

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    # Save errors
    err_path = Path(cfg.paths.errors_file)
    err_path.parent.mkdir(parents=True, exist_ok=True)
    err_df = pd.DataFrame(error_rows)
    if err_path.exists():
        err_df.to_csv(err_path, mode="a", header=False, index=False)
    else:
        err_df.to_csv(err_path, index=False)

    send_telegram("\n".join(lines))
    logger.info("Evening job completed")

if __name__ == "__main__":
    main()

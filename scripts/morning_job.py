#!/usr/bin/env python3
"""
Morning Job - Top 5 Selection + OHLC Prediction + FinBERT Sentiment
Runs every trading day ~08:45 IST
"""
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import traceback

from src.config import cfg
from src.holidays import is_trading_day
from src.data_loader import get_universe_symbols
from src.scoring import select_top5
from src.model import OHLCPredictor
from src.sentiment import get_sentiment_engine
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
    logger.info(f"Morning Job started | {today_str}")

    # --------------------------------------------------
    # 1. Holiday / Weekend check
    # --------------------------------------------------
    if not is_trading_day():
        logger.info("Today is not a trading day. Exiting cleanly.")
        return

    try:
        # --------------------------------------------------
        # 2. Get universe
        # --------------------------------------------------
        symbols = get_universe_symbols()
        logger.info(f"Universe loaded: {len(symbols)} symbols")

        # --------------------------------------------------
        # 3. Select Top 5
        # --------------------------------------------------
        logger.info("Scoring stocks and selecting Top 5...")
        top5_df = select_top5(symbols, top_n=cfg.top_n)

        if top5_df.empty:
            send_telegram(f"⚠️ Morning Job: No stocks passed filters on {today_str}")
            logger.error("No stocks selected")
            return

        logger.info(f"Top 5 selected:\n{top5_df[['symbol', 'score']].to_string(index=False)}")

        # --------------------------------------------------
        # 4. Predict OHLC for each stock
        # --------------------------------------------------
        predictions = {}
        logger.info("Running OHLC predictions...")

        for _, row in top5_df.iterrows():
            symbol = row["symbol"]
            try:
                predictor = OHLCPredictor(symbol)
                pred = predictor.predict_next()
                if pred:
                    predictions[symbol] = pred
                    logger.info(f"{symbol} → C: {pred['Close']}")
                else:
                    logger.warning(f"Prediction failed for {symbol}")
            except Exception as e:
                logger.error(f"Prediction error {symbol}: {e}")

        if not predictions:
            send_telegram(f"⚠️ Morning Job: All predictions failed on {today_str}")
            return

        # --------------------------------------------------
        # 5. Sentiment Analysis (FinBERT + fallback)
        # --------------------------------------------------
        logger.info("Running sentiment analysis...")
        sentiment_engine = get_sentiment_engine()
        sentiments = {}

        for symbol in predictions.keys():
            try:
                sent = sentiment_engine.analyze_stock(
                    symbol, 
                    max_articles=cfg.sentiment.max_articles
                )
                sentiments[symbol] = sent
                logger.info(f"{symbol} sentiment: {sent.overall_score:+.2f} ({sent.overall_label})")
            except Exception as e:
                logger.error(f"Sentiment failed for {symbol}: {e}")
                sentiments[symbol] = None

        # --------------------------------------------------
        # 6. Save predictions (for evening job)
        # --------------------------------------------------
        pred_dir = Path(cfg.paths.predictions)
        pred_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for symbol, pred in predictions.items():
            score_val = top5_df.loc[top5_df["symbol"] == symbol, "score"].values
            records.append({
                "date": today_str,
                "symbol": symbol,
                "Open": pred["Open"],
                "High": pred["High"],
                "Low": pred["Low"],
                "Close": pred["Close"],
                "score": float(score_val[0]) if len(score_val) > 0 else 0.0
            })

        pred_df = pd.DataFrame(records)
        pred_file = pred_dir / f"{today_str}.csv"
        pred_df.to_csv(pred_file, index=False)
        logger.info(f"Predictions saved → {pred_file}")

        # --------------------------------------------------
        # 7. Build clean Telegram Message
        # --------------------------------------------------
        lines = [
            f"*TOP 5 STOCK PREDICTIONS*",
            f"Date: `{today_str}`",
            f"Universe: `{cfg.universe.upper()}`",
            "",
            "```",
            f"{'Stock':<12} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8}",
            "-" * 48
        ]

        for _, row in top5_df.iterrows():
            symbol = row["symbol"]
            clean = symbol.replace(".NS", "")
            pred = predictions.get(symbol)

            if not pred:
                continue

            lines.append(
                f"{clean:<12} {pred['Open']:>8.2f} {pred['High']:>8.2f} "
                f"{pred['Low']:>8.2f} {pred['Close']:>8.2f}"
            )

        lines.append("```")
        lines.append("")

        # Optional: Add short sentiment summary
        lines.append("*Sentiment Summary*")
        for _, row in top5_df.iterrows():
            symbol = row["symbol"]
            clean = symbol.replace(".NS", "")
            sent = sentiments.get(symbol)

            if sent and sent.article_count > 0:
                emoji = "🟢" if sent.overall_score >= 0.15 else "🔴" if sent.overall_score <= -0.15 else "⚪"
                lines.append(f"{emoji} {clean}: `{sent.overall_score:+.2f}` ({sent.overall_label})")

        lines.append("")
        lines.append(f"_Job finished in {(datetime.now() - start_time).seconds}s_")

        message = "\n".join(lines)
        success = send_telegram(message)

        if success:
            logger.info("Telegram message sent successfully")
        else:
            logger.error("Failed to send Telegram message")

    except Exception as e:
        error_msg = f"❌ *Morning Job Failed*\n`{today_str}`\n\n```{str(e)[:800]}```"
        logger.error(traceback.format_exc())
        send_telegram(error_msg)

    logger.info("Morning Job finished")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

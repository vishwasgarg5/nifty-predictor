#!/usr/bin/env python3
"""
Morning Job
- Select Top 5 from Nifty universe (fast 2-stage scoring)
- Predict next-day OHLC
- FinBERT sentiment on Top 5 only
- Clean table output
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import pandas as pd
from datetime import datetime
import traceback

from src.config import cfg
from src.holidays import is_trading_day
from src.universe import get_universe_symbols
from src.scoring import select_top5
from src.model import OHLCPredictor
from src.sentiment import get_sentiment_engine
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
    logger.info(f"Morning Job started | {today_str}")

    if not is_trading_day():
        logger.info("Not a trading day. Exit.")
        return

    try:
        # 1. Universe
        symbols = get_universe_symbols()
        logger.info(f"Universe size: {len(symbols)}")

        # 2. Top 5
        top5_df = select_top5(symbols, top_n=cfg.top_n)
        if top5_df.empty:
            send_telegram(f"⚠️ Morning Job: No stocks passed filters on `{today_str}`")
            return

        # 3. Predictions
        predictions = {}
        for _, row in top5_df.iterrows():
            symbol = row["symbol"]
            try:
                predictor = OHLCPredictor(symbol)
                pred = predictor.predict_next()
                if pred:
                    predictions[symbol] = pred
                    logger.info(f"{symbol} → C:{pred['Close']}")
            except Exception as e:
                logger.error(f"Prediction failed {symbol}: {e}")

        if not predictions:
            send_telegram(f"⚠️ Morning Job: All predictions failed on `{today_str}`")
            return

        # 4. Sentiment (only Top 5)
        sentiments = {}
        try:
            engine = get_sentiment_engine()
            for symbol in predictions:
                try:
                    sentiments[symbol] = engine.analyze_stock(
                        symbol, max_articles=cfg.sentiment.max_articles
                    )
                except Exception as e:
                    logger.warning(f"Sentiment failed {symbol}: {e}")
                    sentiments[symbol] = None
        except Exception as e:
            logger.warning(f"Sentiment engine failed: {e}")

        # 5. Save predictions
        pred_dir = Path(cfg.paths.predictions)
        pred_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for symbol, pred in predictions.items():
            score_val = top5_df.loc[top5_df["symbol"] == symbol, "score"]
            records.append({
                "date": today_str,
                "symbol": symbol,
                "Open": pred["Open"],
                "High": pred["High"],
                "Low": pred["Low"],
                "Close": pred["Close"],
                "score": float(score_val.values[0]) if len(score_val) else 0.0
            })
        pred_df = pd.DataFrame(records)
        pred_file = pred_dir / f"{today_str}.csv"
        pred_df.to_csv(pred_file, index=False)
        logger.info(f"Saved → {pred_file}")

        # 6. Telegram – clean table
        lines = [
            f"*TOP 5 STOCK PREDICTIONS*",
            f"Date: `{today_str}` | Universe: `{cfg.universe.upper()}`",
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
        lines.append("*Sentiment*")

        for _, row in top5_df.iterrows():
            symbol = row["symbol"]
            clean = symbol.replace(".NS", "")
            sent = sentiments.get(symbol)
            if sent and getattr(sent, "article_count", 0) > 0:
                emoji = "🟢" if sent.overall_score >= 0.15 else "🔴" if sent.overall_score <= -0.15 else "⚪"
                lines.append(
                    f"{emoji} {clean}: `{sent.overall_score:+.2f}` ({sent.overall_label})"
                )

        lines.append("")
        lines.append(f"_Finished in {(datetime.now() - start_time).seconds}s_")

        send_telegram("\n".join(lines))
        logger.info("Morning Telegram sent")

    except Exception as e:
        logger.error(traceback.format_exc())
        send_telegram(f"❌ *Morning Job Failed*\n`{today_str}`\n```{str(e)[:700]}```")

    logger.info("Morning Job finished")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

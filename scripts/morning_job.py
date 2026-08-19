import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from src.config import cfg
from src.holidays import is_trading_day
from src.data_loader import get_universe_symbols, download_history
from src.telegram_utils import send_telegram
from src.sentiment import get_sentiment_engine
from src.scoring import select_top5
from src.model import OHLCPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    if not is_trading_day():
        logger.info("Not a trading day. Exiting.")
        return

    logger.info("=== Morning Job Started ===")
    today = datetime.now().strftime("%Y-%m-%d")

    symbols = get_universe_symbols()
    logger.info(f"Universe size: {len(symbols)}")

    symbols = get_universe_symbols()
    top5_df = select_top5(symbols, top_n=cfg.top_n)
    
    predictions = {}
    for _, row in top5_df.iterrows():
        pred = OHLCPredictor(row["symbol"])
        predictions[row["symbol"]] = pred.predict_next()
    # In real use, replace with actual select_top5 + OHLCPredictor
    top5 = [
        {"symbol": "RELIANCE.NS", "score": 8.4},
        {"symbol": "TCS.NS", "score": 7.9},
        {"symbol": "HDFCBANK.NS", "score": 7.6},
        {"symbol": "INFY.NS", "score": 7.2},
        {"symbol": "ICICIBANK.NS", "score": 7.0},
    ]

    engine = get_sentiment_engine()
    sentiments = {}
    for s in top5:
        sentiments[s["symbol"]] = engine.analyze_stock(s["symbol"])

    # Build message
    lines = [f"*Top 5 Predictions + Sentiment* ({today})\n"]
    for item in top5:
        sym = item["symbol"]
        sent = sentiments.get(sym)
        lines.append(f"*{sym.replace('.NS','')}*  Score: {item['score']}")
        lines.append(f"Pred → O: —  H: —  L: —  C: —")  # replace with real preds
        if sent and sent.article_count > 0:
            emoji = "🟢" if sent.overall_score > 0.15 else "🔴" if sent.overall_score < -0.15 else "⚪"
            lines.append(f"{emoji} Sentiment: {sent.overall_score:+.2f} ({sent.overall_label}) [{sent.method}]")
            for title, sc in sent.headlines[:2]:
                lines.append(f"  • {title[:65]}... ({sc:+.2f})")
        lines.append("")

    send_telegram("\n".join(lines))
    logger.info("Morning job completed")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging, traceback
from datetime import datetime
import pandas as pd

from src.config import cfg
from src.holidays import is_trading_day
from src.universe import get_universe_symbols
from src.scoring import select_top5
from src.model import OHLCPredictor
from src.sentiment import get_sentiment_engine
from src.indexes import predict_indexes
from src.ipo import ipo_watchlist_telegram_lines
from src.ipo_gmp import build_ipo_desk_message
from src.telegram_utils import send_telegram

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def _uni_label():
    if hasattr(cfg, "universes") and getattr(cfg.universes, "primary", None):
        return "+".join(str(x).upper() for x in cfg.universes.primary)
    return "NIFTY"

def main():
    start, today = datetime.now(), datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Morning start {today}")
    if not is_trading_day():
        # Still send IPO desk on non-trading days if enabled
        if getattr(cfg, "ipo_desk", None) and cfg.ipo_desk.enabled:
            send_telegram(build_ipo_desk_message(cfg.ipo_desk.max_rows))
        return

    try:
        symbols = get_universe_symbols()
        top5 = select_top5(symbols, cfg.top_n)
        if top5.empty:
            send_telegram(f"⚠️ No stocks selected `{today}`")
            return

        predictions = {}
        for _, row in top5.iterrows():
            p = OHLCPredictor(row["symbol"]).predict_next()
            if p:
                predictions[row["symbol"]] = p

        sentiments = {}
        try:
            eng = get_sentiment_engine()
            for s in predictions:
                try:
                    sentiments[s] = eng.analyze_stock(s, cfg.sentiment.max_articles)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(e)

        pred_dir = Path(cfg.paths.predictions)
        pred_dir.mkdir(parents=True, exist_ok=True)
        recs = []
        for sym, pred in predictions.items():
            sc = top5.loc[top5.symbol == sym, "score"]
            recs.append({"date": today, "symbol": sym, **pred, "score": float(sc.values[0]) if len(sc) else 0})
        pd.DataFrame(recs).to_csv(pred_dir / f"{today}.csv", index=False)

        index_preds = predict_indexes()
        if index_preds:
            pd.DataFrame(index_preds).to_csv(pred_dir / f"{today}_indexes.csv", index=False)

        # ---- Telegram: stocks ----
        lines = [
            f"*TOP 5 STOCK PREDICTIONS*",
            f"Date: `{today}` | `{_uni_label()}`",
            "", "```",
            f"{'Stock':<12} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8}",
            "-" * 48,
        ]
        for _, row in top5.iterrows():
            sym, pred = row["symbol"], predictions.get(row["symbol"])
            if not pred:
                continue
            c = sym.replace(".NS", "")
            lines.append(f"{c:<12} {pred['Open']:>8.2f} {pred['High']:>8.2f} {pred['Low']:>8.2f} {pred['Close']:>8.2f}")
        lines.append("```")

        # indexes
        if index_preds:
            lines += ["", "*INDEX PREDICTIONS*", "```",
                      f"{'Index':<12} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8}", "-" * 48]
            for p in index_preds:
                lines.append(f"{p['name']:<12} {p['Open']:>8.2f} {p['High']:>8.2f} {p['Low']:>8.2f} {p['Close']:>8.2f}")
            lines.append("```")

        # sentiment
        lines.append("")
        lines.append("*Sentiment*")
        for _, row in top5.iterrows():
            s = sentiments.get(row["symbol"])
            if s and s.article_count:
                em = "🟢" if s.overall_score >= 0.15 else "🔴" if s.overall_score <= -0.15 else "⚪"
                lines.append(f"{em} {row['symbol'].replace('.NS','')}: `{s.overall_score:+.2f}` ({s.overall_label})")

        # listed IPO watchlist
        lines += [""] + ipo_watchlist_telegram_lines()

        lines.append(f"\n_Finished in {(datetime.now()-start).seconds}s_")
        send_telegram("\n".join(lines))

        # separate IPO desk message (GMP)
        if getattr(cfg, "ipo_desk", None) and cfg.ipo_desk.enabled:
            send_telegram(build_ipo_desk_message(int(getattr(cfg.ipo_desk, "max_rows", 10))))

        logger.info("Morning done")
    except Exception as e:
        logger.error(traceback.format_exc())
        send_telegram(f"❌ Morning failed\n```{str(e)[:700]}```")

if __name__ == "__main__":
    main()

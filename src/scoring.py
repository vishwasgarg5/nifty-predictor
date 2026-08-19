# src/scoring.py
import logging
import pandas as pd
import yfinance as yf
from src.config import cfg
from src.data_loader import download_history
from src.features import create_features

logger = logging.getLogger(__name__)


def compute_stock_score(symbol: str) -> dict | None:
    """
    Calculate a composite score for a stock.
    Returns None if the stock should be rejected.
    """
    try:
        # 1. Download price history
        hist = download_history(symbol, period="6mo")
        if hist is None or len(hist) < 50:
            logger.info(f"{symbol} → Rejected: insufficient price history")
            return None

        # 2. Create features
        try:
            feat = create_features(hist)
        except Exception as e:
            logger.warning(f"{symbol} → Feature creation failed: {e}")
            return None

        if feat.empty or len(feat) < 30:
            logger.info(f"{symbol} → Rejected: not enough feature rows after dropna")
            return None

        last = feat.iloc[-1]

        # -------------------------------------------------
        # Technical Score
        # -------------------------------------------------
        tech = 0.0

        # RSI
        rsi = last.get("rsi_14", 50)
        if rsi < cfg.scoring.rsi_oversold:
            tech += 1.8
        elif rsi < 50:
            tech += 0.9
        elif rsi > 70:
            tech -= 0.5   # slightly penalize overbought

        # Trend (SMA)
        if last["Close"] > last.get("sma_20", last["Close"]):
            tech += 1.2
        if last["Close"] > last.get("sma_50", last["Close"]):
            tech += 1.0

        # MACD
        macd_val = last.get("MACD_12_26_9", 0)
        macd_signal = last.get("MACDs_12_26_9", 0)
        if macd_val > macd_signal:
            tech += 1.0

        # Volume spike
        vol_ratio = last.get("vol_ratio", 1.0)
        if vol_ratio > cfg.scoring.volume_spike:
            tech += 1.0

        # -------------------------------------------------
        # Risk Filters (soft)
        # -------------------------------------------------
        atr_pct = last.get("atr_pct", 0)
        if atr_pct > cfg.scoring.max_atr_pct:
            logger.info(f"{symbol} → Rejected: high ATR {atr_pct:.2f}% > {cfg.scoring.max_atr_pct}%")
            return None

        # -------------------------------------------------
        # Fundamental Score (soft – don’t reject easily)
        # -------------------------------------------------
        fund = 0.0
        pe = pb = roe = avg_vol = None

        try:
            info = yf.Ticker(symbol).info
            pe = info.get("trailingPE") or info.get("forwardPE")
            pb = info.get("priceToBook")
            roe = info.get("returnOnEquity")
            avg_vol = info.get("averageVolume") or info.get("averageDailyVolume10Day")

            if pe is not None and 0 < pe < cfg.scoring.pe_max:
                fund += 1.2
            if pb is not None and 0 < pb < cfg.scoring.pb_max:
                fund += 1.0
            if roe is not None and roe > cfg.scoring.roe_min:
                fund += 1.3

            # Soft volume filter
            if avg_vol is not None and avg_vol < cfg.scoring.min_avg_volume:
                logger.info(f"{symbol} → Rejected: low volume {avg_vol:,.0f}")
                return None

        except Exception as e:
            logger.debug(f"{symbol} → Fundamental data incomplete: {e}")

        # -------------------------------------------------
        # Final Score
        # -------------------------------------------------
        total = (
            tech * cfg.scoring.weights.technical +
            fund * cfg.scoring.weights.fundamental
        )

        result = {
            "symbol": symbol,
            "score": round(total, 2),
            "close": round(float(last["Close"]), 2),
            "rsi": round(float(rsi), 1),
            "atr_pct": round(float(atr_pct), 2),
            "pe": pe,
            "volume": avg_vol
        }

        logger.info(
            f"{symbol} → ACCEPTED | Score: {total:.2f} | "
            f"Tech: {tech:.1f} | Fund: {fund:.1f} | RSI: {rsi:.1f}"
        )
        return result

    except Exception as e:
        logger.error(f"{symbol} → Unexpected error: {e}")
        return None


def select_top5(symbols: list, top_n: int = 5) -> pd.DataFrame:
    """
    Score all symbols and return the top N.
    """
    results = []
    logger.info(f"Starting scoring of {len(symbols)} symbols...")

    for i, sym in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] Scoring {sym}")
        res = compute_stock_score(sym)
        if res:
            results.append(res)

    logger.info(f"Total stocks that passed filters: {len(results)}")

    if not results:
        logger.warning("No stocks passed the filters!")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    logger.info("Top selected stocks:")
    for _, row in df.iterrows():
        logger.info(f"  {row['symbol']} → {row['score']}")

    return df

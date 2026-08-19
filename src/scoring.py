# src/scoring.py
import logging
import pandas as pd
import yfinance as yf
from src.config import cfg
from src.data_loader import download_history
from src.features import create_features

logger = logging.getLogger(__name__)


def compute_stock_score(symbol: str) -> dict | None:
    try:
        hist = download_history(symbol, period="6mo")
        if hist is None or len(hist) < 60:
            logger.info(f"{symbol} → Rejected: bad/insufficient history")
            return None

        feat = create_features(hist)
        if feat is None or feat.empty or len(feat) < 30:
            logger.info(f"{symbol} → Rejected: feature creation failed or too few rows")
            return None

        last = feat.iloc[-1]

        # ---------- Technical Score (very forgiving) ----------
        tech = 0.0

        rsi = last.get("rsi_14", 50)
        if pd.isna(rsi):
            rsi = 50

        if rsi < 40:
            tech += 1.5
        elif rsi < 55:
            tech += 1.0
        elif rsi > 70:
            tech += 0.3

        if last["Close"] > last.get("sma_20", last["Close"]):
            tech += 1.2
        if last["Close"] > last.get("sma_50", last["Close"]):
            tech += 1.0

        # MACD (safe access)
        macd_col = next((c for c in feat.columns if "MACD_12_26_9" in str(c)), None)
        signal_col = next((c for c in feat.columns if "MACDs_12_26_9" in str(c)), None)

        if macd_col and signal_col:
            if last[macd_col] > last[signal_col]:
                tech += 1.0

        vol_ratio = last.get("vol_ratio", 1.0)
        if not pd.isna(vol_ratio) and vol_ratio > 1.1:
            tech += 0.8

        # ---------- Soft Risk Filter ----------
        atr_pct = last.get("atr_pct", 3.0)
        if pd.isna(atr_pct):
            atr_pct = 3.0

        if atr_pct > 10.0:          # only reject extremely volatile stocks
            logger.info(f"{symbol} → Rejected: extreme ATR {atr_pct:.2f}%")
            return None

        # ---------- Fundamental (bonus only) ----------
        fund = 0.0
        try:
            info = yf.Ticker(symbol).info
            pe = info.get("trailingPE") or info.get("forwardPE")
            pb = info.get("priceToBook")
            roe = info.get("returnOnEquity")

            if pe and 0 < pe < 50:
                fund += 1.0
            if pb and 0 < pb < 8:
                fund += 0.8
            if roe and roe > 0.10:
                fund += 1.0
        except Exception:
            pass

        total = tech * 0.75 + fund * 0.25

        result = {
            "symbol": symbol,
            "score": round(total, 2),
            "close": round(float(last["Close"]), 2),
            "rsi": round(float(rsi), 1),
            "atr_pct": round(float(atr_pct), 2)
        }

        logger.info(f"{symbol} → ACCEPTED | Score: {total:.2f} | RSI: {rsi:.1f}")
        return result

    except Exception as e:
        logger.error(f"{symbol} → Unexpected error: {e}")
        return None


def select_top5(symbols: list, top_n: int = 5) -> pd.DataFrame:
    """
    Score all symbols and return the top N.
    If no stock passes the filters, use an emergency fallback selection.
    """
    results = []
    logger.info(f"Starting scoring of {len(symbols)} symbols...")

    for i, sym in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] Scoring {sym}")
        res = compute_stock_score(sym)
        if res:
            results.append(res)

    logger.info(f"Total stocks that passed filters: {len(results)}")

    # -------------------------------------------------
    # Normal case – we have some stocks
    # -------------------------------------------------
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

        logger.info("Top selected stocks:")
        for _, row in df.iterrows():
            logger.info(f"  {row['symbol']} → Score: {row['score']}")

        return df

    # -------------------------------------------------
    # Emergency Fallback – no stock passed filters
    # -------------------------------------------------
    logger.warning("No stocks passed filters — activating emergency fallback selection")

    emergency = []
    for sym in symbols[:15]:          # try first 15 symbols
        try:
            hist = download_history(sym, period="3mo")
            if hist is not None and len(hist) > 30:
                last_close = float(hist["Close"].iloc[-1])
                emergency.append({
                    "symbol": sym,
                    "score": 5.0,                    # neutral score
                    "close": round(last_close, 2),
                    "rsi": 50.0,
                    "atr_pct": 2.5,
                    "pe": None,
                    "volume": None
                })
                logger.info(f"Emergency selected: {sym}")

                if len(emergency) >= top_n:
                    break
        except Exception as e:
            logger.warning(f"Emergency selection failed for {sym}: {e}")

    if not emergency:
        logger.error("Even emergency fallback failed – returning empty DataFrame")
        return pd.DataFrame()

    df = pd.DataFrame(emergency)
    logger.info(f"Emergency Top {len(df)} stocks selected")
    return df

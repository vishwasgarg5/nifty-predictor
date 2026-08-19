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
            return None

        feat = create_features(hist)
        last = feat.iloc[-1]

        # ----- Technical Score -----
        tech = 0.0

        # RSI
        if last["rsi_14"] < cfg.scoring.rsi_oversold:
            tech += 1.8
        elif last["rsi_14"] < 50:
            tech += 0.8

        # Trend
        if last["Close"] > last["sma_20"]:
            tech += 1.2
        if last["Close"] > last["sma_50"]:
            tech += 1.0

        # MACD
        if last.get("MACD_12_26_9", 0) > last.get("MACDs_12_26_9", 0):
            tech += 1.0

        # Volume
        if last["vol_ratio"] > cfg.scoring.volume_spike:
            tech += 1.0

        # ATR filter (risk)
        if last["atr_pct"] > cfg.scoring.max_atr_pct:
            return None   # too volatile

        # ----- Fundamental Score -----
        fund = 0.0
        try:
            info = yf.Ticker(symbol).info
            pe = info.get("trailingPE") or info.get("forwardPE")
            pb = info.get("priceToBook")
            roe = info.get("returnOnEquity")
            avg_vol = info.get("averageVolume")

            if pe and 0 < pe < cfg.scoring.pe_max:
                fund += 1.2
            if pb and 0 < pb < cfg.scoring.pb_max:
                fund += 1.0
            if roe and roe > cfg.scoring.roe_min:
                fund += 1.3
            if avg_vol and avg_vol < cfg.scoring.min_avg_volume:
                return None   # low liquidity
        except Exception:
            pass

        total = (tech * cfg.scoring.weights.technical +
                 fund * cfg.scoring.weights.fundamental)

        return {
            "symbol": symbol,
            "score": round(total, 2),
            "close": round(last["Close"], 2),
            "rsi": round(last["rsi_14"], 1),
            "atr_pct": round(last["atr_pct"], 2)
        }

    except Exception as e:
        logger.warning(f"Scoring failed for {symbol}: {e}")
        return None


def select_top5(symbols: list, top_n: int = 5) -> pd.DataFrame:
    results = []
    for i, sym in enumerate(symbols):
        logger.info(f"Scoring {i+1}/{len(symbols)} → {sym}")
        res = compute_stock_score(sym)
        if res:
            results.append(res)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False).head(top_n)
    return df.reset_index(drop=True)

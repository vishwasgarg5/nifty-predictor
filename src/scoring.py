import logging
import pandas as pd
import yfinance as yf
from src.config import cfg
from src.data_loader import download_history
from src.features import create_features

logger = logging.getLogger(__name__)

def _cheap_score(symbol: str) -> float | None:
    """Very fast filter using limited history."""
    hist = download_history(symbol, period="3mo", retries=2)
    if hist is None or len(hist) < 40:
        return None
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    close = hist["Close"]
    vol = hist["Volume"]
    ret5 = close.pct_change(5).iloc[-1]
    vol_ratio = vol.iloc[-1] / (vol.rolling(20).mean().iloc[-1] + 1e-9)
    rsi = 50
    try:
        import pandas_ta_classic as ta
        r = ta.rsi(close, length=14)
        if r is not None and not r.empty:
            rsi = float(r.iloc[-1])
    except Exception:
        pass
    score = 0.0
    if ret5 > 0:
        score += 1.0
    if vol_ratio > 1.1:
        score += 0.8
    if rsi < 45:
        score += 1.2
    elif rsi < 55:
        score += 0.6
    return score

def compute_stock_score(symbol: str) -> dict | None:
    hist = download_history(symbol, period="6mo", retries=2)
    if hist is None or len(hist) < cfg.min_history_days:
        return None
    feat = create_features(hist)
    if feat is None or feat.empty or len(feat) < 30:
        return None
    last = feat.iloc[-1]
    tech = 0.0
    rsi = last.get("rsi_14", 50)
    if pd.isna(rsi):
        rsi = 50
    if rsi < cfg.scoring.rsi_oversold:
        tech += 1.6
    elif rsi < 55:
        tech += 0.9
    if last["Close"] > last.get("sma_20", last["Close"]):
        tech += 1.1
    if last["Close"] > last.get("sma_50", last["Close"]):
        tech += 0.9
    atr_pct = last.get("atr_pct", 3.0)
    if pd.isna(atr_pct):
        atr_pct = 3.0
    if atr_pct > cfg.scoring.max_atr_pct:
        return None
    vol_ratio = last.get("vol_ratio", 1.0)
    if not pd.isna(vol_ratio) and vol_ratio > cfg.scoring.volume_spike:
        tech += 0.8
    fund = 0.0
    try:
        info = yf.Ticker(symbol).info
        pe = info.get("trailingPE") or info.get("forwardPE")
        pb = info.get("priceToBook")
        roe = info.get("returnOnEquity")
        if pe and 0 < pe < cfg.scoring.pe_max:
            fund += 1.0
        if pb and 0 < pb < cfg.scoring.pb_max:
            fund += 0.7
        if roe and roe > cfg.scoring.roe_min:
            fund += 0.9
    except Exception:
        pass
    total = tech * cfg.scoring.weights.technical + fund * cfg.scoring.weights.fundamental
    return {
        "symbol": symbol,
        "score": round(total, 2),
        "close": round(float(last["Close"]), 2),
        "rsi": round(float(rsi), 1),
        "atr_pct": round(float(atr_pct), 2)
    }

def select_top5(symbols: list, top_n: int = 5) -> pd.DataFrame:
    """
    2-stage selection for speed on Nifty 500:
    1) Cheap score on all (or first 200)
    2) Full score on prefilter_top
    """
    # Limit for GitHub Actions speed – take first 200 most common / or all if small
    symbols = symbols[:220]
    logger.info(f"Stage-1 cheap scoring on {len(symbols)} symbols...")

    cheap = []
    for i, sym in enumerate(symbols, 1):
        if i % 40 == 0:
            logger.info(f"  cheap progress {i}/{len(symbols)}")
        s = _cheap_score(sym)
        if s is not None:
            cheap.append({"symbol": sym, "cheap": s})

    if not cheap:
        logger.warning("Cheap stage empty – emergency fallback")
        return _emergency(symbols, top_n)

    cheap_df = pd.DataFrame(cheap).sort_values("cheap", ascending=False)
    candidates = cheap_df.head(cfg.scoring.prefilter_top)["symbol"].tolist()
    logger.info(f"Stage-2 full scoring on {len(candidates)} candidates...")

    results = []
    for sym in candidates:
        res = compute_stock_score(sym)
        if res:
            results.append(res)

    if not results:
        return _emergency(symbols, top_n)

    df = pd.DataFrame(results).sort_values("score", ascending=False).head(top_n)
    logger.info("Selected Top stocks:")
    for _, r in df.iterrows():
        logger.info(f"  {r['symbol']} → {r['score']}")
    return df.reset_index(drop=True)

def _emergency(symbols, top_n):
    rows = []
    for sym in symbols[:15]:
        hist = download_history(sym, period="3mo", retries=2)
        if hist is not None and len(hist) > 30:
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            rows.append({
                "symbol": sym,
                "score": 5.0,
                "close": float(hist["Close"].iloc[-1]),
                "rsi": 50.0,
                "atr_pct": 2.5
            })
            if len(rows) >= top_n:
                break
    return pd.DataFrame(rows)

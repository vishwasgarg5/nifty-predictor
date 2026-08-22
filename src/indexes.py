import logging
from src.config import cfg
from src.model import OHLCPredictor
from src.data_loader import get_actual_ohlc

logger = logging.getLogger(__name__)

def get_index_list():
    if not hasattr(cfg, "indexes") or not cfg.indexes.enabled:
        return []
    return list(cfg.indexes.list)

def predict_indexes():
    out = []
    for item in get_index_list():
        symbol = item.symbol if hasattr(item, "symbol") else item["symbol"]
        name = item.name if hasattr(item, "name") else item.get("name", symbol)
        try:
            pred = OHLCPredictor(symbol).predict_next()
            if pred:
                out.append({"symbol": symbol, "name": name, **pred})
        except Exception as e:
            logger.warning(f"index {name}: {e}")
    return out

def compare_indexes(pred_rows):
    rows = []
    for p in pred_rows:
        actual = get_actual_ohlc(p["symbol"])
        if not actual:
            continue
        rows.append({
            "name": p["name"],
            "pred_c": p["Close"], "actual_c": actual["Close"],
            "diff_c": actual["Close"] - p["Close"],
            "pct_c": (actual["Close"] - p["Close"]) / p["Close"] * 100 if p["Close"] else 0,
        })
    return rows

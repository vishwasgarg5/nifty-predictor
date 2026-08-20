import pandas as pd
import pandas_ta_classic as ta
import logging

logger = logging.getLogger(__name__)

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    need = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in need):
        return pd.DataFrame()

    df = df[need].dropna()
    if len(df) < 50:
        return pd.DataFrame()

    try:
        df["ret_1"] = df["Close"].pct_change()
        df["ret_5"] = df["Close"].pct_change(5)
        df["sma_10"] = ta.sma(df["Close"], length=10)
        df["sma_20"] = ta.sma(df["Close"], length=20)
        df["sma_50"] = ta.sma(df["Close"], length=50)
        df["rsi_14"] = ta.rsi(df["Close"], length=14)
        macd = ta.macd(df["Close"])
        if macd is not None and not macd.empty:
            df = pd.concat([df, macd], axis=1)
        df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        df["atr_pct"] = df["atr_14"] / df["Close"] * 100
        df["vol_sma_20"] = df["Volume"].rolling(20).mean()
        df["vol_ratio"] = df["Volume"] / df["vol_sma_20"]
        df["close_sma20_ratio"] = df["Close"] / df["sma_20"]

        df["target_o"] = df["Open"].shift(-1)
        df["target_h"] = df["High"].shift(-1)
        df["target_l"] = df["Low"].shift(-1)
        df["target_c"] = df["Close"].shift(-1)

        return df.dropna()
    except Exception as e:
        logger.error(f"features error: {e}")
        return pd.DataFrame()

# src/features.py
import pandas as pd
import pandas_ta_classic as ta
import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Very defensive feature creation.
    """
    df = df.copy()

    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure required columns exist
    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            logger.warning(f"Missing column: {col}")
            return pd.DataFrame()

    df = df[required].copy()
    df = df.dropna()

    if len(df) < 50:
        return pd.DataFrame()

    try:
        # Returns
        df["ret_1"] = df["Close"].pct_change()
        df["ret_3"] = df["Close"].pct_change(3)
        df["ret_5"] = df["Close"].pct_change(5)

        # Moving averages
        df["sma_10"] = ta.sma(df["Close"], length=10)
        df["sma_20"] = ta.sma(df["Close"], length=20)
        df["sma_50"] = ta.sma(df["Close"], length=50)
        df["ema_12"] = ta.ema(df["Close"], length=12)
        df["ema_26"] = ta.ema(df["Close"], length=26)

        # RSI
        df["rsi_14"] = ta.rsi(df["Close"], length=14)

        # MACD
        macd = ta.macd(df["Close"])
        if macd is not None and not macd.empty:
            df = pd.concat([df, macd], axis=1)

        # Bollinger Bands
        bb = ta.bbands(df["Close"], length=20)
        if bb is not None and not bb.empty:
            df = pd.concat([df, bb], axis=1)

        # ATR
        df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        df["atr_pct"] = (df["atr_14"] / df["Close"]) * 100

        # Volume
        df["vol_sma_20"] = df["Volume"].rolling(20).mean()
        df["vol_ratio"] = df["Volume"] / df["vol_sma_20"]

        # Relative strength
        df["close_sma20_ratio"] = df["Close"] / df["sma_20"]
        df["close_sma50_ratio"] = df["Close"] / df["sma_50"]

        # Targets
        df["target_o"] = df["Open"].shift(-1)
        df["target_h"] = df["High"].shift(-1)
        df["target_l"] = df["Low"].shift(-1)
        df["target_c"] = df["Close"].shift(-1)

        df = df.dropna()
        return df

    except Exception as e:
        logger.error(f"Feature creation error: {e}")
        return pd.DataFrame()

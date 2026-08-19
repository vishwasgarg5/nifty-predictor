import pandas as pd
import pandas_ta as ta
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create technical features + next-day OHLC targets.
    """
    df = df.copy()

    # Basic returns
    df["ret_1"] = df["Close"].pct_change()
    df["ret_3"] = df["Close"].pct_change(3)
    df["ret_5"] = df["Close"].pct_change(5)

    # Moving averages
    df["sma_10"] = ta.sma(df["Close"], length=10)
    df["sma_20"] = ta.sma(df["Close"], length=20)
    df["sma_50"] = ta.sma(df["Close"], length=50)
    df["ema_12"] = ta.ema(df["Close"], length=12)
    df["ema_26"] = ta.ema(df["Close"], length=26)

    # RSI & MACD
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    macd = ta.macd(df["Close"])
    df = pd.concat([df, macd], axis=1)

    # Bollinger Bands & ATR
    bb = ta.bbands(df["Close"], length=20)
    df = pd.concat([df, bb], axis=1)
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["atr_pct"] = df["atr_14"] / df["Close"] * 100

    # Volume
    df["vol_sma_20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_sma_20"]

    # Price relative to MAs
    df["close_sma20_ratio"] = df["Close"] / df["sma_20"]
    df["close_sma50_ratio"] = df["Close"] / df["sma_50"]

    # Targets (next day)
    df["target_o"] = df["Open"].shift(-1)
    df["target_h"] = df["High"].shift(-1)
    df["target_l"] = df["Low"].shift(-1)
    df["target_c"] = df["Close"].shift(-1)

    df = df.dropna()
    return df

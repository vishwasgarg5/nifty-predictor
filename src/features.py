import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA20", "EMA20", "RSI", "MACD",
    "BB_H", "BB_L", "ATR", "OBV",
    "Close_Lag1", "Close_Lag2", "Close_Lag3",
    "Daily_Return", "Volatility",
]


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SMA20"] = SMAIndicator(df["Close"], window=20).sma_indicator()
    df["EMA20"] = EMAIndicator(df["Close"], window=20).ema_indicator()
    df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
    df["MACD"] = MACD(df["Close"]).macd()

    bb = BollingerBands(df["Close"], window=20)
    df["BB_H"] = bb.bollinger_hband()
    df["BB_L"] = bb.bollinger_lband()

    df["ATR"] = AverageTrueRange(
        df["High"], df["Low"], df["Close"]
    ).average_true_range()

    df["OBV"] = OnBalanceVolumeIndicator(
        close=df["Close"],
        volume=df["Volume"],
    ).on_balance_volume()

    df["Close_Lag1"] = df["Close"].shift(1)
    df["Close_Lag2"] = df["Close"].shift(2)
    df["Close_Lag3"] = df["Close"].shift(3)

    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(10).std()

    return df


def create_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Features + next-day targets for model training."""
    df = _add_indicators(df)

    df["target_o"] = df["Open"].shift(-1)
    df["target_h"] = df["High"].shift(-1)
    df["target_l"] = df["Low"].shift(-1)
    df["target_c"] = df["Close"].shift(-1)

    return df.dropna().reset_index(drop=True)


def create_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Latest feature row without dropping the final market day."""
    df = _add_indicators(df)

    latest = df[FEATURE_COLS].iloc[[-1]]
    return latest.dropna()

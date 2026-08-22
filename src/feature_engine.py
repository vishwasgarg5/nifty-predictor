"""Unified feature engineering.

The output is designed for:

    - ranking
    - Return ML
    - Direction ML
    - Risk ML
    - future walk-forward backtesting
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.point_in_time import available_history


FEATURE_VERSION = "features-v2"


def _rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate RSI."""

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        period
    ).mean()

    avg_loss = loss.rolling(
        period
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan,
    )

    return 100 - (
        100 / (1 + rs)
    )


def _atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """Calculate Average True Range."""

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    previous_close = close.shift(1)

    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )

    true_range = ranges.max(
        axis=1
    )

    return true_range.rolling(
        period
    ).mean()


def _zscore(
    series: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Rolling z-score."""

    mean = series.rolling(
        period
    ).mean()

    std = series.rolling(
        period
    ).std()

    return (
        series - mean
    ) / std.replace(
        0,
        np.nan,
    )


def build_feature_frame(
    history: pd.DataFrame,
    as_of=None,
) -> pd.DataFrame:
    """Build a point-in-time safe feature dataframe."""

    df = available_history(
        history,
        as_of,
    ).copy()

    required = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    if (
        df.empty
        or not required.issubset(
            df.columns
        )
    ):
        return pd.DataFrame()

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=list(required)
    )

    if len(df) < 60:
        return pd.DataFrame()

    # -----------------------------
    # RETURNS
    # -----------------------------

    df["return_1d"] = (
        df["Close"].pct_change(1)
    )

    df["return_5d"] = (
        df["Close"].pct_change(5)
    )

    df["return_10d"] = (
        df["Close"].pct_change(10)
    )

    df["return_20d"] = (
        df["Close"].pct_change(20)
    )

    # -----------------------------
    # MOVING AVERAGES
    # -----------------------------

    df["sma_5"] = (
        df["Close"].rolling(5).mean()
    )

    df["sma_10"] = (
        df["Close"].rolling(10).mean()
    )

    df["sma_20"] = (
        df["Close"].rolling(20).mean()
    )

    df["sma_50"] = (
        df["Close"].rolling(50).mean()
    )

    df["dist_sma_5"] = (
        df["Close"] / df["sma_5"]
    ) - 1

    df["dist_sma_20"] = (
        df["Close"] / df["sma_20"]
    ) - 1

    df["dist_sma_50"] = (
        df["Close"] / df["sma_50"]
    ) - 1

    # -----------------------------
    # RSI
    # -----------------------------

    df["rsi_14"] = _rsi(
        df["Close"],
        14,
    )

    # -----------------------------
    # VOLATILITY
    # -----------------------------

    df["volatility_5"] = (
        df["return_1d"]
        .rolling(5)
        .std()
        * np.sqrt(252)
    )

    df["volatility_20"] = (
        df["return_1d"]
        .rolling(20)
        .std()
        * np.sqrt(252)
    )

    # -----------------------------
    # ATR
    # -----------------------------

    df["atr_14"] = _atr(
        df,
        14,
    )

    df["atr_pct"] = (
        df["atr_14"]
        / df["Close"]
    )

    # -----------------------------
    # VOLUME
    # -----------------------------

    df["volume_sma_20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["Volume"]
        / df["volume_sma_20"]
    )

    df["volume_zscore"] = _zscore(
        df["Volume"],
        20,
    )

    # -----------------------------
    # INTRADAY STRUCTURE
    # -----------------------------

    df["range_pct"] = (
        (df["High"] - df["Low"])
        / df["Close"]
    )

    df["body_pct"] = (
        (df["Close"] - df["Open"])
        / df["Open"]
    )

    df["close_position"] = (
        (df["Close"] - df["Low"])
        / (
            df["High"] - df["Low"]
        ).replace(
            0,
            np.nan,
        )
    )

    # -----------------------------
    # BREAKOUT FEATURES
    # -----------------------------

    df["high_20"] = (
        df["High"]
        .rolling(20)
        .max()
        .shift(1)
    )

    df["low_20"] = (
        df["Low"]
        .rolling(20)
        .min()
        .shift(1)
    )

    df["breakout_20"] = (
        df["Close"]
        / df["high_20"]
    ) - 1

    df["breakdown_20"] = (
        df["Close"]
        / df["low_20"]
    ) - 1

    return df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


def feature_columns() -> list[str]:
    """Return the stable ML feature list."""

    return [
        "return_1d",
        "return_5d",
        "return_10d",
        "return_20d",

        "dist_sma_5",
        "dist_sma_20",
        "dist_sma_50",

        "rsi_14",

        "volatility_5",
        "volatility_20",

        "atr_pct",

        "volume_ratio",
        "volume_zscore",

        "range_pct",
        "body_pct",
        "close_position",

        "breakout_20",
        "breakdown_20",
    ]


def latest_features(
    history: pd.DataFrame,
    as_of=None,
) -> dict | None:
    """Return the latest complete feature vector."""

    frame = build_feature_frame(
        history,
        as_of,
    )

    if frame.empty:
        return None

    columns = feature_columns()

    frame = frame.dropna(
        subset=columns
    )

    if frame.empty:
        return None

    row = frame.iloc[-1]

    features = {
        column: float(
            row[column]
        )
        for column in columns
    }

    features["feature_version"] = (
        FEATURE_VERSION
    )

    features["feature_timestamp"] = str(
        frame.index[-1]
    )

    return features


def feature_quality_score(
    features: dict | None,
) -> float:
    """Score feature completeness and validity."""

    if not features:
        return 0.0

    columns = feature_columns()

    valid = 0

    for column in columns:

        value = features.get(
            column
        )

        try:

            numeric = float(value)

            if math.isfinite(numeric):
                valid += 1

        except (
            TypeError,
            ValueError,
        ):
            pass

    return round(
        valid / len(columns),
        4,
    )

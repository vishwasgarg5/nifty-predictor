"""
Training target generation.

Creates point-in-time safe targets for:

    - Return ML
    - Direction ML
    - Risk ML

Targets describe what happens AFTER the
feature observation date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_VERSION = "targets-v1"


# ============================================================
# BUILD TARGET FRAME
# ============================================================

def build_target_frame(
    history: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Build future prediction targets.

    Parameters
    ----------
    history:
        Historical OHLCV dataframe.

    horizon:
        Number of future trading days.

    Creates:

        future_return
        future_direction
        future_volatility
    """

    if history is None or history.empty:

        return pd.DataFrame()

    required = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    if not required.issubset(
        history.columns
    ):

        return pd.DataFrame()

    df = history.copy()

    # --------------------------------------------------------
    # CLEAN CLOSE
    # --------------------------------------------------------

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Close"]
    )

    if len(df) <= horizon:

        return pd.DataFrame()

    # --------------------------------------------------------
    # FUTURE CLOSE
    # --------------------------------------------------------

    future_close = (
        df["Close"]
        .shift(-horizon)
    )

    # --------------------------------------------------------
    # RETURN TARGET
    # --------------------------------------------------------

    df["future_return"] = (
        future_close
        / df["Close"]
    ) - 1

    # --------------------------------------------------------
    # DIRECTION TARGET
    # --------------------------------------------------------

    df["future_direction"] = (
        df["future_return"] > 0
    ).astype(
        float
    )

    # --------------------------------------------------------
    # FUTURE VOLATILITY TARGET
    # --------------------------------------------------------

    future_returns = (
        df["Close"]
        .pct_change()
    )

    df["future_volatility"] = (
        future_returns
        .rolling(horizon)
        .std()
        .shift(-horizon + 1)
        * np.sqrt(252)
    )

    # --------------------------------------------------------
    # REMOVE UNKNOWN FUTURE DATA
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "future_return",
            "future_direction",
            "future_volatility",
        ]
    )

    return df


# ============================================================
# TARGET COLUMNS
# ============================================================

def target_columns() -> list[str]:
    """
    Return stable target column names.
    """

    return [
        "future_return",
        "future_direction",
        "future_volatility",
    ]


# ============================================================
# TARGET QUALITY
# ============================================================

def target_quality_score(
    frame: pd.DataFrame,
) -> float:
    """
    Measure target completeness.
    """

    if frame is None or frame.empty:

        return 0.0

    columns = target_columns()

    valid = frame.dropna(
        subset=columns
    )

    if len(frame) == 0:

        return 0.0

    return round(
        len(valid) / len(frame),
        4,
    )

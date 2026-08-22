"""Leakage-safe ML target generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


FORWARD_HORIZON = 1


def add_ml_targets(
    frame: pd.DataFrame,
    horizon: int = FORWARD_HORIZON,
) -> pd.DataFrame:
    """Add forward-looking targets.

    Features at time T predict outcomes occurring after T.

    Targets:
        target_return
        target_direction
        target_risk
    """

    if frame is None or frame.empty:
        return pd.DataFrame()

    df = frame.copy()

    if "Close" not in df.columns:
        return pd.DataFrame()

    close = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    # Future close.
    future_close = close.shift(
        -horizon
    )

    # Forward return.
    df["target_return"] = (
        future_close / close
    ) - 1

    # Direction classification.
    df["target_direction"] = (
        df["target_return"] > 0
    ).astype(float)

    # Realized future absolute move.
    #
    # For a 1-day horizon this is the
    # absolute next-day return.
    df["target_risk"] = (
        df["target_return"].abs()
    )

    # The final rows do not have future data.
    # They must never be used for training.
    df.loc[
        future_close.isna(),
        [
            "target_return",
            "target_direction",
            "target_risk",
        ],
    ] = np.nan

    return df


def split_training_data(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create clean X and y without leakage."""

    required = (
        list(feature_columns)
        + [target_column]
    )

    available = frame.dropna(
        subset=required
    ).copy()

    if available.empty:
        return (
            pd.DataFrame(
                columns=feature_columns
            ),
            pd.Series(
                dtype=float
            ),
        )

    x = available[
        feature_columns
    ].copy()

    y = available[
        target_column
    ].copy()

    return x, y

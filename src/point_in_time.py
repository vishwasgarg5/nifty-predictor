"""Point-in-time safety utilities.

These helpers ensure that features are built only from information that
would have been available at the prediction timestamp.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a clean DatetimeIndex."""

    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(
            result.index,
            errors="coerce",
        )

    result = result[
        ~result.index.isna()
    ]

    return result.sort_index()


def available_history(
    df: pd.DataFrame,
    as_of: Optional[datetime | str] = None,
) -> pd.DataFrame:
    """Return only rows available at the requested point in time.

    For end-of-day features, use the latest completed candle.
    No rows after ``as_of`` are allowed into the feature pipeline.
    """

    result = ensure_datetime_index(df)

    if result.empty or as_of is None:
        return result

    cutoff = pd.Timestamp(as_of)

    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)

    index = result.index

    if getattr(index, "tz", None) is not None:
        result.index = result.index.tz_localize(None)

    return result[
        result.index <= cutoff
    ].copy()


def latest_completed_row(
    df: pd.DataFrame,
    as_of: Optional[datetime | str] = None,
) -> pd.Series | None:
    """Return the latest row available at ``as_of``."""

    history = available_history(
        df,
        as_of,
    )

    if history.empty:
        return None

    return history.iloc[-1]

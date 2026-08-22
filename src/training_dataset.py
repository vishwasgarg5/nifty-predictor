"""
ML training dataset builder.

Combines:

    Feature Engine
    +
    Target Engine

to create a clean dataset for:

    - Return ML
    - Direction ML
    - Risk ML
"""

from __future__ import annotations

import pandas as pd

from src.features import (
    FEATURE_VERSION,
    build_feature_frame,
    feature_columns,
)

from src.targets import (
    TARGET_VERSION,
    build_target_frame,
    target_columns,
)


def build_training_dataset(
    history: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Build a complete ML training dataset.

    Each row contains:

        Features available on that date

    and:

        Targets describing what happened
        after that date.
    """

    if history is None or history.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # BUILD FEATURES
    # --------------------------------------------------------

    features = build_feature_frame(
        history
    )

    if features.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # BUILD TARGETS
    # --------------------------------------------------------

    targets = build_target_frame(
        history,
        horizon=horizon,
    )

    if targets.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # KEEP TARGET COLUMNS
    # --------------------------------------------------------

    target_data = targets[
        target_columns()
    ].copy()

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    dataset = features.join(
        target_data,
        how="inner",
    )

    # --------------------------------------------------------
    # REMOVE INVALID ROWS
    # --------------------------------------------------------

    required_columns = (
        feature_columns()
        + target_columns()
    )

    dataset = dataset.dropna(
        subset=required_columns
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    dataset = dataset.sort_index()

    return dataset


def dataset_info(
    dataset: pd.DataFrame,
) -> dict:
    """
    Return information about a training dataset.
    """

    if dataset is None or dataset.empty:

        return {
            "rows": 0,
            "features": 0,
            "targets": 0,
            "feature_version": FEATURE_VERSION,
            "target_version": TARGET_VERSION,
        }

    return {
        "rows": len(dataset),

        "features": len(
            feature_columns()
        ),

        "targets": len(
            target_columns()
        ),

        "feature_version": FEATURE_VERSION,

        "target_version": TARGET_VERSION,

        "start": str(
            dataset.index.min()
        ),

        "end": str(
            dataset.index.max()
        ),
    }


def split_features_and_targets(
    dataset: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Split training dataset into:

        X = ML features

        y = ML targets
    """

    if dataset is None or dataset.empty:

        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    X = dataset[
        feature_columns()
    ].copy()

    y = dataset[
        target_columns()
    ].copy()

    return (
        X,
        y,
    )

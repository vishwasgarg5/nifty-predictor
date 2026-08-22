#!/usr/bin/env python3

"""
Production Model Training.

This module:

    1. Builds features from historical market data.
    2. Creates leakage-safe ML targets.
    3. Trains ReturnModel.
    4. Trains DirectionModel.
    5. Trains RiskModel.
    6. Combines them into ProductionModel.
    7. Saves the trained model.

The saved model can later be registered as a
Challenger in model_registry.py.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(
    "train_model"
)


# ============================================================
# TIME
# ============================================================

def utc_now_iso() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# MODEL PATH
# ============================================================

def get_models_directory() -> Path:
    """Return the directory used for saved models."""

    path = (
        PROJECT_ROOT
        / "data"
        / "models"
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def get_default_model_path() -> Path:
    """Return the default trained model path."""

    return (
        get_models_directory()
        / "challenger_model.joblib"
    )


# ============================================================
# TRAINING DATA PREPARATION
# ============================================================

def prepare_training_data(
    history: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[str],
]:
    """
    Build features and ML targets.

    Returns:

        training_frame
        feature_columns
    """

    from src.feature_engine import (
        build_feature_frame,
        feature_columns,
    )

    from src.ml_targets import (
        add_ml_targets,
    )

    if history is None or history.empty:

        raise ValueError(
            "Historical data is empty."
        )

    # --------------------------------------------------------
    # BUILD FEATURES
    # --------------------------------------------------------

    frame = build_feature_frame(
        history
    )

    if frame.empty:

        raise ValueError(
            "Feature generation returned "
            "an empty DataFrame."
        )

    # --------------------------------------------------------
    # ADD TARGETS
    # --------------------------------------------------------

    frame = add_ml_targets(
        frame
    )

    if frame.empty:

        raise ValueError(
            "Target generation returned "
            "an empty DataFrame."
        )

    columns = feature_columns()

    # --------------------------------------------------------
    # CHECK FEATURES
    # --------------------------------------------------------

    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:

        raise ValueError(
            "Missing feature columns: "
            + ", ".join(missing)
        )

    return (
        frame,
        columns,
    )


# ============================================================
# MODEL TRAINING
# ============================================================

def train_production_model(
    history: pd.DataFrame,
) -> tuple[
    Any,
    dict[str, Any],
]:
    """
    Train the complete production model.

    The result contains:

        ReturnModel
        DirectionModel
        RiskModel
        ProductionModel
    """

    from src.return_model import (
        ReturnModel,
    )

    from src.direction_model import (
        DirectionModel,
    )

    from src.risk_model import (
        RiskModel,
    )

    from src.production_model import (
        ProductionModel,
    )

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    frame, columns = (
        prepare_training_data(
            history
        )
    )

    logger.info(
        "Training rows available: %s",
        len(frame),
    )

    # --------------------------------------------------------
    # RETURN MODEL
    # --------------------------------------------------------

    logger.info(
        "Training ReturnModel..."
    )

    return_model = ReturnModel(
        feature_columns=columns
    )

    return_model.fit(
        frame
    )

    # --------------------------------------------------------
    # DIRECTION MODEL
    # --------------------------------------------------------

    logger.info(
        "Training DirectionModel..."
    )

    direction_model = DirectionModel(
        feature_columns=columns
    )

    direction_model.fit(
        frame
    )

    # --------------------------------------------------------
    # RISK MODEL
    # --------------------------------------------------------

    logger.info(
        "Training RiskModel..."
    )

    risk_model = RiskModel(
        feature_columns=columns
    )

    risk_model.fit(
        frame
    )

    # --------------------------------------------------------
    # COMBINE MODELS
    # --------------------------------------------------------

    production_model = ProductionModel(
        return_model=return_model,
        direction_model=direction_model,
        risk_model=risk_model,
        feature_columns=columns,
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = production_model.get_metadata()

    metadata.update(
        {
            "trained_at": utc_now_iso(),

            "training_rows": len(
                frame
            ),

            "feature_columns": columns,

            "return_model_train_size": (
                return_model.train_size
            ),

            "direction_model_train_size": (
                direction_model.train_size
            ),

            "risk_model_train_size": (
                risk_model.train_size
            ),
        }
    )

    logger.info(
        "Production model training complete."
    )

    return (
        production_model,
        metadata,
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_trained_model(
    model: Any,
    metadata: dict[str, Any],
    path: str | Path | None = None,
) -> Path:
    """
    Save the trained ProductionModel.

    The model and metadata are stored together.
    """

    if path is None:

        output_path = (
            get_default_model_path()
        )

    else:

        output_path = Path(
            path
        )

        if not output_path.is_absolute():

            output_path = (
                PROJECT_ROOT
                / output_path
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "model": model,
        "metadata": dict(
            metadata
        ),
    }

    joblib.dump(
        payload,
        output_path,
    )

    logger.info(
        "Trained model saved: %s",
        output_path,
    )

    return output_path


# ============================================================
# TRAIN + SAVE
# ============================================================

def train_and_save(
    history: pd.DataFrame,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Train and save a ProductionModel.

    Returns training information including
    the saved model path.
    """

    model, metadata = (
        train_production_model(
            history
        )
    )

    model_path = save_trained_model(
        model=model,
        metadata=metadata,
        path=path,
    )

    return {
        "model": model,
        "model_path": model_path,
        "metadata": metadata,
    }


# ============================================================
# CLI HELPERS
# ============================================================

def load_history_from_csv(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load historical OHLCV data from CSV.

    Required columns:

        Open
        High
        Low
        Close
        Volume
    """

    csv_path = Path(
        path
    )

    if not csv_path.exists():

        raise FileNotFoundError(
            f"History file not found: "
            f"{csv_path}"
        )

    frame = pd.read_csv(
        csv_path
    )

    required = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:

        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    return frame


# ============================================================
# CLI
# ============================================================

def main() -> int:
    """
    Train a ProductionModel from a CSV file.

    Example:

        python -m src.train_model data/history.csv
    """

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Train the production "
            "stock prediction model."
        )
    )

    parser.add_argument(
        "history_file",
        help=(
            "CSV containing OHLCV "
            "historical data."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/models/"
            "challenger_model.joblib"
        ),
        help=(
            "Output model path."
        ),
    )

    args = parser.parse_args()

    try:

        history = (
            load_history_from_csv(
                args.history_file
            )
        )

        result = train_and_save(
            history=history,
            path=args.output,
        )

        print()

        print("=" * 70)
        print("MODEL TRAINING COMPLETE")
        print("=" * 70)

        print()

        print(
            "Model path:",
            result["model_path"],
        )

        print()

        for key, value in (
            result["metadata"]
            .items()
        ):

            print(
                f"{key}: {value}"
            )

        return 0

    except Exception as error:

        logger.exception(
            "Model training failed."
        )

        print(
            f"\nERROR: {error}"
        )

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

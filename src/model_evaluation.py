#!/usr/bin/env python3

"""
Model Evaluation Module.

Evaluates prediction quality using records from the
prediction ledger.

Metrics
-------
Return Prediction:
    - MAE
    - RMSE
    - Bias
    - Correlation

Direction Prediction:
    - Directional Accuracy

Risk Prediction:
    - MAE
    - RMSE

Confidence:
    - Confidence Calibration
    - Confidence vs Accuracy

Rolling Performance:
    - Recent rolling directional accuracy
    - Recent rolling return MAE

The main public entry points are:

    evaluate(predictions)
    run_evaluation(predictions)

The module is designed to be called from:

    scripts/evaluation_job.py
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger("model_evaluation")


# ============================================================
# TIME
# ============================================================

def utc_now_iso() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# CONFIG HELPERS
# ============================================================

def object_to_dict(
    value: Any,
) -> dict[str, Any]:
    """Convert a configuration object into a dictionary."""

    if value is None:
        return {}

    if isinstance(value, dict):
        return dict(value)

    if hasattr(value, "items"):

        try:
            return dict(value.items())

        except Exception:
            pass

    if hasattr(value, "__dict__"):

        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    return {}


def load_config() -> Any:
    """Load project configuration."""

    try:

        from src.config import cfg

        return cfg

    except Exception as error:

        logger.warning(
            "Could not load config: %s",
            error,
        )

        return None


def get_evaluation_config() -> dict[str, Any]:
    """
    Get model evaluation configuration.

    Supported example:

        evaluation:
            rolling_window: 30
            min_samples: 10
    """

    defaults = {
        "rolling_window": 30,
        "min_samples": 5,
    }

    cfg = load_config()

    if cfg is None:
        return defaults

    section = getattr(
        cfg,
        "evaluation",
        None,
    )

    values = object_to_dict(
        section
    )

    result = defaults.copy()

    for key in defaults:

        if key in values:

            try:

                result[key] = int(
                    values[key]
                )

            except Exception:
                pass

    result["rolling_window"] = max(
        1,
        result["rolling_window"],
    )

    result["min_samples"] = max(
        1,
        result["min_samples"],
    )

    return result


# ============================================================
# DATA HELPERS
# ============================================================

def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Find the first matching column."""

    for column in candidates:

        if column in frame.columns:

            return column

    return None


def numeric_series(
    frame: pd.DataFrame,
    column: str | None,
) -> pd.Series:
    """Return a numeric series."""

    if column is None:
        return pd.Series(
            dtype=float
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def evaluated_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return only successfully evaluated predictions.
    """

    if predictions is None:

        return pd.DataFrame()

    if predictions.empty:

        return predictions.copy()

    frame = predictions.copy()

    if "evaluation_status" not in frame.columns:

        return pd.DataFrame()

    status = (
        frame["evaluation_status"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return frame.loc[
        status == "EVALUATED"
    ].copy()


# ============================================================
# RETURN METRICS
# ============================================================

def calculate_return_metrics(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate predicted return vs actual return metrics.
    """

    predicted_column = find_column(
        frame,
        [
            "predicted_return",
            "expected_return",
            "return_prediction",
        ],
    )

    actual_column = find_column(
        frame,
        [
            "actual_return",
        ],
    )

    if (
        predicted_column is None
        or actual_column is None
    ):

        return {
            "available": False,
            "samples": 0,
        }

    data = pd.DataFrame(
        {
            "predicted": numeric_series(
                frame,
                predicted_column,
            ),
            "actual": numeric_series(
                frame,
                actual_column,
            ),
        }
    ).dropna()

    if data.empty:

        return {
            "available": False,
            "samples": 0,
        }

    errors = (
        data["predicted"]
        - data["actual"]
    )

    mae = float(
        errors.abs().mean()
    )

    rmse = float(
        math.sqrt(
            (errors ** 2).mean()
        )
    )

    bias = float(
        errors.mean()
    )

    correlation = None

    if len(data) >= 2:

        try:

            value = data[
                "predicted"
            ].corr(
                data["actual"]
            )

            if pd.notna(value):

                correlation = float(
                    value
                )

        except Exception:

            correlation = None

    return {
        "available": True,
        "samples": int(
            len(data)
        ),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "correlation": correlation,
        "mean_predicted_return": float(
            data["predicted"].mean()
        ),
        "mean_actual_return": float(
            data["actual"].mean()
        ),
    }


# ============================================================
# DIRECTION METRICS
# ============================================================

def normalize_direction(
    value: Any,
) -> str | None:
    """
    Normalize direction values.
    """

    if value is None:

        return None

    if pd.isna(value):

        return None

    text = str(
        value
    ).strip().upper()

    mapping = {
        "UP": "UP",
        "BUY": "UP",
        "LONG": "UP",
        "1": "UP",
        "DOWN": "DOWN",
        "SELL": "DOWN",
        "SHORT": "DOWN",
        "-1": "DOWN",
        "FLAT": "FLAT",
        "NEUTRAL": "FLAT",
        "0": "FLAT",
    }

    return mapping.get(
        text,
        text,
    )


def calculate_direction_metrics(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate directional prediction accuracy.
    """

    predicted_column = find_column(
        frame,
        [
            "predicted_direction",
            "direction",
            "direction_prediction",
        ],
    )

    actual_column = find_column(
        frame,
        [
            "actual_direction",
        ],
    )

    if (
        predicted_column is None
        or actual_column is None
    ):

        return {
            "available": False,
            "samples": 0,
        }

    data = pd.DataFrame(
        {
            "predicted": frame[
                predicted_column
            ].apply(
                normalize_direction
            ),
            "actual": frame[
                actual_column
            ].apply(
                normalize_direction
            ),
        }
    )

    data = data.dropna()

    if data.empty:

        return {
            "available": False,
            "samples": 0,
        }

    correct = (
        data["predicted"]
        == data["actual"]
    )

    accuracy = float(
        correct.mean() * 100.0
    )

    return {
        "available": True,
        "samples": int(
            len(data)
        ),
        "correct_predictions": int(
            correct.sum()
        ),
        "directional_accuracy": accuracy,
        "up_predictions": int(
            (
                data["predicted"]
                == "UP"
            ).sum()
        ),
        "down_predictions": int(
            (
                data["predicted"]
                == "DOWN"
            ).sum()
        ),
        "flat_predictions": int(
            (
                data["predicted"]
                == "FLAT"
            ).sum()
        ),
    }


# ============================================================
# RISK METRICS
# ============================================================

def calculate_risk_metrics(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate predicted risk vs actual risk metrics.
    """

    predicted_column = find_column(
        frame,
        [
            "predicted_risk",
            "risk_score",
            "risk_prediction",
        ],
    )

    actual_column = find_column(
        frame,
        [
            "actual_risk",
        ],
    )

    if (
        predicted_column is None
        or actual_column is None
    ):

        return {
            "available": False,
            "samples": 0,
        }

    data = pd.DataFrame(
        {
            "predicted": numeric_series(
                frame,
                predicted_column,
            ),
            "actual": numeric_series(
                frame,
                actual_column,
            ),
        }
    ).dropna()

    if data.empty:

        return {
            "available": False,
            "samples": 0,
        }

    errors = (
        data["predicted"]
        - data["actual"]
    )

    mae = float(
        errors.abs().mean()
    )

    rmse = float(
        math.sqrt(
            (errors ** 2).mean()
        )
    )

    bias = float(
        errors.mean()
    )

    return {
        "available": True,
        "samples": int(
            len(data)
        ),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "mean_predicted_risk": float(
            data["predicted"].mean()
        ),
        "mean_actual_risk": float(
            data["actual"].mean()
        ),
    }


# ============================================================
# CONFIDENCE CALIBRATION
# ============================================================

def normalize_confidence(
    value: Any,
) -> float | None:
    """
    Normalize confidence values.

    Supports:

        0.75
        75
        0-100 scales
    """

    try:

        numeric = float(
            value
        )

    except Exception:

        return None

    if pd.isna(numeric):

        return None

    if numeric < 0:

        return None

    if numeric <= 1:

        numeric *= 100.0

    return min(
        100.0,
        max(
            0.0,
            numeric,
        ),
    )


def calculate_confidence_metrics(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate confidence calibration.

    A confidence prediction is compared with whether the
    predicted direction was actually correct.
    """

    confidence_column = find_column(
        frame,
        [
            "confidence",
            "confidence_score",
        ],
    )

    predicted_direction_column = find_column(
        frame,
        [
            "predicted_direction",
            "direction",
            "direction_prediction",
        ],
    )

    actual_direction_column = find_column(
        frame,
        [
            "actual_direction",
        ],
    )

    if (
        confidence_column is None
        or predicted_direction_column is None
        or actual_direction_column is None
    ):

        return {
            "available": False,
            "samples": 0,
        }

    data = pd.DataFrame(
        {
            "confidence": frame[
                confidence_column
            ].apply(
                normalize_confidence
            ),
            "predicted": frame[
                predicted_direction_column
            ].apply(
                normalize_direction
            ),
            "actual": frame[
                actual_direction_column
            ].apply(
                normalize_direction
            ),
        }
    ).dropna()

    if data.empty:

        return {
            "available": False,
            "samples": 0,
        }

    data["correct"] = (
        data["predicted"]
        == data["actual"]
    ).astype(int)

    average_confidence = float(
        data["confidence"].mean()
    )

    actual_accuracy = float(
        data["correct"].mean()
        * 100.0
    )

    calibration_error = abs(
        average_confidence
        - actual_accuracy
    )

    return {
        "available": True,
        "samples": int(
            len(data)
        ),
        "average_confidence": (
            average_confidence
        ),
        "actual_accuracy": (
            actual_accuracy
        ),
        "calibration_error": float(
            calibration_error
        ),
    }


# ============================================================
# ROLLING PERFORMANCE
# ============================================================

def get_sort_column(
    frame: pd.DataFrame,
) -> str | None:
    """
    Find the best timestamp column for chronological evaluation.
    """

    return find_column(
        frame,
        [
            "prediction_date",
            "created_at",
            "evaluation_timestamp",
            "timestamp",
            "date",
        ],
    )


def calculate_rolling_metrics(
    frame: pd.DataFrame,
    window: int,
) -> dict[str, Any]:
    """
    Calculate recent rolling performance metrics.
    """

    if frame.empty:

        return {
            "available": False,
            "samples": 0,
        }

    result = frame.copy()

    sort_column = get_sort_column(
        result
    )

    if sort_column is not None:

        parsed_dates = pd.to_datetime(
            result[sort_column],
            utc=True,
            errors="coerce",
        )

        result = result.assign(
            _evaluation_sort_time=parsed_dates
        )

        result = result.sort_values(
            by="_evaluation_sort_time",
            ascending=True,
            na_position="last",
        )

    result = result.tail(
        max(
            1,
            int(window),
        )
    )

    metrics: dict[str, Any] = {
        "available": True,
        "samples": int(
            len(result)
        ),
        "window": int(
            window
        ),
    }

    direction_metrics = (
        calculate_direction_metrics(
            result
        )
    )

    if direction_metrics.get(
        "available"
    ):

        metrics[
            "directional_accuracy"
        ] = direction_metrics.get(
            "directional_accuracy"
        )

    return_metrics = (
        calculate_return_metrics(
            result
        )
    )

    if return_metrics.get(
        "available"
    ):

        metrics["return_mae"] = (
            return_metrics.get(
                "mae"
            )
        )

        metrics["return_rmse"] = (
            return_metrics.get(
                "rmse"
            )
        )

        metrics["return_correlation"] = (
            return_metrics.get(
                "correlation"
            )
        )

    return metrics


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

def calculate_model_health(
    return_metrics: dict[str, Any],
    direction_metrics: dict[str, Any],
    risk_metrics: dict[str, Any],
    min_samples: int,
) -> dict[str, Any]:
    """
    Produce a simple model health assessment.

    This is intentionally conservative and is primarily used
    as a monitoring signal.
    """

    sample_count = max(
        int(
            return_metrics.get(
                "samples",
                0,
            )
        ),
        int(
            direction_metrics.get(
                "samples",
                0,
            )
        ),
    )

    if sample_count < min_samples:

        return {
            "status": "INSUFFICIENT_DATA",
            "samples": sample_count,
            "minimum_samples": min_samples,
        }

    directional_accuracy = (
        direction_metrics.get(
            "directional_accuracy"
        )
    )

    correlation = (
        return_metrics.get(
            "correlation"
        )
    )

    return_mae = (
        return_metrics.get(
            "mae"
        )
    )

    health_score = 50.0

    if directional_accuracy is not None:

        health_score += (
            directional_accuracy
            - 50.0
        )

    if correlation is not None:

        health_score += (
            correlation
            * 20.0
        )

    if return_mae is not None:

        health_score -= min(
            20.0,
            float(return_mae)
            * 2.0,
        )

    health_score = max(
        0.0,
        min(
            100.0,
            health_score,
        ),
    )

    if health_score >= 70:

        status = "GOOD"

    elif health_score >= 50:

        status = "STABLE"

    elif health_score >= 30:

        status = "DEGRADED"

    else:

        status = "CRITICAL"

    return {
        "status": status,
        "score": float(
            health_score
        ),
        "samples": sample_count,
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """
    Evaluate model performance.

    This is the primary public evaluation function.
    """

    config = get_evaluation_config()

    evaluated = (
        evaluated_predictions(
            predictions
        )
    )

    logger.info(
        "Running model evaluation on %s evaluated prediction(s).",
        len(evaluated),
    )

    if evaluated.empty:

        return {
            "status": "NO_EVALUATED_PREDICTIONS",
            "evaluated_at": utc_now_iso(),
            "total_samples": 0,
            "return_metrics": {
                "available": False,
                "samples": 0,
            },
            "direction_metrics": {
                "available": False,
                "samples": 0,
            },
            "risk_metrics": {
                "available": False,
                "samples": 0,
            },
            "confidence_metrics": {
                "available": False,
                "samples": 0,
            },
            "rolling_metrics": {
                "available": False,
                "samples": 0,
            },
            "model_health": {
                "status": "INSUFFICIENT_DATA",
                "samples": 0,
                "minimum_samples": (
                    config["min_samples"]
                ),
            },
        }

    return_metrics = (
        calculate_return_metrics(
            evaluated
        )
    )

    direction_metrics = (
        calculate_direction_metrics(
            evaluated
        )
    )

    risk_metrics = (
        calculate_risk_metrics(
            evaluated
        )
    )

    confidence_metrics = (
        calculate_confidence_metrics(
            evaluated
        )
    )

    rolling_metrics = (
        calculate_rolling_metrics(
            evaluated,
            window=config[
                "rolling_window"
            ],
        )
    )

    model_health = (
        calculate_model_health(
            return_metrics=return_metrics,
            direction_metrics=direction_metrics,
            risk_metrics=risk_metrics,
            min_samples=config[
                "min_samples"
            ],
        )
    )

    return {
        "status": "SUCCESS",
        "evaluated_at": utc_now_iso(),
        "total_samples": int(
            len(evaluated)
        ),
        "return_metrics": (
            return_metrics
        ),
        "direction_metrics": (
            direction_metrics
        ),
        "risk_metrics": (
            risk_metrics
        ),
        "confidence_metrics": (
            confidence_metrics
        ),
        "rolling_metrics": (
            rolling_metrics
        ),
        "model_health": (
            model_health
        ),
    }


# ============================================================
# COMPATIBILITY ENTRY POINT
# ============================================================

def run_evaluation(
    predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Compatibility entry point.

    When predictions are supplied, evaluate them directly.

    When no DataFrame is supplied, attempt to load the
    prediction ledger.
    """

    if predictions is not None:

        return evaluate(
            predictions
        )

    try:

        from scripts.evaluation_job import (
            load_prediction_ledger,
        )

        ledger = (
            load_prediction_ledger()
        )

        return evaluate(
            ledger
        )

    except Exception as error:

        logger.exception(
            "Could not run model evaluation."
        )

        return {
            "status": "ERROR",
            "evaluated_at": utc_now_iso(),
            "error": str(error),
        }


# ============================================================
# CLI TEST
# ============================================================

def main() -> int:
    """
    Run model evaluation against the prediction ledger.
    """

    result = run_evaluation()

    print()

    print("=" * 70)

    print("MODEL EVALUATION")

    print("=" * 70)

    print(
        f"Status: {result.get('status')}"
    )

    print(
        f"Total samples: "
        f"{result.get('total_samples')}"
    )

    print()

    print("Return Metrics:")

    print(
        result.get(
            "return_metrics",
            {},
        )
    )

    print()

    print("Direction Metrics:")

    print(
        result.get(
            "direction_metrics",
            {},
        )
    )

    print()

    print("Risk Metrics:")

    print(
        result.get(
            "risk_metrics",
            {},
        )
    )

    print()

    print("Confidence Metrics:")

    print(
        result.get(
            "confidence_metrics",
            {},
        )
    )

    print()

    print("Rolling Metrics:")

    print(
        result.get(
            "rolling_metrics",
            {},
        )
    )

    print()

    print("Model Health:")

    print(
        result.get(
            "model_health",
            {},
        )
    )

    return (
        0
        if result.get("status")
        in {
            "SUCCESS",
            "NO_EVALUATED_PREDICTIONS",
        }
        else 1
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

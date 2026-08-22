#!/usr/bin/env python3

"""
Production Drift Detection.

This module detects degradation in model performance by comparing
recent evaluated predictions against historical performance.

Pipeline
--------
Prediction Ledger
        │
        ▼
Evaluated Predictions
        │
        ├───────────────┐
        ▼               ▼
Historical Window   Recent Window
        │               │
        └───────┬───────┘
                ▼
        Performance Comparison
                │
                ├── Direction Accuracy Drift
                ├── Return MAE Drift
                ├── Return RMSE Drift
                ├── Return Bias Drift
                └── Prediction Volume
                │
                ▼
          Drift Severity
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
      NORMAL  WARNING  CRITICAL
                │
                ▼
         Production Monitoring
                │
                ▼
          Circuit Breaker

Public API
----------
detect_drift(predictions)

Compatibility API:
    run_drift_detection(predictions=None)
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger("drift_detection")


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
    """Convert a config object into a dictionary."""

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


def get_drift_config() -> dict[str, Any]:
    """
    Load drift detection configuration.

    Example config:

        drift_detection:
            recent_window: 20
            historical_window: 100
            min_samples: 10

            direction_warning_drop: 10
            direction_critical_drop: 20

            mae_warning_increase: 25
            mae_critical_increase: 50

            rmse_warning_increase: 25
            rmse_critical_increase: 50

            minimum_recent_accuracy: 45
            critical_recent_accuracy: 35
    """

    defaults = {
        "recent_window": 20,
        "historical_window": 100,
        "min_samples": 10,

        "direction_warning_drop": 10.0,
        "direction_critical_drop": 20.0,

        "mae_warning_increase": 25.0,
        "mae_critical_increase": 50.0,

        "rmse_warning_increase": 25.0,
        "rmse_critical_increase": 50.0,

        "minimum_recent_accuracy": 45.0,
        "critical_recent_accuracy": 35.0,
    }

    cfg = load_config()

    if cfg is None:
        return defaults

    section = getattr(
        cfg,
        "drift_detection",
        None,
    )

    values = object_to_dict(
        section
    )

    result = defaults.copy()

    for key, default_value in defaults.items():

        if key not in values:
            continue

        value = values[key]

        try:

            if isinstance(
                default_value,
                int,
            ):

                result[key] = int(value)

            else:

                result[key] = float(value)

        except Exception:

            logger.warning(
                "Invalid drift config value "
                "for %s: %s",
                key,
                value,
            )

    result["recent_window"] = max(
        1,
        int(
            result["recent_window"]
        ),
    )

    result["historical_window"] = max(
        result["recent_window"],
        int(
            result["historical_window"]
        ),
    )

    result["min_samples"] = max(
        1,
        int(
            result["min_samples"]
        ),
    )

    return result


# ============================================================
# DATA HELPERS
# ============================================================

def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Return the first matching column."""

    for column in candidates:

        if column in frame.columns:

            return column

    return None


def get_evaluated_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return only predictions with EVALUATED status.
    """

    if predictions is None:

        return pd.DataFrame()

    if predictions.empty:

        return predictions.copy()

    if "evaluation_status" not in predictions.columns:

        return pd.DataFrame()

    frame = predictions.copy()

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


def get_sort_column(
    frame: pd.DataFrame,
) -> str | None:
    """
    Find the best chronological column.
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


def sort_predictions(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sort predictions chronologically.
    """

    if frame.empty:

        return frame.copy()

    result = frame.copy()

    sort_column = get_sort_column(
        result
    )

    if sort_column is None:

        return result.reset_index(
            drop=True
        )

    timestamps = pd.to_datetime(
        result[sort_column],
        utc=True,
        errors="coerce",
    )

    result = result.assign(
        _drift_sort_time=timestamps
    )

    result = result.sort_values(
        by="_drift_sort_time",
        ascending=True,
        na_position="last",
    )

    return result.drop(
        columns=["_drift_sort_time"]
    ).reset_index(
        drop=True
    )


# ============================================================
# DIRECTION HELPERS
# ============================================================

def normalize_direction(
    value: Any,
) -> str | None:
    """Normalize direction values."""

    if value is None:

        return None

    if pd.isna(value):

        return None

    value = str(
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
        value,
        value,
    )


# ============================================================
# METRIC CALCULATIONS
# ============================================================

def calculate_direction_accuracy(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate directional accuracy."""

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
            "accuracy": None,
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
    ).dropna()

    if data.empty:

        return {
            "available": False,
            "samples": 0,
            "accuracy": None,
        }

    correct = (
        data["predicted"]
        == data["actual"]
    )

    return {
        "available": True,
        "samples": int(
            len(data)
        ),
        "accuracy": float(
            correct.mean() * 100.0
        ),
    }


def calculate_return_errors(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate return MAE and RMSE.
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
            "mae": None,
            "rmse": None,
            "bias": None,
        }

    predicted = pd.to_numeric(
        frame[predicted_column],
        errors="coerce",
    )

    actual = pd.to_numeric(
        frame[actual_column],
        errors="coerce",
    )

    data = pd.DataFrame(
        {
            "predicted": predicted,
            "actual": actual,
        }
    ).dropna()

    if data.empty:

        return {
            "available": False,
            "samples": 0,
            "mae": None,
            "rmse": None,
            "bias": None,
        }

    errors = (
        data["predicted"]
        - data["actual"]
    )

    mae = float(
        errors.abs().mean()
    )

    rmse = float(
        ((errors ** 2).mean()) ** 0.5
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
    }


def calculate_window_metrics(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate all drift comparison metrics.
    """

    direction = (
        calculate_direction_accuracy(
            frame
        )
    )

    returns = (
        calculate_return_errors(
            frame
        )
    )

    samples = max(
        int(
            direction.get(
                "samples",
                0,
            )
        ),
        int(
            returns.get(
                "samples",
                0,
            )
        ),
    )

    return {
        "samples": samples,

        "direction_accuracy": (
            direction.get(
                "accuracy"
            )
        ),

        "return_mae": (
            returns.get(
                "mae"
            )
        ),

        "return_rmse": (
            returns.get(
                "rmse"
            )
        ),

        "return_bias": (
            returns.get(
                "bias"
            )
        ),
    }


# ============================================================
# WINDOW COMPARISON
# ============================================================

def percentage_increase(
    historical: float | None,
    recent: float | None,
) -> float | None:
    """
    Calculate percentage increase.

    Returns None when comparison is impossible.
    """

    if (
        historical is None
        or recent is None
    ):

        return None

    try:

        historical = float(
            historical
        )

        recent = float(
            recent
        )

    except Exception:

        return None

    if historical == 0:

        if recent == 0:

            return 0.0

        return None

    return (
        (
            recent
            - historical
        )
        / abs(historical)
        * 100.0
    )


def compare_windows(
    historical: dict[str, Any],
    recent: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare recent performance with historical performance.
    """

    historical_accuracy = (
        historical.get(
            "direction_accuracy"
        )
    )

    recent_accuracy = (
        recent.get(
            "direction_accuracy"
        )
    )

    accuracy_drop = None

    if (
        historical_accuracy is not None
        and recent_accuracy is not None
    ):

        accuracy_drop = (
            float(
                historical_accuracy
            )
            - float(
                recent_accuracy
            )
        )

    mae_increase = (
        percentage_increase(
            historical.get(
                "return_mae"
            ),
            recent.get(
                "return_mae"
            ),
        )
    )

    rmse_increase = (
        percentage_increase(
            historical.get(
                "return_rmse"
            ),
            recent.get(
                "return_rmse"
            ),
        )
    )

    return {
        "direction_accuracy_drop": (
            accuracy_drop
        ),
        "return_mae_increase_pct": (
            mae_increase
        ),
        "return_rmse_increase_pct": (
            rmse_increase
        ),
    }


# ============================================================
# DRIFT RULES
# ============================================================

def detect_drift_signals(
    historical: dict[str, Any],
    recent: dict[str, Any],
    comparison: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Detect individual drift signals.
    """

    signals: list[
        dict[str, Any]
    ] = []

    accuracy_drop = comparison.get(
        "direction_accuracy_drop"
    )

    if accuracy_drop is not None:

        if (
            accuracy_drop
            >= config[
                "direction_critical_drop"
            ]
        ):

            signals.append(
                {
                    "metric": (
                        "direction_accuracy"
                    ),
                    "severity": "CRITICAL",
                    "value": accuracy_drop,
                    "message": (
                        "Directional accuracy dropped "
                        f"by {accuracy_drop:.2f} points."
                    ),
                }
            )

        elif (
            accuracy_drop
            >= config[
                "direction_warning_drop"
            ]
        ):

            signals.append(
                {
                    "metric": (
                        "direction_accuracy"
                    ),
                    "severity": "WARNING",
                    "value": accuracy_drop,
                    "message": (
                        "Directional accuracy dropped "
                        f"by {accuracy_drop:.2f} points."
                    ),
                }
            )

    recent_accuracy = recent.get(
        "direction_accuracy"
    )

    if recent_accuracy is not None:

        if (
            recent_accuracy
            <= config[
                "critical_recent_accuracy"
            ]
        ):

            signals.append(
                {
                    "metric": (
                        "recent_direction_accuracy"
                    ),
                    "severity": "CRITICAL",
                    "value": recent_accuracy,
                    "message": (
                        "Recent directional accuracy "
                        f"is only {recent_accuracy:.2f}%."
                    ),
                }
            )

        elif (
            recent_accuracy
            <= config[
                "minimum_recent_accuracy"
            ]
        ):

            signals.append(
                {
                    "metric": (
                        "recent_direction_accuracy"
                    ),
                    "severity": "WARNING",
                    "value": recent_accuracy,
                    "message": (
                        "Recent directional accuracy "
                        f"is {recent_accuracy:.2f}%."
                    ),
                }
            )

    mae_increase = comparison.get(
        "return_mae_increase_pct"
    )

    if mae_increase is not None:

        if (
            mae_increase
            >= config[
                "mae_critical_increase"
            ]
        ):

            signals.append(
                {
                    "metric": "return_mae",
                    "severity": "CRITICAL",
                    "value": mae_increase,
                    "message": (
                        "Return MAE increased by "
                        f"{mae_increase:.2f}%."
                    ),
                }
            )

        elif (
            mae_increase
            >= config[
                "mae_warning_increase"
            ]
        ):

            signals.append(
                {
                    "metric": "return_mae",
                    "severity": "WARNING",
                    "value": mae_increase,
                    "message": (
                        "Return MAE increased by "
                        f"{mae_increase:.2f}%."
                    ),
                }
            )

    rmse_increase = comparison.get(
        "return_rmse_increase_pct"
    )

    if rmse_increase is not None:

        if (
            rmse_increase
            >= config[
                "rmse_critical_increase"
            ]
        ):

            signals.append(
                {
                    "metric": "return_rmse",
                    "severity": "CRITICAL",
                    "value": rmse_increase,
                    "message": (
                        "Return RMSE increased by "
                        f"{rmse_increase:.2f}%."
                    ),
                }
            )

        elif (
            rmse_increase
            >= config[
                "rmse_warning_increase"
            ]
        ):

            signals.append(
                {
                    "metric": "return_rmse",
                    "severity": "WARNING",
                    "value": rmse_increase,
                    "message": (
                        "Return RMSE increased by "
                        f"{rmse_increase:.2f}%."
                    ),
                }
            )

    return signals


def determine_drift_status(
    signals: list[dict[str, Any]],
) -> str:
    """
    Determine overall drift severity.
    """

    severities = {
        str(
            signal.get(
                "severity",
                ""
            )
        ).upper()
        for signal in signals
    }

    if "CRITICAL" in severities:

        return "CRITICAL"

    if "WARNING" in severities:

        return "WARNING"

    return "NORMAL"


# ============================================================
# MAIN DRIFT DETECTION
# ============================================================

def detect_drift(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """
    Detect production model drift.

    The recent window is compared against the historical
    baseline window.

    Example:

        historical_window = 100
        recent_window = 20

    Historical baseline:
        previous predictions before the recent 20

    Recent performance:
        latest 20 predictions
    """

    config = get_drift_config()

    evaluated = (
        get_evaluated_predictions(
            predictions
        )
    )

    evaluated = sort_predictions(
        evaluated
    )

    total_samples = len(
        evaluated
    )

    min_samples = config[
        "min_samples"
    ]

    required_samples = (
        min_samples * 2
    )

    logger.info(
        "Drift detection | "
        "evaluated_samples=%s",
        total_samples,
    )

    if total_samples < required_samples:

        return {
            "status": "INSUFFICIENT_DATA",
            "drift_detected": False,
            "evaluated_at": utc_now_iso(),
            "total_samples": total_samples,
            "required_samples": required_samples,
            "signals": [],
            "message": (
                "Not enough evaluated predictions "
                "for drift detection."
            ),
        }

    recent_window = min(
        config["recent_window"],
        total_samples // 2,
    )

    recent = evaluated.tail(
        recent_window
    ).copy()

    historical_available = evaluated.iloc[
        : len(evaluated)
        - recent_window
    ].copy()

    historical_window = min(
        config[
            "historical_window"
        ],
        len(
            historical_available
        ),
    )

    historical = historical_available.tail(
        historical_window
    ).copy()

    if (
        len(recent)
        < min_samples
        or len(historical)
        < min_samples
    ):

        return {
            "status": "INSUFFICIENT_DATA",
            "drift_detected": False,
            "evaluated_at": utc_now_iso(),
            "total_samples": total_samples,
            "recent_samples": len(
                recent
            ),
            "historical_samples": len(
                historical
            ),
            "signals": [],
            "message": (
                "Recent or historical comparison "
                "window has insufficient samples."
            ),
        }

    historical_metrics = (
        calculate_window_metrics(
            historical
        )
    )

    recent_metrics = (
        calculate_window_metrics(
            recent
        )
    )

    comparison = compare_windows(
        historical_metrics,
        recent_metrics,
    )

    signals = detect_drift_signals(
        historical=historical_metrics,
        recent=recent_metrics,
        comparison=comparison,
        config=config,
    )

    status = determine_drift_status(
        signals
    )

    drift_detected = (
        status != "NORMAL"
    )

    logger.info(
        "Drift detection complete | "
        "status=%s | "
        "signals=%s",
        status,
        len(signals),
    )

    return {
        "status": status,
        "drift_detected": drift_detected,
        "evaluated_at": utc_now_iso(),

        "total_samples": total_samples,

        "recent": {
            "window": recent_window,
            "metrics": recent_metrics,
        },

        "historical": {
            "window": historical_window,
            "metrics": historical_metrics,
        },

        "comparison": comparison,

        "signals": signals,

        "config": config,
    }


# ============================================================
# COMPATIBILITY ENTRY POINT
# ============================================================

def run_drift_detection(
    predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Compatibility entry point.

    If predictions are supplied, detect drift directly.

    Otherwise the prediction ledger is loaded.
    """

    if predictions is not None:

        return detect_drift(
            predictions
        )

    try:

        from scripts.evaluation_job import (
            load_prediction_ledger,
        )

        predictions = (
            load_prediction_ledger()
        )

        return detect_drift(
            predictions
        )

    except Exception as error:

        logger.exception(
            "Drift detection failed."
        )

        return {
            "status": "ERROR",
            "drift_detected": True,
            "evaluated_at": utc_now_iso(),
            "error": str(error),
        }


# ============================================================
# CLI
# ============================================================

def main() -> int:
    """Run drift detection from the prediction ledger."""

    result = (
        run_drift_detection()
    )

    print()

    print("=" * 70)

    print("DRIFT DETECTION")

    print("=" * 70)

    print(
        f"Status: {result.get('status')}"
    )

    print(
        f"Drift detected: "
        f"{result.get('drift_detected')}"
    )

    print(
        f"Total samples: "
        f"{result.get('total_samples')}"
    )

    recent = result.get(
        "recent",
        {},
    )

    historical = result.get(
        "historical",
        {},
    )

    if recent:

        print()

        print("Recent Performance:")

        print(
            recent.get(
                "metrics",
                {},
            )
        )

    if historical:

        print()

        print("Historical Performance:")

        print(
            historical.get(
                "metrics",
                {},
            )
        )

    comparison = result.get(
        "comparison",
        {},
    )

    if comparison:

        print()

        print("Comparison:")

        print(
            comparison
        )

    signals = result.get(
        "signals",
        [],
    )

    if signals:

        print()

        print("Drift Signals:")

        for signal in signals:

            print(
                f"- "
                f"[{signal.get('severity')}] "
                f"{signal.get('message')}"
            )

    if result.get("message"):

        print()

        print(
            f"Message: "
            f"{result.get('message')}"
        )

    if result.get("error"):

        print()

        print(
            f"Error: "
            f"{result.get('error')}"
        )

    return (
        0
        if result.get("status")
        in {
            "NORMAL",
            "WARNING",
            "CRITICAL",
            "INSUFFICIENT_DATA",
        }
        else 1
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

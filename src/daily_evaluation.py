# daily_evaluation.py
#!/usr/bin/env python3

"""
Daily Prediction Evaluation.

This module compares stored model predictions with
actual market results.

Evaluation includes:

    - Predicted vs actual return
    - Direction accuracy
    - Prediction error
    - MAE
    - RMSE
    - Mean prediction error
    - Win rate
    - Top-N performance

Typical flow:

    Morning:
        Prediction Pipeline
                |
                v
        PredictionStore
                |
                v
        data/predictions/YYYY-MM-DD.json

    Evening / Next Trading Day:
        Actual OHLC Data
                |
                v
        DailyEvaluator
                |
                v
        Evaluation Metrics
                |
                +--> Telegram Report
                |
                +--> Model Monitoring
                |
                +--> Retraining Decision
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any


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
    "daily_evaluation"
)


# ============================================================
# NUMBER HELPERS
# ============================================================

def to_float(
    value: Any,
    default: float | None = None,
) -> float | None:
    """
    Convert a value to a finite float.
    """

    if value is None:

        return default

    try:

        result = float(value)

    except (
        TypeError,
        ValueError,
    ):

        return default

    if not math.isfinite(result):

        return default

    return result


def first_numeric(
    source: dict[str, Any],
    keys: list[str],
) -> float | None:
    """
    Return the first valid numeric value
    found for the supplied keys.
    """

    for key in keys:

        if key not in source:

            continue

        value = to_float(
            source.get(key)
        )

        if value is not None:

            return value

    return None


def normalize_date(
    value: str | date | datetime | None,
) -> str | None:
    """
    Normalize a date to YYYY-MM-DD.
    """

    if value is None:

        return None

    if isinstance(
        value,
        datetime,
    ):

        return value.date().isoformat()

    if isinstance(
        value,
        date,
    ):

        return value.isoformat()

    text = str(
        value
    ).strip()

    if not text:

        return None

    try:

        return date.fromisoformat(
            text
        ).isoformat()

    except ValueError:

        pass

    try:

        return datetime.fromisoformat(
            text
        ).date().isoformat()

    except ValueError as error:

        raise ValueError(
            f"Invalid date: {value}"
        ) from error


# ============================================================
# PREDICTION VALUE EXTRACTION
# ============================================================

def get_predicted_return(
    prediction: dict[str, Any],
) -> float | None:
    """
    Extract the expected return.

    Supports common field names used by
    prediction and ensemble pipelines.
    """

    return first_numeric(
        prediction,
        [
            "expected_return",
            "predicted_return",
            "return_prediction",
            "prediction",
        ],
    )


def get_probability_up(
    prediction: dict[str, Any],
) -> float | None:
    """
    Extract probability of upward movement.
    """

    value = first_numeric(
        prediction,
        [
            "probability_up",
            "direction_probability",
            "probability",
        ],
    )

    if value is None:

        return None

    # Accept either:
    #
    # 0.65
    #
    # or:
    #
    # 65.0

    if value > 1.0:

        value = value / 100.0

    return max(
        0.0,
        min(
            1.0,
            value,
        ),
    )


def get_predicted_direction(
    prediction: dict[str, Any],
) -> str | None:
    """
    Determine the predicted direction.

    Uses an explicit direction field first,
    then falls back to probability or return.
    """

    direction = prediction.get(
        "direction"
    )

    if direction is not None:

        text = str(
            direction
        ).strip().upper()

        if text in {
            "UP",
            "DOWN",
            "FLAT",
        }:

            return text

    probability_up = (
        get_probability_up(
            prediction
        )
    )

    if probability_up is not None:

        if probability_up > 0.5:

            return "UP"

        if probability_up < 0.5:

            return "DOWN"

        return "FLAT"

    expected_return = (
        get_predicted_return(
            prediction
        )
    )

    if expected_return is not None:

        if expected_return > 0:

            return "UP"

        if expected_return < 0:

            return "DOWN"

        return "FLAT"

    return None


# ============================================================
# ACTUAL VALUE EXTRACTION
# ============================================================

def get_actual_open(
    actual: dict[str, Any],
) -> float | None:
    """
    Extract actual opening price.
    """

    return first_numeric(
        actual,
        [
            "Open",
            "open",
            "actual_open",
        ],
    )


def get_actual_close(
    actual: dict[str, Any],
) -> float | None:
    """
    Extract actual closing price.
    """

    return first_numeric(
        actual,
        [
            "Close",
            "close",
            "actual_close",
        ],
    )


def calculate_actual_return(
    actual: dict[str, Any],
) -> float | None:
    """
    Calculate percentage return.

    Return is measured as:

        (Close - Open) / Open * 100
    """

    actual_open = get_actual_open(
        actual
    )

    actual_close = get_actual_close(
        actual
    )

    if actual_open is None:

        return None

    if actual_close is None:

        return None

    if actual_open == 0:

        return None

    return (
        (
            actual_close
            - actual_open
        )
        / actual_open
    ) * 100.0


def get_actual_direction(
    actual_return: float | None,
    flat_threshold: float = 0.0,
) -> str | None:
    """
    Convert actual return into UP, DOWN or FLAT.
    """

    if actual_return is None:

        return None

    threshold = abs(
        float(flat_threshold)
    )

    if actual_return > threshold:

        return "UP"

    if actual_return < -threshold:

        return "DOWN"

    return "FLAT"


# ============================================================
# SINGLE STOCK EVALUATION
# ============================================================

def evaluate_prediction(
    prediction: dict[str, Any],
    actual_result: dict[str, Any],
    flat_threshold: float = 0.0,
) -> dict[str, Any]:
    """
    Evaluate one stock prediction.

    Returns a structured comparison.
    """

    if not isinstance(
        prediction,
        dict,
    ):

        raise TypeError(
            "prediction must be a dictionary."
        )

    if not isinstance(
        actual_result,
        dict,
    ):

        raise TypeError(
            "actual_result must be a dictionary."
        )

    symbol = str(
        prediction.get(
            "symbol",
            "",
        )
    ).strip()

    predicted_return = (
        get_predicted_return(
            prediction
        )
    )

    actual_return = (
        calculate_actual_return(
            actual_result
        )
    )

    predicted_direction = (
        get_predicted_direction(
            prediction
        )
    )

    actual_direction = (
        get_actual_direction(
            actual_return,
            flat_threshold=flat_threshold,
        )
    )

    prediction_error: float | None = None
    absolute_error: float | None = None
    squared_error: float | None = None

    if (
        predicted_return is not None
        and actual_return is not None
    ):

        prediction_error = (
            predicted_return
            - actual_return
        )

        absolute_error = abs(
            prediction_error
        )

        squared_error = (
            prediction_error
            * prediction_error
        )

    direction_correct: bool | None = None

    if (
        predicted_direction is not None
        and actual_direction is not None
    ):

        direction_correct = (
            predicted_direction
            == actual_direction
        )

    return {
        "symbol": symbol,

        "predicted_return": (
            predicted_return
        ),

        "actual_return": (
            actual_return
        ),

        "prediction_error": (
            prediction_error
        ),

        "absolute_error": (
            absolute_error
        ),

        "squared_error": (
            squared_error
        ),

        "predicted_direction": (
            predicted_direction
        ),

        "actual_direction": (
            actual_direction
        ),

        "direction_correct": (
            direction_correct
        ),

        "actual_ohlc": dict(
            actual_result
        ),
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    evaluations: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    """
    Calculate aggregate evaluation metrics.
    """

    if not evaluations:

        return {
            "evaluation_count": 0,
            "return_evaluation_count": 0,
            "direction_evaluation_count": 0,
            "mae": None,
            "rmse": None,
            "mean_error": None,
            "direction_accuracy": None,
            "win_rate": None,
        }

    errors: list[float] = []
    absolute_errors: list[float] = []
    squared_errors: list[float] = []

    direction_results: list[bool] = []
    wins: list[bool] = []

    for evaluation in evaluations:

        error = to_float(
            evaluation.get(
                "prediction_error"
            )
        )

        absolute_error = to_float(
            evaluation.get(
                "absolute_error"
            )
        )

        squared_error = to_float(
            evaluation.get(
                "squared_error"
            )
        )

        if error is not None:

            errors.append(
                error
            )

        if absolute_error is not None:

            absolute_errors.append(
                absolute_error
            )

        if squared_error is not None:

            squared_errors.append(
                squared_error
            )

        direction_correct = (
            evaluation.get(
                "direction_correct"
            )
        )

        if isinstance(
            direction_correct,
            bool,
        ):

            direction_results.append(
                direction_correct
            )

        actual_return = to_float(
            evaluation.get(
                "actual_return"
            )
        )

        if actual_return is not None:

            wins.append(
                actual_return > 0
            )

    mae: float | None = None

    if absolute_errors:

        mae = (
            sum(
                absolute_errors
            )
            / len(
                absolute_errors
            )
        )

    rmse: float | None = None

    if squared_errors:

        rmse = math.sqrt(
            sum(
                squared_errors
            )
            / len(
                squared_errors
            )
        )

    mean_error: float | None = None

    if errors:

        mean_error = (
            sum(errors)
            / len(errors)
        )

    direction_accuracy: float | None = None

    if direction_results:

        direction_accuracy = (
            sum(direction_results)
            / len(direction_results)
        ) * 100.0

    win_rate: float | None = None

    if wins:

        win_rate = (
            sum(wins)
            / len(wins)
        ) * 100.0

    return {
        "evaluation_count": (
            len(evaluations)
        ),

        "return_evaluation_count": (
            len(errors)
        ),

        "direction_evaluation_count": (
            len(direction_results)
        ),

        "mae": mae,

        "rmse": rmse,

        "mean_error": mean_error,

        "direction_accuracy": (
            direction_accuracy
        ),

        "win_rate": win_rate,
    }


# ============================================================
# DAILY EVALUATOR
# ============================================================

class DailyEvaluator:
    """
    Evaluate stored predictions against
    actual market data.
    """

    def __init__(
        self,
        flat_threshold: float = 0.0,
    ) -> None:

        self.flat_threshold = abs(
            float(
                flat_threshold
            )
        )


    # ========================================================
    # EVALUATE STOCKS
    # ========================================================

    def evaluate_stocks(
        self,
        predictions: list[
            dict[str, Any]
        ],
        actual_results: dict[
            str,
            dict[str, Any]
        ],
    ) -> dict[str, Any]:
        """
        Evaluate multiple stock predictions.

        actual_results example:

            {
                "RELIANCE.NS": {
                    "Open": 1400,
                    "High": 1420,
                    "Low": 1390,
                    "Close": 1415,
                }
            }
        """

        if not isinstance(
            predictions,
            list,
        ):

            raise TypeError(
                "predictions must be a list."
            )

        if not isinstance(
            actual_results,
            dict,
        ):

            raise TypeError(
                "actual_results must be "
                "a dictionary."
            )

        evaluations: list[
            dict[str, Any]
        ] = []

        missing_actuals: list[
            str
        ] = []

        for prediction in predictions:

            if not isinstance(
                prediction,
                dict,
            ):

                continue

            symbol = str(
                prediction.get(
                    "symbol",
                    "",
                )
            ).strip()

            if not symbol:

                continue

            actual_result = (
                actual_results.get(
                    symbol
                )
            )

            if not isinstance(
                actual_result,
                dict,
            ):

                missing_actuals.append(
                    symbol
                )

                continue

            evaluation = (
                evaluate_prediction(
                    prediction=prediction,
                    actual_result=actual_result,
                    flat_threshold=(
                        self.flat_threshold
                    ),
                )
            )

            evaluations.append(
                evaluation
            )

        metrics = calculate_metrics(
            evaluations
        )

        return {
            "evaluations": evaluations,

            "metrics": metrics,

            "missing_actuals": (
                missing_actuals
            ),

            "flat_threshold": (
                self.flat_threshold
            ),
        }


    # ========================================================
    # EVALUATE TOP STOCKS
    # ========================================================

    def evaluate_top_stocks(
        self,
        prediction_result: dict[str, Any],
        actual_results: dict[
            str,
            dict[str, Any]
        ],
    ) -> dict[str, Any]:
        """
        Evaluate the stored Top stocks.
        """

        if not isinstance(
            prediction_result,
            dict,
        ):

            raise TypeError(
                "prediction_result must be "
                "a dictionary."
            )

        top_stocks = (
            prediction_result.get(
                "top_stocks",
                []
            )
        )

        if not isinstance(
            top_stocks,
            list,
        ):

            raise RuntimeError(
                "top_stocks must be a list."
            )

        return self.evaluate_stocks(
            predictions=top_stocks,
            actual_results=actual_results,
        )


# ============================================================
# STORE INTEGRATION
# ============================================================

def evaluate_saved_predictions(
    actual_results: dict[
        str,
        dict[str, Any]
    ],
    prediction_date: (
        str
        | date
        | datetime
        | None
    ) = None,
    flat_threshold: float = 0.0,
    mark_evaluated: bool = True,
) -> dict[str, Any]:
    """
    Load saved predictions and evaluate them.

    This integrates with PredictionStore.
    """

    from src.prediction_store import (
        PredictionStore,
    )

    store = PredictionStore()

    normalized_date = normalize_date(
        prediction_date
    )

    payload = store.load(
        normalized_date
    )

    if payload is None:

        raise FileNotFoundError(
            "No saved predictions found."
        )

    prediction_result = payload.get(
        "predictions"
    )

    if not isinstance(
        prediction_result,
        dict,
    ):

        raise RuntimeError(
            "Stored prediction result "
            "is invalid."
        )

    evaluator = DailyEvaluator(
        flat_threshold=flat_threshold
    )

    evaluation = (
        evaluator.evaluate_top_stocks(
            prediction_result=(
                prediction_result
            ),
            actual_results=actual_results,
        )
    )

    evaluation["prediction_date"] = (
        normalized_date
        or payload.get(
            "prediction_date"
        )
    )

    evaluation["evaluated_at"] = (
        datetime.now().astimezone().isoformat()
    )

    if mark_evaluated:

        store.mark_evaluated(
            prediction_date=(
                normalized_date
            ),
            evaluation=evaluation,
        )

    logger.info(
        "Prediction evaluation complete | "
        "date=%s | stocks=%s | "
        "direction_accuracy=%s",
        evaluation.get(
            "prediction_date"
        ),
        evaluation[
            "metrics"
        ].get(
            "evaluation_count"
        ),
        evaluation[
            "metrics"
        ].get(
            "direction_accuracy"
        ),
    )

    return evaluation


# ============================================================
# RETRAINING SIGNAL
# ============================================================

def needs_retraining(
    metrics: dict[str, Any],
    minimum_direction_accuracy: float = 50.0,
    maximum_mae: float | None = None,
    minimum_samples: int = 20,
) -> dict[str, Any]:
    """
    Generate a basic retraining recommendation.

    This does not retrain automatically.

    It only reports whether performance suggests
    retraining should be considered.
    """

    evaluation_count = int(
        metrics.get(
            "evaluation_count",
            0,
        )
        or 0
    )

    direction_accuracy = to_float(
        metrics.get(
            "direction_accuracy"
        )
    )

    mae = to_float(
        metrics.get(
            "mae"
        )
    )

    reasons: list[str] = []

    if evaluation_count < minimum_samples:

        return {
            "should_retrain": False,

            "reason": (
                "INSUFFICIENT_EVALUATION_DATA"
            ),

            "evaluation_count": (
                evaluation_count
            ),

            "minimum_samples": (
                minimum_samples
            ),

            "details": [
                "More evaluation data is required "
                "before making a retraining decision."
            ],
        }

    if (
        direction_accuracy is not None
        and direction_accuracy
        < minimum_direction_accuracy
    ):

        reasons.append(
            "DIRECTION_ACCURACY_BELOW_THRESHOLD"
        )

    if (
        maximum_mae is not None
        and mae is not None
        and mae > maximum_mae
    ):

        reasons.append(
            "MAE_ABOVE_THRESHOLD"
        )

    return {
        "should_retrain": bool(
            reasons
        ),

        "reason": (
            " | ".join(reasons)
            if reasons
            else "MODEL_PERFORMANCE_ACCEPTABLE"
        ),

        "evaluation_count": (
            evaluation_count
        ),

        "direction_accuracy": (
            direction_accuracy
        ),

        "mae": mae,

        "details": reasons,
    }


# ============================================================
# CLI TEST
# ============================================================

def main() -> int:
    """Display evaluator information."""

    print()

    print("=" * 70)

    print(
        "DAILY PREDICTION EVALUATOR"
    )

    print("=" * 70)

    example_predictions = [
        {
            "symbol": "TEST.NS",
            "expected_return": 2.0,
            "probability_up": 0.70,
            "direction": "UP",
        }
    ]

    example_actuals = {
        "TEST.NS": {
            "Open": 100.0,
            "High": 104.0,
            "Low": 99.0,
            "Close": 102.0,
        }
    }

    evaluator = DailyEvaluator()

    result = evaluator.evaluate_stocks(
        predictions=(
            example_predictions
        ),
        actual_results=(
            example_actuals
        ),
    )

    print()

    print(
        "Metrics:"
    )

    for key, value in (
        result["metrics"].items()
    ):

        print(
            f"{key}: {value}"
        )

    print()

    print(
        "SUCCESS: DailyEvaluator "
        "is ready."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

#!/usr/bin/env python3

"""
Daily Production Orchestrator.

Coordinates the complete daily ML workflow:

    1. Load production Champion model.
    2. Load market data for multiple symbols.
    3. Generate features.
    4. Run predictions.
    5. Rank all stocks.
    6. Select Top-N opportunities.
    7. Send morning Telegram report.
    8. Persist predictions.
    9. Load actual OHLC data.
    10. Compare prediction vs actual.
    11. Calculate performance metrics.
    12. Decide whether retraining is needed.
    13. Send evening Telegram report.

External data providers are intentionally injected through
callbacks. This keeps the orchestrator independent from the
specific market data API.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.model_loader import load_champion_model
from src.telegram_reporter import TelegramReporter


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTION_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "predictions"
)


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(
    "daily_orchestrator"
)


# ============================================================
# TIME
# ============================================================

def utc_now_iso() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def trading_date_string() -> str:
    """Return today's date."""

    return datetime.now().strftime(
        "%Y-%m-%d"
    )


# ============================================================
# HELPERS
# ============================================================

def safe_float(
    value: Any,
    default: float | None = None,
) -> float | None:
    """Convert a value to a finite float."""

    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return default

    if not math.isfinite(
        result
    ):

        return default

    return result


# ============================================================
# PREDICTION STORAGE
# ============================================================

class PredictionStore:
    """
    Persist daily predictions.

    Each trading day is stored separately.
    """

    def __init__(
        self,
        base_path: Path | None = None,
    ) -> None:

        self.base_path = (
            base_path
            or PREDICTION_DIRECTORY
        )

        self.base_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    def get_path(
        self,
        date: str,
    ) -> Path:

        return (
            self.base_path
            / f"{date}.json"
        )

    def save(
        self,
        date: str,
        predictions: list[
            dict[str, Any]
        ],
        metadata: dict[
            str,
            Any,
        ] | None = None,
    ) -> Path:
        """Save daily predictions."""

        path = self.get_path(
            date
        )

        payload = {
            "date": date,
            "saved_at": utc_now_iso(),
            "predictions": predictions,
            "metadata": metadata or {},
        }

        temporary = path.with_suffix(
            ".tmp"
        )

        with temporary.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                indent=2,
                default=str,
            )

        temporary.replace(
            path
        )

        logger.info(
            "Saved predictions: %s",
            path,
        )

        return path

    def load(
        self,
        date: str,
    ) -> dict[str, Any] | None:
        """Load predictions for a trading date."""

        path = self.get_path(
            date
        )

        if not path.exists():

            return None

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(
                file
            )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    comparisons: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    """
    Calculate prediction performance metrics.
    """

    close_errors: list[float] = []

    direction_total = 0

    direction_correct = 0

    for item in comparisons:

        predicted = safe_float(
            item.get(
                "predicted_close"
            )
        )

        actual = safe_float(
            item.get(
                "actual_close"
            )
        )

        if (
            predicted is not None
            and actual is not None
        ):

            close_errors.append(
                abs(
                    actual - predicted
                )
            )

        predicted_direction = str(
            item.get(
                "predicted_direction",
                item.get(
                    "direction",
                    "",
                ),
            )
        ).upper()

        actual_direction = str(
            item.get(
                "actual_direction",
                "",
            )
        ).upper()

        if (
            predicted_direction
            and actual_direction
        ):

            direction_total += 1

            if (
                predicted_direction
                == actual_direction
            ):

                direction_correct += 1

    if close_errors:

        mae = sum(
            close_errors
        ) / len(
            close_errors
        )

        rmse = math.sqrt(
            sum(
                error ** 2
                for error in close_errors
            )
            / len(close_errors)
        )

    else:

        mae = None
        rmse = None

    direction_accuracy = None

    if direction_total:

        direction_accuracy = (
            direction_correct
            / direction_total
        )

    return {
        "sample_count": len(
            comparisons
        ),
        "close_mae": mae,
        "close_rmse": rmse,
        "direction_total": (
            direction_total
        ),
        "direction_correct": (
            direction_correct
        ),
        "direction_accuracy": (
            direction_accuracy
        ),
    }


# ============================================================
# RETRAINING DECISION
# ============================================================

def evaluate_retraining(
    metrics: dict[str, Any],
    minimum_direction_accuracy: float = 0.50,
    maximum_close_mae: float | None = None,
) -> dict[str, Any]:
    """
    Decide whether model retraining is recommended.
    """

    triggered_rules: list[str] = []

    direction_accuracy = safe_float(
        metrics.get(
            "direction_accuracy"
        )
    )

    close_mae = safe_float(
        metrics.get(
            "close_mae"
        )
    )

    if (
        direction_accuracy is not None
        and direction_accuracy
        < minimum_direction_accuracy
    ):

        triggered_rules.append(
            "DIRECTION_ACCURACY_BELOW_THRESHOLD"
        )

    if (
        maximum_close_mae is not None
        and close_mae is not None
        and close_mae
        > maximum_close_mae
    ):

        triggered_rules.append(
            "CLOSE_MAE_ABOVE_THRESHOLD"
        )

    should_retrain = bool(
        triggered_rules
    )

    if should_retrain:

        reason = (
            "Model performance crossed "
            "retraining thresholds."
        )

    else:

        reason = (
            "Current performance is within "
            "configured thresholds."
        )

    return {
        "should_retrain": should_retrain,
        "reason": reason,
        "triggered_rules": (
            triggered_rules
        ),
        "metrics": metrics,
    }


# ============================================================
# DAILY ORCHESTRATOR
# ============================================================

class DailyOrchestrator:
    """
    Execute the daily stock prediction workflow.

    The market data functions must be supplied by the
    application.

    market_data_loader(symbol) -> DataFrame

    actual_data_loader(symbol) -> DataFrame or dict
    """

    def __init__(
        self,
        symbols: list[str],
        market_data_loader: Callable[
            [str],
            pd.DataFrame,
        ],
        actual_data_loader: Callable[
            [str],
            Any,
        ] | None = None,
        top_n: int = 5,
        reporter: TelegramReporter | None = None,
        prediction_store: (
            PredictionStore | None
        ) = None,
    ) -> None:

        self.symbols = list(
            symbols
        )

        self.market_data_loader = (
            market_data_loader
        )

        self.actual_data_loader = (
            actual_data_loader
        )

        self.top_n = max(
            int(top_n),
            1,
        )

        self.reporter = (
            reporter
            or TelegramReporter()
        )

        self.prediction_store = (
            prediction_store
            or PredictionStore()
        )

    # ========================================================
    # FEATURE BUILDING
    # ========================================================

    def build_features(
        self,
        history: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build model features from market history.
        """

        from src.feature_engine import (
            build_feature_frame,
        )

        frame = build_feature_frame(
            history
        )

        if frame.empty:

            raise RuntimeError(
                "Feature generation returned "
                "an empty DataFrame."
            )

        return frame

    # ========================================================
    # PREDICT ONE SYMBOL
    # ========================================================

    def predict_symbol(
        self,
        model: Any,
        symbol: str,
    ) -> dict[str, Any]:
        """
        Generate prediction for one stock.
        """

        history = self.market_data_loader(
            symbol
        )

        if (
            history is None
            or history.empty
        ):

            raise RuntimeError(
                f"No market data for {symbol}"
            )

        features = self.build_features(
            history
        )

        prediction = model.predict(
            features
        )

        if not isinstance(
            prediction,
            dict,
        ):

            raise RuntimeError(
                "Model predict() must return "
                "a dictionary."
            )

        result = dict(
            prediction
        )

        result["symbol"] = symbol

        latest = history.iloc[-1]

        latest_close = safe_float(
            latest.get("Close")
        )

        expected_return = safe_float(
            result.get(
                "expected_return"
            ),
            0.0,
        ) or 0.0

        # ----------------------------------------------------
        # DEFAULT PREDICTED OHLC
        # ----------------------------------------------------

        predicted_close = (
            latest_close
            * (1.0 + expected_return)
            if latest_close is not None
            else None
        )

        predicted_open = safe_float(
            result.get(
                "predicted_open"
            ),
            latest_close,
        )

        expected_risk = safe_float(
            result.get(
                "expected_risk"
            ),
            0.02,
        ) or 0.02

        if predicted_close is not None:

            predicted_high = max(
                predicted_open
                or predicted_close,
                predicted_close,
            ) * (
                1.0
                + expected_risk
            )

            predicted_low = min(
                predicted_open
                or predicted_close,
                predicted_close,
            ) * (
                1.0
                - expected_risk
            )

        else:

            predicted_high = None
            predicted_low = None

        result.setdefault(
            "predicted_open",
            predicted_open,
        )

        result.setdefault(
            "predicted_close",
            predicted_close,
        )

        result.setdefault(
            "predicted_high",
            predicted_high,
        )

        result.setdefault(
            "predicted_low",
            predicted_low,
        )

        result["last_close"] = (
            latest_close
        )

        return result

    # ========================================================
    # MORNING WORKFLOW
    # ========================================================

    def run_morning(
        self,
        date: str | None = None,
        sentiment_summary: dict[
            str,
            Any,
        ] | None = None,
        fundamental_summary: dict[
            str,
            Any,
        ] | None = None,
        index_summary: dict[
            str,
            Any,
        ] | None = None,
        send_telegram: bool = True,
    ) -> dict[str, Any]:
        """
        Execute the morning prediction workflow.
        """

        report_date = (
            date
            or trading_date_string()
        )

        model, model_metadata = (
            load_champion_model()
        )

        predictions: list[
            dict[str, Any]
        ] = []

        failures: list[
            dict[str, str]
        ] = []

        for symbol in self.symbols:

            try:

                prediction = (
                    self.predict_symbol(
                        model=model,
                        symbol=symbol,
                    )
                )

                predictions.append(
                    prediction
                )

            except Exception as error:

                logger.exception(
                    "Prediction failed for %s",
                    symbol,
                )

                failures.append(
                    {
                        "symbol": symbol,
                        "error": str(error),
                    }
                )

        predictions.sort(
            key=lambda item: (
                safe_float(
                    item.get(
                        "opportunity_score"
                    ),
                    float("-inf"),
                )
                or float("-inf")
            ),
            reverse=True,
        )

        top_predictions = predictions[
            :self.top_n
        ]

        path = self.prediction_store.save(
            date=report_date,
            predictions=top_predictions,
            metadata={
                "model": model_metadata,
                "total_symbols": len(
                    self.symbols
                ),
                "successful_predictions": len(
                    predictions
                ),
                "failed_predictions": len(
                    failures
                ),
                "failures": failures,
                "sentiment_summary": (
                    sentiment_summary
                ),
                "fundamental_summary": (
                    fundamental_summary
                ),
                "index_summary": (
                    index_summary
                ),
            },
        )

        telegram_response = None

        if send_telegram:

            telegram_response = (
                self.reporter.send_morning_report(
                    predictions=(
                        top_predictions
                    ),
                    top_n=self.top_n,
                    report_date=report_date,
                    sentiment_summary=(
                        sentiment_summary
                    ),
                    fundamental_summary=(
                        fundamental_summary
                    ),
                    index_summary=(
                        index_summary
                    ),
                )
            )

        return {
            "date": report_date,
            "model_metadata": (
                model_metadata
            ),
            "predictions": (
                top_predictions
            ),
            "prediction_count": len(
                top_predictions
            ),
            "failures": failures,
            "prediction_path": str(
                path
            ),
            "telegram_response": (
                telegram_response
            ),
        }

    # ========================================================
    # ACTUAL DATA
    # ========================================================

    def get_actual_ohlc(
        self,
        symbol: str,
    ) -> dict[str, Any]:
        """
        Load actual OHLC data for one symbol.
        """

        if self.actual_data_loader is None:

            raise RuntimeError(
                "actual_data_loader is not configured."
            )

        data = self.actual_data_loader(
            symbol
        )

        if isinstance(
            data,
            pd.DataFrame,
        ):

            if data.empty:

                raise RuntimeError(
                    f"No actual data for {symbol}"
                )

            row = data.iloc[-1].to_dict()

        elif isinstance(
            data,
            pd.Series,
        ):

            row = data.to_dict()

        elif isinstance(
            data,
            dict,
        ):

            row = dict(
                data
            )

        else:

            raise TypeError(
                "Actual data loader must return "
                "DataFrame, Series, or dict."
            )

        return {
            "actual_open": safe_float(
                row.get("Open")
                if "Open" in row
                else row.get("actual_open")
            ),
            "actual_high": safe_float(
                row.get("High")
                if "High" in row
                else row.get("actual_high")
            ),
            "actual_low": safe_float(
                row.get("Low")
                if "Low" in row
                else row.get("actual_low")
            ),
            "actual_close": safe_float(
                row.get("Close")
                if "Close" in row
                else row.get("actual_close")
            ),
        }

    # ========================================================
    # EVENING WORKFLOW
    # ========================================================

    def run_evening(
        self,
        date: str | None = None,
        send_telegram: bool = True,
        minimum_direction_accuracy: float = 0.50,
        maximum_close_mae: float | None = None,
    ) -> dict[str, Any]:
        """
        Execute the evening evaluation workflow.
        """

        report_date = (
            date
            or trading_date_string()
        )

        stored = self.prediction_store.load(
            report_date
        )

        if stored is None:

            raise RuntimeError(
                "No predictions found for "
                f"{report_date}"
            )

        predictions = stored.get(
            "predictions",
            [],
        )

        comparisons: list[
            dict[str, Any]
        ] = []

        failures: list[
            dict[str, str]
        ] = []

        for prediction in predictions:

            symbol = prediction.get(
                "symbol"
            )

            if not symbol:

                continue

            try:

                actual = (
                    self.get_actual_ohlc(
                        symbol
                    )
                )

                comparison = dict(
                    prediction
                )

                comparison.update(
                    actual
                )

                # --------------------------------------------
                # ACTUAL DIRECTION
                # --------------------------------------------

                previous_close = safe_float(
                    prediction.get(
                        "last_close"
                    )
                )

                actual_close = safe_float(
                    actual.get(
                        "actual_close"
                    )
                )

                if (
                    previous_close is not None
                    and actual_close is not None
                ):

                    if (
                        actual_close
                        > previous_close
                    ):

                        actual_direction = "UP"

                    elif (
                        actual_close
                        < previous_close
                    ):

                        actual_direction = "DOWN"

                    else:

                        actual_direction = "NEUTRAL"

                else:

                    actual_direction = "UNKNOWN"

                comparison[
                    "predicted_direction"
                ] = prediction.get(
                    "direction",
                    "NEUTRAL",
                )

                comparison[
                    "actual_direction"
                ] = actual_direction

                comparisons.append(
                    comparison
                )

            except Exception as error:

                logger.exception(
                    "Evaluation failed for %s",
                    symbol,
                )

                failures.append(
                    {
                        "symbol": str(symbol),
                        "error": str(error),
                    }
                )

        metrics = calculate_metrics(
            comparisons
        )

        retraining_status = (
            evaluate_retraining(
                metrics=metrics,
                minimum_direction_accuracy=(
                    minimum_direction_accuracy
                ),
                maximum_close_mae=(
                    maximum_close_mae
                ),
            )
        )

        telegram_response = None

        if send_telegram:

            telegram_response = (
                self.reporter.send_evening_report(
                    comparisons=comparisons,
                    metrics=metrics,
                    retraining_status=(
                        retraining_status
                    ),
                    report_date=report_date,
                )
            )

        return {
            "date": report_date,
            "comparisons": comparisons,
            "metrics": metrics,
            "retraining_status": (
                retraining_status
            ),
            "failures": failures,
            "telegram_response": (
                telegram_response
            ),
        }


# ============================================================
# CLI
# ============================================================

def main() -> int:
    """
    Basic orchestrator information.

    A real application should instantiate DailyOrchestrator
    with its market data provider.
    """

    print()

    print("=" * 70)
    print("DAILY PRODUCTION ORCHESTRATOR")
    print("=" * 70)

    print()

    print(
        "This module must be connected to "
        "market_data_loader and actual_data_loader."
    )

    print()

    print(
        "Morning workflow:"
    )

    print(
        "Load Champion -> Predict all symbols "
        "-> Rank -> Top 5 -> Telegram"
    )

    print()

    print(
        "Evening workflow:"
    )

    print(
        "Load actual OHLC -> Compare "
        "-> Metrics -> Retraining decision "
        "-> Telegram"
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

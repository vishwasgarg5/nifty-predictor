#!/usr/bin/env python3

"""Evaluate ML predictions against actual market results.

Phase 4 pipeline:

    Prediction Ledger
            │
            ▼
    Find Pending Predictions
            │
            ▼
    Download Actual OHLC
            │
            ▼
    Actual vs Predicted
            │
            ├── Return Error
            ├── Direction Accuracy
            ├── Probability Error
            ├── Risk Error
            └── OHLC Error
                    │
                    ▼
              Update Ledger
                    │
                    ▼
              Evaluation Report

This script should normally run after the market
session for predictions generated earlier.

Example:

    python scripts/evaluate_models.py
"""

from __future__ import annotations

import sys
import logging
import traceback

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.config import cfg

from src.data_loader import (
    download_history,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    __name__
)


# ============================================================
# CONFIG
# ============================================================

def get_ledger_path() -> Path:
    """Resolve prediction ledger path."""

    try:

        paths = getattr(
            cfg,
            "paths",
            None,
        )

        value = getattr(
            paths,
            "ledger",
            None,
        )

        if value:

            return Path(
                value
            )

    except Exception:

        pass

    return (
        PROJECT_ROOT
        / "data"
        / "ledger"
        / "predictions.csv"
    )


def get_reports_dir() -> Path:
    """Resolve evaluation reports directory."""

    try:

        paths = getattr(
            cfg,
            "paths",
            None,
        )

        value = getattr(
            paths,
            "reports",
            None,
        )

        if value:

            path = Path(
                value
            )

            path.mkdir(
                parents=True,
                exist_ok=True,
            )

            return path

    except Exception:

        pass

    path = (
        PROJECT_ROOT
        / "data"
        / "reports"
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


# ============================================================
# DATA HELPERS
# ============================================================

def safe_float(
    value: Any,
    default: float = np.nan,
) -> float:
    """Convert value to float safely."""

    try:

        if value is None:

            return default

        result = float(
            value
        )

        if np.isnan(result):

            return default

        return result

    except (
        TypeError,
        ValueError,
    ):

        return default


def normalize_probability(
    value: Any,
) -> float:
    """Normalize probability to 0-1."""

    probability = safe_float(
        value,
        default=0.5,
    )

    if probability > 1.0:

        probability = (
            probability / 100.0
        )

    return max(
        0.0,
        min(
            1.0,
            probability,
        ),
    )


def parse_market_date(
    value: Any,
) -> pd.Timestamp | None:
    """Parse market date safely."""

    try:

        timestamp = pd.to_datetime(
            value,
            errors="coerce",
        )

        if pd.isna(timestamp):

            return None

        return pd.Timestamp(
            timestamp
        ).normalize()

    except Exception:

        return None


# ============================================================
# LOAD LEDGER
# ============================================================

def load_ledger() -> tuple[
    pd.DataFrame,
    Path,
]:
    """Load prediction ledger."""

    ledger_path = (
        get_ledger_path()
    )

    if not ledger_path.exists():

        logger.warning(
            "Ledger does not exist: %s",
            ledger_path,
        )

        return (
            pd.DataFrame(),
            ledger_path,
        )

    try:

        frame = pd.read_csv(
            ledger_path
        )

        logger.info(
            "Loaded ledger: %s rows",
            len(frame),
        )

        return (
            frame,
            ledger_path,
        )

    except Exception as error:

        logger.error(
            "Failed to load ledger: %s",
            error,
        )

        return (
            pd.DataFrame(),
            ledger_path,
        )


# ============================================================
# ACTUAL MARKET DATA
# ============================================================

def get_actual_ohlc(
    symbol: str,
    market_date: Any,
) -> dict[str, float] | None:
    """Fetch actual OHLC for prediction date."""

    target_date = parse_market_date(
        market_date
    )

    if target_date is None:

        return None

    try:

        history = download_history(
            symbol,
            period="10d",
        )

        if (
            history is None
            or history.empty
        ):

            return None

        history = history.copy()

        history.index = pd.to_datetime(
            history.index
        )

        if getattr(
            history.index,
            "tz",
            None,
        ) is not None:

            history.index = (
                history.index.tz_localize(
                    None
                )
            )

        history_dates = (
            pd.DatetimeIndex(
                history.index
            ).normalize()
        )

        matching = history.loc[
            history_dates
            == target_date
        ]

        if matching.empty:

            return None

        row = matching.iloc[-1]

        required = (
            "Open",
            "High",
            "Low",
            "Close",
        )

        for column in required:

            if column not in row.index:

                return None

        return {

            "actual_open": safe_float(
                row["Open"]
            ),

            "actual_high": safe_float(
                row["High"]
            ),

            "actual_low": safe_float(
                row["Low"]
            ),

            "actual_close": safe_float(
                row["Close"]
            ),
        }

    except Exception as error:

        logger.warning(
            "Actual data failed for %s: %s",
            symbol,
            error,
        )

        return None


# ============================================================
# EVALUATION METRICS
# ============================================================

def calculate_metrics(
    record: pd.Series,
    actual: dict[str, float],
) -> dict[str, Any]:
    """Calculate prediction errors and classification metrics."""

    current_close = safe_float(
        record.get(
            "current_close"
        )
    )

    predicted_close = safe_float(
        record.get(
            "predicted_close"
        )
    )

    actual_close = safe_float(
        actual.get(
            "actual_close"
        )
    )

    predicted_return = safe_float(
        record.get(
            "expected_return"
        )
    )

    actual_return = np.nan

    if (
        np.isfinite(current_close)
        and current_close != 0
        and np.isfinite(actual_close)
    ):

        actual_return = (
            actual_close
            / current_close
        ) - 1.0

    # --------------------------------------------------------
    # RETURN ERROR
    # --------------------------------------------------------

    return_error = np.nan

    if (
        np.isfinite(predicted_return)
        and np.isfinite(actual_return)
    ):

        return_error = (
            actual_return
            - predicted_return
        )

    return_absolute_error = np.nan

    if np.isfinite(
        return_error
    ):

        return_absolute_error = abs(
            return_error
        )

    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    actual_direction = None

    if np.isfinite(
        actual_return
    ):

        actual_direction = (
            "UP"
            if actual_return >= 0
            else "DOWN"
        )

    predicted_direction = str(
        record.get(
            "direction",
            "",
        )
    ).upper()

    direction_correct = np.nan

    if (
        actual_direction
        and predicted_direction
        in (
            "UP",
            "DOWN",
        )
    ):

        direction_correct = int(
            predicted_direction
            == actual_direction
        )

    # --------------------------------------------------------
    # PROBABILITY / BRIER SCORE
    # --------------------------------------------------------

    probability_up = (
        normalize_probability(
            record.get(
                "probability_up"
            )
        )
    )

    actual_up = np.nan

    if actual_direction:

        actual_up = int(
            actual_direction
            == "UP"
        )

    brier_score = np.nan

    if np.isfinite(
        actual_up
    ):

        brier_score = (
            probability_up
            - actual_up
        ) ** 2

    probability_error = np.nan

    if np.isfinite(
        actual_up
    ):

        probability_error = abs(
            probability_up
            - actual_up
        )

    # --------------------------------------------------------
    # OHLC ERRORS
    # --------------------------------------------------------

    predicted_open = safe_float(
        record.get(
            "predicted_open"
        )
    )

    predicted_high = safe_float(
        record.get(
            "predicted_high"
        )
    )

    predicted_low = safe_float(
        record.get(
            "predicted_low"
        )
    )

    open_error = np.nan
    high_error = np.nan
    low_error = np.nan
    close_error = np.nan

    if (
        np.isfinite(
            predicted_open
        )
        and np.isfinite(
            actual["actual_open"]
        )
    ):

        open_error = abs(
            actual["actual_open"]
            - predicted_open
        )

    if (
        np.isfinite(
            predicted_high
        )
        and np.isfinite(
            actual["actual_high"]
        )
    ):

        high_error = abs(
            actual["actual_high"]
            - predicted_high
        )

    if (
        np.isfinite(
            predicted_low
        )
        and np.isfinite(
            actual["actual_low"]
        )
    ):

        low_error = abs(
            actual["actual_low"]
            - predicted_low
        )

    if (
        np.isfinite(
            predicted_close
        )
        and np.isfinite(
            actual["actual_close"]
        )
    ):

        close_error = abs(
            actual["actual_close"]
            - predicted_close
        )

    # --------------------------------------------------------
    # RISK ERROR
    # --------------------------------------------------------

    predicted_risk = safe_float(
        record.get(
            "expected_risk"
        )
    )

    realized_risk = np.nan

    if (
        np.isfinite(current_close)
        and current_close != 0
        and np.isfinite(
            actual["actual_high"]
        )
        and np.isfinite(
            actual["actual_low"]
        )
    ):

        realized_risk = (
            actual["actual_high"]
            - actual["actual_low"]
        ) / current_close

    risk_error = np.nan

    if (
        np.isfinite(
            predicted_risk
        )
        and np.isfinite(
            realized_risk
        )
    ):

        risk_error = (
            realized_risk
            - predicted_risk
        )

    risk_absolute_error = np.nan

    if np.isfinite(
        risk_error
    ):

        risk_absolute_error = abs(
            risk_error
        )

    return {

        **actual,

        "actual_return": (
            actual_return
        ),

        "actual_direction": (
            actual_direction
        ),

        "return_error": (
            return_error
        ),

        "return_absolute_error": (
            return_absolute_error
        ),

        "direction_correct": (
            direction_correct
        ),

        "actual_up": (
            actual_up
        ),

        "brier_score": (
            brier_score
        ),

        "probability_error": (
            probability_error
        ),

        "open_absolute_error": (
            open_error
        ),

        "high_absolute_error": (
            high_error
        ),

        "low_absolute_error": (
            low_error
        ),

        "close_absolute_error": (
            close_error
        ),

        "realized_risk": (
            realized_risk
        ),

        "risk_error": (
            risk_error
        ),

        "risk_absolute_error": (
            risk_absolute_error
        ),

        "evaluation_status": (
            "EVALUATED"
        ),

        "evaluated_at": (
            datetime.now().isoformat()
        ),
    }


# ============================================================
# FIND PENDING RECORDS
# ============================================================

def get_pending_records(
    ledger: pd.DataFrame,
) -> list[int]:
    """Return indexes that still require evaluation."""

    if ledger.empty:

        return []

    if "evaluation_status" not in ledger.columns:

        return list(
            ledger.index
        )

    pending: list[int] = []

    for index, row in ledger.iterrows():

        status = str(
            row.get(
                "evaluation_status",
                "",
            )
        ).upper()

        actual_close = safe_float(
            row.get(
                "actual_close"
            )
        )

        if (
            status != "EVALUATED"
            or not np.isfinite(
                actual_close
            )
        ):

            pending.append(
                index
            )

    return pending


# ============================================================
# SUMMARY METRICS
# ============================================================

def mean_or_nan(
    frame: pd.DataFrame,
    column: str,
) -> float | None:
    """Safely calculate mean."""

    if column not in frame.columns:

        return None

    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).dropna()

    if values.empty:

        return None

    return float(
        values.mean()
    )


def calculate_summary(
    evaluated: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate overall evaluation summary."""

    if evaluated.empty:

        return {}

    total = len(
        evaluated
    )

    direction_values = (
        pd.to_numeric(
            evaluated.get(
                "direction_correct",
                pd.Series(
                    dtype=float
                ),
            ),
            errors="coerce",
        )
        .dropna()
    )

    direction_accuracy = None

    if not direction_values.empty:

        direction_accuracy = float(
            direction_values.mean()
        )

    return {

        "evaluated_predictions": total,

        "return_mae": mean_or_nan(
            evaluated,
            "return_absolute_error",
        ),

        "direction_accuracy": (
            direction_accuracy
        ),

        "mean_brier_score": mean_or_nan(
            evaluated,
            "brier_score",
        ),

        "mean_probability_error": (
            mean_or_nan(
                evaluated,
                "probability_error",
            )
        ),

        "risk_mae": mean_or_nan(
            evaluated,
            "risk_absolute_error",
        ),

        "open_mae": mean_or_nan(
            evaluated,
            "open_absolute_error",
        ),

        "high_mae": mean_or_nan(
            evaluated,
            "high_absolute_error",
        ),

        "low_mae": mean_or_nan(
            evaluated,
            "low_absolute_error",
        ),

        "close_mae": mean_or_nan(
            evaluated,
            "close_absolute_error",
        ),
    }


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    evaluated: pd.DataFrame,
    summary: dict[str, Any],
) -> Path:
    """Save evaluation report."""

    reports_dir = (
        get_reports_dir()
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_path = (
        reports_dir
        / f"evaluation_{timestamp}.csv"
    )

    evaluated.to_csv(
        report_path,
        index=False,
    )

    summary_path = (
        reports_dir
        / f"evaluation_{timestamp}_summary.csv"
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        summary_path,
        index=False,
    )

    logger.info(
        "Evaluation report: %s",
        report_path,
    )

    logger.info(
        "Summary report: %s",
        summary_path,
    )

    return report_path


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    """Evaluate pending predictions."""

    started_at = datetime.now()

    logger.info(
        "=" * 70
    )

    logger.info(
        "PHASE 4 MODEL EVALUATION STARTED"
    )

    logger.info(
        "=" * 70
    )


    # --------------------------------------------------------
    # LOAD LEDGER
    # --------------------------------------------------------

    ledger, ledger_path = (
        load_ledger()
    )

    if ledger.empty:

        logger.info(
            "No ledger records available."
        )

        return 0


    # --------------------------------------------------------
    # PENDING RECORDS
    # --------------------------------------------------------

    pending_indexes = (
        get_pending_records(
            ledger
        )
    )

    logger.info(
        "Pending evaluations: %s",
        len(pending_indexes),
    )

    if not pending_indexes:

        logger.info(
            "No pending predictions."
        )

        return 0


    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    evaluated_indexes: list[
        int
    ] = []

    for position, index in enumerate(
        pending_indexes,
        start=1,
    ):

        row = ledger.loc[
            index
        ]

        symbol = str(
            row.get(
                "symbol",
                "",
            )
        )

        market_date = row.get(
            "market_date"
        )

        if not symbol:

            continue

        logger.info(
            "[%s/%s] Evaluating %s | %s",
            position,
            len(pending_indexes),
            symbol,
            market_date,
        )

        try:

            actual = get_actual_ohlc(

                symbol=symbol,

                market_date=market_date,
            )

            if not actual:

                logger.info(
                    "%s | actual data not available yet",
                    symbol,
                )

                continue

            metrics = calculate_metrics(

                record=row,

                actual=actual,
            )

            for key, value in metrics.items():

                ledger.loc[
                    index,
                    key,
                ] = value

            evaluated_indexes.append(
                index
            )

            logger.info(

                "%s EVALUATED | "
                "return_error=%+.4f | "
                "direction=%s | "
                "brier=%.4f",

                symbol,

                safe_float(
                    metrics.get(
                        "return_error"
                    ),
                    default=0.0,
                ),

                metrics.get(
                    "direction_correct"
                ),

                safe_float(
                    metrics.get(
                        "brier_score"
                    ),
                    default=0.0,
                ),
            )

        except Exception as error:

            logger.warning(
                "Evaluation failed for %s: %s",
                symbol,
                error,
            )

            logger.debug(
                traceback.format_exc()
            )


    # --------------------------------------------------------
    # SAVE UPDATED LEDGER
    # --------------------------------------------------------

    if evaluated_indexes:

        ledger_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        ledger.to_csv(
            ledger_path,
            index=False,
        )

        logger.info(
            "Updated ledger: %s",
            ledger_path,
        )


    # --------------------------------------------------------
    # BUILD EVALUATED DATASET
    # --------------------------------------------------------

    if (
        "evaluation_status"
        not in ledger.columns
    ):

        logger.info(
            "No evaluated records."
        )

        return 0

    evaluated = ledger.loc[
        ledger[
            "evaluation_status"
        ]
        .astype(str)
        .str.upper()
        == "EVALUATED"
    ].copy()


    if evaluated.empty:

        logger.info(
            "No records evaluated yet."
        )

        return 0


    # --------------------------------------------------------
    # CALCULATE SUMMARY
    # --------------------------------------------------------

    summary = (
        calculate_summary(
            evaluated
        )
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        "MODEL EVALUATION SUMMARY"
    )

    logger.info(
        "Evaluated predictions: %s",
        summary.get(
            "evaluated_predictions"
        ),
    )

    logger.info(
        "Return MAE: %s",
        format(
            summary["return_mae"],
            ".6f",
        )
        if summary.get(
            "return_mae"
        ) is not None
        else "N/A",
    )

    logger.info(
        "Direction Accuracy: %s",
        format(
            summary["direction_accuracy"],
            ".2%",
        )
        if summary.get(
            "direction_accuracy"
        ) is not None
        else "N/A",
    )

    logger.info(
        "Brier Score: %s",
        format(
            summary["mean_brier_score"],
            ".6f",
        )
        if summary.get(
            "mean_brier_score"
        ) is not None
        else "N/A",
    )

    logger.info(
        "Risk MAE: %s",
        format(
            summary["risk_mae"],
            ".6f",
        )
        if summary.get(
            "risk_mae"
        ) is not None
        else "N/A",
    )

    logger.info(
        "=" * 70
    )


    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    save_report(
        evaluated,
        summary,
    )


    elapsed = int(
        (
            datetime.now()
            - started_at
        ).total_seconds()
    )

    logger.info(
        "Evaluation completed in %ss",
        elapsed,
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

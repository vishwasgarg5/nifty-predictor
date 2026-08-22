#!/usr/bin/env python3

"""
Evaluation Job.

Pipeline
--------
1. Load prediction ledger.
2. Ensure every record has a stable prediction_id.
3. Find pending predictions.
4. Resolve actual market outcomes.
5. Merge outcomes using prediction_id.
6. Save updated prediction ledger.
7. Calculate evaluation metrics.
8. Save latest evaluation report.
9. Run production monitoring.

IMPORTANT
---------
Ledger records are NEVER merged using DataFrame indexes.

All outcome updates use:

    prediction_id
"""

from __future__ import annotations

import hashlib
import logging
import sys
import traceback
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

logger = logging.getLogger("evaluation_job")


# ============================================================
# TIME HELPERS
# ============================================================

def utc_now() -> datetime:
    """Return current UTC datetime."""

    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return current UTC time as ISO string."""

    return utc_now().isoformat()


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

    from src.config import cfg

    return cfg


# ============================================================
# PATH HELPERS
# ============================================================

def resolve_path(
    value: str | Path,
) -> Path:
    """Resolve a project-relative path."""

    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_ledger_path() -> Path:
    """Get prediction ledger path."""

    cfg = load_config()

    candidates: list[Any] = []

    for section_name in [
        "ledger",
        "data",
        "paths",
    ]:

        section = getattr(
            cfg,
            section_name,
            None,
        )

        values = object_to_dict(section)

        for key in [
            "path",
            "prediction_ledger",
            "ledger",
        ]:

            value = values.get(key)

            if value:
                candidates.append(value)

    for candidate in candidates:

        return resolve_path(candidate)

    return (
        PROJECT_ROOT
        / "data"
        / "ledger"
        / "predictions.csv"
    )


def get_reports_dir() -> Path:
    """Get reports directory."""

    cfg = load_config()

    for section_name in [
        "data",
        "paths",
    ]:

        section = getattr(
            cfg,
            section_name,
            None,
        )

        values = object_to_dict(section)

        for key in [
            "reports_dir",
            "reports",
        ]:

            value = values.get(key)

            if value:
                return resolve_path(value)

    return PROJECT_ROOT / "data" / "reports"


# ============================================================
# GENERIC HELPERS
# ============================================================

def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Find first existing column."""

    for column in candidates:

        if column in frame.columns:
            return column

    return None


def normalize_value(
    value: Any,
) -> str:
    """Normalize a value for ID generation."""

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# PREDICTION ID
# ============================================================

def generate_prediction_id(
    row: pd.Series,
) -> str:
    """
    Generate a stable prediction ID.

    Primary identity:

        market_date
        symbol
        model_version

    The final ID is deterministic, meaning the
    same prediction data generates the same ID.
    """

    market_date = ""

    for column in [
        "market_date",
        "prediction_date",
        "date",
        "created_at",
        "timestamp",
    ]:

        if column in row.index:

            market_date = normalize_value(
                row.get(column)
            )

            if market_date:
                break

    symbol = ""

    for column in [
        "symbol",
        "ticker",
        "stock",
    ]:

        if column in row.index:

            symbol = normalize_value(
                row.get(column)
            ).upper()

            if symbol:
                break

    model_version = ""

    for column in [
        "model_version",
        "model_name",
        "model",
    ]:

        if column in row.index:

            model_version = normalize_value(
                row.get(column)
            )

            if model_version:
                break

    raw_value = (
        f"{market_date}|"
        f"{symbol}|"
        f"{model_version}"
    )

    return hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()[:24]


def ensure_prediction_ids(
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure every ledger record has prediction_id.

    Existing IDs are preserved.
    Missing IDs are generated.
    """

    if ledger.empty:
        return ledger.copy()

    result = ledger.copy()

    if "prediction_id" not in result.columns:

        result["prediction_id"] = pd.NA

    missing_mask = (
        result["prediction_id"].isna()
        |
        result["prediction_id"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    missing_count = int(
        missing_mask.sum()
    )

    if missing_count == 0:
        return result

    logger.info(
        "Generating %s missing prediction_id value(s).",
        missing_count,
    )

    for index in result.index:

        current_id = result.at[
            index,
            "prediction_id",
        ]

        if (
            pd.notna(current_id)
            and str(current_id).strip()
        ):
            continue

        result.at[
            index,
            "prediction_id",
        ] = generate_prediction_id(
            result.loc[index]
        )

    return result


# ============================================================
# LOAD / SAVE LEDGER
# ============================================================

def load_prediction_ledger() -> pd.DataFrame:
    """Load prediction ledger."""

    path = get_ledger_path()

    if not path.exists():

        logger.warning(
            "Prediction ledger does not exist: %s",
            path,
        )

        return pd.DataFrame()

    try:

        ledger = pd.read_csv(path)

        ledger = ensure_prediction_ids(
            ledger
        )

        logger.info(
            "Loaded %s ledger record(s).",
            len(ledger),
        )

        return ledger

    except pd.errors.EmptyDataError:

        return pd.DataFrame()

    except Exception as error:

        raise RuntimeError(
            f"Could not load prediction ledger: {error}"
        ) from error


def save_prediction_ledger(
    ledger: pd.DataFrame,
) -> Path:
    """Save prediction ledger."""

    path = get_ledger_path()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ledger.to_csv(
        path,
        index=False,
    )

    logger.info(
        "Saved prediction ledger: %s",
        path,
    )

    return path


# ============================================================
# PENDING PREDICTIONS
# ============================================================

def get_pending_predictions(
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find predictions requiring evaluation.

    Supported statuses:

        PENDING
        WAITING
        UNEVALUATED
    """

    if ledger.empty:
        return ledger.copy()

    result = ledger.copy()

    if "evaluation_status" not in result.columns:

        result["evaluation_status"] = "PENDING"

    status = (
        result["evaluation_status"]
        .fillna("PENDING")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    pending_statuses = [
        "PENDING",
        "WAITING",
        "UNEVALUATED",
    ]

    return result.loc[
        status.isin(
            pending_statuses
        )
    ].copy()


# ============================================================
# ACTUAL OUTCOMES
# ============================================================

def resolve_actual_outcomes(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resolve actual outcomes.

    Expected return columns:

        prediction_id
        actual_return
        actual_direction
        actual_risk
        evaluation_status
    """

    if predictions.empty:
        return predictions.copy()

    try:

        from src.actuals import (
            resolve_actual_outcomes as resolver,
        )

        logger.info(
            "Resolving actual market outcomes."
        )

        outcomes = resolver(
            predictions.copy()
        )

        if not isinstance(
            outcomes,
            pd.DataFrame,
        ):

            raise TypeError(
                "Actual outcome resolver must return "
                "a pandas DataFrame."
            )

        outcomes = ensure_prediction_ids(
            outcomes
        )

        return outcomes

    except Exception as error:

        logger.exception(
            "Actual outcome resolution failed."
        )

        result = predictions.copy()

        result["evaluation_status"] = (
            "WAITING"
        )

        result["evaluation_error"] = (
            f"Outcome resolution failed: {error}"
        )

        return result


# ============================================================
# MERGE OUTCOMES BY PREDICTION ID
# ============================================================

def update_ledger_with_actuals(
    ledger: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Update prediction ledger using stable prediction_id.

    DataFrame indexes are NEVER used for merging.
    """

    if ledger.empty:
        return ledger.copy()

    updated = ensure_prediction_ids(
        ledger
    )

    if outcomes.empty:
        return updated

    outcomes = ensure_prediction_ids(
        outcomes
    )

    if "prediction_id" not in outcomes.columns:

        raise ValueError(
            "Actual outcomes do not contain prediction_id."
        )

    outcome_columns = [
        "actual_return",
        "actual_direction",
        "actual_risk",
        "evaluation_status",
        "evaluation_timestamp",
        "evaluation_error",
        "exit_price",
        "exit_timestamp",
    ]

    available_columns = [
        column
        for column in outcome_columns
        if column in outcomes.columns
    ]

    if not available_columns:

        logger.warning(
            "No outcome columns available for merge."
        )

        return updated

    outcome_map = (
        outcomes[
            ["prediction_id"]
            + available_columns
        ]
        .drop_duplicates(
            subset=["prediction_id"],
            keep="last",
        )
        .set_index(
            "prediction_id"
        )
    )

    for column in available_columns:

        if column not in updated.columns:

            updated[column] = pd.NA

        mapped_values = (
            updated["prediction_id"]
            .map(
                outcome_map[column]
            )
        )

        has_value = mapped_values.notna()

        updated.loc[
            has_value,
            column,
        ] = mapped_values.loc[
            has_value
        ]

    logger.info(
        "Merged outcomes using prediction_id | "
        "outcomes=%s",
        len(outcome_map),
    )

    return updated


# ============================================================
# EVALUATION METRICS
# ============================================================

def calculate_evaluation_metrics(
    ledger: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate evaluation metrics."""

    result: dict[str, Any] = {
        "model_name": "ALL",
        "sample_count": 0,
        "return_mae": None,
        "direction_accuracy": None,
        "brier_score": None,
        "risk_mae": None,
    }

    if ledger.empty:
        return result

    evaluated = ledger.copy()

    if "evaluation_status" in evaluated.columns:

        status = (
            evaluated["evaluation_status"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
        )

        evaluated = evaluated.loc[
            status.eq("EVALUATED")
        ].copy()

    if evaluated.empty:
        return result

    result["sample_count"] = int(
        len(evaluated)
    )

    # --------------------------------------------------------
    # RETURN MAE
    # --------------------------------------------------------

    predicted_return_column = find_column(
        evaluated,
        [
            "predicted_return",
            "expected_return",
            "return_prediction",
        ],
    )

    actual_return_column = find_column(
        evaluated,
        [
            "actual_return",
        ],
    )

    if (
        predicted_return_column
        and actual_return_column
    ):

        predicted = pd.to_numeric(
            evaluated[predicted_return_column],
            errors="coerce",
        )

        actual = pd.to_numeric(
            evaluated[actual_return_column],
            errors="coerce",
        )

        valid = (
            predicted.notna()
            & actual.notna()
        )

        if valid.any():

            result["return_mae"] = float(
                (
                    predicted[valid]
                    - actual[valid]
                )
                .abs()
                .mean()
            )

    # --------------------------------------------------------
    # DIRECTION ACCURACY
    # --------------------------------------------------------

    predicted_direction_column = find_column(
        evaluated,
        [
            "predicted_direction",
            "direction",
            "direction_prediction",
        ],
    )

    actual_direction_column = find_column(
        evaluated,
        [
            "actual_direction",
        ],
    )

    if (
        predicted_direction_column
        and actual_direction_column
    ):

        predicted = (
            evaluated[
                predicted_direction_column
            ]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        actual = (
            evaluated[
                actual_direction_column
            ]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        valid = (
            evaluated[
                predicted_direction_column
            ].notna()
            &
            evaluated[
                actual_direction_column
            ].notna()
        )

        if valid.any():

            result[
                "direction_accuracy"
            ] = float(
                (
                    predicted[valid]
                    == actual[valid]
                )
                .mean()
            )

    # --------------------------------------------------------
    # BRIER SCORE
    # --------------------------------------------------------

    confidence_column = find_column(
        evaluated,
        [
            "confidence",
            "confidence_score",
            "direction_probability",
        ],
    )

    if (
        confidence_column
        and predicted_direction_column
        and actual_direction_column
    ):

        confidence = pd.to_numeric(
            evaluated[confidence_column],
            errors="coerce",
        )

        predicted = (
            evaluated[
                predicted_direction_column
            ]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        actual = (
            evaluated[
                actual_direction_column
            ]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        valid = (
            confidence.notna()
            &
            evaluated[
                predicted_direction_column
            ].notna()
            &
            evaluated[
                actual_direction_column
            ].notna()
        )

        if valid.any():

            probabilities = (
                confidence[valid]
                .copy()
            )

            if probabilities.max() > 1:
                probabilities = (
                    probabilities / 100.0
                )

            probabilities = probabilities.clip(
                lower=0.0,
                upper=1.0,
            )

            correct = (
                predicted[valid]
                == actual[valid]
            ).astype(float)

            result["brier_score"] = float(
                (
                    probabilities
                    - correct
                )
                .pow(2)
                .mean()
            )

    # --------------------------------------------------------
    # RISK MAE
    # --------------------------------------------------------

    predicted_risk_column = find_column(
        evaluated,
        [
            "predicted_risk",
            "risk_score",
            "risk",
        ],
    )

    actual_risk_column = find_column(
        evaluated,
        [
            "actual_risk",
        ],
    )

    if (
        predicted_risk_column
        and actual_risk_column
    ):

        predicted = pd.to_numeric(
            evaluated[predicted_risk_column],
            errors="coerce",
        )

        actual = pd.to_numeric(
            evaluated[actual_risk_column],
            errors="coerce",
        )

        valid = (
            predicted.notna()
            & actual.notna()
        )

        if valid.any():

            result["risk_mae"] = float(
                (
                    predicted[valid]
                    - actual[valid]
                )
                .abs()
                .mean()
            )

    return result


# ============================================================
# SAVE EVALUATION REPORT
# ============================================================

def save_evaluation_report(
    metrics: dict[str, Any],
) -> Path:
    """Save latest evaluation report."""

    reports_dir = get_reports_dir()

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        reports_dir
        / "latest_evaluation.csv"
    )

    report = pd.DataFrame(
        [metrics]
    )

    report["generated_at"] = (
        utc_now_iso()
    )

    report.to_csv(
        path,
        index=False,
    )

    logger.info(
        "Saved evaluation report: %s",
        path,
    )

    return path


# ============================================================
# MONITORING
# ============================================================

def run_production_monitoring() -> dict[str, Any]:
    """Run production monitoring."""

    try:

        from src.monitoring import (
            run_monitoring,
        )

        result = run_monitoring()

        if isinstance(
            result,
            dict,
        ):
            return result

        return {
            "status": "UNKNOWN",
        }

    except Exception as error:

        logger.exception(
            "Production monitoring failed."
        )

        return {
            "status": "ERROR",
            "health_status": "CRITICAL",
            "health_score": 0,
            "error": str(error),
        }


# ============================================================
# MAIN JOB
# ============================================================

def run_evaluation_job() -> dict[str, Any]:
    """Run complete evaluation pipeline."""

    result: dict[str, Any] = {
        "started_at": utc_now_iso(),
        "finished_at": None,
        "status": "STARTED",
        "ledger_records": 0,
        "pending_predictions": 0,
        "evaluated_predictions": 0,
        "metrics": {},
        "monitoring": {},
        "error": None,
    }

    try:

        logger.info(
            "Starting evaluation job."
        )

        # Step 1
        ledger = load_prediction_ledger()

        result["ledger_records"] = len(
            ledger
        )

        if ledger.empty:

            result["status"] = "NO_LEDGER_DATA"

            return result

        # Save generated IDs immediately.
        save_prediction_ledger(ledger)

        # Step 2
        pending = get_pending_predictions(
            ledger
        )

        result[
            "pending_predictions"
        ] = len(pending)

        # Step 3 and 4
        if not pending.empty:

            outcomes = (
                resolve_actual_outcomes(
                    pending
                )
            )

            ledger = (
                update_ledger_with_actuals(
                    ledger,
                    outcomes,
                )
            )

            save_prediction_ledger(
                ledger
            )

        # Step 5
        metrics = (
            calculate_evaluation_metrics(
                ledger
            )
        )

        result["metrics"] = metrics

        result[
            "evaluated_predictions"
        ] = metrics.get(
            "sample_count",
            0,
        )

        # Step 6
        report_path = (
            save_evaluation_report(
                metrics
            )
        )

        result["report_path"] = str(
            report_path
        )

        # Step 7
        monitoring = (
            run_production_monitoring()
        )

        result["monitoring"] = (
            monitoring
        )

        result["status"] = "SUCCESS"

        return result

    except Exception as error:

        logger.exception(
            "Evaluation job failed."
        )

        result["status"] = "FAILED"

        result["error"] = str(error)

        result["traceback"] = (
            traceback.format_exc()
        )

        try:

            from src.circuit_breaker import (
                register_failure,
            )

            register_failure(
                reason=(
                    "Evaluation job failed: "
                    f"{error}"
                ),
                health_score=0,
                health_status="CRITICAL",
            )

        except Exception as breaker_error:

            logger.error(
                "Circuit breaker update failed: %s",
                breaker_error,
            )

        return result

    finally:

        result["finished_at"] = (
            utc_now_iso()
        )

        logger.info(
            "Evaluation job finished | status=%s",
            result.get("status"),
        )


# ============================================================
# CLI
# ============================================================

def main() -> int:

    result = run_evaluation_job()

    print()

    print("=" * 70)

    print("EVALUATION JOB RESULT")

    print("=" * 70)

    for key, value in result.items():

        if key == "traceback":
            continue

        print(f"{key}: {value}")

    return (
        0
        if result.get("status")
        in {
            "SUCCESS",
            "NO_LEDGER_DATA",
        }
        else 1
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

#!/usr/bin/env python3

"""
Evaluation Job.

Pipeline
--------
1. Load the prediction ledger.
2. Find predictions ready for evaluation.
3. Fetch/resolve actual market outcomes.
4. Update the prediction ledger.
5. Generate evaluation metrics.
6. Save the latest evaluation report.
7. Run production monitoring.
8. Update the circuit breaker through monitoring.

This job does NOT send Telegram messages.

Expected flow:

Prediction Ledger
       │
       ▼
Find PENDING / WAITING predictions
       │
       ▼
Resolve actual outcomes
       │
       ▼
Update Ledger
       │
       ▼
Model Evaluation
       │
       ▼
Evaluation Report
       │
       ▼
Production Monitoring
       │
       ▼
Circuit Breaker
"""

from __future__ import annotations

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
    """Return the current UTC datetime."""

    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return the current UTC datetime as ISO text."""

    return utc_now().isoformat()


# ============================================================
# CONFIG HELPERS
# ============================================================

def object_to_dict(value: Any) -> dict[str, Any]:
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

def resolve_path(value: str | Path) -> Path:
    """Resolve a project-relative path."""

    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_ledger_path() -> Path:
    """Get prediction ledger path."""

    cfg = load_config()

    candidates: list[Any] = []

    for section_name in ["data", "paths"]:

        section = getattr(
            cfg,
            section_name,
            None,
        )

        values = object_to_dict(section)

        for key in [
            "prediction_ledger",
            "ledger",
        ]:
            if values.get(key):
                candidates.append(values[key])

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

    for section_name in ["data", "paths"]:

        section = getattr(
            cfg,
            section_name,
            None,
        )

        values = object_to_dict(section)

        for key in [
            "reports",
            "reports_dir",
        ]:
            if values.get(key):
                return resolve_path(
                    values[key]
                )

    return PROJECT_ROOT / "data" / "reports"


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


def to_numeric_series(
    frame: pd.DataFrame,
    column: str | None,
) -> pd.Series:
    """Safely convert a DataFrame column to numeric."""

    if column is None:
        return pd.Series(
            index=frame.index,
            dtype="float64",
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


# ============================================================
# LOAD LEDGER
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

        frame = pd.read_csv(path)

        logger.info(
            "Loaded %s ledger record(s).",
            len(frame),
        )

        return frame

    except pd.errors.EmptyDataError:

        return pd.DataFrame()

    except Exception as error:

        raise RuntimeError(
            f"Could not load ledger: {error}"
        ) from error


def save_prediction_ledger(
    frame: pd.DataFrame,
) -> Path:
    """Save the updated prediction ledger."""

    path = get_ledger_path()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        path,
        index=False,
    )

    logger.info(
        "Saved updated prediction ledger: %s",
        path,
    )

    return path


# ============================================================
# FIND PREDICTIONS READY FOR EVALUATION
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

    if "evaluation_status" not in ledger.columns:

        result = ledger.copy()

        result["evaluation_status"] = (
            "PENDING"
        )

        return result

    status = (
        ledger["evaluation_status"]
        .fillna("PENDING")
        .astype(str)
        .str.upper()
    )

    mask = status.isin(
        [
            "PENDING",
            "WAITING",
            "UNEVALUATED",
        ]
    )

    return ledger.loc[
        mask
    ].copy()


# ============================================================
# ACTUAL OUTCOME RESOLUTION
# ============================================================

def resolve_actual_outcomes(
    pending: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resolve actual outcomes for pending predictions.

    This function attempts to use an existing project
    outcome/evaluation module.

    Supported module patterns:

        src.evaluation.resolve_actual_outcomes(df)
        src.evaluation.evaluate_predictions(df)
        src.actuals.resolve_actual_outcomes(df)
        src.actuals.fetch_actuals(df)

    The returned DataFrame should contain one or more of:

        actual_return
        actual_direction
        actual_risk
        evaluation_status
    """

    if pending.empty:
        return pending.copy()

    attempts: list[str] = []

    # --------------------------------------------------------
    # src.evaluation
    # --------------------------------------------------------

    try:

        from src import evaluation

        for function_name in [
            "resolve_actual_outcomes",
            "evaluate_predictions",
        ]:

            function = getattr(
                evaluation,
                function_name,
                None,
            )

            if callable(function):

                logger.info(
                    "Using src.evaluation.%s()",
                    function_name,
                )

                result = function(
                    pending.copy()
                )

                if isinstance(
                    result,
                    pd.DataFrame,
                ):
                    return result

                if isinstance(result, dict):
                    return pd.DataFrame(
                        [result]
                    )

    except Exception as error:

        attempts.append(
            f"src.evaluation: {error}"
        )

    # --------------------------------------------------------
    # src.actuals
    # --------------------------------------------------------

    try:

        from src import actuals

        for function_name in [
            "resolve_actual_outcomes",
            "fetch_actuals",
        ]:

            function = getattr(
                actuals,
                function_name,
                None,
            )

            if callable(function):

                logger.info(
                    "Using src.actuals.%s()",
                    function_name,
                )

                result = function(
                    pending.copy()
                )

                if isinstance(
                    result,
                    pd.DataFrame,
                ):
                    return result

                if isinstance(result, dict):
                    return pd.DataFrame(
                        [result]
                    )

    except Exception as error:

        attempts.append(
            f"src.actuals: {error}"
        )

    logger.warning(
        "No actual outcome resolver was available. "
        "Predictions remain WAITING. Attempts: %s",
        " | ".join(attempts),
    )

    result = pending.copy()

    result["evaluation_status"] = (
        "WAITING"
    )

    return result


# ============================================================
# UPDATE LEDGER
# ============================================================

def update_ledger_with_actuals(
    ledger: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge actual outcomes back into the prediction ledger.

    Uses index alignment when possible.
    """

    if ledger.empty or outcomes.empty:
        return ledger.copy()

    updated = ledger.copy()

    outcome_columns = [
        "actual_return",
        "actual_direction",
        "actual_risk",
        "evaluation_status",
    ]

    for column in outcome_columns:

        if column not in outcomes.columns:
            continue

        if column not in updated.columns:

            updated[column] = pd.NA

        for index in outcomes.index:

            if index in updated.index:

                value = outcomes.at[
                    index,
                    column,
                ]

                if pd.notna(value):

                    updated.at[
                        index,
                        column,
                    ] = value

    if "evaluation_status" not in updated.columns:

        updated["evaluation_status"] = (
            "PENDING"
        )

    actual_return_column = (
        "actual_return"
        if "actual_return" in updated.columns
        else None
    )

    if actual_return_column:

        has_actual = (
            pd.to_numeric(
                updated[actual_return_column],
                errors="coerce",
            )
            .notna()
        )

        updated.loc[
            has_actual,
            "evaluation_status",
        ] = "EVALUATED"

    return updated


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_evaluation_metrics(
    ledger: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate production evaluation metrics.

    Metrics:

        sample_count
        return_mae
        direction_accuracy
        brier_score
        risk_mae
    """

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

    # --------------------------------------------------------
    # FILTER EVALUATED RECORDS
    # --------------------------------------------------------

    evaluated = ledger.copy()

    if "evaluation_status" in evaluated.columns:

        status = (
            evaluated["evaluation_status"]
            .fillna("")
            .astype(str)
            .str.upper()
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

        predicted = to_numeric_series(
            evaluated,
            predicted_return_column,
        )

        actual = to_numeric_series(
            evaluated,
            actual_return_column,
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

        predicted_direction = (
            evaluated[
                predicted_direction_column
            ]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        actual_direction = (
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
            & evaluated[
                actual_direction_column
            ].notna()
        )

        if valid.any():

            result[
                "direction_accuracy"
            ] = float(
                (
                    predicted_direction[valid]
                    == actual_direction[valid]
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

        confidence = to_numeric_series(
            evaluated,
            confidence_column,
        )

        predicted_direction = (
            evaluated[
                predicted_direction_column
            ]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        actual_direction = (
            evaluated[
                actual_direction_column
            ]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        valid = (
            confidence.notna()
            & evaluated[
                predicted_direction_column
            ].notna()
            & evaluated[
                actual_direction_column
            ].notna()
        )

        if valid.any():

            probabilities = confidence[
                valid
            ].copy()

            # Convert percentage confidence
            # such as 75 -> 0.75.
            if probabilities.max() > 1:
                probabilities = (
                    probabilities / 100.0
                )

            probabilities = probabilities.clip(
                lower=0.0,
                upper=1.0,
            )

            actual_correct = (
                predicted_direction[valid]
                == actual_direction[valid]
            ).astype(float)

            result["brier_score"] = float(
                (
                    probabilities
                    - actual_correct
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

        predicted_risk = to_numeric_series(
            evaluated,
            predicted_risk_column,
        )

        actual_risk = to_numeric_series(
            evaluated,
            actual_risk_column,
        )

        valid = (
            predicted_risk.notna()
            & actual_risk.notna()
        )

        if valid.any():

            result["risk_mae"] = float(
                (
                    predicted_risk[valid]
                    - actual_risk[valid]
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
    """
    Save latest evaluation report.

    Required by src.monitoring.
    """

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
# RUN MONITORING
# ============================================================

def run_production_monitoring() -> dict[str, Any]:
    """
    Run monitoring after evaluation.

    Monitoring automatically checks:

        - evaluation quality
        - stale data
        - model registry
        - ledger health

    It then updates the circuit breaker.
    """

    try:

        from src.monitoring import (
            run_monitoring,
        )

        logger.info(
            "Running production monitoring."
        )

        result = run_monitoring()

        if isinstance(result, dict):
            return result

        return {
            "status": "UNKNOWN",
            "health_status": "UNKNOWN",
            "health_score": None,
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
    """
    Run the complete evaluation pipeline.
    """

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

    logger.info(
        "=" * 70
    )

    logger.info(
        "STARTING EVALUATION JOB"
    )

    logger.info(
        "=" * 70
    )

    try:

        # ----------------------------------------------------
        # STEP 1: LOAD LEDGER
        # ----------------------------------------------------

        logger.info(
            "Step 1: Loading prediction ledger."
        )

        ledger = (
            load_prediction_ledger()
        )

        result["ledger_records"] = len(
            ledger
        )

        if ledger.empty:

            result["status"] = "NO_LEDGER_DATA"

            logger.warning(
                "No ledger records available."
            )

            return result

        # ----------------------------------------------------
        # STEP 2: FIND PENDING PREDICTIONS
        # ----------------------------------------------------

        logger.info(
            "Step 2: Finding predictions "
            "ready for evaluation."
        )

        pending = (
            get_pending_predictions(
                ledger
            )
        )

        result[
            "pending_predictions"
        ] = len(pending)

        if pending.empty:

            logger.info(
                "No pending predictions found."
            )

        else:

            # ------------------------------------------------
            # STEP 3: RESOLVE ACTUAL OUTCOMES
            # ------------------------------------------------

            logger.info(
                "Step 3: Resolving actual outcomes."
            )

            outcomes = (
                resolve_actual_outcomes(
                    pending
                )
            )

            # ------------------------------------------------
            # STEP 4: UPDATE LEDGER
            # ------------------------------------------------

            logger.info(
                "Step 4: Updating prediction ledger."
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

        # ----------------------------------------------------
        # STEP 5: CALCULATE METRICS
        # ----------------------------------------------------

        logger.info(
            "Step 5: Calculating evaluation metrics."
        )

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

        # ----------------------------------------------------
        # STEP 6: SAVE EVALUATION REPORT
        # ----------------------------------------------------

        logger.info(
            "Step 6: Saving evaluation report."
        )

        report_path = (
            save_evaluation_report(
                metrics
            )
        )

        result["report_path"] = str(
            report_path
        )

        # ----------------------------------------------------
        # STEP 7: RUN MONITORING
        # ----------------------------------------------------

        logger.info(
            "Step 7: Running production monitoring."
        )

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

        result["error"] = str(
            error
        )

        result["traceback"] = (
            traceback.format_exc()
        )

        # Fail-safe circuit breaker registration.
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
                "Could not register evaluation "
                "failure with circuit breaker: %s",
                breaker_error,
            )

        return result

    finally:

        result["finished_at"] = (
            utc_now_iso()
        )

        logger.info(
            "=" * 70
        )

        logger.info(
            "EVALUATION JOB FINISHED | STATUS=%s",
            result.get("status"),
        )

        logger.info(
            "=" * 70
        )


# ============================================================
# CLI
# ============================================================

def main() -> int:
    """CLI entry point."""

    result = run_evaluation_job()

    print()

    print("=" * 70)

    print("EVALUATION JOB RESULT")

    print("=" * 70)

    print(
        "Status: "
        f"{result.get('status')}"
    )

    print(
        "Ledger records: "
        f"{result.get('ledger_records')}"
    )

    print(
        "Pending predictions: "
        f"{result.get('pending_predictions')}"
    )

    print(
        "Evaluated predictions: "
        f"{result.get('evaluated_predictions')}"
    )

    metrics = result.get(
        "metrics",
        {},
    )

    if metrics:

        print()

        print("METRICS")

        for key in [
            "sample_count",
            "return_mae",
            "direction_accuracy",
            "brier_score",
            "risk_mae",
        ]:

            print(
                f"{key}: "
                f"{metrics.get(key)}"
            )

    monitoring = result.get(
        "monitoring",
        {},
    )

    if monitoring:

        print()

        print("SYSTEM HEALTH")

        print(
            "Status: "
            f"{monitoring.get('health_status')}"
        )

        print(
            "Score: "
            f"{monitoring.get('health_score')}"
        )

        breaker = monitoring.get(
            "circuit_breaker",
            {},
        )

        if breaker:

            print(
                "Circuit Breaker: "
                f"{breaker.get('state')}"
            )

            print(
                "Predictions Allowed: "
                f"{breaker.get('predictions_allowed')}"
            )

    if result.get("error"):

        print()

        print(
            "ERROR: "
            f"{result.get('error')}"
        )

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

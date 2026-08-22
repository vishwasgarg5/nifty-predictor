#!/usr/bin/env python3

"""
Prediction Evaluation Job.

This job evaluates past predictions stored in the prediction
ledger and updates them with actual market outcomes.

Pipeline
--------
Prediction Ledger
       │
       ▼
Load predictions
       │
       ▼
Find PENDING / WAITING predictions
       │
       ▼
Resolve actual market outcomes
       │
       ▼
Update prediction ledger
       │
       ▼
Run model evaluation
       │
       ▼
Run drift detection
       │
       ▼
Update Champion / Challenger
       │
       ▼
Finish

Important
---------
Only predictions whose evaluation horizon has been reached will
be marked as EVALUATED.

Predictions that cannot yet be evaluated remain WAITING.

The job rewrites the ledger safely using a temporary file.
"""

from __future__ import annotations

import logging
import sys
import tempfile
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

def object_to_dict(
    value: Any,
) -> dict[str, Any]:
    """
    Convert common configuration objects into dictionaries.
    """

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
    """
    Load the project configuration.
    """

    try:

        from src.config import cfg

        return cfg

    except Exception as error:

        logger.error(
            "Could not import src.config.cfg: %s",
            error,
        )

        raise


# ============================================================
# LEDGER PATH
# ============================================================

def get_ledger_path() -> Path:
    """
    Get prediction ledger location.

    Supported configuration locations:

        data:
            prediction_ledger: ...

        paths:
            prediction_ledger: ...
    """

    cfg = load_config()

    candidates: list[Any] = []

    data_section = getattr(
        cfg,
        "data",
        None,
    )

    if data_section is not None:

        values = object_to_dict(
            data_section
        )

        for key in [
            "prediction_ledger",
            "ledger",
        ]:

            if key in values:

                candidates.append(
                    values[key]
                )

    paths_section = getattr(
        cfg,
        "paths",
        None,
    )

    if paths_section is not None:

        values = object_to_dict(
            paths_section
        )

        for key in [
            "prediction_ledger",
            "ledger",
        ]:

            if key in values:

                candidates.append(
                    values[key]
                )

    for candidate in candidates:

        if candidate:

            path = Path(
                str(candidate)
            )

            if not path.is_absolute():

                path = PROJECT_ROOT / path

            return path

    return (
        PROJECT_ROOT
        / "data"
        / "ledger"
        / "predictions.csv"
    )


# ============================================================
# LEDGER LOADING
# ============================================================

def load_prediction_ledger() -> pd.DataFrame:
    """
    Load the prediction ledger.

    Returns an empty DataFrame when the ledger does not yet exist.
    """

    path = get_ledger_path()

    if not path.exists():

        logger.warning(
            "Prediction ledger does not exist: %s",
            path,
        )

        return pd.DataFrame()

    try:

        frame = pd.read_csv(
            path
        )

    except pd.errors.EmptyDataError:

        logger.warning(
            "Prediction ledger is empty: %s",
            path,
        )

        return pd.DataFrame()

    logger.info(
        "Loaded %s prediction record(s) from %s",
        len(frame),
        path,
    )

    return frame


# ============================================================
# LEDGER STATUS HELPERS
# ============================================================

def ensure_evaluation_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure all required evaluation columns exist.
    """

    result = frame.copy()

    defaults = {
        "actual_return": pd.NA,
        "actual_direction": pd.NA,
        "actual_risk": pd.NA,
        "evaluation_status": "PENDING",
        "evaluation_timestamp": pd.NA,
        "evaluation_error": pd.NA,
        "exit_price": pd.NA,
        "exit_timestamp": pd.NA,
    }

    for column, default_value in defaults.items():

        if column not in result.columns:

            result[column] = default_value

    result["evaluation_status"] = (
        result["evaluation_status"]
        .fillna("PENDING")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return result


def get_pending_predictions(
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return predictions that still require evaluation.

    Eligible statuses:

        PENDING
        WAITING
        RETRY

    Existing EVALUATED predictions are preserved.
    """

    if ledger.empty:

        return ledger.copy()

    frame = ensure_evaluation_columns(
        ledger
    )

    eligible_statuses = {
        "PENDING",
        "WAITING",
        "RETRY",
        "",
        "NAN",
        "NONE",
    }

    mask = (
        frame["evaluation_status"]
        .astype(str)
        .str.upper()
        .isin(eligible_statuses)
    )

    pending = frame.loc[
        mask
    ].copy()

    logger.info(
        "Found %s prediction(s) requiring evaluation.",
        len(pending),
    )

    return pending


# ============================================================
# ROW IDENTIFICATION
# ============================================================

def ensure_record_id(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure the ledger has a stable internal record ID.

    Existing record IDs are preserved.

    This is important because the evaluation result must be
    merged back into the correct ledger rows.
    """

    result = frame.copy()

    if "_record_id" not in result.columns:

        result["_record_id"] = pd.NA

    missing = result["_record_id"].isna()

    if missing.any():

        result.loc[
            missing,
            "_record_id",
        ] = [
            f"record_{index}"
            for index in result.index[
                missing
            ]
        ]

    result["_record_id"] = (
        result["_record_id"]
        .astype(str)
    )

    return result


# ============================================================
# MERGE EVALUATION RESULTS
# ============================================================

def merge_evaluation_results(
    ledger: pd.DataFrame,
    evaluated: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge evaluated prediction records back into the full ledger.
    """

    if ledger.empty:

        return ledger.copy()

    result = ledger.copy()

    if evaluated is None or evaluated.empty:

        return result

    if "_record_id" not in result.columns:

        raise RuntimeError(
            "Ledger is missing _record_id."
        )

    if "_record_id" not in evaluated.columns:

        raise RuntimeError(
            "Evaluation result is missing _record_id."
        )

    evaluated = evaluated.copy()

    evaluated = evaluated.set_index(
        "_record_id",
        drop=False,
    )

    evaluation_columns = [
        "actual_return",
        "actual_direction",
        "actual_risk",
        "evaluation_status",
        "evaluation_timestamp",
        "evaluation_error",
        "entry_price",
        "entry_timestamp",
        "exit_price",
        "exit_timestamp",
    ]

    for column in evaluation_columns:

        if column not in evaluated.columns:

            continue

        if column not in result.columns:

            result[column] = pd.NA

        mapping = (
            evaluated[column]
            .to_dict()
        )

        result[column] = (
            result["_record_id"]
            .map(mapping)
            .where(
                result["_record_id"].isin(
                    mapping.keys()
                ),
                result[column],
            )
        )

    return result


# ============================================================
# SAFE LEDGER SAVE
# ============================================================

def save_prediction_ledger(
    ledger: pd.DataFrame,
) -> Path:
    """
    Save the prediction ledger safely.

    A temporary file is written first and then atomically
    replaces the existing ledger.
    """

    path = get_ledger_path()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".csv",
            prefix="predictions_",
            dir=path.parent,
            delete=False,
        ) as temporary_file:

            temporary_path = Path(
                temporary_file.name
            )

            ledger.to_csv(
                temporary_file,
                index=False,
            )

        temporary_path.replace(
            path
        )

        logger.info(
            "Prediction ledger saved successfully: %s",
            path,
        )

        return path

    except Exception:

        if (
            temporary_path is not None
            and temporary_path.exists()
        ):

            try:

                temporary_path.unlink()

            except Exception:
                pass

        raise


# ============================================================
# ACTUAL OUTCOME RESOLUTION
# ============================================================

def run_actual_outcome_resolution(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resolve actual market outcomes.
    """

    if predictions.empty:

        return predictions.copy()

    try:

        from src.actuals import (
            resolve_actual_outcomes,
        )

    except ImportError as error:

        raise RuntimeError(
            "Could not import "
            "src.actuals.resolve_actual_outcomes."
        ) from error

    logger.info(
        "Resolving actual outcomes for %s prediction(s).",
        len(predictions),
    )

    result = resolve_actual_outcomes(
        predictions
    )

    if not isinstance(
        result,
        pd.DataFrame,
    ):

        raise RuntimeError(
            "Actual outcome resolver did not "
            "return a DataFrame."
        )

    return result


# ============================================================
# MODEL EVALUATION
# ============================================================

def run_model_evaluation(
    ledger: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run model evaluation when the project evaluation module
    is available.

    The job supports several common project interfaces:

        src.model_evaluation.run_evaluation()
        src.model_evaluation.evaluate()
        src.evaluation.run_evaluation()
        src.evaluation.evaluate()
    """

    attempts: list[str] = []

    modules = [
        (
            "src.model_evaluation",
            [
                "run_evaluation",
                "evaluate",
            ],
        ),
        (
            "src.evaluation",
            [
                "run_evaluation",
                "evaluate",
            ],
        ),
    ]

    for module_name, function_names in modules:

        try:

            module = __import__(
                module_name,
                fromlist=["*"],
            )

            for function_name in function_names:

                function = getattr(
                    module,
                    function_name,
                    None,
                )

                if not callable(function):

                    continue

                logger.info(
                    "Running %s.%s()",
                    module_name,
                    function_name,
                )

                try:

                    result = function(
                        ledger
                    )

                except TypeError:

                    result = function()

                if isinstance(
                    result,
                    dict,
                ):

                    return result

                return {
                    "status": "COMPLETED",
                    "result": result,
                }

        except ImportError:

            attempts.append(
                f"{module_name}: unavailable"
            )

        except Exception as error:

            logger.exception(
                "Model evaluation failed in %s.",
                module_name,
            )

            return {
                "status": "ERROR",
                "error": str(error),
            }

    logger.info(
        "No compatible model evaluation module found."
    )

    return {
        "status": "SKIPPED",
        "reason": (
            "No model evaluation module available."
        ),
        "attempts": attempts,
    }


# ============================================================
# DRIFT DETECTION
# ============================================================

def run_drift_detection(
    ledger: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run drift detection when available.

    Supported interfaces:

        src.drift_detection.run_drift_detection()
        src.drift_detection.detect_drift()
        src.drift.run_drift_detection()
        src.drift.detect_drift()
    """

    attempts: list[str] = []

    modules = [
        (
            "src.drift_detection",
            [
                "run_drift_detection",
                "detect_drift",
            ],
        ),
        (
            "src.drift",
            [
                "run_drift_detection",
                "detect_drift",
            ],
        ),
    ]

    for module_name, function_names in modules:

        try:

            module = __import__(
                module_name,
                fromlist=["*"],
            )

            for function_name in function_names:

                function = getattr(
                    module,
                    function_name,
                    None,
                )

                if not callable(function):

                    continue

                logger.info(
                    "Running %s.%s()",
                    module_name,
                    function_name,
                )

                try:

                    result = function(
                        ledger
                    )

                except TypeError:

                    result = function()

                if isinstance(
                    result,
                    dict,
                ):

                    return result

                return {
                    "status": "COMPLETED",
                    "result": result,
                }

        except ImportError:

            attempts.append(
                f"{module_name}: unavailable"
            )

        except Exception as error:

            logger.exception(
                "Drift detection failed in %s.",
                module_name,
            )

            return {
                "status": "ERROR",
                "error": str(error),
            }

    logger.info(
        "No compatible drift detection module found."
    )

    return {
        "status": "SKIPPED",
        "reason": (
            "No drift detection module available."
        ),
        "attempts": attempts,
    }


# ============================================================
# CHAMPION / CHALLENGER
# ============================================================

def run_champion_challenger(
    ledger: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run Champion / Challenger model management.

    Supported interfaces:

        src.champion_challenger.run()
        src.champion_challenger.evaluate()
        src.model_manager.run()
        src.model_manager.evaluate()
    """

    attempts: list[str] = []

    modules = [
        (
            "src.champion_challenger",
            [
                "run",
                "evaluate",
                "update",
            ],
        ),
        (
            "src.model_manager",
            [
                "run",
                "evaluate",
                "update",
            ],
        ),
    ]

    for module_name, function_names in modules:

        try:

            module = __import__(
                module_name,
                fromlist=["*"],
            )

            for function_name in function_names:

                function = getattr(
                    module,
                    function_name,
                    None,
                )

                if not callable(function):

                    continue

                logger.info(
                    "Running %s.%s()",
                    module_name,
                    function_name,
                )

                try:

                    result = function(
                        ledger
                    )

                except TypeError:

                    result = function()

                if isinstance(
                    result,
                    dict,
                ):

                    return result

                return {
                    "status": "COMPLETED",
                    "result": result,
                }

        except ImportError:

            attempts.append(
                f"{module_name}: unavailable"
            )

        except Exception as error:

            logger.exception(
                "Champion / Challenger processing failed."
            )

            return {
                "status": "ERROR",
                "error": str(error),
            }

    logger.info(
        "No compatible Champion / Challenger module found."
    )

    return {
        "status": "SKIPPED",
        "reason": (
            "No Champion / Challenger module available."
        ),
        "attempts": attempts,
    }


# ============================================================
# JOB
# ============================================================

def run_evaluation_job() -> dict[str, Any]:
    """
    Run the complete prediction evaluation pipeline.
    """

    result: dict[str, Any] = {
        "started_at": utc_now_iso(),
        "finished_at": None,
        "status": "STARTED",
        "ledger_path": None,
        "ledger_records": 0,
        "pending_predictions": 0,
        "evaluated_predictions": 0,
        "waiting_predictions": 0,
        "invalid_predictions": 0,
        "ledger_updated": False,
        "model_evaluation": {},
        "drift_detection": {},
        "champion_challenger": {},
        "error": None,
    }

    logger.info(
        "=" * 70
    )

    logger.info(
        "STARTING PREDICTION EVALUATION JOB"
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

        ledger = load_prediction_ledger()

        if ledger.empty:

            result["status"] = "NO_LEDGER_RECORDS"

            return result

        ledger = ensure_record_id(
            ledger
        )

        ledger = ensure_evaluation_columns(
            ledger
        )

        result["ledger_records"] = len(
            ledger
        )

        result["ledger_path"] = str(
            get_ledger_path()
        )

        # ----------------------------------------------------
        # STEP 2: FIND PENDING RECORDS
        # ----------------------------------------------------

        logger.info(
            "Step 2: Finding pending predictions."
        )

        pending = get_pending_predictions(
            ledger
        )

        result[
            "pending_predictions"
        ] = len(
            pending
        )

        if pending.empty:

            result["status"] = (
                "NO_PENDING_PREDICTIONS"
            )

            logger.info(
                "No predictions require evaluation."
            )

            return result

        # ----------------------------------------------------
        # STEP 3: RESOLVE ACTUAL OUTCOMES
        # ----------------------------------------------------

        logger.info(
            "Step 3: Resolving actual outcomes."
        )

        evaluated = (
            run_actual_outcome_resolution(
                pending
            )
        )

        # ----------------------------------------------------
        # STEP 4: COUNT RESULTS
        # ----------------------------------------------------

        if "evaluation_status" in evaluated.columns:

            statuses = (
                evaluated["evaluation_status"]
                .fillna("WAITING")
                .astype(str)
                .str.upper()
            )

            result[
                "evaluated_predictions"
            ] = int(
                (statuses == "EVALUATED").sum()
            )

            result[
                "waiting_predictions"
            ] = int(
                (statuses == "WAITING").sum()
            )

            result[
                "invalid_predictions"
            ] = int(
                (statuses == "INVALID").sum()
            )

        # ----------------------------------------------------
        # STEP 5: MERGE RESULTS
        # ----------------------------------------------------

        logger.info(
            "Step 4: Updating ledger records."
        )

        updated_ledger = (
            merge_evaluation_results(
                ledger,
                evaluated,
            )
        )

        # ----------------------------------------------------
        # STEP 6: SAVE LEDGER
        # ----------------------------------------------------

        logger.info(
            "Step 5: Saving prediction ledger."
        )

        ledger_path = (
            save_prediction_ledger(
                updated_ledger
            )
        )

        result["ledger_updated"] = True

        result["ledger_path"] = str(
            ledger_path
        )

        # ----------------------------------------------------
        # STEP 7: MODEL EVALUATION
        # ----------------------------------------------------

        logger.info(
            "Step 6: Running model evaluation."
        )

        result[
            "model_evaluation"
        ] = run_model_evaluation(
            updated_ledger
        )

        # ----------------------------------------------------
        # STEP 8: DRIFT DETECTION
        # ----------------------------------------------------

        logger.info(
            "Step 7: Running drift detection."
        )

        result[
            "drift_detection"
        ] = run_drift_detection(
            updated_ledger
        )

        # ----------------------------------------------------
        # STEP 9: CHAMPION / CHALLENGER
        # ----------------------------------------------------

        logger.info(
            "Step 8: Running Champion / Challenger."
        )

        result[
            "champion_challenger"
        ] = run_champion_challenger(
            updated_ledger
        )

        # ----------------------------------------------------
        # FINAL STATUS
        # ----------------------------------------------------

        if (
            result[
                "evaluated_predictions"
            ] > 0
        ):

            result["status"] = "SUCCESS"

        else:

            result["status"] = (
                "COMPLETED_NO_NEW_EVALUATIONS"
            )

        return result

    except Exception as error:

        logger.exception(
            "Prediction evaluation job failed."
        )

        result["status"] = "FAILED"

        result["error"] = str(
            error
        )

        result["traceback"] = (
            traceback.format_exc()
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
    """
    CLI entry point.
    """

    result = run_evaluation_job()

    print()

    print("=" * 70)

    print("PREDICTION EVALUATION JOB RESULT")

    print("=" * 70)

    print(
        f"Status: {result.get('status')}"
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

    print(
        "Waiting predictions: "
        f"{result.get('waiting_predictions')}"
    )

    print(
        "Invalid predictions: "
        f"{result.get('invalid_predictions')}"
    )

    print(
        "Ledger updated: "
        f"{result.get('ledger_updated')}"
    )

    if result.get("ledger_path"):

        print(
            "Ledger path: "
            f"{result.get('ledger_path')}"
        )

    if result.get("error"):

        print()

        print(
            "Error: "
            f"{result.get('error')}"
        )

    print()

    print("Model Evaluation:")

    print(
        result.get(
            "model_evaluation",
            {},
        )
    )

    print()

    print("Drift Detection:")

    print(
        result.get(
            "drift_detection",
            {},
        )
    )

    print()

    print("Champion / Challenger:")

    print(
        result.get(
            "champion_challenger",
            {},
        )
    )

    return (
        0
        if result.get("status")
        in {
            "SUCCESS",
            "NO_LEDGER_RECORDS",
            "NO_PENDING_PREDICTIONS",
            "COMPLETED_NO_NEW_EVALUATIONS",
        }
        else 1
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

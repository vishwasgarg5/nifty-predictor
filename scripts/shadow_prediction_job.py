#!/usr/bin/env python3

"""
Shadow Prediction Job.

Runs the current Champion and Challenger models against the
same market data / feature set and stores their predictions
for later evaluation.

Pipeline
--------
1. Load Champion and Challenger from model registry.
2. Build one shared prediction dataset.
3. Run Champion prediction.
4. Run Challenger prediction.
5. Combine both prediction outputs.
6. Save results to the shadow prediction ledger.

Important
---------
The Challenger does NOT send Telegram signals.

Both models receive the same input data so that future
performance comparisons are fair.

Expected output:

    prediction_id
    model_role
    model_name
    model_version
    symbol
    prediction_date
    predicted_return
    predicted_direction
    predicted_risk
    confidence
    ...

The resulting shadow predictions can later be evaluated
against actual market outcomes.
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

logger = logging.getLogger(
    "shadow_prediction_job"
)


# ============================================================
# TIME
# ============================================================

def utc_now() -> datetime:
    """Return current UTC datetime."""

    return datetime.now(
        timezone.utc
    )


def utc_now_iso() -> str:
    """Return current UTC timestamp."""

    return utc_now().isoformat()


# ============================================================
# CONFIG HELPERS
# ============================================================

def object_to_dict(
    value: Any,
) -> dict[str, Any]:
    """Convert configuration objects into dictionaries."""

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
            "Could not load configuration: %s",
            error,
        )

        return None


def get_shadow_config() -> dict[str, Any]:
    """
    Get Shadow Prediction configuration.

    Supported config:

        shadow_predictions:
            enabled: true
            ledger_file: data/ledger/shadow_predictions.csv
    """

    defaults = {
        "enabled": True,
        "ledger_file": (
            "data/ledger/"
            "shadow_predictions.csv"
        ),
    }

    cfg = load_config()

    if cfg is None:
        return defaults

    section = getattr(
        cfg,
        "shadow_predictions",
        None,
    )

    values = object_to_dict(
        section
    )

    result = defaults.copy()

    for key in defaults:

        if key in values:
            result[key] = values[key]

    return result


# ============================================================
# DATAFRAME HELPERS
# ============================================================

def ensure_dataframe(
    value: Any,
) -> pd.DataFrame:
    """Convert common outputs into a DataFrame."""

    if value is None:
        return pd.DataFrame()

    if isinstance(value, pd.DataFrame):
        return value.copy()

    if isinstance(value, pd.Series):
        return value.to_frame().T

    if isinstance(value, list):
        try:
            return pd.DataFrame(value)
        except Exception:
            return pd.DataFrame()

    if isinstance(value, dict):
        try:
            return pd.DataFrame([value])
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Find the first matching column."""

    for column in candidates:

        if column in frame.columns:
            return column

    return None


# ============================================================
# MODEL REGISTRY
# ============================================================

def load_registered_models() -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """
    Load the active Champion and Challenger.
    """

    from src.model_registry import (
        get_champion,
        get_challenger,
    )

    champion = get_champion()

    challenger = get_challenger()

    return champion, challenger


def get_model_name(
    model: dict[str, Any] | None,
    fallback: str,
) -> str:
    """Get model name safely."""

    if not isinstance(model, dict):
        return fallback

    for key in [
        "name",
        "model_name",
        "id",
    ]:

        value = model.get(key)

        if value:
            return str(value)

    return fallback


def get_model_version(
    model: dict[str, Any] | None,
) -> str:
    """Get model version safely."""

    if not isinstance(model, dict):
        return "unknown"

    for key in [
        "version",
        "model_version",
    ]:

        value = model.get(key)

        if value is not None:
            return str(value)

    return "unknown"


# ============================================================
# SHARED DATASET
# ============================================================

def build_shared_prediction_data() -> pd.DataFrame:
    """
    Build the prediction dataset once.

    Champion and Challenger must receive this exact same data.

    Supported project entry points:

        src.pipeline.prepare_prediction_data()
        src.pipeline.build_prediction_data()
        src.pipeline.get_prediction_data()

        src.prediction_pipeline.prepare_prediction_data()
        src.prediction_pipeline.build_prediction_data()

    If no dedicated data preparation entry point exists,
    the normal prediction pipeline is attempted.
    """

    attempts: list[str] = []

    modules = [
        (
            "src.pipeline",
            [
                "prepare_prediction_data",
                "build_prediction_data",
                "get_prediction_data",
            ],
        ),
        (
            "src.prediction_pipeline",
            [
                "prepare_prediction_data",
                "build_prediction_data",
                "get_prediction_data",
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
                    "Building shared dataset using "
                    "%s.%s()",
                    module_name,
                    function_name,
                )

                result = function()

                frame = ensure_dataframe(
                    result
                )

                if not frame.empty:
                    return frame

        except Exception as error:

            attempts.append(
                f"{module_name}: {error}"
            )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    try:

        from src import pipeline

        function = getattr(
            pipeline,
            "run_pipeline",
            None,
        )

        if callable(function):

            logger.warning(
                "No dedicated shared-data function "
                "was found. Falling back to "
                "src.pipeline.run_pipeline()."
            )

            result = function()

            return ensure_dataframe(
                result
            )

    except Exception as error:

        attempts.append(
            f"pipeline fallback: {error}"
        )

    raise RuntimeError(
        "Could not build shared prediction data. "
        "Attempts: "
        + " | ".join(attempts)
    )


# ============================================================
# MODEL LOADING
# ============================================================

def extract_model_object(
    model_record: dict[str, Any],
) -> Any:
    """
    Extract an already-loaded model object when available.
    """

    for key in [
        "model",
        "model_object",
        "estimator",
    ]:

        value = model_record.get(key)

        if value is not None:
            return value

    return None


def load_model(
    model_record: dict[str, Any],
) -> Any:
    """
    Load a model from the registry record.

    Supported record patterns:

        {
            "model": <loaded model>
        }

        {
            "path": "models/model.joblib"
        }

        {
            "model_path": "models/model.joblib"
        }

        {
            "file": "models/model.joblib"
        }
    """

    existing_model = extract_model_object(
        model_record
    )

    if existing_model is not None:
        return existing_model

    model_path = None

    for key in [
        "path",
        "model_path",
        "file",
    ]:

        value = model_record.get(key)

        if value:
            model_path = value
            break

    if not model_path:

        raise RuntimeError(
            "Model registry record does not contain "
            "a model object or model path."
        )

    path = Path(
        str(model_path)
    )

    if not path.is_absolute():

        path = PROJECT_ROOT / path

    if not path.exists():

        raise FileNotFoundError(
            f"Model file does not exist: {path}"
        )

    try:

        import joblib

        logger.info(
            "Loading model: %s",
            path,
        )

        return joblib.load(
            path
        )

    except ImportError as error:

        raise RuntimeError(
            "joblib is required to load "
            "registered models."
        ) from error


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_feature_columns(
    model_record: dict[str, Any],
    frame: pd.DataFrame,
) -> list[str]:
    """
    Determine feature columns.

    Priority:

        1. Registry feature_columns.
        2. Registry features.
        3. Numeric DataFrame columns excluding
           prediction / metadata columns.
    """

    for key in [
        "feature_columns",
        "features",
    ]:

        value = model_record.get(key)

        if isinstance(value, list):

            columns = [
                str(column)
                for column in value
                if str(column) in frame.columns
            ]

            if columns:
                return columns

    excluded = {
        "symbol",
        "ticker",
        "stock",
        "sector",
        "date",
        "timestamp",
        "prediction_date",
        "created_at",
        "predicted_return",
        "predicted_direction",
        "predicted_risk",
        "confidence",
        "opportunity_score",
        "quality_score",
        "actual_return",
        "actual_direction",
        "actual_risk",
        "evaluation_status",
    }

    numeric_columns = frame.select_dtypes(
        include="number"
    ).columns.tolist()

    return [
        column
        for column in numeric_columns
        if column not in excluded
    ]


# ============================================================
# PREDICTION
# ============================================================

def predict_with_model(
    model: Any,
    model_record: dict[str, Any],
    shared_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate predictions using one model.

    Supports:

        model.predict(X)

    Optionally:

        model.predict_proba(X)

    The output is normalized into standard columns.
    """

    if shared_data.empty:
        return pd.DataFrame()

    feature_columns = (
        get_feature_columns(
            model_record,
            shared_data,
        )
    )

    if not feature_columns:

        raise RuntimeError(
            "Could not determine model feature columns."
        )

    features = (
        shared_data[
            feature_columns
        ]
        .copy()
    )

    features = features.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    features = features.fillna(
        0
    )

    if not hasattr(
        model,
        "predict",
    ):

        raise RuntimeError(
            "Loaded model does not provide predict()."
        )

    raw_predictions = (
        model.predict(
            features
        )
    )

    predictions = (
        shared_data.copy()
    )

    raw_series = pd.Series(
        raw_predictions,
        index=predictions.index,
    )

    predictions[
        "model_prediction"
    ] = raw_series

    # --------------------------------------------------------
    # STANDARD RETURN PREDICTION
    # --------------------------------------------------------

    if "predicted_return" not in predictions.columns:

        numeric = pd.to_numeric(
            raw_series,
            errors="coerce",
        )

        if numeric.notna().any():

            predictions[
                "predicted_return"
            ] = numeric

    # --------------------------------------------------------
    # STANDARD DIRECTION
    # --------------------------------------------------------

    if "predicted_direction" not in predictions.columns:

        def to_direction(
            value: Any,
        ) -> str | None:

            if pd.isna(value):
                return None

            if isinstance(
                value,
                str,
            ):

                text = value.strip().upper()

                if text in {
                    "UP",
                    "DOWN",
                    "FLAT",
                }:

                    return text

            try:

                numeric_value = float(value)

                if numeric_value > 0:
                    return "UP"

                if numeric_value < 0:
                    return "DOWN"

                return "FLAT"

            except Exception:

                return None

        predictions[
            "predicted_direction"
        ] = raw_series.map(
            to_direction
        )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba",
    ):

        try:

            probabilities = (
                model.predict_proba(
                    features
                )
            )

            probability_frame = pd.DataFrame(
                probabilities,
                index=predictions.index,
            )

            predictions[
                "confidence"
            ] = probability_frame.max(
                axis=1
            )

        except Exception as error:

            logger.warning(
                "Could not calculate confidence: %s",
                error,
            )

    return predictions


# ============================================================
# SHADOW RECORDS
# ============================================================

def get_symbol_column(
    frame: pd.DataFrame,
) -> str | None:
    """Find symbol column."""

    return find_column(
        frame,
        [
            "symbol",
            "ticker",
            "stock",
        ],
    )


def prepare_shadow_records(
    predictions: pd.DataFrame,
    model_role: str,
    model_record: dict[str, Any],
) -> pd.DataFrame:
    """
    Add model and evaluation metadata.
    """

    frame = predictions.copy()

    if frame.empty:
        return frame

    timestamp = utc_now_iso()

    frame["model_role"] = (
        model_role.upper()
    )

    frame["model_name"] = (
        get_model_name(
            model_record,
            fallback=model_role,
        )
    )

    frame["model_version"] = (
        get_model_version(
            model_record
        )
    )

    frame["prediction_date"] = (
        timestamp
    )

    frame["created_at"] = (
        timestamp
    )

    frame["evaluation_status"] = (
        "PENDING"
    )

    for column in [
        "actual_return",
        "actual_direction",
        "actual_risk",
    ]:

        if column not in frame.columns:

            frame[column] = pd.NA

    symbol_column = get_symbol_column(
        frame
    )

    if symbol_column:

        frame["prediction_id"] = (
            frame[symbol_column]
            .astype(str)
            .str.upper()
            + "_"
            + model_role.lower()
            + "_"
            + timestamp.replace(
                ":",
                ""
            ).replace(
                "+",
                ""
            )
        )

    else:

        frame["prediction_id"] = [
            (
                f"{model_role.lower()}_"
                f"{timestamp}_"
                f"{index}"
            )
            for index in range(
                len(frame)
            )
        ]

    return frame


# ============================================================
# SHADOW LEDGER
# ============================================================

def get_shadow_ledger_path() -> Path:
    """Get shadow prediction ledger path."""

    config = get_shadow_config()

    path = Path(
        str(
            config[
                "ledger_file"
            ]
        )
    )

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            / path
        )

    return path


def append_to_shadow_ledger(
    records: pd.DataFrame,
) -> Path:
    """
    Append Champion and Challenger predictions
    to the shadow ledger.
    """

    path = get_shadow_ledger_path()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if records.empty:

        logger.warning(
            "No shadow predictions to save."
        )

        return path

    file_exists = path.exists()

    records.to_csv(
        path,
        mode="a",
        header=not file_exists,
        index=False,
    )

    logger.info(
        "Saved %s shadow prediction(s) to %s",
        len(records),
        path,
    )

    return path


# ============================================================
# MAIN JOB
# ============================================================

def run_shadow_prediction_job() -> dict[str, Any]:
    """
    Run Champion and Challenger predictions
    against the same dataset.
    """

    result: dict[str, Any] = {
        "started_at": utc_now_iso(),
        "status": "STARTED",
        "champion_predictions": 0,
        "challenger_predictions": 0,
        "shared_rows": 0,
        "ledger_updated": False,
        "ledger_path": None,
        "error": None,
    }

    logger.info(
        "=" * 70
    )

    logger.info(
        "STARTING SHADOW PREDICTION JOB"
    )

    logger.info(
        "=" * 70
    )

    try:

        # ----------------------------------------------------
        # STEP 1: CONFIG
        # ----------------------------------------------------

        config = get_shadow_config()

        if not config.get(
            "enabled",
            True,
        ):

            result["status"] = "DISABLED"

            return result

        # ----------------------------------------------------
        # STEP 2: LOAD MODELS
        # ----------------------------------------------------

        logger.info(
            "Loading Champion and Challenger."
        )

        champion_record, challenger_record = (
            load_registered_models()
        )

        if champion_record is None:

            result["status"] = (
                "NO_CHAMPION"
            )

            return result

        if challenger_record is None:

            result["status"] = (
                "NO_CHALLENGER"
            )

            return result

        # ----------------------------------------------------
        # STEP 3: BUILD SHARED DATA
        # ----------------------------------------------------

        logger.info(
            "Building shared prediction dataset."
        )

        shared_data = (
            build_shared_prediction_data()
        )

        result["shared_rows"] = len(
            shared_data
        )

        if shared_data.empty:

            result["status"] = (
                "NO_SHARED_DATA"
            )

            return result

        # ----------------------------------------------------
        # STEP 4: LOAD CHAMPION
        # ----------------------------------------------------

        logger.info(
            "Loading Champion model."
        )

        champion_model = load_model(
            champion_record
        )

        # ----------------------------------------------------
        # STEP 5: LOAD CHALLENGER
        # ----------------------------------------------------

        logger.info(
            "Loading Challenger model."
        )

        challenger_model = load_model(
            challenger_record
        )

        # ----------------------------------------------------
        # STEP 6: RUN CHAMPION
        # ----------------------------------------------------

        logger.info(
            "Running Champion predictions."
        )

        champion_predictions = (
            predict_with_model(
                model=champion_model,
                model_record=champion_record,
                shared_data=shared_data,
            )
        )

        champion_records = (
            prepare_shadow_records(
                predictions=champion_predictions,
                model_role="CHAMPION",
                model_record=champion_record,
            )
        )

        result[
            "champion_predictions"
        ] = len(
            champion_records
        )

        # ----------------------------------------------------
        # STEP 7: RUN CHALLENGER
        # ----------------------------------------------------

        logger.info(
            "Running Challenger predictions."
        )

        challenger_predictions = (
            predict_with_model(
                model=challenger_model,
                model_record=challenger_record,
                shared_data=shared_data,
            )
        )

        challenger_records = (
            prepare_shadow_records(
                predictions=challenger_predictions,
                model_role="CHALLENGER",
                model_record=challenger_record,
            )
        )

        result[
            "challenger_predictions"
        ] = len(
            challenger_records
        )

        # ----------------------------------------------------
        # STEP 8: COMBINE RESULTS
        # ----------------------------------------------------

        records = pd.concat(
            [
                champion_records,
                challenger_records,
            ],
            ignore_index=True,
            sort=False,
        )

        # ----------------------------------------------------
        # STEP 9: SAVE SHADOW LEDGER
        # ----------------------------------------------------

        ledger_path = (
            append_to_shadow_ledger(
                records
            )
        )

        result[
            "ledger_updated"
        ] = True

        result[
            "ledger_path"
        ] = str(
            ledger_path
        )

        result["status"] = "SUCCESS"

        logger.info(
            "Shadow prediction job completed."
        )

        return result

    except Exception as error:

        logger.exception(
            "Shadow prediction job failed."
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
            "SHADOW PREDICTION JOB FINISHED | "
            "STATUS=%s",
            result.get(
                "status"
            ),
        )

        logger.info(
            "=" * 70
        )


# ============================================================
# CLI
# ============================================================

def main() -> int:
    """CLI entry point."""

    result = (
        run_shadow_prediction_job()
    )

    print()

    print("=" * 70)

    print("SHADOW PREDICTION JOB RESULT")

    print("=" * 70)

    print(
        "Status: "
        f"{result.get('status')}"
    )

    print(
        "Shared data rows: "
        f"{result.get('shared_rows')}"
    )

    print(
        "Champion predictions: "
        f"{result.get('champion_predictions')}"
    )

    print(
        "Challenger predictions: "
        f"{result.get('challenger_predictions')}"
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

    return (
        0
        if result.get("status")
        not in {
            "FAILED",
        }
        else 1
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

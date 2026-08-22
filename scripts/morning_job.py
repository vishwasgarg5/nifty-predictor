#!/usr/bin/env python3

"""
Morning Prediction Pipeline.

Pipeline
--------
1. Run market prediction pipeline.
2. Generate ranked opportunities.
3. Generate stable prediction_id values.
4. Safely update the prediction ledger.
5. Run production monitoring.
6. Check the circuit breaker.
7. Send predictions to Telegram only when allowed.

Important
---------
A circuit breaker block does NOT stop prediction generation
or ledger recording. It only blocks Telegram delivery.

Ledger identity
---------------
Every prediction receives a stable prediction_id.

The ID is generated from:

    market_date
    symbol
    model_version

The prediction ledger is updated using prediction_id.

Duplicate prediction records are prevented by checking
prediction_id before inserting new rows.
"""

from __future__ import annotations

import hashlib
import logging
import os
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

logger = logging.getLogger("morning_job")


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
    Convert configuration objects into dictionaries.
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
# DATAFRAME HELPERS
# ============================================================

def ensure_dataframe(
    value: Any,
) -> pd.DataFrame:
    """
    Convert common prediction outputs into a DataFrame.
    """

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
    """
    Find the first matching column.
    """

    for column in candidates:

        if column in frame.columns:
            return column

    return None


def normalize_value(
    value: Any,
) -> str:
    """
    Normalize a value for stable ID generation.
    """

    if value is None:
        return ""

    try:

        if pd.isna(value):
            return ""

    except Exception:
        pass

    return str(value).strip()


# ============================================================
# PREDICTION ID
# ============================================================

def get_prediction_identity_values(
    row: pd.Series,
) -> tuple[str, str, str]:
    """
    Extract stable identity values from a prediction row.

    Returns:

        market_date
        symbol
        model_version
    """

    # --------------------------------------------------------
    # MARKET DATE
    # --------------------------------------------------------

    market_date = ""

    for column in [
        "market_date",
        "prediction_date",
        "date",
    ]:

        if column in row.index:

            value = normalize_value(
                row.get(column)
            )

            if value:
                try:

                    parsed = pd.to_datetime(
                        value,
                        errors="coerce",
                    )

                    if pd.notna(parsed):

                        market_date = (
                            parsed.strftime(
                                "%Y-%m-%d"
                            )
                        )

                        break

                except Exception:
                    pass

                market_date = value

                break

    if not market_date:

        market_date = utc_now().strftime(
            "%Y-%m-%d"
        )

    # --------------------------------------------------------
    # SYMBOL
    # --------------------------------------------------------

    symbol = ""

    for column in [
        "symbol",
        "ticker",
        "stock",
    ]:

        if column in row.index:

            value = normalize_value(
                row.get(column)
            ).upper()

            if value:

                symbol = value

                break

    # --------------------------------------------------------
    # MODEL VERSION
    # --------------------------------------------------------

    model_version = ""

    for column in [
        "model_version",
        "model_name",
        "model",
    ]:

        if column in row.index:

            value = normalize_value(
                row.get(column)
            )

            if value:

                model_version = value

                break

    if not model_version:

        model_version = "default"

    return (
        market_date,
        symbol,
        model_version,
    )


def generate_prediction_id(
    row: pd.Series,
) -> str:
    """
    Generate a deterministic stable prediction ID.

    Identity:

        market_date | symbol | model_version

    Example:

        2026-08-22|RELIANCE|default

    The same identity always produces the same ID.
    """

    (
        market_date,
        symbol,
        model_version,
    ) = get_prediction_identity_values(
        row
    )

    raw_value = (
        f"{market_date}|"
        f"{symbol}|"
        f"{model_version}"
    )

    return hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()[:24]


def ensure_prediction_ids(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure every prediction contains prediction_id.

    Existing IDs are preserved.
    Missing IDs are generated.
    """

    frame = predictions.copy()

    if frame.empty:
        return frame

    if "prediction_id" not in frame.columns:

        frame["prediction_id"] = pd.NA

    for index in frame.index:

        current_id = frame.at[
            index,
            "prediction_id",
        ]

        if (
            pd.notna(current_id)
            and str(current_id).strip()
        ):
            continue

        frame.at[
            index,
            "prediction_id",
        ] = generate_prediction_id(
            frame.loc[index]
        )

    return frame


# ============================================================
# PIPELINE EXECUTION
# ============================================================

def run_prediction_pipeline() -> pd.DataFrame:
    """
    Run the project's prediction pipeline.

    The function tries the available project pipeline entry
    points in order.

    Supported patterns include:

        src.pipeline.run_pipeline()
        src.pipeline.run()

        src.prediction_pipeline.run_pipeline()
        src.prediction_pipeline.run()

        src.predict.run_predictions()
        src.predict.run()
    """

    attempts: list[str] = []

    # --------------------------------------------------------
    # src.pipeline
    # --------------------------------------------------------

    try:

        from src import pipeline

        for function_name in [
            "run_pipeline",
            "run",
        ]:

            function = getattr(
                pipeline,
                function_name,
                None,
            )

            if callable(function):

                logger.info(
                    "Running src.pipeline.%s()",
                    function_name,
                )

                result = function()

                return ensure_dataframe(
                    result
                )

    except Exception as error:

        attempts.append(
            f"src.pipeline: {error}"
        )

    # --------------------------------------------------------
    # src.prediction_pipeline
    # --------------------------------------------------------

    try:

        from src import prediction_pipeline

        for function_name in [
            "run_pipeline",
            "run",
        ]:

            function = getattr(
                prediction_pipeline,
                function_name,
                None,
            )

            if callable(function):

                logger.info(
                    "Running "
                    "src.prediction_pipeline.%s()",
                    function_name,
                )

                result = function()

                return ensure_dataframe(
                    result
                )

    except Exception as error:

        attempts.append(
            f"src.prediction_pipeline: {error}"
        )

    # --------------------------------------------------------
    # src.predict
    # --------------------------------------------------------

    try:

        from src import predict

        for function_name in [
            "run_predictions",
            "run",
        ]:

            function = getattr(
                predict,
                function_name,
                None,
            )

            if callable(function):

                logger.info(
                    "Running src.predict.%s()",
                    function_name,
                )

                result = function()

                return ensure_dataframe(
                    result
                )

    except Exception as error:

        attempts.append(
            f"src.predict: {error}"
        )

    raise RuntimeError(
        "Could not find a compatible prediction pipeline. "
        "Attempts: "
        + " | ".join(attempts)
    )


# ============================================================
# TOP OPPORTUNITY SELECTION
# ============================================================

def select_top_opportunities(
    predictions: pd.DataFrame,
    limit: int = 5,
) -> pd.DataFrame:
    """
    Select the highest-quality opportunities.

    Ranking preference:

        1. opportunity_score
        2. quality_score
        3. confidence
        4. predicted_return
    """

    frame = predictions.copy()

    if frame.empty:
        return frame

    score_column = find_column(
        frame,
        [
            "opportunity_score",
            "quality_score",
            "confidence",
            "predicted_return",
        ],
    )

    if score_column is not None:

        frame[score_column] = pd.to_numeric(
            frame[score_column],
            errors="coerce",
        )

        frame = frame.sort_values(
            by=score_column,
            ascending=False,
            na_position="last",
        )

    return frame.head(
        max(1, int(limit))
    ).reset_index(
        drop=True
    )


# ============================================================
# LEDGER PATH
# ============================================================

def get_ledger_path() -> Path:
    """
    Get prediction ledger location.
    """

    cfg = load_config()

    candidates: list[Any] = []

    # --------------------------------------------------------
    # ledger
    # --------------------------------------------------------

    ledger_section = getattr(
        cfg,
        "ledger",
        None,
    )

    ledger_values = object_to_dict(
        ledger_section
    )

    for key in [
        "path",
        "prediction_ledger",
        "ledger",
    ]:

        value = ledger_values.get(key)

        if value:
            candidates.append(value)

    # --------------------------------------------------------
    # data
    # --------------------------------------------------------

    data_section = getattr(
        cfg,
        "data",
        None,
    )

    data_values = object_to_dict(
        data_section
    )

    for key in [
        "prediction_ledger",
        "ledger",
    ]:

        value = data_values.get(key)

        if value:
            candidates.append(value)

    # --------------------------------------------------------
    # paths
    # --------------------------------------------------------

    paths_section = getattr(
        cfg,
        "paths",
        None,
    )

    path_values = object_to_dict(
        paths_section
    )

    for key in [
        "prediction_ledger",
        "ledger",
    ]:

        value = path_values.get(key)

        if value:
            candidates.append(value)

    # --------------------------------------------------------
    # Resolve first valid candidate.
    # --------------------------------------------------------

    for candidate in candidates:

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
# LEDGER PREPARATION
# ============================================================

def prepare_ledger_records(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare prediction records for the evaluation ledger.

    Adds:

        prediction_date
        created_at
        evaluation_status
        actual_return
        actual_direction
        actual_risk
        prediction_id
    """

    frame = predictions.copy()

    if frame.empty:
        return frame

    now = utc_now_iso()

    # --------------------------------------------------------
    # MARKET DATE
    # --------------------------------------------------------

    if "market_date" not in frame.columns:

        frame["market_date"] = (
            utc_now().strftime(
                "%Y-%m-%d"
            )
        )

    # --------------------------------------------------------
    # PREDICTION DATE
    # --------------------------------------------------------

    if "prediction_date" not in frame.columns:

        frame["prediction_date"] = now

    # --------------------------------------------------------
    # CREATED AT
    # --------------------------------------------------------

    if "created_at" not in frame.columns:

        frame["created_at"] = now

    # --------------------------------------------------------
    # EVALUATION STATUS
    # --------------------------------------------------------

    if "evaluation_status" not in frame.columns:

        frame["evaluation_status"] = (
            "PENDING"
        )

    # --------------------------------------------------------
    # ACTUAL RETURN
    # --------------------------------------------------------

    if "actual_return" not in frame.columns:

        frame["actual_return"] = pd.NA

    # --------------------------------------------------------
    # ACTUAL DIRECTION
    # --------------------------------------------------------

    if "actual_direction" not in frame.columns:

        frame["actual_direction"] = pd.NA

    # --------------------------------------------------------
    # ACTUAL RISK
    # --------------------------------------------------------

    if "actual_risk" not in frame.columns:

        frame["actual_risk"] = pd.NA

    # --------------------------------------------------------
    # GENERATE STABLE IDS
    # --------------------------------------------------------

    frame = ensure_prediction_ids(
        frame
    )

    return frame


# ============================================================
# LEDGER LOAD
# ============================================================

def load_existing_ledger(
    path: Path,
) -> pd.DataFrame:
    """
    Load the existing prediction ledger.

    Old records are automatically upgraded with
    prediction_id when missing.
    """

    if not path.exists():

        return pd.DataFrame()

    try:

        ledger = pd.read_csv(path)

    except pd.errors.EmptyDataError:

        return pd.DataFrame()

    except Exception as error:

        raise RuntimeError(
            f"Could not load prediction ledger: "
            f"{error}"
        ) from error

    if ledger.empty:
        return ledger

    ledger = ensure_prediction_ids(
        ledger
    )

    return ledger


# ============================================================
# LEDGER UPDATE
# ============================================================

def merge_new_predictions(
    existing: pd.DataFrame,
    new_predictions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    int,
]:
    """
    Merge new predictions into the ledger.

    The merge key is prediction_id.

    Existing predictions are preserved.

    New predictions with an existing prediction_id
    are NOT duplicated.

    Returns:

        updated_ledger
        inserted_count
    """

    new_records = ensure_prediction_ids(
        new_predictions
    )

    if new_records.empty:

        return (
            existing.copy(),
            0,
        )

    # --------------------------------------------------------
    # Remove duplicates inside the new batch.
    # --------------------------------------------------------

    new_records = (
        new_records
        .drop_duplicates(
            subset=["prediction_id"],
            keep="first",
        )
        .copy()
    )

    # --------------------------------------------------------
    # No existing ledger.
    # --------------------------------------------------------

    if existing.empty:

        return (
            new_records.reset_index(
                drop=True
            ),
            len(new_records),
        )

    existing_records = ensure_prediction_ids(
        existing
    )

    existing_ids = set(
        existing_records[
            "prediction_id"
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Only insert genuinely new predictions.
    # --------------------------------------------------------

    insert_mask = ~(
        new_records[
            "prediction_id"
        ]
        .astype(str)
        .str.strip()
        .isin(existing_ids)
    )

    records_to_insert = (
        new_records
        .loc[insert_mask]
        .copy()
    )

    inserted_count = len(
        records_to_insert
    )

    if records_to_insert.empty:

        return (
            existing_records.reset_index(
                drop=True
            ),
            0,
        )

    # --------------------------------------------------------
    # Preserve all columns from both DataFrames.
    # --------------------------------------------------------

    updated = pd.concat(
        [
            existing_records,
            records_to_insert,
        ],
        ignore_index=True,
        sort=False,
    )

    return (
        updated,
        inserted_count,
    )


def save_ledger(
    ledger: pd.DataFrame,
    path: Path,
) -> Path:
    """
    Save the complete prediction ledger safely.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    ledger.to_csv(
        temp_path,
        index=False,
    )

    temp_path.replace(
        path
    )

    logger.info(
        "Prediction ledger saved: %s",
        path,
    )

    return path


def append_to_ledger(
    predictions: pd.DataFrame,
) -> tuple[
    Path,
    int,
]:
    """
    Safely update the persistent prediction ledger.

    Despite the historical function name, this does NOT
    blindly append rows.

    Process:

        1. Load existing ledger.
        2. Upgrade old records with prediction_id.
        3. Prepare new records.
        4. Remove duplicate prediction_id values.
        5. Insert only new predictions.
        6. Atomically save the complete ledger.

    Returns:

        ledger_path
        inserted_count
    """

    path = get_ledger_path()

    records = prepare_ledger_records(
        predictions
    )

    if records.empty:

        logger.warning(
            "No prediction records to write "
            "to the ledger."
        )

        return (
            path,
            0,
        )

    existing = load_existing_ledger(
        path
    )

    updated_ledger, inserted_count = (
        merge_new_predictions(
            existing,
            records,
        )
    )

    # Save even if inserted_count == 0 because
    # existing records may have been upgraded with IDs.
    save_ledger(
        updated_ledger,
        path,
    )

    logger.info(
        "Ledger update complete | "
        "new_predictions=%s | "
        "total_records=%s",
        inserted_count,
        len(updated_ledger),
    )

    return (
        path,
        inserted_count,
    )


# ============================================================
# MONITORING
# ============================================================

def run_production_monitoring() -> dict[str, Any]:
    """
    Run production monitoring after prediction generation.
    """

    try:

        from src.monitoring import (
            run_monitoring,
        )

        logger.info(
            "Running production monitoring."
        )

        result = run_monitoring()

        if not isinstance(
            result,
            dict,
        ):

            return {
                "status": "UNKNOWN",
                "health_status": "UNKNOWN",
                "health_score": None,
                "error": (
                    "Monitoring returned an "
                    "unexpected result."
                ),
            }

        return result

    except Exception as error:

        logger.exception(
            "Production monitoring failed."
        )

        return {
            "status": "ERROR",
            "health_status": "CRITICAL",
            "health_score": 0,
            "error": str(error),
            "circuit_breaker": {
                "state": "ERROR",
                "predictions_allowed": False,
                "message": (
                    "Monitoring failed. "
                    "Telegram delivery blocked."
                ),
            },
        }


# ============================================================
# CIRCUIT BREAKER
# ============================================================

def check_circuit_breaker() -> tuple[
    bool,
    str,
    dict[str, Any],
]:
    """
    Check whether predictions may be sent.

    Returns:

        allowed
        reason
        breaker_status
    """

    try:

        from src.circuit_breaker import (
            can_send_predictions,
            get_status,
        )

        allowed, reason = (
            can_send_predictions()
        )

        status = get_status()

        logger.info(
            "Circuit breaker | "
            "state=%s | "
            "allowed=%s",
            status.get("state"),
            allowed,
        )

        return (
            bool(allowed),
            str(reason),
            status,
        )

    except Exception as error:

        logger.exception(
            "Circuit breaker check failed."
        )

        return (
            False,
            (
                "Circuit breaker check failed. "
                "Telegram delivery blocked for safety. "
                f"Error: {error}"
            ),
            {
                "state": "ERROR",
                "predictions_allowed": False,
                "error": str(error),
            },
        )


# ============================================================
# TELEGRAM CONFIGURATION
# ============================================================

def get_telegram_config() -> dict[str, Any]:
    """
    Load Telegram configuration.

    Environment variables override config values:

        TELEGRAM_BOT_TOKEN
        TELEGRAM_CHAT_ID
    """

    cfg = load_config()

    telegram_section = getattr(
        cfg,
        "telegram",
        None,
    )

    values = object_to_dict(
        telegram_section
    )

    token = (
        os.getenv(
            "TELEGRAM_BOT_TOKEN"
        )
        or values.get("bot_token")
        or values.get("token")
    )

    chat_id = (
        os.getenv(
            "TELEGRAM_CHAT_ID"
        )
        or values.get("chat_id")
    )

    enabled = values.get(
        "enabled",
        True,
    )

    return {
        "enabled": bool(enabled),
        "bot_token": token,
        "chat_id": chat_id,
    }


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def format_number(
    value: Any,
    decimals: int = 2,
    suffix: str = "",
) -> str:
    """
    Format a numeric value safely.
    """

    try:

        numeric = float(value)

        if pd.isna(numeric):
            return "N/A"

        return (
            f"{numeric:.{decimals}f}"
            f"{suffix}"
        )

    except Exception:

        return "N/A"


def format_prediction_message(
    predictions: pd.DataFrame,
    monitoring: dict[str, Any] | None = None,
) -> str:
    """
    Build the Telegram Top 5 prediction message.
    """

    if predictions.empty:

        return (
            "📊 Market Prediction Update\n\n"
            "No qualified opportunities were found."
        )

    lines = [
        "📈 TOP MARKET OPPORTUNITIES",
        "",
        (
            "Generated: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
        ),
        "",
    ]

    symbol_column = find_column(
        predictions,
        [
            "symbol",
            "ticker",
            "stock",
        ],
    )

    return_column = find_column(
        predictions,
        [
            "predicted_return",
            "expected_return",
            "return_prediction",
        ],
    )

    direction_column = find_column(
        predictions,
        [
            "predicted_direction",
            "direction",
            "direction_prediction",
        ],
    )

    confidence_column = find_column(
        predictions,
        [
            "confidence",
            "confidence_score",
        ],
    )

    risk_column = find_column(
        predictions,
        [
            "predicted_risk",
            "risk_score",
            "risk",
        ],
    )

    opportunity_column = find_column(
        predictions,
        [
            "opportunity_score",
            "quality_score",
        ],
    )

    for position, (_, row) in enumerate(
        predictions.iterrows(),
        start=1,
    ):

        symbol = (
            str(
                row.get(
                    symbol_column,
                    "UNKNOWN",
                )
            )
            if symbol_column
            else "UNKNOWN"
        )

        lines.append(
            f"{position}. {symbol}"
        )

        if return_column:

            value = format_number(
                row.get(return_column),
                decimals=2,
                suffix="%",
            )

            lines.append(
                f"   Expected Return: {value}"
            )

        if direction_column:

            direction = row.get(
                direction_column
            )

            if pd.notna(direction):

                lines.append(
                    f"   Direction: {direction}"
                )

        if confidence_column:

            confidence = format_number(
                row.get(
                    confidence_column
                ),
                decimals=2,
            )

            lines.append(
                f"   Confidence: {confidence}"
            )

        if risk_column:

            risk = format_number(
                row.get(
                    risk_column
                ),
                decimals=2,
            )

            lines.append(
                f"   Risk: {risk}"
            )

        if opportunity_column:

            score = format_number(
                row.get(
                    opportunity_column
                ),
                decimals=2,
            )

            lines.append(
                f"   Opportunity Score: {score}"
            )

        lines.append("")

    if monitoring:

        health_status = monitoring.get(
            "health_status"
        )

        health_score = monitoring.get(
            "health_score"
        )

        if health_status is not None:

            lines.extend(
                [
                    "──────────────────",
                    (
                        "System Health: "
                        f"{health_status}"
                    ),
                    (
                        "Health Score: "
                        f"{health_score}"
                    ),
                ]
            )

    return "\n".join(lines)


# ============================================================
# TELEGRAM DELIVERY
# ============================================================

def send_telegram_message(
    message: str,
) -> bool:
    """
    Send a message through Telegram.

    Uses the project's telegram module when available.
    Falls back to the Telegram Bot API through requests.
    """

    telegram_config = (
        get_telegram_config()
    )

    if not telegram_config.get(
        "enabled",
        True,
    ):

        logger.warning(
            "Telegram is disabled."
        )

        return False

    token = telegram_config.get(
        "bot_token"
    )

    chat_id = telegram_config.get(
        "chat_id"
    )

    if not token or not chat_id:

        raise RuntimeError(
            "Telegram configuration is incomplete. "
            "TELEGRAM_BOT_TOKEN and "
            "TELEGRAM_CHAT_ID are required."
        )

    # --------------------------------------------------------
    # Try project Telegram module first.
    # --------------------------------------------------------

    try:

        from src.telegram import (
            send_message,
        )

        result = send_message(
            message
        )

        if result is None:
            return True

        return bool(result)

    except ImportError:

        pass

    except Exception as error:

        logger.warning(
            "Project Telegram module failed. "
            "Using direct API fallback: %s",
            error,
        )

    # --------------------------------------------------------
    # Telegram Bot API fallback.
    # --------------------------------------------------------

    try:

        import requests

    except ImportError as error:

        raise RuntimeError(
            "requests package is required for "
            "Telegram delivery."
        ) from error

    endpoint = (
        f"https://api.telegram.org/bot"
        f"{token}/sendMessage"
    )

    response = requests.post(
        endpoint,
        json={
            "chat_id": chat_id,
            "text": message,
        },
        timeout=30,
    )

    response.raise_for_status()

    logger.info(
        "Telegram message sent successfully."
    )

    return True


# ============================================================
# MAIN JOB
# ============================================================

def run_morning_job() -> dict[str, Any]:
    """
    Run the complete morning prediction job.
    """

    result: dict[str, Any] = {
        "started_at": utc_now_iso(),
        "status": "STARTED",
        "predictions_generated": 0,
        "top_predictions": 0,
        "ledger_predictions_inserted": 0,
        "ledger_updated": False,
        "telegram_sent": False,
        "telegram_blocked": False,
        "circuit_breaker": {},
        "monitoring": {},
        "error": None,
    }

    logger.info(
        "=" * 70
    )

    logger.info(
        "STARTING MORNING PREDICTION JOB"
    )

    logger.info(
        "=" * 70
    )

    try:

        # ----------------------------------------------------
        # STEP 1: GENERATE PREDICTIONS
        # ----------------------------------------------------

        logger.info(
            "Step 1: Generating predictions."
        )

        predictions = (
            run_prediction_pipeline()
        )

        result[
            "predictions_generated"
        ] = len(predictions)

        logger.info(
            "Generated %s prediction(s).",
            len(predictions),
        )

        if predictions.empty:

            result["status"] = "NO_PREDICTIONS"

            logger.warning(
                "No predictions were generated."
            )

            return result

        # ----------------------------------------------------
        # STEP 2: SELECT TOP 5
        # ----------------------------------------------------

        logger.info(
            "Step 2: Selecting top opportunities."
        )

        top_predictions = (
            select_top_opportunities(
                predictions,
                limit=5,
            )
        )

        result[
            "top_predictions"
        ] = len(top_predictions)

        # ----------------------------------------------------
        # STEP 3: PREPARE STABLE IDENTITIES
        # ----------------------------------------------------

        logger.info(
            "Step 3: Generating stable prediction IDs."
        )

        top_predictions = (
            prepare_ledger_records(
                top_predictions
            )
        )

        # ----------------------------------------------------
        # STEP 4: UPDATE LEDGER
        # ----------------------------------------------------

        logger.info(
            "Step 4: Updating prediction ledger."
        )

        (
            ledger_path,
            inserted_count,
        ) = append_to_ledger(
            top_predictions
        )

        result[
            "ledger_predictions_inserted"
        ] = inserted_count

        result["ledger_updated"] = True

        result["ledger_path"] = str(
            ledger_path
        )

        # ----------------------------------------------------
        # STEP 5: RUN MONITORING
        # ----------------------------------------------------

        logger.info(
            "Step 5: Running production monitoring."
        )

        monitoring = (
            run_production_monitoring()
        )

        result[
            "monitoring"
        ] = monitoring

        # ----------------------------------------------------
        # STEP 6: CHECK CIRCUIT BREAKER
        # ----------------------------------------------------

        logger.info(
            "Step 6: Checking circuit breaker."
        )

        (
            allowed,
            reason,
            breaker_status,
        ) = check_circuit_breaker()

        result[
            "circuit_breaker"
        ] = breaker_status

        # ----------------------------------------------------
        # STEP 7: BLOCK OR SEND TELEGRAM
        # ----------------------------------------------------

        if not allowed:

            result[
                "telegram_blocked"
            ] = True

            result["status"] = (
                "TELEGRAM_BLOCKED"
            )

            result[
                "telegram_block_reason"
            ] = reason

            logger.warning(
                "Telegram delivery BLOCKED: %s",
                reason,
            )

            logger.warning(
                "Predictions were generated and "
                "ledger was updated, but no market "
                "signals were sent because the "
                "circuit breaker is not allowing "
                "delivery."
            )

            return result

        logger.info(
            "Step 7: Sending Telegram message."
        )

        message = (
            format_prediction_message(
                top_predictions,
                monitoring,
            )
        )

        telegram_sent = (
            send_telegram_message(
                message
            )
        )

        result[
            "telegram_sent"
        ] = telegram_sent

        if telegram_sent:

            result["status"] = "SUCCESS"

        else:

            result["status"] = (
                "TELEGRAM_NOT_SENT"
            )

        return result

    except Exception as error:

        logger.exception(
            "Morning prediction job failed."
        )

        result["status"] = "FAILED"

        result["error"] = str(error)

        result["traceback"] = (
            traceback.format_exc()
        )

        # Register critical pipeline failure.
        try:

            from src.circuit_breaker import (
                register_failure,
            )

            register_failure(
                reason=(
                    "Morning job failed: "
                    f"{error}"
                ),
                health_score=0,
                health_status="CRITICAL",
            )

        except Exception as breaker_error:

            logger.error(
                "Could not register failure "
                "with circuit breaker: %s",
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
            "MORNING JOB FINISHED | STATUS=%s",
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

    result = run_morning_job()

    print()

    print("=" * 70)

    print("MORNING JOB RESULT")

    print("=" * 70)

    print(
        "Status: "
        f"{result.get('status')}"
    )

    print(
        "Predictions generated: "
        f"{result.get('predictions_generated')}"
    )

    print(
        "Top predictions: "
        f"{result.get('top_predictions')}"
    )

    print(
        "New ledger predictions: "
        f"{result.get('ledger_predictions_inserted')}"
    )

    print(
        "Ledger updated: "
        f"{result.get('ledger_updated')}"
    )

    print(
        "Telegram sent: "
        f"{result.get('telegram_sent')}"
    )

    print(
        "Telegram blocked: "
        f"{result.get('telegram_blocked')}"
    )

    breaker = result.get(
        "circuit_breaker",
        {},
    )

    if breaker:

        print()

        print("Circuit Breaker:")

        print(
            "  State: "
            f"{breaker.get('state')}"
        )

        print(
            "  Predictions allowed: "
            f"{breaker.get('predictions_allowed')}"
        )

        print(
            "  Reason: "
            f"{breaker.get('reason', breaker.get('message'))}"
        )

    monitoring = result.get(
        "monitoring",
        {},
    )

    if monitoring:

        print()

        print("System Health:")

        print(
            "  Status: "
            f"{monitoring.get('health_status')}"
        )

        print(
            "  Score: "
            f"{monitoring.get('health_score')}"
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
        in {
            "SUCCESS",
            "TELEGRAM_BLOCKED",
            "NO_PREDICTIONS",
        }
        else 1
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

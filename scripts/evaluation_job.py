#!/usr/bin/env python3

"""
Daily Model Evaluation Job.

Responsibilities
----------------
1. Load pending predictions from the prediction ledger.
2. Fetch actual market data for each prediction.
3. Calculate actual return and direction.
4. Update the prediction ledger.
5. Run model evaluation.
6. Run drift detection when available.
7. Run Champion / Challenger comparison.
8. Run production monitoring.
9. Send important monitoring alerts to Telegram.
10. Save evaluation and monitoring reports.

Run:
    python scripts/evaluation_job.py
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("evaluation_job")


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_CONFIG = {
    "paths": {
        "ledger": "data/ledger/predictions.csv",
        "reports": "data/reports",
        "registry": "data/model_registry.json",
        "monitoring": "data/monitoring",
    },
    "evaluation": {
        "enabled": True,
        "recent_window": 50,
        "save_reports": True,
    },
    "drift": {
        "enabled": True,
    },
    "champion_challenger": {
        "enabled": True,
    },
    "monitoring": {
        "enabled": True,
    },
    "telegram_alerts": {
        "enabled": True,
        "minimum_level": "CRITICAL",
        "send_healthy_report": False,
    },
}


# ============================================================
# CONFIG HELPERS
# ============================================================

def _object_to_dict(value: Any) -> dict[str, Any]:
    """Convert config-like objects to a normal dictionary."""

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


def _merge_section(
    target: dict[str, Any],
    values: dict[str, Any],
) -> None:
    """Merge one configuration section safely."""

    for key, value in values.items():

        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, dict)
        ):
            target[key].update(value)

        else:
            target[key] = value


def load_config() -> dict[str, Any]:
    """
    Load configuration from src.config.cfg.

    Falls back to DEFAULT_CONFIG if the project config cannot
    be imported.
    """

    config = {
        "paths": dict(DEFAULT_CONFIG["paths"]),
        "evaluation": dict(DEFAULT_CONFIG["evaluation"]),
        "drift": dict(DEFAULT_CONFIG["drift"]),
        "champion_challenger": dict(
            DEFAULT_CONFIG["champion_challenger"]
        ),
        "monitoring": dict(
            DEFAULT_CONFIG["monitoring"]
        ),
        "telegram_alerts": dict(
            DEFAULT_CONFIG["telegram_alerts"]
        ),
    }

    try:
        from src.config import cfg

        for section_name in config:

            section = getattr(
                cfg,
                section_name,
                None,
            )

            values = _object_to_dict(
                section
            )

            if values:
                _merge_section(
                    config[section_name],
                    values,
                )

    except Exception as error:
        logger.warning(
            "Using default configuration because project "
            "config could not be loaded: %s",
            error,
        )

    return config


# ============================================================
# PATH HELPERS
# ============================================================

def resolve_project_path(
    value: str | Path,
) -> Path:
    """Resolve a path relative to the project root."""

    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# ============================================================
# LEDGER HELPERS
# ============================================================

def load_prediction_ledger(
    ledger_path: str | Path,
) -> pd.DataFrame:
    """Load the prediction ledger safely."""

    path = resolve_project_path(
        ledger_path
    )

    if not path.exists():

        logger.warning(
            "Prediction ledger does not exist: %s",
            path,
        )

        return pd.DataFrame()

    try:

        return pd.read_csv(path)

    except pd.errors.EmptyDataError:

        return pd.DataFrame()

    except Exception as error:

        logger.exception(
            "Unable to read prediction ledger: %s",
            error,
        )

        return pd.DataFrame()


def atomic_save_ledger(
    frame: pd.DataFrame,
    ledger_path: str | Path,
) -> Path:
    """Safely write the updated ledger."""

    path = resolve_project_path(
        ledger_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        f"{path.suffix}.tmp"
    )

    frame.to_csv(
        temporary,
        index=False,
    )

    shutil.move(
        str(temporary),
        str(path),
    )

    return path


def ensure_ledger_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure all evaluation columns exist."""

    defaults: dict[str, Any] = {
        "evaluation_status": "PENDING",
        "actual_open": np.nan,
        "actual_high": np.nan,
        "actual_low": np.nan,
        "actual_close": np.nan,
        "actual_return": np.nan,
        "actual_direction": np.nan,
        "actual_risk": np.nan,
        "actual_date": pd.NA,
        "evaluated_at": pd.NA,
    }

    result = frame.copy()

    for column, default in defaults.items():

        if column not in result.columns:

            result[column] = default

    return result


# ============================================================
# DATE AND SYMBOL HELPERS
# ============================================================

def parse_date(
    value: Any,
) -> pd.Timestamp | None:
    """Safely parse a date value."""

    if value is None or pd.isna(value):
        return None

    try:

        parsed = pd.to_datetime(
            value,
            errors="coerce",
        )

        if pd.isna(parsed):
            return None

        return pd.Timestamp(
            parsed
        ).normalize()

    except Exception:

        return None


def find_prediction_date_column(
    frame: pd.DataFrame,
) -> str | None:
    """Find the prediction date column."""

    candidates = [
        "prediction_date",
        "market_date",
        "date",
        "created_at",
    ]

    for column in candidates:

        if column in frame.columns:
            return column

    return None


def find_symbol_column(
    frame: pd.DataFrame,
) -> str | None:
    """Find the symbol column."""

    for column in (
        "symbol",
        "ticker",
    ):

        if column in frame.columns:
            return column

    return None


def normalize_yahoo_symbol(
    symbol: Any,
) -> str | None:
    """
    Convert NSE symbols to Yahoo Finance symbols.

    Examples:
        RELIANCE -> RELIANCE.NS
        RELIANCE.NS -> RELIANCE.NS
        ^NSEI -> ^NSEI
    """

    if symbol is None or pd.isna(symbol):
        return None

    value = str(symbol).strip().upper()

    if not value:
        return None

    if value.startswith("^"):
        return value

    if "." in value:
        return value

    return f"{value}.NS"


# ============================================================
# MARKET DATA
# ============================================================

def fetch_actual_market_data(
    symbol: str,
    prediction_date: pd.Timestamp,
) -> dict[str, Any] | None:
    """
    Fetch the first available trading session after prediction_date.
    """

    yahoo_symbol = normalize_yahoo_symbol(
        symbol
    )

    if yahoo_symbol is None:
        return None

    start = (
        prediction_date
        + pd.Timedelta(days=1)
    )

    end = (
        prediction_date
        + pd.Timedelta(days=10)
    )

    try:

        history = yf.download(
            yahoo_symbol,
            start=start.strftime(
                "%Y-%m-%d"
            ),
            end=end.strftime(
                "%Y-%m-%d"
            ),
            progress=False,
            auto_adjust=False,
            threads=False,
        )

    except Exception as error:

        logger.warning(
            "Market data download failed for %s: %s",
            yahoo_symbol,
            error,
        )

        return None

    if history is None or history.empty:
        return None

    if isinstance(
        history.columns,
        pd.MultiIndex,
    ):

        history.columns = [
            column[0]
            if isinstance(
                column,
                tuple,
            )
            else column
            for column in history.columns
        ]

    history = history.dropna(
        how="all"
    )

    if history.empty:
        return None

    row = history.iloc[0]

    def get_value(
        column: str,
    ) -> float:

        if column not in row.index:
            return np.nan

        try:
            return float(row[column])

        except Exception:
            return np.nan

    return {
        "actual_date": str(
            pd.Timestamp(
                history.index[0]
            ).date()
        ),
        "actual_open": get_value(
            "Open"
        ),
        "actual_high": get_value(
            "High"
        ),
        "actual_low": get_value(
            "Low"
        ),
        "actual_close": get_value(
            "Close"
        ),
    }


def calculate_actual_return(
    prediction_row: pd.Series,
    actual_close: float,
) -> float:
    """Calculate realized return."""

    baseline_candidates = [
        "current_close",
        "close",
        "reference_close",
        "predicted_open",
    ]

    for column in baseline_candidates:

        if column not in prediction_row.index:
            continue

        baseline = pd.to_numeric(
            pd.Series(
                [
                    prediction_row[column]
                ]
            ),
            errors="coerce",
        ).iloc[0]

        if (
            pd.notna(baseline)
            and baseline != 0
            and pd.notna(actual_close)
        ):

            return float(
                (actual_close - baseline)
                / baseline
            )

    actual_open = pd.to_numeric(
        pd.Series(
            [
                prediction_row.get(
                    "actual_open"
                )
            ]
        ),
        errors="coerce",
    ).iloc[0]

    if (
        pd.notna(actual_open)
        and actual_open != 0
        and pd.notna(actual_close)
    ):

        return float(
            (actual_close - actual_open)
            / actual_open
        )

    return np.nan


def calculate_actual_risk(
    actual_open: float,
    actual_high: float,
    actual_low: float,
    actual_close: float,
) -> float:
    """Estimate realized daily risk."""

    if (
        pd.isna(actual_high)
        or pd.isna(actual_low)
        or pd.isna(actual_close)
        or actual_close == 0
    ):

        return np.nan

    return float(
        abs(
            actual_high - actual_low
        )
        / abs(actual_close)
    )


# ============================================================
# EVALUATE PENDING PREDICTIONS
# ============================================================

def evaluate_pending_predictions(
    ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Fetch actual market outcomes and update pending predictions.
    """

    result = ensure_ledger_columns(
        ledger
    )

    stats = {
        "pending": 0,
        "evaluated": 0,
        "waiting": 0,
        "failed": 0,
    }

    if result.empty:
        return result, stats

    symbol_column = find_symbol_column(
        result
    )

    date_column = find_prediction_date_column(
        result
    )

    if symbol_column is None:

        logger.error(
            "Cannot evaluate ledger: "
            "symbol/ticker column missing."
        )

        return result, stats

    if date_column is None:

        logger.error(
            "Cannot evaluate ledger: "
            "prediction date column missing."
        )

        return result, stats

    pending_mask = (
        result["evaluation_status"]
        .fillna("PENDING")
        .astype(str)
        .str.upper()
        .isin(
            {
                "PENDING",
                "UNEVALUATED",
                "WAITING",
            }
        )
    )

    pending_indexes = result.index[
        pending_mask
    ].tolist()

    stats["pending"] = len(
        pending_indexes
    )

    if not pending_indexes:
        return result, stats

    cache: dict[
        tuple[str, str],
        dict[str, Any] | None,
    ] = {}

    for index in pending_indexes:

        row = result.loc[index]

        symbol = row.get(
            symbol_column
        )

        prediction_date = parse_date(
            row.get(
                date_column
            )
        )

        if prediction_date is None:

            logger.warning(
                "Skipping row %s: "
                "invalid prediction date.",
                index,
            )

            result.at[
                index,
                "evaluation_status",
            ] = "FAILED"

            result.at[
                index,
                "evaluated_at",
            ] = datetime.now().isoformat()

            stats["failed"] += 1

            continue

        key = (
            str(symbol),
            prediction_date.strftime(
                "%Y-%m-%d"
            ),
        )

        if key not in cache:

            cache[key] = (
                fetch_actual_market_data(
                    symbol=str(symbol),
                    prediction_date=prediction_date,
                )
            )

        actual = cache[key]

        if actual is None:

            result.at[
                index,
                "evaluation_status",
            ] = "WAITING"

            stats["waiting"] += 1

            continue

        actual_open = actual[
            "actual_open"
        ]

        actual_high = actual[
            "actual_high"
        ]

        actual_low = actual[
            "actual_low"
        ]

        actual_close = actual[
            "actual_close"
        ]

        actual_return = (
            calculate_actual_return(
                row,
                actual_close,
            )
        )

        if pd.notna(actual_return):

            actual_direction = (
                "UP"
                if actual_return >= 0
                else "DOWN"
            )

        else:

            actual_direction = pd.NA

        actual_risk = (
            calculate_actual_risk(
                actual_open,
                actual_high,
                actual_low,
                actual_close,
            )
        )

        result.at[
            index,
            "actual_open",
        ] = actual_open

        result.at[
            index,
            "actual_high",
        ] = actual_high

        result.at[
            index,
            "actual_low",
        ] = actual_low

        result.at[
            index,
            "actual_close",
        ] = actual_close

        result.at[
            index,
            "actual_return",
        ] = actual_return

        result.at[
            index,
            "actual_direction",
        ] = actual_direction

        result.at[
            index,
            "actual_risk",
        ] = actual_risk

        result.at[
            index,
            "actual_date",
        ] = actual[
            "actual_date"
        ]

        result.at[
            index,
            "evaluation_status",
        ] = "EVALUATED"

        result.at[
            index,
            "evaluated_at",
        ] = datetime.now().isoformat()

        stats["evaluated"] += 1

        logger.info(
            "Evaluated %s | "
            "prediction=%s | "
            "actual=%s",
            symbol,
            prediction_date.date(),
            actual["actual_date"],
        )

    return result, stats


# ============================================================
# METRIC HELPERS
# ============================================================

def numeric_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> pd.Series | None:
    """Return the first usable numeric column."""

    for column in candidates:

        if column in frame.columns:

            return pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    return None


def calculate_direction_accuracy(
    frame: pd.DataFrame,
) -> float | None:
    """Calculate direction accuracy."""

    if (
        "predicted_direction"
        not in frame.columns
        or "actual_direction"
        not in frame.columns
    ):

        return None

    predicted = (
        frame["predicted_direction"]
        .astype(str)
        .str.upper()
        .replace(
            {
                "1": "UP",
                "1.0": "UP",
                "-1": "DOWN",
                "-1.0": "DOWN",
                "TRUE": "UP",
                "FALSE": "DOWN",
            }
        )
    )

    actual = (
        frame["actual_direction"]
        .astype(str)
        .str.upper()
        .replace(
            {
                "1": "UP",
                "1.0": "UP",
                "-1": "DOWN",
                "-1.0": "DOWN",
                "TRUE": "UP",
                "FALSE": "DOWN",
            }
        )
    )

    valid = (
        predicted.isin(
            ["UP", "DOWN"]
        )
        & actual.isin(
            ["UP", "DOWN"]
        )
    )

    if not valid.any():
        return None

    return float(
        (
            predicted[valid]
            == actual[valid]
        ).mean()
    )


def calculate_brier_score(
    frame: pd.DataFrame,
) -> float | None:
    """Calculate Brier score."""

    if "actual_direction" not in frame.columns:
        return None

    probability = numeric_column(
        frame,
        [
            "direction_probability",
            "probability_up",
            "confidence",
        ],
    )

    if probability is None:
        return None

    actual = (
        frame["actual_direction"]
        .astype(str)
        .str.upper()
        .replace(
            {
                "1": "UP",
                "1.0": "UP",
                "-1": "DOWN",
                "-1.0": "DOWN",
            }
        )
        .map(
            {
                "UP": 1.0,
                "DOWN": 0.0,
            }
        )
    )

    valid = (
        probability.notna()
        & actual.notna()
    )

    if not valid.any():
        return None

    probability = probability.clip(
        0.0,
        1.0,
    )

    return float(
        np.mean(
            (
                probability[valid]
                - actual[valid]
            )
            ** 2
        )
    )


def calculate_simple_mae_metrics(
    frame: pd.DataFrame,
) -> dict[str, float | None]:
    """Calculate return and risk MAE."""

    result: dict[
        str,
        float | None,
    ] = {
        "return_mae": None,
        "risk_mae": None,
    }

    predicted_return = numeric_column(
        frame,
        [
            "predicted_return",
            "expected_return",
        ],
    )

    actual_return = numeric_column(
        frame,
        [
            "actual_return",
            "realized_return",
        ],
    )

    if (
        predicted_return is not None
        and actual_return is not None
    ):

        valid = (
            predicted_return.notna()
            & actual_return.notna()
        )

        if valid.any():

            result[
                "return_mae"
            ] = float(
                np.mean(
                    np.abs(
                        predicted_return[valid]
                        - actual_return[valid]
                    )
                )
            )

    predicted_risk = numeric_column(
        frame,
        [
            "predicted_risk",
            "expected_risk",
        ],
    )

    actual_risk = numeric_column(
        frame,
        [
            "actual_risk",
            "realized_risk",
        ],
    )

    if (
        predicted_risk is not None
        and actual_risk is not None
    ):

        valid = (
            predicted_risk.notna()
            & actual_risk.notna()
        )

        if valid.any():

            result[
                "risk_mae"
            ] = float(
                np.mean(
                    np.abs(
                        predicted_risk[valid]
                        - actual_risk[valid]
                    )
                )
            )

    return result


def calculate_evaluation_metrics(
    evaluated: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate global and per-model metrics."""

    if evaluated.empty:

        return {
            "sample_count": 0,
            "return_mae": None,
            "direction_accuracy": None,
            "brier_score": None,
            "risk_mae": None,
            "models": {},
        }

    simple_metrics = (
        calculate_simple_mae_metrics(
            evaluated
        )
    )

    metrics = {
        "sample_count": int(
            len(evaluated)
        ),
        "return_mae": simple_metrics.get(
            "return_mae"
        ),
        "direction_accuracy": (
            calculate_direction_accuracy(
                evaluated
            )
        ),
        "brier_score": (
            calculate_brier_score(
                evaluated
            )
        ),
        "risk_mae": simple_metrics.get(
            "risk_mae"
        ),
        "models": {},
    }

    if "model_name" in evaluated.columns:

        for (
            model_name,
            group,
        ) in evaluated.groupby(
            "model_name",
            dropna=False,
        ):

            model_simple = (
                calculate_simple_mae_metrics(
                    group
                )
            )

            metrics["models"][
                str(model_name)
            ] = {
                "sample_count": int(
                    len(group)
                ),
                "return_mae": (
                    model_simple.get(
                        "return_mae"
                    )
                ),
                "direction_accuracy": (
                    calculate_direction_accuracy(
                        group
                    )
                ),
                "brier_score": (
                    calculate_brier_score(
                        group
                    )
                ),
                "risk_mae": (
                    model_simple.get(
                        "risk_mae"
                    )
                ),
            }

    return metrics


# ============================================================
# REPORTING
# ============================================================

def save_evaluation_report(
    metrics: dict[str, Any],
    reports_path: str | Path,
) -> Path:
    """Save evaluation metrics."""

    path = resolve_project_path(
        reports_path
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = (
        path
        / "latest_evaluation.csv"
    )

    rows: list[
        dict[str, Any]
    ] = []

    rows.append(
        {
            "model_name": "ALL",
            "sample_count": metrics.get(
                "sample_count"
            ),
            "return_mae": metrics.get(
                "return_mae"
            ),
            "direction_accuracy": metrics.get(
                "direction_accuracy"
            ),
            "brier_score": metrics.get(
                "brier_score"
            ),
            "risk_mae": metrics.get(
                "risk_mae"
            ),
            "generated_at": (
                datetime.now().isoformat()
            ),
        }
    )

    for (
        model_name,
        model_metrics,
    ) in metrics.get(
        "models",
        {},
    ).items():

        rows.append(
            {
                "model_name": model_name,
                "sample_count": (
                    model_metrics.get(
                        "sample_count"
                    )
                ),
                "return_mae": (
                    model_metrics.get(
                        "return_mae"
                    )
                ),
                "direction_accuracy": (
                    model_metrics.get(
                        "direction_accuracy"
                    )
                ),
                "brier_score": (
                    model_metrics.get(
                        "brier_score"
                    )
                ),
                "risk_mae": (
                    model_metrics.get(
                        "risk_mae"
                    )
                ),
                "generated_at": (
                    datetime.now().isoformat()
                ),
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        report,
        index=False,
    )

    return report


# ============================================================
# DRIFT DETECTION
# ============================================================

def run_drift_detection(
    ledger_path: str | Path,
) -> Any:
    """
    Run the available drift detector.
    """

    candidates = [
        (
            "src.drift_detection",
            "detect_drift",
        ),
        (
            "src.drift_detection",
            "run_drift_detection",
        ),
        (
            "src.drift",
            "detect_drift",
        ),
        (
            "src.drift",
            "run_drift_detection",
        ),
    ]

    for (
        module_name,
        function_name,
    ) in candidates:

        try:

            module = __import__(
                module_name,
                fromlist=[
                    function_name
                ],
            )

            function = getattr(
                module,
                function_name,
            )

            try:

                return function(
                    ledger_path=ledger_path
                )

            except TypeError:

                return function()

        except (
            ImportError,
            AttributeError,
        ):

            continue

        except Exception as error:

            logger.exception(
                "Drift detection failed: %s",
                error,
            )

            return None

    logger.info(
        "No drift detection entry point found."
    )

    return None


# ============================================================
# CHAMPION / CHALLENGER
# ============================================================

def run_champion_challenger_evaluation(
    ledger_path: str | Path,
) -> list[dict[str, Any]]:
    """Run Champion / Challenger comparison."""

    try:

        from src.champion_challenger import (
            evaluate_and_maybe_promote,
        )

        return evaluate_and_maybe_promote(
            ledger_path=ledger_path
        )

    except ImportError:

        logger.warning(
            "Champion/Challenger module "
            "not available."
        )

        return []

    except Exception as error:

        logger.exception(
            "Champion/Challenger evaluation failed: %s",
            error,
        )

        return []


# ============================================================
# MONITORING
# ============================================================

def run_production_monitoring() -> dict[str, Any]:
    """
    Run production monitoring.

    Monitoring failures are isolated so they do not stop
    the evaluation pipeline.
    """

    try:

        from src.monitoring import (
            run_monitoring,
        )

        result = run_monitoring()

        if not isinstance(
            result,
            dict,
        ):

            return {
                "status": "INVALID_RESULT",
                "health_status": "UNKNOWN",
                "health_score": None,
                "alerts": [],
            }

        return result

    except ImportError as error:

        logger.warning(
            "Monitoring module not available: %s",
            error,
        )

        return {
            "status": "UNAVAILABLE",
            "health_status": "UNKNOWN",
            "health_score": None,
            "alerts": [],
        }

    except Exception as error:

        logger.exception(
            "Production monitoring failed: %s",
            error,
        )

        return {
            "status": "FAILED",
            "health_status": "UNKNOWN",
            "health_score": None,
            "alerts": [
                {
                    "level": "WARNING",
                    "message": (
                        "Monitoring execution failed: "
                        f"{error}"
                    ),
                }
            ],
        }


# ============================================================
# TELEGRAM ALERTS
# ============================================================

ALERT_LEVELS = {
    "INFO": 1,
    "WARNING": 2,
    "CRITICAL": 3,
}


def should_send_alert(
    alert_level: str,
    minimum_level: str,
) -> bool:
    """Check whether an alert passes the threshold."""

    current = ALERT_LEVELS.get(
        str(alert_level).upper(),
        1,
    )

    minimum = ALERT_LEVELS.get(
        str(minimum_level).upper(),
        3,
    )

    return current >= minimum


def get_telegram_credentials() -> tuple[
    str | None,
    str | None,
]:
    """Read Telegram credentials from environment."""

    token = os.getenv(
        "TELEGRAM_BOT_TOKEN"
    )

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID"
    )

    if token:
        token = token.strip()

    if chat_id:
        chat_id = chat_id.strip()

    return token, chat_id


def send_telegram_message(
    message: str,
) -> bool:
    """Send a Telegram message."""

    token, chat_id = (
        get_telegram_credentials()
    )

    if not token or not chat_id:

        logger.warning(
            "Telegram credentials are missing. "
            "Alert was not sent."
        )

        return False

    url = (
        f"https://api.telegram.org/bot"
        f"{token}/sendMessage"
    )

    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=15,
        )

        if response.ok:

            logger.info(
                "Telegram alert sent successfully."
            )

            return True

        logger.error(
            "Telegram API returned %s: %s",
            response.status_code,
            response.text,
        )

        return False

    except Exception as error:

        logger.exception(
            "Telegram alert failed: %s",
            error,
        )

        return False


def format_monitoring_alert(
    monitoring_result: dict[str, Any],
    alerts: list[dict[str, Any]],
) -> str:
    """Create a readable Telegram monitoring alert."""

    health_status = str(
        monitoring_result.get(
            "health_status",
            "UNKNOWN",
        )
    ).upper()

    health_score = monitoring_result.get(
        "health_score",
        "N/A",
    )

    ledger = monitoring_result.get(
        "ledger",
        {},
    )

    models = monitoring_result.get(
        "models",
        {},
    )

    lines = [
        "🚨 <b>STOCK PREDICTION SYSTEM ALERT</b>",
        "",
        f"<b>Health:</b> {health_status}",
        f"<b>Score:</b> {health_score}/100",
        "",
        "<b>Pipeline:</b>",
        (
            "Predictions: "
            f"{ledger.get('total_predictions', 0)}"
        ),
        (
            "Evaluated: "
            f"{ledger.get('evaluated', 0)}"
        ),
        (
            "Pending: "
            f"{ledger.get('pending', 0)}"
        ),
        (
            "Champion: "
            f"{models.get('champion', 'UNKNOWN')}"
        ),
        "",
        "<b>Alerts:</b>",
    ]

    for alert in alerts:

        level = str(
            alert.get(
                "level",
                "INFO",
            )
        ).upper()

        message = str(
            alert.get(
                "message",
                "Unknown alert",
            )
        )

        if level == "CRITICAL":
            icon = "🔴"

        elif level == "WARNING":
            icon = "🟠"

        else:
            icon = "🔵"

        lines.append(
            f"{icon} <b>{level}</b>: "
            f"{message}"
        )

    lines.extend(
        [
            "",
            (
                f"<b>Time:</b> "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ),
        ]
    )

    return "\n".join(
        lines
    )


def send_monitoring_alerts(
    monitoring_result: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    """
    Send Telegram alerts according to configured severity.
    """

    settings = config.get(
        "telegram_alerts",
        {},
    )

    if not bool(
        settings.get(
            "enabled",
            True,
        )
    ):

        logger.info(
            "Telegram monitoring alerts are disabled."
        )

        return False

    minimum_level = str(
        settings.get(
            "minimum_level",
            "CRITICAL",
        )
    ).upper()

    alerts = monitoring_result.get(
        "alerts",
        [],
    )

    selected_alerts = [
        alert
        for alert in alerts
        if should_send_alert(
            alert.get(
                "level",
                "INFO",
            ),
            minimum_level,
        )
    ]

    health_status = str(
        monitoring_result.get(
            "health_status",
            "UNKNOWN",
        )
    ).upper()

    send_healthy_report = bool(
        settings.get(
            "send_healthy_report",
            False,
        )
    )

    if (
        not selected_alerts
        and not (
            send_healthy_report
            and health_status == "HEALTHY"
        )
    ):

        logger.info(
            "No Telegram monitoring alert "
            "meets the configured threshold."
        )

        return False

    if (
        not selected_alerts
        and send_healthy_report
    ):

        selected_alerts = [
            {
                "level": "INFO",
                "message": (
                    "System is healthy. "
                    "No active alerts."
                ),
            }
        ]

    message = format_monitoring_alert(
        monitoring_result,
        selected_alerts,
    )

    return send_telegram_message(
        message
    )


# ============================================================
# MAIN JOB
# ============================================================

def run_evaluation_job() -> dict[str, Any]:
    """Run the complete daily evaluation workflow."""

    config = load_config()

    if not bool(
        config["evaluation"].get(
            "enabled",
            True,
        )
    ):

        logger.info(
            "Evaluation job is disabled."
        )

        return {
            "status": "DISABLED",
        }

    ledger_path = config[
        "paths"
    ].get(
        "ledger",
        "data/ledger/predictions.csv",
    )

    reports_path = config[
        "paths"
    ].get(
        "reports",
        "data/reports",
    )

    logger.info(
        "=" * 60
    )

    logger.info(
        "STARTING DAILY EVALUATION JOB"
    )

    logger.info(
        "=" * 60
    )

    # --------------------------------------------------------
    # LOAD LEDGER
    # --------------------------------------------------------

    ledger = load_prediction_ledger(
        ledger_path
    )

    if ledger.empty:

        logger.warning(
            "Prediction ledger is empty."
        )

        monitoring_result = (
            run_production_monitoring()
        )

        alert_sent = send_monitoring_alerts(
            monitoring_result,
            config,
        )

        return {
            "status": "NO_DATA",
            "evaluated_now": 0,
            "monitoring": monitoring_result,
            "telegram_alert_sent": alert_sent,
        }

    # --------------------------------------------------------
    # EVALUATE PENDING PREDICTIONS
    # --------------------------------------------------------

    updated_ledger, stats = (
        evaluate_pending_predictions(
            ledger
        )
    )

    atomic_save_ledger(
        updated_ledger,
        ledger_path,
    )

    # --------------------------------------------------------
    # GET EVALUATED PREDICTIONS
    # --------------------------------------------------------

    evaluated = updated_ledger.loc[
        updated_ledger[
            "evaluation_status"
        ]
        .fillna("")
        .astype(str)
        .str.upper()
        .eq("EVALUATED")
    ].copy()

    recent_window = int(
        config["evaluation"].get(
            "recent_window",
            50,
        )
    )

    if recent_window > 0:

        evaluation_frame = (
            evaluated.tail(
                recent_window
            )
        )

    else:

        evaluation_frame = evaluated

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    metrics = calculate_evaluation_metrics(
        evaluation_frame
    )

    report_path = None

    if bool(
        config["evaluation"].get(
            "save_reports",
            True,
        )
    ):

        report_path = (
            save_evaluation_report(
                metrics,
                reports_path,
            )
        )

    # --------------------------------------------------------
    # DRIFT DETECTION
    # --------------------------------------------------------

    drift_result = None

    if bool(
        config["drift"].get(
            "enabled",
            True,
        )
    ):

        drift_result = (
            run_drift_detection(
                ledger_path
            )
        )

    # --------------------------------------------------------
    # CHAMPION / CHALLENGER
    # --------------------------------------------------------

    comparison_results: list[
        dict[str, Any]
    ] = []

    if bool(
        config[
            "champion_challenger"
        ].get(
            "enabled",
            True,
        )
    ):

        comparison_results = (
            run_champion_challenger_evaluation(
                ledger_path
            )
        )

    # --------------------------------------------------------
    # PRODUCTION MONITORING
    # --------------------------------------------------------

    monitoring_result = (
        run_production_monitoring()
    )

    # --------------------------------------------------------
    # TELEGRAM ALERTS
    # --------------------------------------------------------

    alert_sent = send_monitoring_alerts(
        monitoring_result,
        config,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {
        "status": "SUCCESS",
        "evaluated_now": stats[
            "evaluated"
        ],
        "waiting": stats[
            "waiting"
        ],
        "failed": stats[
            "failed"
        ],
        "total_evaluated": int(
            len(evaluated)
        ),
        "metrics": metrics,
        "report_path": (
            str(report_path)
            if report_path
            else None
        ),
        "drift_result": drift_result,
        "challenger_comparisons": (
            comparison_results
        ),
        "monitoring": monitoring_result,
        "telegram_alert_sent": alert_sent,
    }

    logger.info(
        "Evaluation complete | "
        "New=%s | Waiting=%s | Failed=%s",
        stats["evaluated"],
        stats["waiting"],
        stats["failed"],
    )

    logger.info(
        "System health | "
        "Status=%s | Score=%s",
        monitoring_result.get(
            "health_status",
            "UNKNOWN",
        ),
        monitoring_result.get(
            "health_score",
            "N/A",
        ),
    )

    return summary


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    result = run_evaluation_job()

    print()
    print("=" * 60)
    print("EVALUATION JOB SUMMARY")
    print("=" * 60)

    for key, value in result.items():

        if key in (
            "metrics",
            "monitoring",
        ):
            continue

        if key == "challenger_comparisons":

            print(
                f"{key}: "
                f"{len(value)} comparison(s)"
            )

            continue

        print(
            f"{key}: {value}"
        )

    print()

    metrics = result.get(
        "metrics",
        {},
    )

    if metrics:

        print("METRICS")

        for key in (
            "sample_count",
            "return_mae",
            "direction_accuracy",
            "brier_score",
            "risk_mae",
        ):

            print(
                f"{key}: "
                f"{metrics.get(key)}"
            )

    print()

    monitoring = result.get(
        "monitoring",
        {},
    )

    if monitoring:

        print("SYSTEM HEALTH")

        print(
            "Status: "
            f"{monitoring.get('health_status')}"
        )

        print(
            "Score: "
            f"{monitoring.get('health_score')}"
        )

        alerts = monitoring.get(
            "alerts",
            [],
        )

        print(
            f"Alerts: {len(alerts)}"
        )

        for alert in alerts:

            print(
                f"[{alert.get('level')}] "
                f"{alert.get('message')}"
            )

    print()

    print(
        "Telegram alert sent: "
        f"{result.get('telegram_alert_sent')}"
    )

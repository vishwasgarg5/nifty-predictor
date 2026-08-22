#!/usr/bin/env python3

"""
Production Monitoring System.

Responsibilities
----------------
1. Monitor prediction ledger health.
2. Monitor evaluation metrics.
3. Monitor model registry.
4. Detect stale predictions and stale evaluation data.
5. Calculate a production health score.
6. Generate WARNING and CRITICAL alerts.
7. Persist latest health status.
8. Maintain monitoring history.
9. Automatically update the production circuit breaker.

Output
------
data/monitoring/latest_health.json
data/monitoring/monitoring_history.csv
data/monitoring/circuit_breaker.json
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("monitoring")


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,

    "paths": {
        "ledger": "data/ledger/predictions.csv",
        "reports": "data/reports",
        "registry": "data/model_registry.json",
        "monitoring": "data/monitoring",
    },

    "thresholds": {
        "minimum_direction_accuracy": 0.45,
        "maximum_return_mae": 0.10,
        "maximum_brier_score": 0.30,
        "maximum_risk_mae": 0.10,
        "maximum_pending_predictions": 100,
        "minimum_evaluated_predictions": 10,

        # Stale-data protection
        "maximum_prediction_age_hours": 36,
        "maximum_evaluation_age_hours": 72,
        "maximum_monitoring_age_hours": 48,
    },

    "health": {
        "healthy_score": 80,
        "warning_score": 50,
    },
}


# ============================================================
# CONFIG HELPERS
# ============================================================

def _object_to_dict(
    value: Any,
) -> dict[str, Any]:
    """
    Convert a config-like object to a dictionary.
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


def _deep_merge(
    base: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    """
    Recursively merge configuration dictionaries.
    """

    result = dict(base)

    for key, value in updates.items():

        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(
                result[key],
                value,
            )

        else:
            result[key] = value

    return result


def load_monitoring_config() -> dict[str, Any]:
    """
    Load monitoring configuration from src.config.cfg.

    Falls back to defaults if project configuration
    cannot be imported.
    """

    config = {
        "enabled": DEFAULT_CONFIG["enabled"],

        "paths": dict(
            DEFAULT_CONFIG["paths"]
        ),

        "thresholds": dict(
            DEFAULT_CONFIG["thresholds"]
        ),

        "health": dict(
            DEFAULT_CONFIG["health"]
        ),
    }

    try:
        from src.config import cfg

        monitoring_section = getattr(
            cfg,
            "monitoring",
            None,
        )

        values = _object_to_dict(
            monitoring_section
        )

        if values:
            config = _deep_merge(
                config,
                values,
            )

    except Exception as error:

        logger.warning(
            "Could not load monitoring config. "
            "Using defaults: %s",
            error,
        )

    return config


# ============================================================
# PATH HELPERS
# ============================================================

def resolve_project_path(
    value: str | Path,
) -> Path:
    """
    Resolve a path relative to PROJECT_ROOT.
    """

    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_paths() -> dict[str, Path]:
    """
    Return resolved monitoring paths.
    """

    config = load_monitoring_config()

    paths = config.get(
        "paths",
        {},
    )

    monitoring_dir = resolve_project_path(
        paths.get(
            "monitoring",
            DEFAULT_CONFIG["paths"]["monitoring"],
        )
    )

    return {
        "ledger": resolve_project_path(
            paths.get(
                "ledger",
                DEFAULT_CONFIG["paths"]["ledger"],
            )
        ),

        "reports": resolve_project_path(
            paths.get(
                "reports",
                DEFAULT_CONFIG["paths"]["reports"],
            )
        ),

        "registry": resolve_project_path(
            paths.get(
                "registry",
                DEFAULT_CONFIG["paths"]["registry"],
            )
        ),

        "monitoring": monitoring_dir,

        "latest_health": (
            monitoring_dir
            / "latest_health.json"
        ),

        "history": (
            monitoring_dir
            / "monitoring_history.csv"
        ),
    }


# ============================================================
# TIME HELPERS
# ============================================================

def utc_now() -> datetime:
    """
    Return current UTC time.
    """

    return datetime.now(
        timezone.utc
    )


def utc_now_iso() -> str:
    """
    Return current UTC time as ISO string.
    """

    return utc_now().isoformat()


def parse_datetime(
    value: Any,
) -> datetime | None:
    """
    Safely parse a datetime value.
    """

    if value is None:
        return None

    if isinstance(
        value,
        pd.Timestamp,
    ):

        if pd.isna(value):
            return None

        value = value.to_pydatetime()

    if isinstance(
        value,
        datetime,
    ):

        parsed = value

    else:

        try:
            parsed = pd.to_datetime(
                value,
                errors="coerce",
                utc=True,
            )

            if pd.isna(parsed):
                return None

            parsed = parsed.to_pydatetime()

        except Exception:
            return None

    if parsed.tzinfo is None:

        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.astimezone(
        timezone.utc
    )


def age_hours(
    value: Any,
) -> float | None:
    """
    Calculate age in hours.
    """

    parsed = parse_datetime(
        value
    )

    if parsed is None:
        return None

    age = utc_now() - parsed

    return round(
        age.total_seconds() / 3600,
        2,
    )


# ============================================================
# FILE HELPERS
# ============================================================

def get_file_age_hours(
    path: Path,
) -> float | None:
    """
    Return the age of a file in hours.
    """

    if not path.exists():
        return None

    try:

        modified = datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=timezone.utc,
        )

        age = utc_now() - modified

        return round(
            age.total_seconds() / 3600,
            2,
        )

    except Exception:

        return None


def atomic_write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    """
    Atomically write JSON data.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
            suffix=".tmp",
        ) as temporary:

            json.dump(
                data,
                temporary,
                indent=2,
                default=str,
            )

            temporary_path = Path(
                temporary.name
            )

        os.replace(
            temporary_path,
            path,
        )

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
# LEDGER HELPERS
# ============================================================

def load_ledger(
    path: Path,
) -> pd.DataFrame:
    """
    Load prediction ledger safely.
    """

    if not path.exists():

        logger.warning(
            "Prediction ledger not found: %s",
            path,
        )

        return pd.DataFrame()

    try:

        return pd.read_csv(path)

    except pd.errors.EmptyDataError:

        return pd.DataFrame()

    except Exception as error:

        logger.error(
            "Could not load prediction ledger: %s",
            error,
        )

        return pd.DataFrame()


def find_date_column(
    frame: pd.DataFrame,
) -> str | None:
    """
    Find the most appropriate prediction timestamp column.
    """

    candidates = [
        "prediction_date",
        "market_date",
        "date",
        "created_at",
        "generated_at",
        "timestamp",
    ]

    for column in candidates:

        if column in frame.columns:
            return column

    return None


# ============================================================
# ALERT HELPERS
# ============================================================

def create_alert(
    level: str,
    message: str,
    component: str = "SYSTEM",
) -> dict[str, Any]:
    """
    Create a standard monitoring alert.
    """

    return {
        "level": str(
            level
        ).upper(),

        "component": str(
            component
        ).upper(),

        "message": str(
            message
        ),

        "created_at": utc_now_iso(),
    }


def alert_counts(
    alerts: list[dict[str, Any]],
) -> dict[str, int]:
    """
    Count alerts by severity.
    """

    counts = {
        "INFO": 0,
        "WARNING": 0,
        "CRITICAL": 0,
    }

    for alert in alerts:

        level = str(
            alert.get(
                "level",
                "INFO",
            )
        ).upper()

        if level not in counts:
            level = "INFO"

        counts[level] += 1

    return counts


# ============================================================
# LEDGER MONITORING
# ============================================================

def monitor_ledger() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """
    Monitor prediction ledger health.
    """

    config = load_monitoring_config()

    thresholds = config.get(
        "thresholds",
        {},
    )

    paths = get_paths()

    ledger_path = paths["ledger"]

    alerts: list[
        dict[str, Any]
    ] = []

    result: dict[str, Any] = {
        "path": str(ledger_path),
        "exists": ledger_path.exists(),
        "total_predictions": 0,
        "evaluated": 0,
        "pending": 0,
        "waiting": 0,
        "failed": 0,
        "latest_prediction_at": None,
        "latest_prediction_age_hours": None,
        "file_age_hours": get_file_age_hours(
            ledger_path
        ),
        "stale": False,
    }

    if not ledger_path.exists():

        alerts.append(
            create_alert(
                "CRITICAL",
                "Prediction ledger is missing or unavailable.",
                "LEDGER",
            )
        )

        result["stale"] = True

        return result, alerts

    frame = load_ledger(
        ledger_path
    )

    if frame.empty:

        alerts.append(
            create_alert(
                "WARNING",
                "Prediction ledger exists but contains no records.",
                "LEDGER",
            )
        )

        return result, alerts

    result["total_predictions"] = int(
        len(frame)
    )

    if "evaluation_status" in frame.columns:

        status = (
            frame["evaluation_status"]
            .fillna("PENDING")
            .astype(str)
            .str.upper()
        )

        result["evaluated"] = int(
            status.eq("EVALUATED").sum()
        )

        result["pending"] = int(
            status.isin(
                [
                    "PENDING",
                    "UNEVALUATED",
                ]
            ).sum()
        )

        result["waiting"] = int(
            status.eq("WAITING").sum()
        )

        result["failed"] = int(
            status.eq("FAILED").sum()
        )

    else:

        result["pending"] = int(
            len(frame)
        )

    # --------------------------------------------------------
    # PENDING THRESHOLD
    # --------------------------------------------------------

    maximum_pending = int(
        thresholds.get(
            "maximum_pending_predictions",
            DEFAULT_CONFIG["thresholds"][
                "maximum_pending_predictions"
            ],
        )
    )

    if result["pending"] > maximum_pending:

        alerts.append(
            create_alert(
                "WARNING",
                (
                    "Pending predictions exceed threshold: "
                    f"{result['pending']} > "
                    f"{maximum_pending}."
                ),
                "LEDGER",
            )
        )

    # --------------------------------------------------------
    # FAILED RECORDS
    # --------------------------------------------------------

    if result["failed"] > 0:

        alerts.append(
            create_alert(
                "WARNING",
                (
                    f"{result['failed']} prediction "
                    "record(s) failed evaluation."
                ),
                "LEDGER",
            )
        )

    # --------------------------------------------------------
    # STALE PREDICTION DETECTION
    # --------------------------------------------------------

    date_column = find_date_column(
        frame
    )

    if date_column is not None:

        dates = pd.to_datetime(
            frame[date_column],
            errors="coerce",
            utc=True,
        )

        latest = dates.max()

        if pd.notna(latest):

            latest_datetime = (
                latest.to_pydatetime()
            )

            result[
                "latest_prediction_at"
            ] = latest_datetime.isoformat()

            prediction_age = age_hours(
                latest_datetime
            )

            result[
                "latest_prediction_age_hours"
            ] = prediction_age

            maximum_age = float(
                thresholds.get(
                    "maximum_prediction_age_hours",
                    DEFAULT_CONFIG["thresholds"][
                        "maximum_prediction_age_hours"
                    ],
                )
            )

            if (
                prediction_age is not None
                and prediction_age > maximum_age
            ):

                result["stale"] = True

                alerts.append(
                    create_alert(
                        "CRITICAL",
                        (
                            "Prediction data is stale: "
                            f"{prediction_age:.2f} hours old. "
                            f"Maximum allowed: "
                            f"{maximum_age:.2f} hours."
                        ),
                        "LEDGER",
                    )
                )

    else:

        file_age = result[
            "file_age_hours"
        ]

        maximum_age = float(
            thresholds.get(
                "maximum_prediction_age_hours",
                DEFAULT_CONFIG["thresholds"][
                    "maximum_prediction_age_hours"
                ],
            )
        )

        if (
            file_age is not None
            and file_age > maximum_age
        ):

            result["stale"] = True

            alerts.append(
                create_alert(
                    "CRITICAL",
                    (
                        "Prediction ledger file is stale: "
                        f"{file_age:.2f} hours old."
                    ),
                    "LEDGER",
                )
            )

    return result, alerts


# ============================================================
# EVALUATION REPORT MONITORING
# ============================================================

def monitor_evaluation_report() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """
    Monitor latest evaluation report.
    """

    config = load_monitoring_config()

    thresholds = config.get(
        "thresholds",
        {},
    )

    paths = get_paths()

    report_path = (
        paths["reports"]
        / "latest_evaluation.csv"
    )

    alerts: list[
        dict[str, Any]
    ] = []

    result: dict[str, Any] = {
        "path": str(report_path),
        "exists": report_path.exists(),
        "file_age_hours": get_file_age_hours(
            report_path
        ),
        "stale": False,
        "sample_count": 0,
        "return_mae": None,
        "direction_accuracy": None,
        "brier_score": None,
        "risk_mae": None,
    }

    if not report_path.exists():

        alerts.append(
            create_alert(
                "WARNING",
                "Latest evaluation report does not exist.",
                "EVALUATION",
            )
        )

        return result, alerts

    maximum_age = float(
        thresholds.get(
            "maximum_evaluation_age_hours",
            DEFAULT_CONFIG["thresholds"][
                "maximum_evaluation_age_hours"
            ],
        )
    )

    file_age = result[
        "file_age_hours"
    ]

    if (
        file_age is not None
        and file_age > maximum_age
    ):

        result["stale"] = True

        alerts.append(
            create_alert(
                "CRITICAL",
                (
                    "Evaluation report is stale: "
                    f"{file_age:.2f} hours old. "
                    f"Maximum allowed: "
                    f"{maximum_age:.2f} hours."
                ),
                "EVALUATION",
            )
        )

    try:

        frame = pd.read_csv(
            report_path
        )

    except Exception as error:

        alerts.append(
            create_alert(
                "WARNING",
                (
                    "Could not read evaluation report: "
                    f"{error}"
                ),
                "EVALUATION",
            )
        )

        return result, alerts

    if frame.empty:

        alerts.append(
            create_alert(
                "WARNING",
                "Evaluation report is empty.",
                "EVALUATION",
            )
        )

        return result, alerts

    # --------------------------------------------------------
    # Prefer ALL row
    # --------------------------------------------------------

    row = frame.iloc[0]

    if "model_name" in frame.columns:

        all_rows = frame[
            frame["model_name"]
            .astype(str)
            .str.upper()
            .eq("ALL")
        ]

        if not all_rows.empty:
            row = all_rows.iloc[0]

    def get_numeric(
        column: str,
    ) -> float | None:

        if column not in row.index:
            return None

        value = pd.to_numeric(
            pd.Series(
                [row[column]]
            ),
            errors="coerce",
        ).iloc[0]

        if pd.isna(value):
            return None

        return float(value)

    result["sample_count"] = int(
        get_numeric("sample_count") or 0
    )

    result["return_mae"] = get_numeric(
        "return_mae"
    )

    result["direction_accuracy"] = get_numeric(
        "direction_accuracy"
    )

    result["brier_score"] = get_numeric(
        "brier_score"
    )

    result["risk_mae"] = get_numeric(
        "risk_mae"
    )

    # --------------------------------------------------------
    # MINIMUM SAMPLE COUNT
    # --------------------------------------------------------

    minimum_samples = int(
        thresholds.get(
            "minimum_evaluated_predictions",
            DEFAULT_CONFIG["thresholds"][
                "minimum_evaluated_predictions"
            ],
        )
    )

    if (
        result["sample_count"] > 0
        and result["sample_count"] < minimum_samples
    ):

        alerts.append(
            create_alert(
                "WARNING",
                (
                    "Evaluation sample size is below threshold: "
                    f"{result['sample_count']} < "
                    f"{minimum_samples}."
                ),
                "EVALUATION",
            )
        )

    # --------------------------------------------------------
    # DIRECTION ACCURACY
    # --------------------------------------------------------

    minimum_accuracy = float(
        thresholds.get(
            "minimum_direction_accuracy",
            DEFAULT_CONFIG["thresholds"][
                "minimum_direction_accuracy"
            ],
        )
    )

    accuracy = result[
        "direction_accuracy"
    ]

    if (
        accuracy is not None
        and result["sample_count"] >= minimum_samples
        and accuracy < minimum_accuracy
    ):

        alerts.append(
            create_alert(
                "WARNING",
                (
                    "Direction accuracy below threshold: "
                    f"{accuracy:.4f} < "
                    f"{minimum_accuracy:.4f}."
                ),
                "MODEL",
            )
        )

    # --------------------------------------------------------
    # RETURN MAE
    # --------------------------------------------------------

    maximum_return_mae = float(
        thresholds.get(
            "maximum_return_mae",
            DEFAULT_CONFIG["thresholds"][
                "maximum_return_mae"
            ],
        )
    )

    return_mae = result[
        "return_mae"
    ]

    if (
        return_mae is not None
        and result["sample_count"] >= minimum_samples
        and return_mae > maximum_return_mae
    ):

        alerts.append(
            create_alert(
                "WARNING",
                (
                    "Return MAE exceeds threshold: "
                    f"{return_mae:.4f} > "
                    f"{maximum_return_mae:.4f}."
                ),
                "MODEL",
            )
        )

    # --------------------------------------------------------
    # BRIER SCORE
    # --------------------------------------------------------

    maximum_brier = float(
        thresholds.get(
            "maximum_brier_score",
            DEFAULT_CONFIG["thresholds"][
                "maximum_brier_score"
            ],
        )
    )

    brier = result[
        "brier_score"
    ]

    if (
        brier is not None
        and result["sample_count"] >= minimum_samples
        and brier > maximum_brier
    ):

        alerts.append(
            create_alert(
                "WARNING",
                (
                    "Brier score exceeds threshold: "
                    f"{brier:.4f} > "
                    f"{maximum_brier:.4f}."
                ),
                "MODEL",
            )
        )

    # --------------------------------------------------------
    # RISK MAE
    # --------------------------------------------------------

    maximum_risk_mae = float(
        thresholds.get(
            "maximum_risk_mae",
            DEFAULT_CONFIG["thresholds"][
                "maximum_risk_mae"
            ],
        )
    )

    risk_mae = result[
        "risk_mae"
    ]

    if (
        risk_mae is not None
        and result["sample_count"] >= minimum_samples
        and risk_mae > maximum_risk_mae
    ):

        alerts.append(
            create_alert(
                "WARNING",
                (
                    "Risk MAE exceeds threshold: "
                    f"{risk_mae:.4f} > "
                    f"{maximum_risk_mae:.4f}."
                ),
                "MODEL",
            )
        )

    return result, alerts


# ============================================================
# MODEL REGISTRY MONITORING
# ============================================================

def monitor_model_registry() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """
    Monitor Champion / Challenger model registry.
    """

    paths = get_paths()

    registry_path = paths[
        "registry"
    ]

    alerts: list[
        dict[str, Any]
    ] = []

    result: dict[str, Any] = {
        "path": str(registry_path),
        "exists": registry_path.exists(),
        "champion": None,
        "challengers": 0,
        "valid": False,
    }

    if not registry_path.exists():

        alerts.append(
            create_alert(
                "CRITICAL",
                "Model registry is missing.",
                "MODELS",
            )
        )

        return result, alerts

    try:

        with registry_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            registry = json.load(
                file
            )

    except Exception as error:

        alerts.append(
            create_alert(
                "CRITICAL",
                (
                    "Could not read model registry: "
                    f"{error}"
                ),
                "MODELS",
            )
        )

        return result, alerts

    if not isinstance(
        registry,
        dict,
    ):

        alerts.append(
            create_alert(
                "CRITICAL",
                "Model registry has an invalid structure.",
                "MODELS",
            )
        )

        return result, alerts

    champion = registry.get(
        "champion"
    )

    if isinstance(
        champion,
        dict,
    ):

        champion_name = (
            champion.get("name")
            or champion.get("model_name")
            or champion.get("id")
        )

    else:

        champion_name = champion

    if champion_name is not None:

        champion_name = str(
            champion_name
        )

    result["champion"] = (
        champion_name
    )

    challengers = registry.get(
        "challengers",
        [],
    )

    if isinstance(
        challengers,
        list,
    ):

        result["challengers"] = len(
            challengers
        )

    if not champion_name:

        alerts.append(
            create_alert(
                "CRITICAL",
                "No Champion model is configured.",
                "MODELS",
            )
        )

        return result, alerts

    result["valid"] = True

    return result, alerts


# ============================================================
# STALE MONITORING CHECK
# ============================================================

def monitor_monitoring_freshness() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """
    Check whether the previous monitoring output is stale.

    This protects against a pipeline that silently stops
    running.
    """

    config = load_monitoring_config()

    thresholds = config.get(
        "thresholds",
        {},
    )

    paths = get_paths()

    latest_health_path = paths[
        "latest_health"
    ]

    alerts: list[
        dict[str, Any]
    ] = []

    result: dict[str, Any] = {
        "path": str(
            latest_health_path
        ),
        "exists": latest_health_path.exists(),
        "file_age_hours": get_file_age_hours(
            latest_health_path
        ),
        "stale": False,
    }

    if not latest_health_path.exists():

        return result, alerts

    maximum_age = float(
        thresholds.get(
            "maximum_monitoring_age_hours",
            DEFAULT_CONFIG["thresholds"][
                "maximum_monitoring_age_hours"
            ],
        )
    )

    age = result[
        "file_age_hours"
    ]

    if (
        age is not None
        and age > maximum_age
    ):

        result["stale"] = True

        alerts.append(
            create_alert(
                "WARNING",
                (
                    "Previous monitoring result is stale: "
                    f"{age:.2f} hours old."
                ),
                "MONITORING",
            )
        )

    return result, alerts


# ============================================================
# HEALTH SCORE
# ============================================================

def calculate_health_score(
    alerts: list[dict[str, Any]],
) -> tuple[
    float,
    str,
]:
    """
    Calculate overall production health.

    Scoring:
        Start at 100.

        WARNING  -> -10 points
        CRITICAL -> -30 points

    The score is then mapped to:

        HEALTHY  >= healthy_score
        WARNING  >= warning_score
        CRITICAL < warning_score
    """

    config = load_monitoring_config()

    health_config = config.get(
        "health",
        {},
    )

    score = 100.0

    for alert in alerts:

        level = str(
            alert.get(
                "level",
                "INFO",
            )
        ).upper()

        if level == "CRITICAL":

            score -= 30.0

        elif level == "WARNING":

            score -= 10.0

    score = max(
        0.0,
        min(
            100.0,
            score,
        ),
    )

    healthy_threshold = float(
        health_config.get(
            "healthy_score",
            DEFAULT_CONFIG["health"][
                "healthy_score"
            ],
        )
    )

    warning_threshold = float(
        health_config.get(
            "warning_score",
            DEFAULT_CONFIG["health"][
                "warning_score"
            ],
        )
    )

    if score >= healthy_threshold:

        status = "HEALTHY"

    elif score >= warning_threshold:

        status = "WARNING"

    else:

        status = "CRITICAL"

    # A critical alert always prevents HEALTHY status.
    has_critical = any(
        str(
            alert.get(
                "level",
                "",
            )
        ).upper()
        == "CRITICAL"
        for alert in alerts
    )

    if has_critical and status == "HEALTHY":

        status = "WARNING"

    return round(
        score,
        2,
    ), status


# ============================================================
# CIRCUIT BREAKER INTEGRATION
# ============================================================

def update_circuit_breaker(
    health_score: float,
    health_status: str,
    alerts: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Send monitoring health information to the circuit breaker.

    Monitoring must never fail because of breaker integration,
    so errors are isolated.
    """

    critical_messages = [
        str(
            alert.get(
                "message",
                "Unknown critical failure",
            )
        )
        for alert in alerts
        if str(
            alert.get(
                "level",
                "",
            )
        ).upper()
        == "CRITICAL"
    ]

    if critical_messages:

        reason = " | ".join(
            critical_messages
        )

    else:

        reason = (
            f"Health status: {health_status}, "
            f"score: {health_score}"
        )

    try:

        from src.circuit_breaker import (
            get_status,
            update_from_health,
        )

        update_from_health(
            health_score=health_score,
            health_status=health_status,
            reason=reason,
        )

        return get_status()

    except ImportError as error:

        logger.warning(
            "Circuit breaker module unavailable: %s",
            error,
        )

        return {
            "enabled": False,
            "state": "UNAVAILABLE",
            "predictions_allowed": True,
            "message": (
                "Circuit breaker module unavailable."
            ),
            "error": str(error),
        }

    except Exception as error:

        logger.exception(
            "Circuit breaker update failed: %s",
            error,
        )

        return {
            "enabled": True,
            "state": "ERROR",
            "predictions_allowed": False,
            "message": (
                "Circuit breaker update failed. "
                "Predictions should be blocked for safety."
            ),
            "error": str(error),
        }


# ============================================================
# SAVE MONITORING RESULT
# ============================================================

def save_monitoring_result(
    result: dict[str, Any],
) -> Path:
    """
    Save the latest monitoring result.
    """

    paths = get_paths()

    output = paths[
        "latest_health"
    ]

    atomic_write_json(
        output,
        result,
    )

    return output


def append_monitoring_history(
    result: dict[str, Any],
) -> Path:
    """
    Append a compact monitoring record to CSV history.
    """

    paths = get_paths()

    history_path = paths[
        "history"
    ]

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    alerts = result.get(
        "alerts",
        [],
    )

    counts = alert_counts(
        alerts
    )

    ledger = result.get(
        "ledger",
        {},
    )

    evaluation = result.get(
        "evaluation",
        {},
    )

    models = result.get(
        "models",
        {},
    )

    circuit_breaker = result.get(
        "circuit_breaker",
        {},
    )

    row = pd.DataFrame(
        [
            {
                "generated_at": result.get(
                    "generated_at"
                ),

                "health_status": result.get(
                    "health_status"
                ),

                "health_score": result.get(
                    "health_score"
                ),

                "info_alerts": counts[
                    "INFO"
                ],

                "warning_alerts": counts[
                    "WARNING"
                ],

                "critical_alerts": counts[
                    "CRITICAL"
                ],

                "total_predictions": ledger.get(
                    "total_predictions"
                ),

                "evaluated": ledger.get(
                    "evaluated"
                ),

                "pending": ledger.get(
                    "pending"
                ),

                "ledger_stale": ledger.get(
                    "stale"
                ),

                "evaluation_sample_count": evaluation.get(
                    "sample_count"
                ),

                "direction_accuracy": evaluation.get(
                    "direction_accuracy"
                ),

                "return_mae": evaluation.get(
                    "return_mae"
                ),

                "brier_score": evaluation.get(
                    "brier_score"
                ),

                "risk_mae": evaluation.get(
                    "risk_mae"
                ),

                "evaluation_stale": evaluation.get(
                    "stale"
                ),

                "champion": models.get(
                    "champion"
                ),

                "model_registry_valid": models.get(
                    "valid"
                ),

                "circuit_breaker_state": (
                    circuit_breaker.get(
                        "state"
                    )
                ),

                "predictions_allowed": (
                    circuit_breaker.get(
                        "predictions_allowed"
                    )
                ),
            }
        ]
    )

    try:

        row.to_csv(
            history_path,
            mode="a",
            header=not history_path.exists(),
            index=False,
        )

    except Exception as error:

        logger.error(
            "Could not update monitoring history: %s",
            error,
        )

    return history_path


# ============================================================
# MAIN MONITORING FUNCTION
# ============================================================

def run_monitoring() -> dict[str, Any]:
    """
    Run the complete production monitoring system.
    """

    config = load_monitoring_config()

    if not bool(
        config.get(
            "enabled",
            True,
        )
    ):

        return {
            "status": "DISABLED",
            "generated_at": utc_now_iso(),
            "health_status": "UNKNOWN",
            "health_score": None,
            "alerts": [],
            "circuit_breaker": {
                "state": "DISABLED",
                "predictions_allowed": True,
            },
        }

    logger.info(
        "=" * 60
    )

    logger.info(
        "STARTING PRODUCTION MONITORING"
    )

    logger.info(
        "=" * 60
    )

    all_alerts: list[
        dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # LEDGER
    # --------------------------------------------------------

    ledger_result, ledger_alerts = (
        monitor_ledger()
    )

    all_alerts.extend(
        ledger_alerts
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    evaluation_result, evaluation_alerts = (
        monitor_evaluation_report()
    )

    all_alerts.extend(
        evaluation_alerts
    )

    # --------------------------------------------------------
    # MODEL REGISTRY
    # --------------------------------------------------------

    models_result, models_alerts = (
        monitor_model_registry()
    )

    all_alerts.extend(
        models_alerts
    )

    # --------------------------------------------------------
    # PREVIOUS MONITORING FRESHNESS
    # --------------------------------------------------------

    freshness_result, freshness_alerts = (
        monitor_monitoring_freshness()
    )

    all_alerts.extend(
        freshness_alerts
    )

    # --------------------------------------------------------
    # HEALTH SCORE
    # --------------------------------------------------------

    health_score, health_status = (
        calculate_health_score(
            all_alerts
        )
    )

    # --------------------------------------------------------
    # CIRCUIT BREAKER
    # --------------------------------------------------------

    circuit_breaker_result = (
        update_circuit_breaker(
            health_score=health_score,
            health_status=health_status,
            alerts=all_alerts,
        )
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    result: dict[str, Any] = {
        "status": "SUCCESS",

        "generated_at": utc_now_iso(),

        "health_status": health_status,

        "health_score": health_score,

        "ledger": ledger_result,

        "evaluation": evaluation_result,

        "models": models_result,

        "monitoring_freshness": (
            freshness_result
        ),

        "circuit_breaker": (
            circuit_breaker_result
        ),

        "alerts": all_alerts,

        "alert_counts": alert_counts(
            all_alerts
        ),
    }

    # --------------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------------

    try:

        latest_path = (
            save_monitoring_result(
                result
            )
        )

        result["latest_health_path"] = str(
            latest_path
        )

    except Exception as error:

        logger.exception(
            "Failed to save latest monitoring result: %s",
            error,
        )

        result["alerts"].append(
            create_alert(
                "WARNING",
                (
                    "Could not save latest health report: "
                    f"{error}"
                ),
                "MONITORING",
            )
        )

    try:

        history_path = (
            append_monitoring_history(
                result
            )
        )

        result["history_path"] = str(
            history_path
        )

    except Exception as error:

        logger.exception(
            "Failed to save monitoring history: %s",
            error,
        )

    logger.info(
        "Monitoring complete | "
        "Health=%s | "
        "Score=%s | "
        "Alerts=%s | "
        "Breaker=%s",
        result["health_status"],
        result["health_score"],
        len(result["alerts"]),
        result[
            "circuit_breaker"
        ].get(
            "state",
            "UNKNOWN",
        ),
    )

    return result


# ============================================================
# CLI
# ============================================================

def main() -> int:
    """
    Run monitoring from the command line.
    """

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    result = run_monitoring()

    print()

    print("=" * 60)

    print("PRODUCTION MONITORING")

    print("=" * 60)

    print(
        "Status: "
        f"{result.get('status')}"
    )

    print(
        "Health: "
        f"{result.get('health_status')}"
    )

    print(
        "Score: "
        f"{result.get('health_score')}"
    )

    print()

    ledger = result.get(
        "ledger",
        {},
    )

    print("LEDGER")

    print(
        "Total predictions: "
        f"{ledger.get('total_predictions')}"
    )

    print(
        "Evaluated: "
        f"{ledger.get('evaluated')}"
    )

    print(
        "Pending: "
        f"{ledger.get('pending')}"
    )

    print(
        "Stale: "
        f"{ledger.get('stale')}"
    )

    print()

    evaluation = result.get(
        "evaluation",
        {},
    )

    print("EVALUATION")

    print(
        "Samples: "
        f"{evaluation.get('sample_count')}"
    )

    print(
        "Direction accuracy: "
        f"{evaluation.get('direction_accuracy')}"
    )

    print(
        "Return MAE: "
        f"{evaluation.get('return_mae')}"
    )

    print(
        "Brier score: "
        f"{evaluation.get('brier_score')}"
    )

    print()

    models = result.get(
        "models",
        {},
    )

    print("MODELS")

    print(
        "Champion: "
        f"{models.get('champion')}"
    )

    print(
        "Registry valid: "
        f"{models.get('valid')}"
    )

    print()

    breaker = result.get(
        "circuit_breaker",
        {},
    )

    print("CIRCUIT BREAKER")

    print(
        "State: "
        f"{breaker.get('state')}"
    )

    print(
        "Predictions allowed: "
        f"{breaker.get('predictions_allowed')}"
    )

    print(
        "Reason: "
        f"{breaker.get('reason', breaker.get('message'))}"
    )

    print()

    print("ALERTS")

    alerts = result.get(
        "alerts",
        [],
    )

    if not alerts:

        print(
            "No active alerts."
        )

    else:

        for alert in alerts:

            print(
                f"[{alert.get('level')}] "
                f"[{alert.get('component')}] "
                f"{alert.get('message')}"
            )

    print()

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

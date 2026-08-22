#!/usr/bin/env python3

"""
Production Monitoring and Safety Dashboard.

Responsibilities
----------------
1. Check prediction pipeline health.
2. Analyze prediction ledger status.
3. Analyze model evaluation performance.
4. Check model registry and Champion status.
5. Read drift detection results when available.
6. Generate a system health score.
7. Generate alerts for important failures.
8. Save a JSON and CSV monitoring report.

Run:
    python src/monitoring.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


logger = logging.getLogger("monitoring")


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_CONFIG = {
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
    },

    "health": {
        "healthy_score": 80,
        "warning_score": 50,
    },
}


# ============================================================
# CONFIG LOADING
# ============================================================

def object_to_dict(value: Any) -> dict[str, Any]:
    """Convert configuration objects safely."""

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


def load_monitoring_config() -> dict[str, Any]:
    """Load monitoring configuration."""

    config = {
        "enabled": DEFAULT_CONFIG["enabled"],
        "paths": dict(DEFAULT_CONFIG["paths"]),
        "thresholds": dict(
            DEFAULT_CONFIG["thresholds"]
        ),
        "health": dict(
            DEFAULT_CONFIG["health"]
        ),
    }

    try:
        from src.config import cfg

        monitoring = getattr(
            cfg,
            "monitoring",
            None,
        )

        monitoring_dict = object_to_dict(
            monitoring
        )

        if monitoring_dict:
            for key, value in monitoring_dict.items():

                if (
                    key in config
                    and isinstance(config[key], dict)
                    and isinstance(value, dict)
                ):
                    config[key].update(value)

                else:
                    config[key] = value

    except Exception as error:
        logger.warning(
            "Could not load monitoring config: %s",
            error,
        )

    return config


# ============================================================
# PATH HELPERS
# ============================================================

def resolve_path(
    value: str | Path,
) -> Path:

    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# ============================================================
# LEDGER HEALTH
# ============================================================

def load_ledger(
    ledger_path: str | Path,
) -> pd.DataFrame:

    path = resolve_path(ledger_path)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except Exception as error:

        logger.error(
            "Unable to load ledger: %s",
            error,
        )

        return pd.DataFrame()


def analyze_ledger(
    ledger: pd.DataFrame,
) -> dict[str, Any]:

    if ledger.empty:

        return {
            "exists": False,
            "total_predictions": 0,
            "pending": 0,
            "evaluated": 0,
            "failed": 0,
            "status": "NO_DATA",
        }

    status_column = (
        "evaluation_status"
        if "evaluation_status" in ledger.columns
        else None
    )

    pending = 0
    evaluated = 0
    failed = 0

    if status_column:

        status = (
            ledger[status_column]
            .fillna("PENDING")
            .astype(str)
            .str.upper()
        )

        pending = int(
            status.isin(
                [
                    "PENDING",
                    "WAITING",
                    "UNEVALUATED",
                ]
            ).sum()
        )

        evaluated = int(
            status.eq("EVALUATED").sum()
        )

        failed = int(
            status.eq("FAILED").sum()
        )

    return {
        "exists": True,
        "total_predictions": int(
            len(ledger)
        ),
        "pending": pending,
        "evaluated": evaluated,
        "failed": failed,
        "status": "OK",
    }


# ============================================================
# EVALUATION METRICS
# ============================================================

def load_latest_evaluation(
    reports_path: str | Path,
) -> dict[str, Any]:

    path = (
        resolve_path(reports_path)
        / "latest_evaluation.csv"
    )

    if not path.exists():
        return {}

    try:

        frame = pd.read_csv(path)

        if frame.empty:
            return {}

        if "model_name" in frame.columns:

            all_models = frame[
                frame["model_name"]
                .astype(str)
                .eq("ALL")
            ]

            if not all_models.empty:

                row = all_models.iloc[0]

            else:

                row = frame.iloc[0]

        else:

            row = frame.iloc[0]

        result = {}

        for column in row.index:

            value = row[column]

            if pd.isna(value):
                result[column] = None

            elif isinstance(
                value,
                np.generic,
            ):
                result[column] = value.item()

            else:
                result[column] = value

        return result

    except Exception as error:

        logger.error(
            "Unable to load evaluation report: %s",
            error,
        )

        return {}


# ============================================================
# MODEL REGISTRY
# ============================================================

def load_model_registry(
    registry_path: str | Path,
) -> dict[str, Any]:

    path = resolve_path(registry_path)

    if not path.exists():

        return {
            "champion": None,
            "challengers": [],
            "history": [],
        }

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            registry = json.load(file)

        registry.setdefault(
            "champion",
            None,
        )

        registry.setdefault(
            "challengers",
            [],
        )

        registry.setdefault(
            "history",
            [],
        )

        return registry

    except Exception as error:

        logger.error(
            "Unable to load model registry: %s",
            error,
        )

        return {
            "champion": None,
            "challengers": [],
            "history": [],
        }


def analyze_model_registry(
    registry: dict[str, Any],
) -> dict[str, Any]:

    champion = registry.get(
        "champion"
    )

    if isinstance(
        champion,
        dict,
    ):
        champion_name = champion.get(
            "model_name"
        )
        champion_status = champion.get(
            "status"
        )
    else:
        champion_name = champion
        champion_status = None

    challengers = registry.get(
        "challengers",
        [],
    )

    active_challengers = []

    for challenger in challengers:

        if not isinstance(
            challenger,
            dict,
        ):
            continue

        status = str(
            challenger.get(
                "status",
                "CHALLENGER",
            )
        ).upper()

        if status in (
            "CHALLENGER",
            "ACTIVE",
            "EVALUATING",
        ):

            active_challengers.append(
                challenger.get(
                    "model_name"
                )
            )

    return {
        "champion": champion_name,
        "champion_status": champion_status,
        "challenger_count": len(
            active_challengers
        ),
        "challengers": active_challengers,
    }


# ============================================================
# DRIFT STATUS
# ============================================================

def find_drift_report(
    reports_path: str | Path,
) -> dict[str, Any]:

    path = resolve_path(
        reports_path
    )

    candidates = [
        "latest_drift.json",
        "drift_report.json",
        "latest_drift.csv",
    ]

    for filename in candidates:

        file_path = path / filename

        if not file_path.exists():
            continue

        try:

            if file_path.suffix == ".json":

                with open(
                    file_path,
                    "r",
                    encoding="utf-8",
                ) as file:

                    return json.load(file)

            if file_path.suffix == ".csv":

                frame = pd.read_csv(
                    file_path
                )

                if not frame.empty:

                    return (
                        frame.iloc[0]
                        .replace(
                            {
                                np.nan: None,
                            }
                        )
                        .to_dict()
                    )

        except Exception as error:

            logger.warning(
                "Unable to read drift report %s: %s",
                file_path,
                error,
            )

    return {
        "status": "UNKNOWN",
    }


# ============================================================
# ALERT ENGINE
# ============================================================

def generate_alerts(
    ledger_stats: dict[str, Any],
    metrics: dict[str, Any],
    model_stats: dict[str, Any],
    drift: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, str]]:

    alerts = []

    thresholds = config.get(
        "thresholds",
        {},
    )

    # Ledger missing

    if not ledger_stats.get(
        "exists",
        False,
    ):

        alerts.append(
            {
                "level": "CRITICAL",
                "message": (
                    "Prediction ledger is missing "
                    "or unavailable."
                ),
            }
        )

    # Too many pending

    max_pending = thresholds.get(
        "maximum_pending_predictions",
        100,
    )

    if (
        ledger_stats.get(
            "pending",
            0,
        )
        > max_pending
    ):

        alerts.append(
            {
                "level": "WARNING",
                "message": (
                    f"Too many pending predictions: "
                    f"{ledger_stats['pending']}"
                ),
            }
        )

    # Not enough evaluation data

    min_evaluated = thresholds.get(
        "minimum_evaluated_predictions",
        10,
    )

    if (
        ledger_stats.get(
            "evaluated",
            0,
        )
        < min_evaluated
    ):

        alerts.append(
            {
                "level": "INFO",
                "message": (
                    "Not enough evaluated predictions "
                    "for reliable monitoring."
                ),
            }
        )

    # Direction accuracy

    accuracy = metrics.get(
        "direction_accuracy"
    )

    min_accuracy = thresholds.get(
        "minimum_direction_accuracy",
        0.45,
    )

    if (
        accuracy is not None
        and float(accuracy)
        < float(min_accuracy)
    ):

        alerts.append(
            {
                "level": "WARNING",
                "message": (
                    f"Direction accuracy below threshold: "
                    f"{float(accuracy):.2%}"
                ),
            }
        )

    # Return MAE

    return_mae = metrics.get(
        "return_mae"
    )

    max_return_mae = thresholds.get(
        "maximum_return_mae",
        0.10,
    )

    if (
        return_mae is not None
        and float(return_mae)
        > float(max_return_mae)
    ):

        alerts.append(
            {
                "level": "WARNING",
                "message": (
                    f"Return MAE too high: "
                    f"{float(return_mae):.4f}"
                ),
            }
        )

    # Brier score

    brier_score = metrics.get(
        "brier_score"
    )

    max_brier = thresholds.get(
        "maximum_brier_score",
        0.30,
    )

    if (
        brier_score is not None
        and float(brier_score)
        > float(max_brier)
    ):

        alerts.append(
            {
                "level": "WARNING",
                "message": (
                    f"Brier score too high: "
                    f"{float(brier_score):.4f}"
                ),
            }
        )

    # Risk MAE

    risk_mae = metrics.get(
        "risk_mae"
    )

    max_risk_mae = thresholds.get(
        "maximum_risk_mae",
        0.10,
    )

    if (
        risk_mae is not None
        and float(risk_mae)
        > float(max_risk_mae)
    ):

        alerts.append(
            {
                "level": "WARNING",
                "message": (
                    f"Risk MAE too high: "
                    f"{float(risk_mae):.4f}"
                ),
            }
        )

    # Champion missing

    if not model_stats.get(
        "champion"
    ):

        alerts.append(
            {
                "level": "CRITICAL",
                "message": (
                    "No Champion model is configured."
                ),
            }
        )

    # Drift

    drift_status = str(
        drift.get(
            "status",
            drift.get(
                "drift_status",
                "UNKNOWN",
            ),
        )
    ).upper()

    if drift_status in (
        "CRITICAL",
        "SEVERE",
        "HIGH",
    ):

        alerts.append(
            {
                "level": "CRITICAL",
                "message": (
                    f"Model drift detected: "
                    f"{drift_status}"
                ),
            }
        )

    elif drift_status in (
        "WARNING",
        "MODERATE",
        "MEDIUM",
    ):

        alerts.append(
            {
                "level": "WARNING",
                "message": (
                    f"Possible model drift: "
                    f"{drift_status}"
                ),
            }
        )

    return alerts


# ============================================================
# HEALTH SCORE
# ============================================================

def calculate_health_score(
    alerts: list[dict[str, str]],
) -> tuple[int, str]:

    score = 100

    for alert in alerts:

        level = alert.get(
            "level",
            "INFO",
        ).upper()

        if level == "CRITICAL":

            score -= 35

        elif level == "WARNING":

            score -= 15

        elif level == "INFO":

            score -= 3

    score = max(
        0,
        min(
            100,
            score,
        ),
    )

    if score >= 80:

        status = "HEALTHY"

    elif score >= 50:

        status = "WARNING"

    else:

        status = "CRITICAL"

    return score, status


# ============================================================
# REPORT WRITING
# ============================================================

def save_monitoring_report(
    report: dict[str, Any],
    monitoring_path: str | Path,
) -> tuple[Path, Path]:

    directory = resolve_path(
        monitoring_path
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    json_path = (
        directory
        / f"health_{timestamp}.json"
    )

    latest_json_path = (
        directory
        / "latest_health.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            default=str,
        )

    with open(
        latest_json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            default=str,
        )

    summary_row = {
        "generated_at": report.get(
            "generated_at"
        ),
        "health_score": report.get(
            "health_score"
        ),
        "health_status": report.get(
            "health_status"
        ),
        "total_predictions": report[
            "ledger"
        ].get(
            "total_predictions"
        ),
        "pending": report[
            "ledger"
        ].get(
            "pending"
        ),
        "evaluated": report[
            "ledger"
        ].get(
            "evaluated"
        ),
        "failed": report[
            "ledger"
        ].get(
            "failed"
        ),
        "champion": report[
            "models"
        ].get(
            "champion"
        ),
        "challenger_count": report[
            "models"
        ].get(
            "challenger_count"
        ),
        "alert_count": len(
            report.get(
                "alerts",
                [],
            )
        ),
    }

    csv_path = (
        directory
        / "latest_health.csv"
    )

    pd.DataFrame(
        [summary_row]
    ).to_csv(
        csv_path,
        index=False,
    )

    return latest_json_path, csv_path


# ============================================================
# MAIN MONITORING ENGINE
# ============================================================

def run_monitoring() -> dict[str, Any]:

    config = load_monitoring_config()

    if not bool(
        config.get(
            "enabled",
            True,
        )
    ):

        return {
            "status": "DISABLED",
        }

    paths = config.get(
        "paths",
        {},
    )

    # Load data

    ledger = load_ledger(
        paths.get(
            "ledger",
            "data/ledger/predictions.csv",
        )
    )

    ledger_stats = analyze_ledger(
        ledger
    )

    metrics = load_latest_evaluation(
        paths.get(
            "reports",
            "data/reports",
        )
    )

    registry = load_model_registry(
        paths.get(
            "registry",
            "data/model_registry.json",
        )
    )

    model_stats = analyze_model_registry(
        registry
    )

    drift = find_drift_report(
        paths.get(
            "reports",
            "data/reports",
        )
    )

    # Generate alerts

    alerts = generate_alerts(
        ledger_stats=ledger_stats,
        metrics=metrics,
        model_stats=model_stats,
        drift=drift,
        config=config,
    )

    health_score, health_status = (
        calculate_health_score(
            alerts
        )
    )

    report = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "health_score": health_score,

        "health_status": health_status,

        "ledger": ledger_stats,

        "metrics": metrics,

        "models": model_stats,

        "drift": drift,

        "alerts": alerts,
    }

    json_path, csv_path = (
        save_monitoring_report(
            report,
            paths.get(
                "monitoring",
                "data/monitoring",
            ),
        )
    )

    report["json_report"] = str(
        json_path
    )

    report["csv_report"] = str(
        csv_path
    )

    logger.info(
        "Monitoring complete | "
        "Health=%s | Score=%s | Alerts=%s",
        health_status,
        health_score,
        len(alerts),
    )

    return report


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

    result = run_monitoring()

    print()
    print("=" * 60)
    print("SYSTEM HEALTH REPORT")
    print("=" * 60)

    print(
        f"Status: {result.get('health_status')}"
    )

    print(
        f"Score: {result.get('health_score')}/100"
    )

    print()

    print("ALERTS")

    alerts = result.get(
        "alerts",
        [],
    )

    if not alerts:
        print("No alerts.")

    else:

        for alert in alerts:

            print(
                f"[{alert['level']}] "
                f"{alert['message']}"
            )

    print()

    print(
        f"JSON: {result.get('json_report')}"
    )

    print(
        f"CSV: {result.get('csv_report')}"
    )

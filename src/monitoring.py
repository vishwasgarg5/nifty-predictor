#!/usr/bin/env python3

"""
Production Monitoring.

This module combines production health checks, model evaluation,
and drift detection into a single monitoring result.

Pipeline
--------
Prediction Ledger
       │
       ▼
Actual Outcome Evaluation
       │
       ▼
Model Evaluation
       │
       ▼
Drift Detection
       │
       ▼
Production Monitoring
       │
       ├── Data Health
       ├── Model Performance
       ├── Drift Status
       └── Pipeline Health
       │
       ▼
Health Score
       │
       ▼
NORMAL / WARNING / CRITICAL
       │
       ▼
Circuit Breaker

Public API
----------
run_monitoring()

Compatibility aliases:
    monitor()
    get_system_health()
"""

from __future__ import annotations

import logging
import sys
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

logger = logging.getLogger("monitoring")


# ============================================================
# TIME HELPERS
# ============================================================

def utc_now_iso() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


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
    """Load the project configuration."""

    try:

        from src.config import cfg

        return cfg

    except Exception as error:

        logger.warning(
            "Could not load configuration: %s",
            error,
        )

        return None


def get_monitoring_config() -> dict[str, Any]:
    """
    Get monitoring configuration.

    Supported config:

        monitoring:
            enabled: true

            healthy_score: 80
            warning_score: 60
            critical_score: 40

            insufficient_data_penalty: 0
            warning_drift_penalty: 20
            critical_drift_penalty: 50

            model_warning_penalty: 15
            model_critical_penalty: 40

            pipeline_error_penalty: 50
    """

    defaults = {
        "enabled": True,

        "healthy_score": 80,
        "warning_score": 60,
        "critical_score": 40,

        "insufficient_data_penalty": 0,
        "warning_drift_penalty": 20,
        "critical_drift_penalty": 50,

        "model_warning_penalty": 15,
        "model_critical_penalty": 40,

        "pipeline_error_penalty": 50,
    }

    cfg = load_config()

    if cfg is None:
        return defaults

    section = getattr(
        cfg,
        "monitoring",
        None,
    )

    values = object_to_dict(
        section
    )

    result = defaults.copy()

    for key in defaults:

        if key not in values:
            continue

        value = values[key]

        if key == "enabled":

            result[key] = bool(value)

            continue

        try:

            result[key] = float(
                value
            )

        except Exception:

            logger.warning(
                "Invalid monitoring config "
                "for %s: %s",
                key,
                value,
            )

    return result


# ============================================================
# LEDGER
# ============================================================

def get_ledger_path() -> Path:
    """
    Find the prediction ledger path.
    """

    cfg = load_config()

    candidates: list[Any] = []

    if cfg is not None:

        for section_name in [
            "data",
            "paths",
        ]:

            section = getattr(
                cfg,
                section_name,
                None,
            )

            values = object_to_dict(
                section
            )

            for key in [
                "prediction_ledger",
                "ledger",
            ]:

                if values.get(key):

                    candidates.append(
                        values[key]
                    )

    for candidate in candidates:

        path = Path(
            str(candidate)
        )

        if not path.is_absolute():

            path = (
                PROJECT_ROOT
                / path
            )

        return path

    return (
        PROJECT_ROOT
        / "data"
        / "ledger"
        / "predictions.csv"
    )


def load_prediction_ledger() -> pd.DataFrame:
    """
    Load the prediction ledger.
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

        logger.info(
            "Loaded %s ledger record(s).",
            len(frame),
        )

        return frame

    except Exception as error:

        logger.error(
            "Could not load prediction ledger: %s",
            error,
        )

        raise


# ============================================================
# MODEL EVALUATION
# ============================================================

def run_model_evaluation(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run model evaluation.

    Tries available project APIs without breaking
    the production monitoring pipeline.
    """

    try:

        from src.model_evaluation import (
            evaluate_predictions,
        )

        result = evaluate_predictions(
            predictions
        )

        if isinstance(result, dict):

            return result

        return {
            "status": "UNKNOWN",
            "error": (
                "Model evaluation returned "
                "an unexpected result."
            ),
        }

    except ImportError:

        logger.warning(
            "src.model_evaluation not available."
        )

        return {
            "status": "UNAVAILABLE",
            "error": (
                "Model evaluation module "
                "not available."
            ),
        }

    except Exception as error:

        logger.exception(
            "Model evaluation failed."
        )

        return {
            "status": "ERROR",
            "error": str(error),
        }


# ============================================================
# DRIFT DETECTION
# ============================================================

def run_drift_check(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run production drift detection.
    """

    try:

        from src.drift_detection import (
            detect_drift,
        )

        logger.info(
            "Running drift detection."
        )

        result = detect_drift(
            predictions
        )

        if not isinstance(
            result,
            dict,
        ):

            return {
                "status": "ERROR",
                "drift_detected": True,
                "error": (
                    "Drift detection returned "
                    "an unexpected result."
                ),
            }

        return result

    except ImportError as error:

        logger.warning(
            "Drift detection module unavailable: %s",
            error,
        )

        return {
            "status": "UNAVAILABLE",
            "drift_detected": False,
            "error": str(error),
        }

    except Exception as error:

        logger.exception(
            "Drift detection failed."
        )

        # Fail-safe: an unknown drift failure is treated
        # as a monitoring error.
        return {
            "status": "ERROR",
            "drift_detected": True,
            "error": str(error),
        }


# ============================================================
# STATUS NORMALIZATION
# ============================================================

def normalize_status(
    value: Any,
) -> str:
    """Normalize health status."""

    if value is None:
        return "UNKNOWN"

    return str(
        value
    ).strip().upper()


def determine_model_health(
    evaluation: dict[str, Any],
) -> str:
    """
    Extract a model health status from the
    evaluation result.
    """

    if not evaluation:

        return "UNKNOWN"

    for key in [
        "health_status",
        "status",
        "model_status",
    ]:

        value = evaluation.get(
            key
        )

        if value is None:
            continue

        status = normalize_status(
            value
        )

        if status in {
            "NORMAL",
            "HEALTHY",
            "WARNING",
            "CRITICAL",
            "ERROR",
        }:

            return status

    return "UNKNOWN"


# ============================================================
# HEALTH SCORE
# ============================================================

def calculate_health_score(
    drift: dict[str, Any],
    evaluation: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate production health score.

    Starts at 100 and applies penalties for:

        - Model performance degradation
        - Drift warnings
        - Critical drift
        - Monitoring failures
    """

    score = 100.0

    penalties: list[
        dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # DRIFT
    # --------------------------------------------------------

    drift_status = normalize_status(
        drift.get(
            "status"
        )
    )

    if drift_status == "CRITICAL":

        penalty = config[
            "critical_drift_penalty"
        ]

        score -= penalty

        penalties.append(
            {
                "source": "drift",
                "severity": "CRITICAL",
                "penalty": penalty,
            }
        )

    elif drift_status == "WARNING":

        penalty = config[
            "warning_drift_penalty"
        ]

        score -= penalty

        penalties.append(
            {
                "source": "drift",
                "severity": "WARNING",
                "penalty": penalty,
            }
        )

    elif drift_status == "INSUFFICIENT_DATA":

        penalty = config[
            "insufficient_data_penalty"
        ]

        score -= penalty

        if penalty > 0:

            penalties.append(
                {
                    "source": "drift",
                    "severity": (
                        "INSUFFICIENT_DATA"
                    ),
                    "penalty": penalty,
                }
            )

    elif drift_status == "ERROR":

        penalty = config[
            "pipeline_error_penalty"
        ]

        score -= penalty

        penalties.append(
            {
                "source": "drift",
                "severity": "ERROR",
                "penalty": penalty,
            }
        )

    # --------------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------------

    model_status = (
        determine_model_health(
            evaluation
        )
    )

    if model_status in {
        "CRITICAL",
        "ERROR",
    }:

        penalty = config[
            "model_critical_penalty"
        ]

        score -= penalty

        penalties.append(
            {
                "source": "model",
                "severity": model_status,
                "penalty": penalty,
            }
        )

    elif model_status == "WARNING":

        penalty = config[
            "model_warning_penalty"
        ]

        score -= penalty

        penalties.append(
            {
                "source": "model",
                "severity": "WARNING",
                "penalty": penalty,
            }
        )

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    score = max(
        0.0,
        min(
            100.0,
            score,
        ),
    )

    return {
        "score": round(
            score,
            2,
        ),
        "penalties": penalties,
        "drift_status": drift_status,
        "model_status": model_status,
    }


# ============================================================
# HEALTH STATUS
# ============================================================

def determine_health_status(
    score: float,
    drift: dict[str, Any],
    config: dict[str, Any],
) -> str:
    """
    Convert health score into health status.

    Critical drift always overrides the numeric score.
    """

    drift_status = normalize_status(
        drift.get(
            "status"
        )
    )

    if drift_status == "CRITICAL":

        return "CRITICAL"

    if drift_status == "ERROR":

        return "CRITICAL"

    if score >= config[
        "healthy_score"
    ]:

        return "HEALTHY"

    if score >= config[
        "warning_score"
    ]:

        return "WARNING"

    return "CRITICAL"


# ============================================================
# CIRCUIT BREAKER REGISTRATION
# ============================================================

def update_circuit_breaker(
    health_status: str,
    health_score: float,
    drift: dict[str, Any],
) -> dict[str, Any]:
    """
    Register monitoring results with the circuit breaker.

    The monitoring module does not directly decide whether
    Telegram is sent. The circuit breaker remains the final
    authority.
    """

    try:

        from src.circuit_breaker import (
            register_failure,
        )

        if health_status == "CRITICAL":

            drift_status = normalize_status(
                drift.get(
                    "status"
                )
            )

            reason = (
                "Production monitoring "
                f"critical. Drift status={drift_status}."
            )

            register_failure(
                reason=reason,
                health_score=health_score,
                health_status=health_status,
            )

            return {
                "updated": True,
                "action": (
                    "FAILURE_REGISTERED"
                ),
                "reason": reason,
            }

        return {
            "updated": False,
            "action": "NO_FAILURE",
        }

    except ImportError:

        return {
            "updated": False,
            "action": "UNAVAILABLE",
            "error": (
                "Circuit breaker module "
                "not available."
            ),
        }

    except Exception as error:

        logger.exception(
            "Could not update circuit breaker."
        )

        return {
            "updated": False,
            "action": "ERROR",
            "error": str(error),
        }


# ============================================================
# MAIN MONITORING
# ============================================================

def run_monitoring() -> dict[str, Any]:
    """
    Run complete production monitoring.

    Returns a dictionary compatible with:

        scripts/morning_job.py
        src.circuit_breaker.py
    """

    started_at = utc_now_iso()

    config = get_monitoring_config()

    logger.info(
        "=" * 70
    )

    logger.info(
        "STARTING PRODUCTION MONITORING"
    )

    logger.info(
        "=" * 70
    )

    result: dict[str, Any] = {
        "status": "STARTED",
        "started_at": started_at,
        "health_status": "UNKNOWN",
        "health_score": None,
        "prediction_count": 0,
        "evaluation": {},
        "drift": {},
        "penalties": [],
        "circuit_breaker_update": {},
        "error": None,
    }

    if not config.get(
        "enabled",
        True,
    ):

        result.update(
            {
                "status": "DISABLED",
                "health_status": "UNKNOWN",
                "health_score": 100.0,
                "finished_at": utc_now_iso(),
            }
        )

        return result

    try:

        # ----------------------------------------------------
        # STEP 1: LOAD LEDGER
        # ----------------------------------------------------

        logger.info(
            "Step 1: Loading prediction ledger."
        )

        predictions = (
            load_prediction_ledger()
        )

        result[
            "prediction_count"
        ] = len(
            predictions
        )

        # ----------------------------------------------------
        # STEP 2: MODEL EVALUATION
        # ----------------------------------------------------

        logger.info(
            "Step 2: Running model evaluation."
        )

        evaluation = (
            run_model_evaluation(
                predictions
            )
        )

        result[
            "evaluation"
        ] = evaluation

        # ----------------------------------------------------
        # STEP 3: DRIFT DETECTION
        # ----------------------------------------------------

        logger.info(
            "Step 3: Running drift detection."
        )

        drift = run_drift_check(
            predictions
        )

        result[
            "drift"
        ] = drift

        # ----------------------------------------------------
        # STEP 4: HEALTH SCORE
        # ----------------------------------------------------

        logger.info(
            "Step 4: Calculating health score."
        )

        health = (
            calculate_health_score(
                drift=drift,
                evaluation=evaluation,
                config=config,
            )
        )

        health_score = health[
            "score"
        ]

        result[
            "health_score"
        ] = health_score

        result[
            "penalties"
        ] = health[
            "penalties"
        ]

        # ----------------------------------------------------
        # STEP 5: HEALTH STATUS
        # ----------------------------------------------------

        health_status = (
            determine_health_status(
                score=health_score,
                drift=drift,
                config=config,
            )
        )

        result[
            "health_status"
        ] = health_status

        result[
            "status"
        ] = health_status

        # ----------------------------------------------------
        # STEP 6: CIRCUIT BREAKER
        # ----------------------------------------------------

        logger.info(
            "Step 5: Updating circuit breaker."
        )

        breaker_update = (
            update_circuit_breaker(
                health_status=health_status,
                health_score=health_score,
                drift=drift,
            )
        )

        result[
            "circuit_breaker_update"
        ] = breaker_update

        logger.info(
            "Monitoring complete | "
            "health_status=%s | "
            "health_score=%s | "
            "drift=%s",
            health_status,
            health_score,
            drift.get(
                "status"
            ),
        )

        return result

    except Exception as error:

        logger.exception(
            "Production monitoring failed."
        )

        result.update(
            {
                "status": "ERROR",
                "health_status": "CRITICAL",
                "health_score": 0.0,
                "error": str(error),
            }
        )

        return result

    finally:

        result[
            "finished_at"
        ] = utc_now_iso()

        logger.info(
            "=" * 70
        )

        logger.info(
            "PRODUCTION MONITORING FINISHED | "
            "STATUS=%s",
            result.get(
                "status"
            ),
        )

        logger.info(
            "=" * 70
        )


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

def monitor() -> dict[str, Any]:
    """Compatibility alias."""

    return run_monitoring()


def get_system_health() -> dict[str, Any]:
    """Compatibility alias."""

    return run_monitoring()


# ============================================================
# CLI
# ============================================================

def main() -> int:
    """Run production monitoring."""

    result = run_monitoring()

    print()

    print("=" * 70)

    print("PRODUCTION MONITORING")

    print("=" * 70)

    print(
        f"Status: "
        f"{result.get('status')}"
    )

    print(
        f"Health Status: "
        f"{result.get('health_status')}"
    )

    print(
        f"Health Score: "
        f"{result.get('health_score')}"
    )

    print(
        f"Prediction Count: "
        f"{result.get('prediction_count')}"
    )

    drift = result.get(
        "drift",
        {},
    )

    if drift:

        print()

        print(
            "Drift Status: "
            f"{drift.get('status')}"
        )

        print(
            "Drift Detected: "
            f"{drift.get('drift_detected')}"
        )

        signals = drift.get(
            "signals",
            [],
        )

        if signals:

            print()

            print(
                "Drift Signals:"
            )

            for signal in signals:

                print(
                    f"- "
                    f"[{signal.get('severity')}] "
                    f"{signal.get('message')}"
                )

    penalties = result.get(
        "penalties",
        [],
    )

    if penalties:

        print()

        print(
            "Health Penalties:"
        )

        for penalty in penalties:

            print(
                f"- "
                f"{penalty.get('source')} | "
                f"{penalty.get('severity')} | "
                f"-{penalty.get('penalty')}"
            )

    breaker = result.get(
        "circuit_breaker_update",
        {},
    )

    if breaker:

        print()

        print(
            "Circuit Breaker Update:"
        )

        print(
            f"  Action: "
            f"{breaker.get('action')}"
        )

    if result.get(
        "error"
    ):

        print()

        print(
            f"Error: "
            f"{result.get('error')}"
        )

    return (
        0
        if result.get(
            "status"
        )
        in {
            "HEALTHY",
            "WARNING",
            "CRITICAL",
            "DISABLED",
        }
        else 1
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

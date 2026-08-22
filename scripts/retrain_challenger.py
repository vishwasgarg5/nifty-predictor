#!/usr/bin/env python3

"""Automatic challenger retraining orchestration.

Pipeline:

    Drift Detection
          │
          ▼
    Check Severity
          │
          ▼
    Retraining Required?
          │
     ┌────┴─────┐
     │          │
    NO         YES
     │          │
     ▼          ▼
   Exit    Train Challenger
                 │
                 ▼
          Save Challenger Model
                 │
                 ▼
          Update Model Registry
                 │
                 ▼
          Ready for Parallel
             Evaluation

This script does not automatically replace the Champion.
Promotion is handled separately after sufficient evaluated
predictions are collected and compare_models.py recommends it.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import traceback

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.config import cfg


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
    __name__
)


# ============================================================
# CONFIG HELPERS
# ============================================================

def config_to_dict(
    value: Any,
) -> dict[str, Any]:
    """Convert config section to dictionary."""

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
        try:
            return {
                key: item
                for key, item in value.__dict__.items()
                if not key.startswith("_")
            }
        except Exception:
            pass

    return {}


def get_config_section(
    name: str,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Safely retrieve config section."""

    if default is None:
        default = {}

    try:
        section = getattr(
            cfg,
            name,
            None,
        )

        result = config_to_dict(
            section
        )

        if result:
            return result

    except Exception as error:

        logger.warning(
            "Unable to read config section '%s': %s",
            name,
            error,
        )

    return dict(default)


def get_config_value(
    section: str,
    key: str,
    default: Any = None,
) -> Any:
    """Safely retrieve one config value."""

    try:

        config_section = getattr(
            cfg,
            section,
            None,
        )

        if config_section is None:
            return default

        if isinstance(
            config_section,
            dict,
        ):
            return config_section.get(
                key,
                default,
            )

        return getattr(
            config_section,
            key,
            default,
        )

    except Exception:
        return default


# ============================================================
# PATH HELPERS
# ============================================================

def resolve_project_path(
    value: str | Path,
) -> Path:
    """Resolve a path relative to project root."""

    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_reports_dir() -> Path:
    """Get reports directory."""

    path = get_config_value(
        "paths",
        "reports",
        "data/reports",
    )

    directory = resolve_project_path(
        path
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def get_models_dir() -> Path:
    """Get models directory."""

    path = get_config_value(
        "paths",
        "models",
        "models",
    )

    directory = resolve_project_path(
        path
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


# ============================================================
# DRIFT REPORT
# ============================================================

def find_latest_drift_report() -> Path | None:
    """Find the latest drift JSON report."""

    reports_dir = get_reports_dir()

    reports = sorted(
        reports_dir.glob(
            "drift_report_*.json"
        ),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )

    if not reports:
        return None

    return reports[0]


def load_drift_report() -> dict[str, Any] | None:
    """Load latest drift report."""

    report_path = find_latest_drift_report()

    if report_path is None:

        logger.warning(
            "No drift report found."
        )

        return None

    try:

        with open(
            report_path,
            "r",
            encoding="utf-8",
        ) as file:

            payload = json.load(
                file
            )

        logger.info(
            "Loaded drift report: %s",
            report_path,
        )

        return payload

    except Exception as error:

        logger.error(
            "Failed to load drift report: %s",
            error,
        )

        return None


def extract_drift_result(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Extract drift result from report."""

    if not report:
        return {}

    result = report.get(
        "result"
    )

    if isinstance(result, dict):
        return result

    return report


# ============================================================
# RETRAIN DECISION
# ============================================================

def should_retrain(
    drift_result: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[bool, str]:
    """Determine whether retraining should run."""

    if not bool(
        settings.get(
            "enabled",
            True,
        )
    ):
        return (
            False,
            "Automatic retraining is disabled.",
        )

    severity = str(
        drift_result.get(
            "severity",
            "UNKNOWN",
        )
    ).upper()

    drift_detected = bool(
        drift_result.get(
            "drift_detected",
            False,
        )
    )

    retrain_on_warning = bool(
        settings.get(
            "retrain_on_warning",
            False,
        )
    )

    retrain_on_drift = bool(
        settings.get(
            "retrain_on_drift",
            True,
        )
    )

    retrain_on_critical = bool(
        settings.get(
            "retrain_on_critical",
            True,
        )
    )

    if (
        severity == "CRITICAL"
        and retrain_on_critical
    ):
        return (
            True,
            "Critical model drift detected.",
        )

    if (
        severity == "DRIFT"
        and retrain_on_drift
    ):
        return (
            True,
            "Performance drift detected.",
        )

    if (
        severity == "WARNING"
        and retrain_on_warning
    ):
        return (
            True,
            "Warning-level drift detected.",
        )

    if drift_detected:
        return (
            False,
            f"Drift detected but severity "
            f"'{severity}' is not configured "
            "for retraining.",
        )

    return (
        False,
        "No qualifying model drift detected.",
    )


# ============================================================
# CHALLENGER NAME
# ============================================================

def generate_challenger_name(
    settings: dict[str, Any],
) -> str:
    """Generate unique challenger model name."""

    prefix = str(
        settings.get(
            "challenger_prefix",
            "challenger",
        )
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    return (
        f"{prefix}_{timestamp}"
    )


# ============================================================
# TRAINING
# ============================================================

def get_training_command(
    challenger_name: str,
    settings: dict[str, Any],
) -> list[str]:
    """Build training command.

    Default:

        python scripts/train_models.py
            --model-name <challenger>

    Adjust this if your repository uses a
    different training entry point.
    """

    configured_command = settings.get(
        "training_command"
    )

    if configured_command:

        if isinstance(
            configured_command,
            str,
        ):

            command = configured_command.format(
                challenger_name=challenger_name
            )

            return command.split()

        if isinstance(
            configured_command,
            list,
        ):

            return [
                str(item).format(
                    challenger_name=challenger_name
                )
                for item in configured_command
            ]

    return [

        sys.executable,

        str(
            PROJECT_ROOT
            / "scripts"
            / "train_models.py"
        ),

        "--model-name",

        challenger_name,
    ]


def run_training(
    challenger_name: str,
    settings: dict[str, Any],
) -> tuple[bool, str]:
    """Run challenger training process."""

    command = get_training_command(
        challenger_name,
        settings,
    )

    logger.info(
        "Training challenger: %s",
        challenger_name,
    )

    logger.info(
        "Command: %s",
        " ".join(command),
    )

    timeout = int(
        settings.get(
            "training_timeout_seconds",
            7200,
        )
    )

    try:

        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        output = (
            (process.stdout or "")
            + "\n"
            + (process.stderr or "")
        ).strip()

        if process.returncode != 0:

            logger.error(
                "Training failed with return code %s.",
                process.returncode,
            )

            return (
                False,
                output,
            )

        logger.info(
            "Training completed successfully."
        )

        return (
            True,
            output,
        )

    except subprocess.TimeoutExpired:

        return (
            False,
            "Training process timed out.",
        )

    except Exception as error:

        return (
            False,
            str(error),
        )


# ============================================================
# MODEL DISCOVERY
# ============================================================

def find_newest_model(
    started_at: datetime,
) -> Path | None:
    """Find newest model created during training."""

    models_dir = get_models_dir()

    candidates: list[
        Path
    ] = []

    extensions = (
        "*.pkl",
        "*.joblib",
        "*.pt",
        "*.pth",
        "*.bin",
    )

    for pattern in extensions:

        candidates.extend(
            models_dir.rglob(
                pattern
            )
        )

    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )

    for candidate in candidates:

        modified_at = datetime.fromtimestamp(
            candidate.stat().st_mtime
        )

        if modified_at >= started_at:
            return candidate

    return candidates[0]


def save_challenger_copy(
    source_model: Path,
    challenger_name: str,
) -> Path:
    """Save a named challenger copy."""

    models_dir = get_models_dir()

    challenger_dir = (
        models_dir
        / "challengers"
    )

    challenger_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        challenger_dir
        / f"{challenger_name}"
        f"{source_model.suffix}"
    )

    shutil.copy2(
        source_model,
        destination,
    )

    logger.info(
        "Challenger model saved: %s",
        destination,
    )

    return destination


# ============================================================
# MODEL REGISTRY
# ============================================================

def get_registry_path() -> Path:
    """Get model registry path."""

    configured = get_config_value(
        "paths",
        "model_registry",
        "data/model_registry.json",
    )

    path = resolve_project_path(
        configured
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def load_registry() -> dict[str, Any]:
    """Load model registry."""

    path = get_registry_path()

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

            return json.load(
                file
            )

    except Exception:

        return {
            "champion": None,
            "challengers": [],
            "history": [],
        }


def update_registry(
    challenger_name: str,
    challenger_path: Path | None,
    drift_result: dict[str, Any],
) -> Path:
    """Register newly trained challenger."""

    registry = load_registry()

    registry.setdefault(
        "challengers",
        []
    )

    registry.setdefault(
        "history",
        []
    )

    entry = {

        "model_name": challenger_name,

        "model_path": (
            str(challenger_path)
            if challenger_path
            else None
        ),

        "created_at": (
            datetime.now().isoformat()
        ),

        "status": "CHALLENGER",

        "trigger": "DRIFT_DETECTION",

        "drift_severity": (
            drift_result.get(
                "severity"
            )
        ),

        "drift_score": (
            drift_result.get(
                "score"
            )
        ),
    }

    registry["challengers"].append(
        entry
    )

    registry["history"].append(
        {
            "timestamp": (
                datetime.now().isoformat()
            ),
            "event": (
                "CHALLENGER_CREATED"
            ),
            "details": entry,
        }
    )

    registry_path = get_registry_path()

    with open(
        registry_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            registry,
            file,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Model registry updated: %s",
        registry_path,
    )

    return registry_path


# ============================================================
# REPORT
# ============================================================

def make_json_safe(
    value: Any,
) -> Any:
    """Convert values to JSON-safe objects."""

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        (
            np.integer,
            np.int64,
            np.int32,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            np.float64,
            np.float32,
        ),
    ):
        if np.isnan(value) or np.isinf(value):
            return None

        return float(value)

    if isinstance(value, datetime):
        return value.isoformat()

    return value


def save_report(
    payload: dict[str, Any],
) -> Path:
    """Save retraining report."""

    reports_dir = get_reports_dir()

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_path = (
        reports_dir
        / f"retraining_{timestamp}.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            make_json_safe(payload),
            file,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Retraining report saved: %s",
        report_path,
    )

    return report_path


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(
    message: str,
) -> bool:
    """Send retraining notification."""

    if not bool(
        get_config_value(
            "telegram",
            "enabled",
            False,
        )
    ):
        return False

    try:

        from src.telegram import (
            send_message,
        )

        send_message(
            message
        )

        return True

    except ImportError:
        pass

    except Exception as error:

        logger.warning(
            "Telegram failed: %s",
            error,
        )

        return False

    try:

        from src.telegram_bot import (
            send_message,
        )

        send_message(
            message
        )

        return True

    except Exception as error:

        logger.warning(
            "Telegram unavailable: %s",
            error,
        )

        return False


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    """Run automatic challenger retraining."""

    started_at = datetime.now()

    logger.info(
        "=" * 72
    )

    logger.info(
        "AUTOMATIC CHALLENGER RETRAINING"
    )

    logger.info(
        "=" * 72
    )

    settings = get_config_section(
        "retraining",
        default={},
    )

    # --------------------------------------------------------
    # LOAD DRIFT REPORT
    # --------------------------------------------------------

    drift_report = load_drift_report()

    if drift_report is None:

        logger.warning(
            "Cannot evaluate retraining without "
            "a drift report."
        )

        return 0

    drift_result = extract_drift_result(
        drift_report
    )

    # --------------------------------------------------------
    # RETRAIN DECISION
    # --------------------------------------------------------

    retrain, reason = should_retrain(
        drift_result,
        settings,
    )

    logger.info(
        "Retraining decision: %s",
        retrain,
    )

    logger.info(
        "Reason: %s",
        reason,
    )

    if not retrain:

        report_path = save_report(
            {
                "timestamp": (
                    datetime.now().isoformat()
                ),
                "status": "SKIPPED",
                "reason": reason,
                "drift_result": drift_result,
            }
        )

        logger.info(
            "Retraining skipped. Report: %s",
            report_path,
        )

        return 0

    # --------------------------------------------------------
    # GENERATE CHALLENGER NAME
    # --------------------------------------------------------

    challenger_name = (
        generate_challenger_name(
            settings
        )
    )

    logger.info(
        "New challenger: %s",
        challenger_name,
    )

    # --------------------------------------------------------
    # RUN TRAINING
    # --------------------------------------------------------

    training_started = datetime.now()

    success, training_output = run_training(
        challenger_name,
        settings,
    )

    if not success:

        report_path = save_report(
            {
                "timestamp": (
                    datetime.now().isoformat()
                ),
                "status": "FAILED",
                "challenger_name": challenger_name,
                "reason": (
                    "Training failed."
                ),
                "training_output": (
                    training_output
                ),
                "drift_result": drift_result,
            }
        )

        logger.error(
            "Challenger training failed."
        )

        send_telegram(
            (
                "🚨 *CHALLENGER TRAINING FAILED*\n\n"
                f"Model: `{challenger_name}`\n"
                f"Reason: {training_output[-1000:]}"
            )
        )

        logger.info(
            "Failure report: %s",
            report_path,
        )

        return 1

    # --------------------------------------------------------
    # FIND NEW MODEL
    # --------------------------------------------------------

    source_model = find_newest_model(
        training_started
    )

    challenger_path = None

    if source_model:

        challenger_path = (
            save_challenger_copy(
                source_model,
                challenger_name,
            )
        )

    else:

        logger.warning(
            "Training succeeded but no model "
            "artifact was found."
        )

    # --------------------------------------------------------
    # UPDATE REGISTRY
    # --------------------------------------------------------

    registry_path = update_registry(
        challenger_name=challenger_name,
        challenger_path=challenger_path,
        drift_result=drift_result,
    )

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    report = {

        "timestamp": (
            datetime.now().isoformat()
        ),

        "status": "SUCCESS",

        "challenger_name": challenger_name,

        "challenger_path": (
            str(challenger_path)
            if challenger_path
            else None
        ),

        "registry_path": (
            str(registry_path)
        ),

        "reason": reason,

        "drift_result": drift_result,

        "training_output": (
            training_output[-5000:]
        ),
    }

    report_path = save_report(
        report
    )

    # --------------------------------------------------------
    # TELEGRAM
    # --------------------------------------------------------

    send_telegram(
        (
            "🔄 *CHALLENGER TRAINED*\n\n"
            f"*Model:* `{challenger_name}`\n"
            f"*Drift Severity:* "
            f"{drift_result.get('severity', 'UNKNOWN')}\n"
            f"*Status:* Ready for parallel evaluation"
        )
    )

    elapsed = (
        datetime.now()
        - started_at
    ).total_seconds()

    logger.info(
        "Retraining completed in %.2f seconds.",
        elapsed,
    )

    logger.info(
        "Report: %s",
        report_path,
    )

    return 0


if __name__ == "__main__":

    try:

        raise SystemExit(
            main()
        )

    except KeyboardInterrupt:

        logger.warning(
            "Retraining interrupted."
        )

        raise SystemExit(130)

    except Exception as error:

        logger.error(
            "Fatal retraining error: %s",
            error,
        )

        logger.debug(
            traceback.format_exc()
        )

        raise SystemExit(1)

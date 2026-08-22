#!/usr/bin/env python3

"""Run model performance drift detection.

Pipeline:

    Prediction Ledger
            │
            ▼
    Load Evaluated Predictions
            │
            ▼
    Drift Detector
            │
            ├── Historical Performance
            │
            └── Recent Performance
                    │
                    ▼
                Drift Score
                    │
                    ▼
        STABLE / WARNING / DRIFT / CRITICAL
                    │
                    ├── Save JSON report
                    ├── Save CSV summary
                    └── Telegram notification
"""

from __future__ import annotations

import json
import logging
import sys
import traceback

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.config import cfg
from src.drift_detector import detect_performance_drift


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ============================================================
# CONFIG HELPERS
# ============================================================

def get_nested_config(
    section: str,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Safely load a configuration section."""

    if default is None:
        default = {}

    try:
        value = getattr(cfg, section, None)

        if value is None:
            return default.copy()

        if isinstance(value, dict):
            return value

        if hasattr(value, "items"):
            return dict(value)

        if hasattr(value, "__dict__"):
            return {
                key: value
                for key, value in value.__dict__.items()
                if not key.startswith("_")
            }

    except Exception as error:
        logger.warning(
            "Could not load config section '%s': %s",
            section,
            error,
        )

    return default.copy()


def get_path(
    name: str,
    fallback: Path,
) -> Path:
    """Get a configured path safely."""

    try:
        paths = getattr(
            cfg,
            "paths",
            None,
        )

        value = getattr(
            paths,
            name,
            None,
        )

        if value:
            return Path(value)

    except Exception:
        pass

    return fallback


def resolve_project_path(
    path: Path,
) -> Path:
    """Resolve relative paths from project root."""

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# ============================================================
# PATHS
# ============================================================

def get_ledger_path() -> Path:
    """Return prediction ledger path."""

    path = get_path(
        "ledger",
        Path("data/ledger/predictions.csv"),
    )

    return resolve_project_path(path)


def get_reports_dir() -> Path:
    """Return reports directory."""

    path = get_path(
        "reports",
        Path("data/reports"),
    )

    path = resolve_project_path(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


# ============================================================
# LEDGER
# ============================================================

def load_ledger() -> pd.DataFrame:
    """Load prediction ledger."""

    ledger_path = get_ledger_path()

    if not ledger_path.exists():

        logger.warning(
            "Prediction ledger not found: %s",
            ledger_path,
        )

        return pd.DataFrame()

    try:

        ledger = pd.read_csv(
            ledger_path
        )

        logger.info(
            "Loaded prediction ledger: %s rows",
            len(ledger),
        )

        return ledger

    except Exception as error:

        logger.error(
            "Failed to load ledger: %s",
            error,
        )

        return pd.DataFrame()


# ============================================================
# JSON SERIALIZATION
# ============================================================

def make_json_safe(
    value: Any,
) -> Any:
    """Convert NumPy and Pandas values to JSON-safe types."""

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):
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

        if np.isnan(value):
            return None

        if np.isinf(value):
            return None

        return float(value)

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if value is None:
        return None

    if isinstance(
        value,
        float,
    ):

        if np.isnan(value):
            return None

        if np.isinf(value):
            return None

    return value


# ============================================================
# REPORT SAVING
# ============================================================

def save_json_report(
    result: dict[str, Any],
) -> Path:
    """Save complete drift report as JSON."""

    reports_dir = get_reports_dir()

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_path = (
        reports_dir
        / f"drift_report_{timestamp}.json"
    )

    payload = {

        "generated_at": (
            datetime.now().isoformat()
        ),

        "report_type": (
            "model_performance_drift"
        ),

        "result": (
            make_json_safe(result)
        ),
    }

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Drift JSON report saved: %s",
        report_path,
    )

    return report_path


def save_csv_summary(
    result: dict[str, Any],
) -> Path:
    """Save drift summary as CSV."""

    reports_dir = get_reports_dir()

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_path = (
        reports_dir
        / f"drift_summary_{timestamp}.csv"
    )

    historical = result.get(
        "historical_metrics",
        {},
    )

    recent = result.get(
        "recent_metrics",
        {},
    )

    changes = result.get(
        "metric_changes",
        {},
    )

    rows: list[dict[str, Any]] = []

    metric_mapping = {

        "return_mae": (
            "return_mae_change"
        ),

        "direction_accuracy": (
            "direction_accuracy_change"
        ),

        "brier_score": (
            "brier_score_change"
        ),

        "risk_mae": (
            "risk_mae_change"
        ),
    }

    for metric, change_key in (
        metric_mapping.items()
    ):

        rows.append(
            {

                "metric": metric,

                "historical": (
                    historical.get(
                        metric
                    )
                ),

                "recent": (
                    recent.get(
                        metric
                    )
                ),

                "change": (
                    changes.get(
                        change_key
                    )
                ),

                "severity": (
                    result.get(
                        "severity"
                    )
                ),

                "drift_score": (
                    result.get(
                        "score"
                    )
                ),
            }
        )

    frame = pd.DataFrame(
        rows
    )

    frame.to_csv(
        report_path,
        index=False,
    )

    logger.info(
        "Drift CSV summary saved: %s",
        report_path,
    )

    return report_path


# ============================================================
# MESSAGE FORMATTING
# ============================================================

def format_percent(
    value: Any,
) -> str:
    """Format decimal as percentage."""

    try:

        value = float(value)

        if not np.isfinite(value):
            return "N/A"

        return f"{value:.2%}"

    except (
        TypeError,
        ValueError,
    ):

        return "N/A"


def format_number(
    value: Any,
) -> str:
    """Format numeric metric."""

    try:

        value = float(value)

        if not np.isfinite(value):
            return "N/A"

        return f"{value:.6f}"

    except (
        TypeError,
        ValueError,
    ):

        return "N/A"


def format_drift_message(
    result: dict[str, Any],
) -> str:
    """Create Telegram-friendly drift message."""

    severity = str(
        result.get(
            "severity",
            "UNKNOWN",
        )
    ).upper()

    score = result.get(
        "score",
        0.0,
    )

    drift_detected = bool(
        result.get(
            "drift_detected",
            False,
        )
    )

    if severity == "CRITICAL":
        icon = "🚨"

    elif severity == "DRIFT":
        icon = "⚠️"

    elif severity == "WARNING":
        icon = "🟡"

    elif severity == "STABLE":
        icon = "🟢"

    else:
        icon = "ℹ️"

    lines = [

        f"{icon} *MODEL DRIFT REPORT*",

        "",

        f"*Severity:* {severity}",

        f"*Drift Detected:* "
        f"{'YES' if drift_detected else 'NO'}",

        f"*Drift Score:* {score}",

        "",
    ]

    historical = result.get(
        "historical_metrics",
        {},
    )

    recent = result.get(
        "recent_metrics",
        {},
    )

    if historical and recent:

        lines.extend(
            [

                "*Performance Comparison*",

                (
                    "Return MAE: "
                    f"{format_percent(historical.get('return_mae'))} "
                    "→ "
                    f"{format_percent(recent.get('return_mae'))}"
                ),

                (
                    "Direction Accuracy: "
                    f"{format_percent(historical.get('direction_accuracy'))} "
                    "→ "
                    f"{format_percent(recent.get('direction_accuracy'))}"
                ),

                (
                    "Brier Score: "
                    f"{format_number(historical.get('brier_score'))} "
                    "→ "
                    f"{format_number(recent.get('brier_score'))}"
                ),

                (
                    "Risk MAE: "
                    f"{format_percent(historical.get('risk_mae'))} "
                    "→ "
                    f"{format_percent(recent.get('risk_mae'))}"
                ),

                "",
            ]
        )

    reasons = result.get(
        "reasons",
        [],
    )

    if reasons:

        lines.append(
            "*Reasons:*"
        )

        for reason in reasons:

            lines.append(
                f"• {reason}"
            )

        lines.append("")

    recommendation = result.get(
        "recommendation"
    )

    if recommendation:

        lines.extend(
            [

                "*Recommendation:*",

                recommendation,
            ]
        )

    return "\n".join(
        lines
    )


# ============================================================
# TELEGRAM
# ============================================================

def should_notify(
    result: dict[str, Any],
    drift_config: dict[str, Any],
) -> bool:
    """Determine whether a notification should be sent."""

    if not bool(
        drift_config.get(
            "enabled",
            True,
        )
    ):

        return False

    severity = str(
        result.get(
            "severity",
            "",
        )
    ).upper()

    if severity == "CRITICAL":

        return bool(
            drift_config.get(
                "notify_on_critical",
                True,
            )
        )

    if severity == "DRIFT":

        return bool(
            drift_config.get(
                "notify_on_drift",
                True,
            )
        )

    if severity == "WARNING":

        return bool(
            drift_config.get(
                "notify_on_warning",
                False,
            )
        )

    return False


def send_telegram_notification(
    message: str,
) -> bool:
    """Attempt to send a Telegram notification.

    This function supports common Telegram implementations
    used in the project. If Telegram is unavailable, drift
    detection continues normally.
    """

    try:

        from src.telegram import (
            send_message,
        )

        send_message(
            message
        )

        logger.info(
            "Telegram drift notification sent."
        )

        return True

    except ImportError:

        pass

    except Exception as error:

        logger.warning(
            "Telegram notification failed: %s",
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

        logger.info(
            "Telegram drift notification sent."
        )

        return True

    except ImportError:

        logger.warning(
            "Telegram module not found. "
            "Skipping notification."
        )

    except Exception as error:

        logger.warning(
            "Telegram notification failed: %s",
            error,
        )

    return False


# ============================================================
# CONSOLE OUTPUT
# ============================================================

def print_result(
    result: dict[str, Any],
) -> None:
    """Print readable drift analysis."""

    logger.info(
        "=" * 70
    )

    logger.info(
        "MODEL PERFORMANCE DRIFT RESULT"
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        "Severity: %s",
        result.get(
            "severity"
        ),
    )

    logger.info(
        "Drift detected: %s",
        result.get(
            "drift_detected"
        ),
    )

    logger.info(
        "Drift score: %s",
        result.get(
            "score"
        ),
    )

    logger.info(
        "-"
    )

    historical = result.get(
        "historical_metrics",
        {},
    )

    recent = result.get(
        "recent_metrics",
        {},
    )

    for metric in (
        "return_mae",
        "direction_accuracy",
        "brier_score",
        "risk_mae",
    ):

        logger.info(

            "%s | historical=%s | recent=%s",

            metric,

            historical.get(
                metric,
                "N/A",
            ),

            recent.get(
                metric,
                "N/A",
            ),
        )

    logger.info(
        "-"
    )

    for reason in result.get(
        "reasons",
        [],
    ):

        logger.info(
            "Reason: %s",
            reason,
        )

    logger.info(
        "-"
    )

    logger.info(
        "Recommendation: %s",
        result.get(
            "recommendation"
        ),
    )

    logger.info(
        "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    """Run drift detection."""

    started_at = datetime.now()

    logger.info(
        "=" * 70
    )

    logger.info(
        "PHASE 4 DRIFT DETECTION STARTED"
    )

    logger.info(
        "=" * 70
    )

    # --------------------------------------------------------
    # LOAD CONFIG
    # --------------------------------------------------------

    drift_config = get_nested_config(
        "drift",
        default={},
    )

    if not bool(
        drift_config.get(
            "enabled",
            True,
        )
    ):

        logger.info(
            "Drift detection is disabled."
        )

        return 0

    # --------------------------------------------------------
    # LOAD LEDGER
    # --------------------------------------------------------

    ledger = load_ledger()

    if ledger.empty:

        logger.info(
            "No prediction ledger data available."
        )

        return 0

    # --------------------------------------------------------
    # RUN DETECTOR
    # --------------------------------------------------------

    logger.info(
        "Running performance drift analysis..."
    )

    result_object = detect_performance_drift(
        ledger=ledger,
        config=drift_config,
    )

    result = result_object.to_dict()

    # --------------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------------

    print_result(
        result
    )

    # --------------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------------

    json_path = save_json_report(
        result
    )

    csv_path = save_csv_summary(
        result
    )

    logger.info(
        "Reports generated:"
    )

    logger.info(
        "JSON: %s",
        json_path,
    )

    logger.info(
        "CSV: %s",
        csv_path,
    )

    # --------------------------------------------------------
    # TELEGRAM
    # --------------------------------------------------------

    if should_notify(
        result=result,
        drift_config=drift_config,
    ):

        message = format_drift_message(
            result
        )

        send_telegram_notification(
            message
        )

    else:

        logger.info(
            "No drift notification required."
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    elapsed = (
        datetime.now()
        - started_at
    ).total_seconds()

    logger.info(
        "Drift detection completed in %.2f seconds.",
        elapsed,
    )

    return 0


if __name__ == "__main__":

    try:

        raise SystemExit(
            main()
        )

    except KeyboardInterrupt:

        logger.warning(
            "Drift detection interrupted."
        )

        raise SystemExit(
            130
        )

    except Exception as error:

        logger.error(
            "Fatal drift detection error: %s",
            error,
        )

        logger.debug(
            traceback.format_exc()
        )

        raise SystemExit(
            1
        )

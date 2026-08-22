#!/usr/bin/env python3

"""Compare Champion and Challenger model performance.

Pipeline:

    Prediction Ledger
            │
            ▼
    Load Evaluated Predictions
            │
            ├── Champion
            │
            └── Challenger
                    │
                    ▼
          Champion / Challenger Engine
                    │
                    ▼
            Comparison Decision
                    │
            ┌───────┼────────┐
            ▼       ▼        ▼
        CHAMPION  TIE   CHALLENGER
                            │
                            ▼
                    Promotion Recommended
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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.config import cfg
from src.champion_challenger import compare_models


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
    """Convert a config section into a plain dictionary."""

    if value is None:
        return {}

    if isinstance(
        value,
        dict,
    ):
        return dict(value)

    if hasattr(
        value,
        "items",
    ):
        try:
            return dict(
                value.items()
            )
        except Exception:
            pass

    if hasattr(
        value,
        "__dict__",
    ):
        try:
            return {
                key: item
                for key, item
                in value.__dict__.items()
                if not key.startswith("_")
            }
        except Exception:
            pass

    return {}


def get_config_section(
    name: str,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Safely load a configuration section."""

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
            "Unable to read config section "
            "'%s': %s",
            name,
            error,
        )

    return dict(default)


def get_config_value(
    section: str,
    key: str,
    default: Any = None,
) -> Any:
    """Get one config value safely."""

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
    path: str | Path,
) -> Path:
    """Resolve relative paths from project root."""

    value = Path(path)

    if value.is_absolute():
        return value

    return (
        PROJECT_ROOT
        / value
    )


def get_ledger_path() -> Path:
    """Get prediction ledger path."""

    path = get_config_value(
        section="paths",
        key="ledger",
        default="data/ledger/predictions.csv",
    )

    return resolve_project_path(
        path
    )


def get_reports_dir() -> Path:
    """Get and create reports directory."""

    path = get_config_value(
        section="paths",
        key="reports",
        default="data/reports",
    )

    reports_dir = resolve_project_path(
        path
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return reports_dir


# ============================================================
# LEDGER
# ============================================================

def load_ledger() -> pd.DataFrame:
    """Load prediction ledger CSV."""

    ledger_path = get_ledger_path()

    if not ledger_path.exists():

        logger.warning(
            "Prediction ledger does not exist: %s",
            ledger_path,
        )

        return pd.DataFrame()

    try:

        ledger = pd.read_csv(
            ledger_path
        )

        logger.info(
            "Loaded %s prediction ledger rows.",
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
# JSON SAFETY
# ============================================================

def make_json_safe(
    value: Any,
) -> Any:
    """Convert Pandas and NumPy values to JSON-safe values."""

    if isinstance(
        value,
        dict,
    ):

        return {
            str(key): make_json_safe(item)
            for key, item
            in value.items()
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

    if isinstance(
        value,
        datetime,
    ):

        return value.isoformat()

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
# REPORTS
# ============================================================

def save_json_report(
    result: dict[str, Any],
) -> Path:
    """Save full comparison report as JSON."""

    reports_dir = get_reports_dir()

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_path = (
        reports_dir
        / f"model_comparison_{timestamp}.json"
    )

    payload = {

        "generated_at": (
            datetime.now().isoformat()
        ),

        "report_type": (
            "champion_challenger_comparison"
        ),

        "result": make_json_safe(
            result
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
        "JSON comparison report saved: %s",
        report_path,
    )

    return report_path


def save_csv_report(
    result: dict[str, Any],
) -> Path:
    """Save metric comparison report as CSV."""

    reports_dir = get_reports_dir()

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_path = (
        reports_dir
        / f"model_comparison_{timestamp}.csv"
    )

    champion_metrics = result.get(
        "champion_metrics",
        {},
    )

    challenger_metrics = result.get(
        "challenger_metrics",
        {},
    )

    improvements = result.get(
        "metric_improvements",
        {},
    )

    metric_names = sorted(
        set(
            champion_metrics.keys()
        )
        | set(
            challenger_metrics.keys()
        )
        | set(
            improvements.keys()
        )
    )

    rows: list[
        dict[str, Any]
    ] = []

    for metric in metric_names:

        rows.append(
            {

                "champion": result.get(
                    "champion_name"
                ),

                "challenger": result.get(
                    "challenger_name"
                ),

                "metric": metric,

                "champion_value": (
                    champion_metrics.get(
                        metric
                    )
                ),

                "challenger_value": (
                    challenger_metrics.get(
                        metric
                    )
                ),

                "improvement": (
                    improvements.get(
                        metric
                    )
                ),

                "winner": result.get(
                    "winner"
                ),

                "promotion_recommended": (
                    result.get(
                        "promotion_recommended"
                    )
                ),

                "overall_improvement": (
                    result.get(
                        "overall_improvement"
                    )
                ),
            }
        )

    if not rows:

        rows.append(
            {

                "champion": result.get(
                    "champion_name"
                ),

                "challenger": result.get(
                    "challenger_name"
                ),

                "metric": None,

                "champion_value": None,

                "challenger_value": None,

                "improvement": None,

                "winner": result.get(
                    "winner"
                ),

                "promotion_recommended": (
                    result.get(
                        "promotion_recommended"
                    )
                ),

                "overall_improvement": (
                    result.get(
                        "overall_improvement"
                    )
                ),
            }
        )

    report = pd.DataFrame(
        rows
    )

    report.to_csv(
        report_path,
        index=False,
    )

    logger.info(
        "CSV comparison report saved: %s",
        report_path,
    )

    return report_path


# ============================================================
# MESSAGE FORMATTING
# ============================================================

def format_percentage(
    value: Any,
) -> str:
    """Format a decimal as percentage."""

    try:

        number = float(value)

        if not np.isfinite(number):
            return "N/A"

        return (
            f"{number:.2%}"
        )

    except (
        TypeError,
        ValueError,
    ):

        return "N/A"


def format_number(
    value: Any,
) -> str:
    """Format a numeric metric."""

    try:

        number = float(value)

        if not np.isfinite(number):
            return "N/A"

        return (
            f"{number:.6f}"
        )

    except (
        TypeError,
        ValueError,
    ):

        return "N/A"


def format_comparison_message(
    result: dict[str, Any],
) -> str:
    """Format a Telegram-friendly comparison message."""

    winner = str(
        result.get(
            "winner",
            "UNKNOWN",
        )
    )

    promotion = bool(
        result.get(
            "promotion_recommended",
            False,
        )
    )

    if winner == "CHALLENGER":

        icon = "🚀"

    elif winner == "CHAMPION":

        icon = "🏆"

    elif winner == "TIE":

        icon = "🤝"

    elif winner == "INSUFFICIENT_DATA":

        icon = "📊"

    else:

        icon = "ℹ️"

    champion_name = result.get(
        "champion_name",
        "Champion",
    )

    challenger_name = result.get(
        "challenger_name",
        "Challenger",
    )

    lines = [

        (
            f"{icon} *MODEL COMPARISON*"
        ),

        "",

        (
            f"*Champion:* `{champion_name}`"
        ),

        (
            f"*Challenger:* `{challenger_name}`"
        ),

        "",

        (
            f"*Winner:* {winner}"
        ),

        (
            "*Overall Improvement:* "
            f"{format_percentage(result.get('overall_improvement'))}"
        ),

        (
            "*Promotion:* "
            f"{'RECOMMENDED' if promotion else 'NOT RECOMMENDED'}"
        ),

        "",
    ]

    champion_metrics = result.get(
        "champion_metrics",
        {},
    )

    challenger_metrics = result.get(
        "challenger_metrics",
        {},
    )

    improvements = result.get(
        "metric_improvements",
        {},
    )

    if (
        champion_metrics
        or challenger_metrics
    ):

        lines.append(
            "*Performance*"
        )

        metric_labels = {

            "return_mae": (
                "Return MAE"
            ),

            "direction_accuracy": (
                "Direction Accuracy"
            ),

            "brier_score": (
                "Brier Score"
            ),

            "risk_mae": (
                "Risk MAE"
            ),
        }

        for (
            metric,
            label,
        ) in metric_labels.items():

            champion_value = (
                champion_metrics.get(
                    metric
                )
            )

            challenger_value = (
                challenger_metrics.get(
                    metric
                )
            )

            improvement = (
                improvements.get(
                    metric
                )
            )

            if (
                champion_value is None
                and challenger_value is None
            ):
                continue

            if metric == "direction_accuracy":

                champion_text = (
                    format_percentage(
                        champion_value
                    )
                )

                challenger_text = (
                    format_percentage(
                        challenger_value
                    )
                )

            else:

                champion_text = (
                    format_number(
                        champion_value
                    )
                )

                challenger_text = (
                    format_number(
                        challenger_value
                    )
                )

            improvement_text = (
                format_percentage(
                    improvement
                )
            )

            lines.append(

                f"• *{label}*: "
                f"{champion_text} → "
                f"{challenger_text} "
                f"({improvement_text})"
            )

        lines.append(
            ""
        )

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

def telegram_enabled() -> bool:
    """Check whether Telegram is enabled."""

    return bool(
        get_config_value(
            section="telegram",
            key="enabled",
            default=False,
        )
    )


def send_telegram_notification(
    message: str,
) -> bool:
    """Send comparison result through available Telegram module."""

    if not telegram_enabled():

        logger.info(
            "Telegram notifications disabled."
        )

        return False

    try:

        from src.telegram import (
            send_message,
        )

        send_message(
            message
        )

        logger.info(
            "Telegram comparison sent."
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
            "Telegram comparison sent."
        )

        return True

    except ImportError:

        logger.warning(
            "No compatible Telegram module found."
        )

    except Exception as error:

        logger.warning(
            "Telegram notification failed: %s",
            error,
        )

    return False


# ============================================================
# CONSOLE REPORT
# ============================================================

def print_result(
    result: dict[str, Any],
) -> None:
    """Print model comparison result."""

    logger.info(
        "=" * 72
    )

    logger.info(
        "CHAMPION / CHALLENGER COMPARISON"
    )

    logger.info(
        "=" * 72
    )

    logger.info(
        "Champion: %s",
        result.get(
            "champion_name"
        ),
    )

    logger.info(
        "Challenger: %s",
        result.get(
            "challenger_name"
        ),
    )

    logger.info(
        "Champion evaluations: %s",
        result.get(
            "champion_records"
        ),
    )

    logger.info(
        "Challenger evaluations: %s",
        result.get(
            "challenger_records"
        ),
    )

    logger.info(
        "Winner: %s",
        result.get(
            "winner"
        ),
    )

    logger.info(
        "Overall improvement: %s",
        result.get(
            "overall_improvement"
        ),
    )

    logger.info(
        "Promotion recommended: %s",
        result.get(
            "promotion_recommended"
        ),
    )

    logger.info(
        "-" * 72
    )

    champion_metrics = result.get(
        "champion_metrics",
        {},
    )

    challenger_metrics = result.get(
        "challenger_metrics",
        {},
    )

    improvements = result.get(
        "metric_improvements",
        {},
    )

    for metric in sorted(
        set(champion_metrics)
        | set(challenger_metrics)
        | set(improvements)
    ):

        logger.info(
            "%s | champion=%s | challenger=%s | improvement=%s",
            metric,
            champion_metrics.get(
                metric,
                "N/A",
            ),
            challenger_metrics.get(
                metric,
                "N/A",
            ),
            improvements.get(
                metric,
                "N/A",
            ),
        )

    logger.info(
        "-" * 72
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
        "-" * 72
    )

    logger.info(
        "Recommendation: %s",
        result.get(
            "recommendation"
        ),
    )

    logger.info(
        "=" * 72
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    """Run Champion / Challenger comparison."""

    started_at = datetime.now()

    logger.info(
        "=" * 72
    )

    logger.info(
        "MODEL COMPARISON STARTED"
    )

    logger.info(
        "=" * 72
    )

    # --------------------------------------------------------
    # LOAD CONFIG
    # --------------------------------------------------------

    comparison_config = (
        get_config_section(
            "champion_challenger",
            default={},
        )
    )

    if not bool(
        comparison_config.get(
            "enabled",
            True,
        )
    ):

        logger.info(
            "Champion / Challenger comparison "
            "is disabled."
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
    # COMPARE MODELS
    # --------------------------------------------------------

    logger.info(
        "Running Champion / Challenger comparison..."
    )

    result_object = compare_models(
        ledger=ledger,
        config=comparison_config,
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

    csv_path = save_csv_report(
        result
    )

    logger.info(
        "Comparison reports created."
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

    send_comparison = bool(
        get_config_value(
            section="telegram",
            key="send_training_status",
            default=False,
        )
    )

    if (
        send_comparison
        and result.get("winner")
        not in (
            "INSUFFICIENT_DATA",
            "NO_COMPARISON",
        )
    ):

        message = (
            format_comparison_message(
                result
            )
        )

        send_telegram_notification(
            message
        )

    else:

        logger.info(
            "Comparison Telegram notification "
            "not required."
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    elapsed = (
        datetime.now()
        - started_at
    ).total_seconds()

    logger.info(
        "Model comparison completed "
        "in %.2f seconds.",
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
            "Model comparison interrupted."
        )

        raise SystemExit(
            130
        )

    except Exception as error:

        logger.error(
            "Fatal model comparison error: %s",
            error,
        )

        logger.debug(
            traceback.format_exc()
        )

        raise SystemExit(
            1
        )

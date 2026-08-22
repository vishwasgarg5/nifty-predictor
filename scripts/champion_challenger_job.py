#!/usr/bin/env python3

"""
Champion / Challenger Automation Job.

Pipeline
--------
1. Load current Champion.
2. Load current Challenger.
3. Run model evaluation.
4. Update available model metrics.
5. Compare Champion vs Challenger.
6. Promote Challenger when promotion criteria are met.

This job is designed to be called by the scheduled
daily pipeline.

Important:
----------
A promotion decision updates the model registry only.
The actual production model loader must read the Champion
record from the registry for a promoted model to become
active in future prediction runs.
"""

from __future__ import annotations

import json
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
    "champion_challenger_job"
)


# ============================================================
# TIME
# ============================================================

def utc_now_iso() -> str:
    """Return current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# CONFIG HELPERS
# ============================================================

def object_to_dict(
    value: Any,
) -> dict[str, Any]:
    """Convert config objects into dictionaries."""

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


def get_job_config() -> dict[str, Any]:
    """
    Load Champion / Challenger job settings.
    """

    defaults = {
        "enabled": True,
        "history_file": (
            "data/models/"
            "champion_challenger_history.json"
        ),
    }

    cfg = load_config()

    if cfg is None:
        return defaults

    section = getattr(
        cfg,
        "champion_challenger",
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
# HISTORY
# ============================================================

def get_history_path() -> Path:
    """Return comparison history path."""

    config = get_job_config()

    path = Path(
        str(
            config["history_file"]
        )
    )

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            / path
        )

    return path


def load_history() -> list[dict[str, Any]]:
    """Load previous comparison history."""

    path = get_history_path()

    if not path.exists():
        return []

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

        if isinstance(data, list):
            return data

        return []

    except Exception as error:

        logger.warning(
            "Could not load comparison history: %s",
            error,
        )

        return []


def save_history(
    record: dict[str, Any],
) -> None:
    """Append comparison result to history."""

    path = get_history_path()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = load_history()

    history.append(
        record
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=2,
            default=str,
        )


# ============================================================
# METRIC EXTRACTION
# ============================================================

def safe_float(
    value: Any,
) -> float | None:
    """Convert value to float safely."""

    if value is None:
        return None

    try:

        numeric = float(value)

        if pd.isna(numeric):
            return None

        return numeric

    except Exception:

        return None


def normalize_metrics(
    metrics: dict[str, Any] | None,
) -> dict[str, float]:
    """Keep only valid numeric metrics."""

    if not isinstance(
        metrics,
        dict,
    ):
        return {}

    result: dict[str, float] = {}

    for key, value in metrics.items():

        numeric = safe_float(
            value
        )

        if numeric is not None:

            result[str(key)] = numeric

    return result


# ============================================================
# MODEL EVALUATION
# ============================================================

def run_model_evaluation() -> dict[str, Any]:
    """
    Run the project's model evaluation module.

    Expected output is preferably a dictionary
    containing model metrics.

    Supported entry points:

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

                if not callable(
                    function
                ):
                    continue

                logger.info(
                    "Running %s.%s()",
                    module_name,
                    function_name,
                )

                result = function()

                if isinstance(
                    result,
                    dict,
                ):
                    return result

                return {
                    "status": "SUCCESS",
                    "result": result,
                }

        except Exception as error:

            attempts.append(
                f"{module_name}: {error}"
            )

    logger.warning(
        "No compatible model evaluation "
        "entry point found. Attempts: %s",
        " | ".join(attempts),
    )

    return {
        "status": "UNAVAILABLE",
        "metrics": {},
        "error": (
            "No compatible model evaluation "
            "module found."
        ),
    }


def extract_metrics(
    evaluation_result: dict[str, Any],
) -> dict[str, float]:
    """
    Extract standard metrics from evaluation output.
    """

    if not isinstance(
        evaluation_result,
        dict,
    ):
        return {}

    metric_sources = [
        evaluation_result.get(
            "metrics"
        ),
        evaluation_result.get(
            "model_metrics"
        ),
        evaluation_result,
    ]

    for source in metric_sources:

        if not isinstance(
            source,
            dict,
        ):
            continue

        metrics = normalize_metrics(
            source
        )

        standard = {}

        for key in [
            "direction_accuracy",
            "mae",
            "rmse",
            "average_return_error",
        ]:

            if key in metrics:

                standard[key] = metrics[
                    key
                ]

        if standard:
            return standard

    return {}


# ============================================================
# REGISTRY UPDATE
# ============================================================

def update_champion_metrics(
    metrics: dict[str, float],
) -> dict[str, Any] | None:
    """Update Champion metrics."""

    from src.model_registry import (
        get_champion,
        update_model_metrics,
    )

    champion = get_champion()

    if champion is None:

        logger.warning(
            "No Champion model exists."
        )

        return None

    current = champion.get(
        "metrics",
        {},
    )

    if not isinstance(
        current,
        dict,
    ):
        current = {}

    updated = dict(current)

    updated.update(
        metrics
    )

    return update_model_metrics(
        "champion",
        updated,
    )


# ============================================================
# MAIN JOB
# ============================================================

def run_champion_challenger_job() -> dict[str, Any]:
    """
    Run the automated Champion / Challenger job.
    """

    result: dict[str, Any] = {
        "started_at": utc_now_iso(),
        "status": "STARTED",
        "evaluation": {},
        "champion": None,
        "challenger": None,
        "comparison": {},
        "promoted": False,
        "error": None,
    }

    logger.info(
        "=" * 70
    )

    logger.info(
        "STARTING CHAMPION / CHALLENGER JOB"
    )

    logger.info(
        "=" * 70
    )

    try:

        config = get_job_config()

        if not config.get(
            "enabled",
            True,
        ):

            result["status"] = "DISABLED"

            return result

        # ----------------------------------------------------
        # STEP 1: LOAD REGISTRY
        # ----------------------------------------------------

        from src.model_registry import (
            get_champion,
            get_challenger,
        )

        champion = get_champion()

        challenger = get_challenger()

        result["champion"] = champion
        result["challenger"] = challenger

        if champion is None:

            result["status"] = (
                "NO_CHAMPION"
            )

            return result

        if challenger is None:

            result["status"] = (
                "NO_CHALLENGER"
            )

            return result

        # ----------------------------------------------------
        # STEP 2: RUN MODEL EVALUATION
        # ----------------------------------------------------

        evaluation_result = (
            run_model_evaluation()
        )

        result["evaluation"] = (
            evaluation_result
        )

        metrics = extract_metrics(
            evaluation_result
        )

        # ----------------------------------------------------
        # STEP 3: UPDATE CHAMPION METRICS
        # ----------------------------------------------------

        if metrics:

            update_champion_metrics(
                metrics
            )

        # ----------------------------------------------------
        # STEP 4: EVALUATE PROMOTION
        # ----------------------------------------------------

        from src.champion_challenger import (
            evaluate_promotion,
        )

        comparison = (
            evaluate_promotion()
        )

        result["comparison"] = (
            comparison
        )

        result["promoted"] = bool(
            comparison.get(
                "promoted",
                False,
            )
        )

        result["status"] = (
            comparison.get(
                "status",
                "COMPLETED",
            )
        )

        # ----------------------------------------------------
        # STEP 5: SAVE HISTORY
        # ----------------------------------------------------

        history_record = {
            "timestamp": utc_now_iso(),
            "status": result["status"],
            "promoted": result[
                "promoted"
            ],
            "champion": get_champion(),
            "challenger": get_challenger(),
            "evaluation_metrics": metrics,
            "comparison": comparison,
        }

        save_history(
            history_record
        )

        return result

    except Exception as error:

        logger.exception(
            "Champion / Challenger job failed."
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
            "CHAMPION / CHALLENGER JOB FINISHED | "
            "STATUS=%s",
            result.get(
                "status"
            ),
        )


# ============================================================
# CLI
# ============================================================

def main() -> int:

    result = (
        run_champion_challenger_job()
    )

    print()

    print("=" * 70)

    print("CHAMPION / CHALLENGER JOB RESULT")

    print("=" * 70)

    print(
        "Status: "
        f"{result.get('status')}"
    )

    print(
        "Promoted: "
        f"{result.get('promoted')}"
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

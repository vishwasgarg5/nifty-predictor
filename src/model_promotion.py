#!/usr/bin/env python3

"""
Automatic Champion Promotion Engine.

This module promotes the Challenger model to Champion only when
the Champion / Challenger evaluation engine recommends promotion.

Safety features
---------------
1. Promotion eligibility is calculated first.
2. Current Champion metadata is backed up.
3. Promotion history is recorded.
4. Previous Champion becomes rollback candidate.
5. Model files are never deleted.
6. Promotion failures preserve the existing Champion.
7. Promotion can be disabled through configuration.

Pipeline
--------
Shadow Predictions
        │
        ▼
Actual Outcome Evaluation
        │
        ▼
Champion / Challenger Comparison
        │
        ▼
Promotion Eligibility
        │
        ├── Not eligible ──► Keep current Champion
        │
        ▼
Eligible
        │
        ▼
Backup current Champion metadata
        │
        ▼
Promote Challenger
        │
        ▼
Record promotion history
        │
        ▼
Previous Champion available for rollback
"""

from __future__ import annotations

import json
import logging
import shutil
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

logger = logging.getLogger(
    "model_promotion"
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
    """Convert config object into dictionary."""

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
            "Could not load config: %s",
            error,
        )

        return None


def get_promotion_config() -> dict[str, Any]:
    """
    Load model promotion settings.
    """

    defaults = {
        "enabled": False,
        "auto_promote": False,
        "metadata_path": (
            "data/models/"
            "model_registry.json"
        ),
        "history_path": (
            "data/models/"
            "promotion_history.jsonl"
        ),
        "backup_directory": (
            "data/models/backups"
        ),
    }

    cfg = load_config()

    if cfg is None:
        return defaults

    section = getattr(
        cfg,
        "model_promotion",
        None,
    )

    values = object_to_dict(
        section
    )

    result = defaults.copy()

    for key in defaults:

        if key in values:
            result[key] = values[key]

    result["enabled"] = bool(
        result["enabled"]
    )

    result["auto_promote"] = bool(
        result["auto_promote"]
    )

    return result


# ============================================================
# PATH HELPERS
# ============================================================

def resolve_project_path(
    value: str | Path,
) -> Path:
    """Resolve project-relative path."""

    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_registry_path() -> Path:
    """Return model registry path."""

    config = get_promotion_config()

    return resolve_project_path(
        config["metadata_path"]
    )


def get_history_path() -> Path:
    """Return promotion history path."""

    config = get_promotion_config()

    return resolve_project_path(
        config["history_path"]
    )


def get_backup_directory() -> Path:
    """Return backup directory."""

    config = get_promotion_config()

    return resolve_project_path(
        config["backup_directory"]
    )


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(
    path: Path,
    default: Any,
) -> Any:
    """Load JSON safely."""

    if not path.exists():
        return default

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception as error:

        logger.warning(
            "Could not load JSON %s: %s",
            path,
            error,
        )

        return default


def save_json(
    path: Path,
    data: Any,
) -> None:
    """Save JSON safely using temporary file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            default=str,
        )

    temporary_path.replace(
        path
    )


# ============================================================
# MODEL REGISTRY
# ============================================================

def default_registry() -> dict[str, Any]:
    """
    Return default model registry.
    """

    return {
        "champion": None,
        "challenger": None,
        "previous_champion": None,
        "updated_at": None,
    }


def load_registry() -> dict[str, Any]:
    """Load model registry."""

    path = get_registry_path()

    registry = load_json(
        path,
        default_registry(),
    )

    if not isinstance(
        registry,
        dict,
    ):
        registry = default_registry()

    for key, value in default_registry().items():

        if key not in registry:
            registry[key] = value

    return registry


def save_registry(
    registry: dict[str, Any],
) -> Path:
    """Save model registry."""

    path = get_registry_path()

    registry["updated_at"] = (
        utc_now_iso()
    )

    save_json(
        path,
        registry,
    )

    return path


# ============================================================
# BACKUP
# ============================================================

def backup_registry() -> Path | None:
    """
    Create timestamped backup of current registry.
    """

    registry_path = get_registry_path()

    if not registry_path.exists():
        return None

    backup_directory = (
        get_backup_directory()
    )

    backup_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = utc_now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_path = (
        backup_directory
        / f"model_registry_{timestamp}.json"
    )

    shutil.copy2(
        registry_path,
        backup_path,
    )

    logger.info(
        "Registry backup created: %s",
        backup_path,
    )

    return backup_path


# ============================================================
# PROMOTION HISTORY
# ============================================================

def append_promotion_history(
    record: dict[str, Any],
) -> Path:
    """
    Append promotion event to JSONL history.
    """

    path = get_history_path()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(
                record,
                default=str,
            )
        )

        file.write("\n")

    return path


# ============================================================
# SHADOW LEDGER
# ============================================================

def get_shadow_ledger_path() -> Path:
    """Return shadow prediction ledger path."""

    return (
        PROJECT_ROOT
        / "data"
        / "ledger"
        / "shadow_predictions.csv"
    )


def load_shadow_predictions() -> pd.DataFrame:
    """Load shadow predictions."""

    path = get_shadow_ledger_path()

    if not path.exists():

        logger.warning(
            "Shadow ledger not found: %s",
            path,
        )

        return pd.DataFrame()

    try:

        return pd.read_csv(
            path
        )

    except Exception as error:

        logger.error(
            "Could not load shadow ledger: %s",
            error,
        )

        return pd.DataFrame()


# ============================================================
# MODEL METADATA
# ============================================================

def get_model_name(
    model: Any,
    default_name: str,
) -> str:
    """
    Extract model name from registry metadata.
    """

    if model is None:
        return default_name

    if isinstance(model, str):
        return model

    if isinstance(model, dict):

        for key in [
            "name",
            "model_name",
            "id",
            "version",
        ]:

            value = model.get(key)

            if value:
                return str(value)

    return default_name


def build_model_metadata(
    existing: Any,
    role: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """
    Build updated model metadata.
    """

    metadata = {}

    if isinstance(existing, dict):
        metadata.update(existing)

    metadata["role"] = role
    metadata["updated_at"] = utc_now_iso()

    metadata["performance"] = {
        "sample_size": metrics.get(
            "sample_size"
        ),
        "direction_accuracy": metrics.get(
            "direction_accuracy"
        ),
        "average_actual_return": metrics.get(
            "average_actual_return"
        ),
        "median_actual_return": metrics.get(
            "median_actual_return"
        ),
        "return_mae": metrics.get(
            "return_mae"
        ),
        "return_rmse": metrics.get(
            "return_rmse"
        ),
        "average_actual_risk": metrics.get(
            "average_actual_risk"
        ),
        "risk_mae": metrics.get(
            "risk_mae"
        ),
        "performance_score": metrics.get(
            "performance_score"
        ),
    }

    return metadata


# ============================================================
# PROMOTION DECISION
# ============================================================

def get_promotion_evaluation(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run Champion / Challenger comparison.
    """

    from src.champion_challenger_evaluation import (
        compare_models,
    )

    return compare_models(
        predictions
    )


# ============================================================
# PROMOTE
# ============================================================

def promote_challenger(
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Evaluate and promote Challenger when eligible.

    The function is safe by default because:

    - Promotion must be enabled in config.
    - Challenger must pass Step 30 eligibility.
    - Existing registry is backed up.
    - Previous Champion is retained.
    - Model files are never deleted.
    """

    result: dict[str, Any] = {
        "started_at": utc_now_iso(),
        "status": "STARTED",
        "promoted": False,
        "dry_run": dry_run,
        "backup_path": None,
        "registry_path": str(
            get_registry_path()
        ),
        "history_path": str(
            get_history_path()
        ),
        "evaluation": None,
        "error": None,
    }

    try:

        config = (
            get_promotion_config()
        )

        # ----------------------------------------------------
        # PROMOTION ENABLED?
        # ----------------------------------------------------

        if not config.get(
            "enabled",
            False,
        ):

            result["status"] = (
                "PROMOTION_DISABLED"
            )

            return result

        if not config.get(
            "auto_promote",
            False,
        ):

            result["status"] = (
                "AUTO_PROMOTION_DISABLED"
            )

            return result

        # ----------------------------------------------------
        # LOAD SHADOW PREDICTIONS
        # ----------------------------------------------------

        predictions = (
            load_shadow_predictions()
        )

        if predictions.empty:

            result["status"] = (
                "NO_SHADOW_DATA"
            )

            return result

        # ----------------------------------------------------
        # EVALUATE
        # ----------------------------------------------------

        evaluation = (
            get_promotion_evaluation(
                predictions
            )
        )

        result["evaluation"] = evaluation

        promotion = evaluation.get(
            "promotion",
            {},
        )

        eligible = bool(
            promotion.get(
                "eligible",
                False,
            )
        )

        if not eligible:

            result["status"] = (
                "CHALLENGER_NOT_ELIGIBLE"
            )

            result["reasons"] = (
                promotion.get(
                    "reasons",
                    [],
                )
            )

            return result

        # ----------------------------------------------------
        # LOAD REGISTRY
        # ----------------------------------------------------

        registry = load_registry()

        current_champion = (
            registry.get(
                "champion"
            )
        )

        current_challenger = (
            registry.get(
                "challenger"
            )
        )

        if current_challenger is None:

            result["status"] = (
                "NO_CHALLENGER_REGISTERED"
            )

            return result

        # ----------------------------------------------------
        # DRY RUN
        # ----------------------------------------------------

        if dry_run:

            result["status"] = (
                "PROMOTION_READY"
            )

            result["current_champion"] = (
                get_model_name(
                    current_champion,
                    "UNKNOWN_CHAMPION",
                )
            )

            result["challenger"] = (
                get_model_name(
                    current_challenger,
                    "UNKNOWN_CHALLENGER",
                )
            )

            result[
                "performance_improvement"
            ] = promotion.get(
                "performance_improvement"
            )

            return result

        # ----------------------------------------------------
        # BACKUP REGISTRY
        # ----------------------------------------------------

        backup_path = (
            backup_registry()
        )

        if backup_path is not None:

            result["backup_path"] = str(
                backup_path
            )

        # ----------------------------------------------------
        # UPDATE MODEL ROLES
        # ----------------------------------------------------

        champion_metrics = (
            evaluation.get(
                "champion",
                {},
            )
        )

        challenger_metrics = (
            evaluation.get(
                "challenger",
                {},
            )
        )

        promoted_model = (
            build_model_metadata(
                existing=current_challenger,
                role="CHAMPION",
                metrics=challenger_metrics,
            )
        )

        previous_model = (
            build_model_metadata(
                existing=current_champion,
                role="PREVIOUS_CHAMPION",
                metrics=champion_metrics,
            )
        )

        # ----------------------------------------------------
        # REGISTRY UPDATE
        # ----------------------------------------------------

        registry["previous_champion"] = (
            previous_model
        )

        registry["champion"] = (
            promoted_model
        )

        registry["challenger"] = (
            None
        )

        save_registry(
            registry
        )

        # ----------------------------------------------------
        # PROMOTION HISTORY
        # ----------------------------------------------------

        history_record = {
            "timestamp": utc_now_iso(),
            "event": "MODEL_PROMOTION",
            "previous_champion": (
                get_model_name(
                    current_champion,
                    "UNKNOWN_CHAMPION",
                )
            ),
            "new_champion": (
                get_model_name(
                    current_challenger,
                    "UNKNOWN_CHALLENGER",
                )
            ),
            "performance_improvement": (
                promotion.get(
                    "performance_improvement"
                )
            ),
            "champion_metrics": (
                champion_metrics
            ),
            "challenger_metrics": (
                challenger_metrics
            ),
            "promotion_requirements": (
                promotion.get(
                    "requirements",
                    {},
                )
            ),
        }

        append_promotion_history(
            history_record
        )

        result["status"] = (
            "PROMOTED"
        )

        result["promoted"] = True

        result["previous_champion"] = (
            history_record[
                "previous_champion"
            ]
        )

        result["new_champion"] = (
            history_record[
                "new_champion"
            ]
        )

        result[
            "performance_improvement"
        ] = (
            history_record[
                "performance_improvement"
            ]
        )

        logger.info(
            "Challenger promoted successfully."
        )

        return result

    except Exception as error:

        logger.exception(
            "Model promotion failed."
        )

        result["status"] = (
            "PROMOTION_FAILED"
        )

        result["error"] = str(
            error
        )

        return result

    finally:

        result["finished_at"] = (
            utc_now_iso()
        )


# ============================================================
# ROLLBACK
# ============================================================

def rollback_to_previous_champion() -> dict[str, Any]:
    """
    Roll back current Champion to previous Champion.

    This does not delete any model files.
    """

    result: dict[str, Any] = {
        "started_at": utc_now_iso(),
        "status": "STARTED",
        "rolled_back": False,
        "error": None,
    }

    try:

        registry = load_registry()

        current_champion = (
            registry.get(
                "champion"
            )
        )

        previous_champion = (
            registry.get(
                "previous_champion"
            )
        )

        if previous_champion is None:

            result["status"] = (
                "NO_PREVIOUS_CHAMPION"
            )

            return result

        backup_path = (
            backup_registry()
        )

        if backup_path:

            result["backup_path"] = str(
                backup_path
            )

        registry["champion"] = (
            build_model_metadata(
                existing=previous_champion,
                role="CHAMPION",
                metrics=(
                    previous_champion.get(
                        "performance",
                        {}
                    )
                    if isinstance(
                        previous_champion,
                        dict,
                    )
                    else {}
                ),
            )
        )

        registry["previous_champion"] = (
            current_champion
        )

        save_registry(
            registry
        )

        history_record = {
            "timestamp": utc_now_iso(),
            "event": "MODEL_ROLLBACK",
            "restored_champion": (
                get_model_name(
                    previous_champion,
                    "UNKNOWN_PREVIOUS_CHAMPION",
                )
            ),
            "replaced_champion": (
                get_model_name(
                    current_champion,
                    "UNKNOWN_CHAMPION",
                )
            ),
        }

        append_promotion_history(
            history_record
        )

        result["status"] = (
            "ROLLED_BACK"
        )

        result["rolled_back"] = True

        return result

    except Exception as error:

        logger.exception(
            "Model rollback failed."
        )

        result["status"] = (
            "ROLLBACK_FAILED"
        )

        result["error"] = str(
            error
        )

        return result

    finally:

        result["finished_at"] = (
            utc_now_iso()
        )


# ============================================================
# CLI
# ============================================================

def main() -> int:
    """CLI entry point."""

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Champion / Challenger "
            "promotion engine."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Check promotion eligibility "
            "without modifying the registry."
        ),
    )

    parser.add_argument(
        "--rollback",
        action="store_true",
        help=(
            "Restore the previous Champion."
        ),
    )

    args = parser.parse_args()

    if args.rollback:

        result = (
            rollback_to_previous_champion()
        )

    else:

        result = (
            promote_challenger(
                dry_run=args.dry_run
            )
        )

    print()

    print("=" * 70)

    print("MODEL PROMOTION RESULT")

    print("=" * 70)

    for key, value in result.items():

        if key == "evaluation":
            continue

        print(
            f"{key}: {value}"
        )

    return (
        0
        if result.get("status")
        not in {
            "PROMOTION_FAILED",
            "ROLLBACK_FAILED",
        }
        else 1
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

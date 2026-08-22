#!/usr/bin/env python3

"""
Production Model Registry.

This module manages the registry that identifies which model
is currently active in production.

The registry stores information about:

    - Champion model
    - Previous champion
    - Challenger model
    - Model paths
    - Promotion timestamps
    - Model metadata

The registry is stored as JSON.

Default location:

    data/models/model_registry.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(
    "model_registry"
)


# ============================================================
# TIME
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
    """Convert configuration objects to dictionaries."""

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


# ============================================================
# REGISTRY PATH
# ============================================================

def get_registry_path() -> Path:
    """
    Get the production model registry path.

    Supported configuration:

        model_registry:
            path: data/models/model_registry.json

    or:

        models:
            registry_path: data/models/model_registry.json
    """

    cfg = load_config()

    candidates: list[Any] = []

    if cfg is not None:

        registry_section = getattr(
            cfg,
            "model_registry",
            None,
        )

        registry_values = object_to_dict(
            registry_section
        )

        for key in [
            "path",
            "registry_path",
        ]:

            if registry_values.get(key):
                candidates.append(
                    registry_values[key]
                )

        models_section = getattr(
            cfg,
            "models",
            None,
        )

        models_values = object_to_dict(
            models_section
        )

        for key in [
            "registry_path",
            "model_registry",
        ]:

            if models_values.get(key):
                candidates.append(
                    models_values[key]
                )

    for candidate in candidates:

        if candidate:

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
        / "models"
        / "model_registry.json"
    )


# ============================================================
# DEFAULT REGISTRY
# ============================================================

def get_default_registry() -> dict[str, Any]:
    """
    Return an empty production model registry.
    """

    return {
        "version": 1,
        "updated_at": utc_now_iso(),
        "champion": None,
        "previous_champion": None,
        "challenger": None,
        "history": [],
    }


# ============================================================
# REGISTRY IO
# ============================================================

def load_registry() -> dict[str, Any]:
    """
    Load the production model registry.

    If the registry does not exist, an empty registry
    structure is returned.
    """

    path = get_registry_path()

    if not path.exists():

        logger.warning(
            "Model registry does not exist: %s",
            path,
        )

        return get_default_registry()

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

    except Exception as error:

        logger.error(
            "Could not load model registry: %s",
            error,
        )

        raise RuntimeError(
            "Model registry could not be loaded."
        ) from error

    if not isinstance(
        data,
        dict,
    ):

        raise RuntimeError(
            "Model registry must contain "
            "a JSON object."
        )

    defaults = get_default_registry()

    for key, value in defaults.items():

        if key not in data:

            data[key] = value

    return data


def save_registry(
    registry: dict[str, Any],
) -> Path:
    """
    Save the production model registry safely.
    """

    if not isinstance(
        registry,
        dict,
    ):

        raise TypeError(
            "Registry must be a dictionary."
        )

    path = get_registry_path()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry = dict(
        registry
    )

    registry["updated_at"] = (
        utc_now_iso()
    )

    temporary_path = (
        path.with_suffix(
            ".tmp"
        )
    )

    try:

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                registry,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        temporary_path.replace(
            path
        )

    except Exception as error:

        logger.error(
            "Could not save model registry: %s",
            error,
        )

        if temporary_path.exists():

            try:
                temporary_path.unlink()
            except Exception:
                pass

        raise

    logger.info(
        "Model registry saved: %s",
        path,
    )

    return path


# ============================================================
# MODEL PATH HELPERS
# ============================================================

def resolve_model_path(
    value: Any,
) -> Path | None:
    """
    Convert a model path into an absolute Path.
    """

    if value is None:
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    path = Path(
        text
    )

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            / path
        )

    return path


def get_model_path(
    model_info: dict[str, Any] | None,
) -> Path | None:
    """
    Extract the model file path from registry metadata.
    """

    if not model_info:

        return None

    for key in [
        "path",
        "model_path",
        "file_path",
        "artifact_path",
    ]:

        if model_info.get(key):

            return resolve_model_path(
                model_info[key]
            )

    return None


# ============================================================
# CHAMPION
# ============================================================

def get_champion() -> dict[str, Any] | None:
    """
    Return the currently registered Champion.
    """

    registry = load_registry()

    champion = registry.get(
        "champion"
    )

    if not isinstance(
        champion,
        dict,
    ):

        return None

    return dict(
        champion
    )


def get_champion_path() -> Path | None:
    """
    Return the current Champion model path.
    """

    champion = get_champion()

    return get_model_path(
        champion
    )


def has_champion() -> bool:
    """
    Check whether a valid Champion exists.
    """

    path = get_champion_path()

    if path is None:
        return False

    return path.exists()


# ============================================================
# CHALLENGER
# ============================================================

def get_challenger() -> dict[str, Any] | None:
    """
    Return the currently registered Challenger.
    """

    registry = load_registry()

    challenger = registry.get(
        "challenger"
    )

    if not isinstance(
        challenger,
        dict,
    ):

        return None

    return dict(
        challenger
    )


# ============================================================
# MODEL REGISTRATION
# ============================================================

def create_model_record(
    model_name: str,
    model_path: str | Path,
    model_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a normalized model registry record.
    """

    path = resolve_model_path(
        model_path
    )

    if path is None:

        raise ValueError(
            "Model path is required."
        )

    try:

        relative_path = path.relative_to(
            PROJECT_ROOT
        )

        stored_path = str(
            relative_path
        )

    except ValueError:

        stored_path = str(
            path
        )

    record: dict[str, Any] = {
        "name": str(
            model_name
        ),
        "path": stored_path,
        "model_type": (
            model_type
            if model_type
            else "unknown"
        ),
        "registered_at": utc_now_iso(),
    }

    if metadata:

        record["metadata"] = dict(
            metadata
        )

    return record


def register_challenger(
    model_name: str,
    model_path: str | Path,
    model_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Register a Challenger model.
    """

    registry = load_registry()

    challenger = (
        create_model_record(
            model_name=model_name,
            model_path=model_path,
            model_type=model_type,
            metadata=metadata,
        )
    )

    registry["challenger"] = (
        challenger
    )

    history = registry.setdefault(
        "history",
        [],
    )

    history.append(
        {
            "event": (
                "CHALLENGER_REGISTERED"
            ),
            "timestamp": utc_now_iso(),
            "model": challenger,
        }
    )

    save_registry(
        registry
    )

    logger.info(
        "Registered Challenger: %s",
        challenger.get("name"),
    )

    return challenger


def set_champion(
    model_name: str,
    model_path: str | Path,
    model_type: str | None = None,
    metadata: dict[str, Any] | None = None,
    reason: str = "MANUAL",
) -> dict[str, Any]:
    """
    Set a model as the production Champion.

    The existing Champion is automatically preserved
    as previous_champion.
    """

    registry = load_registry()

    old_champion = registry.get(
        "champion"
    )

    champion = (
        create_model_record(
            model_name=model_name,
            model_path=model_path,
            model_type=model_type,
            metadata=metadata,
        )
    )

    if old_champion:

        registry[
            "previous_champion"
        ] = old_champion

    registry["champion"] = (
        champion
    )

    history = registry.setdefault(
        "history",
        [],
    )

    history.append(
        {
            "event": "CHAMPION_SET",
            "timestamp": utc_now_iso(),
            "reason": reason,
            "model": champion,
            "previous_champion": (
                old_champion
            ),
        }
    )

    save_registry(
        registry
    )

    logger.info(
        "Champion set: %s",
        champion.get("name"),
    )

    return champion


# ============================================================
# PROMOTE REGISTERED CHALLENGER
# ============================================================

def promote_registered_challenger(
    reason: str = "AUTOMATIC_PROMOTION",
) -> dict[str, Any]:
    """
    Promote the currently registered Challenger.

    The current Champion becomes previous_champion.
    """

    registry = load_registry()

    challenger = registry.get(
        "challenger"
    )

    if not isinstance(
        challenger,
        dict,
    ):

        raise RuntimeError(
            "No Challenger is registered."
        )

    challenger_path = get_model_path(
        challenger
    )

    if challenger_path is None:

        raise RuntimeError(
            "Challenger model path is missing."
        )

    if not challenger_path.exists():

        raise RuntimeError(
            "Challenger model file does not exist: "
            f"{challenger_path}"
        )

    previous_champion = registry.get(
        "champion"
    )

    registry[
        "previous_champion"
    ] = previous_champion

    promoted_champion = dict(
        challenger
    )

    promoted_champion[
        "promoted_at"
    ] = utc_now_iso()

    promoted_champion[
        "promotion_reason"
    ] = reason

    registry["champion"] = (
        promoted_champion
    )

    registry["challenger"] = None

    history = registry.setdefault(
        "history",
        [],
    )

    history.append(
        {
            "event": "CHALLENGER_PROMOTED",
            "timestamp": utc_now_iso(),
            "reason": reason,
            "champion": (
                promoted_champion
            ),
            "previous_champion": (
                previous_champion
            ),
        }
    )

    save_registry(
        registry
    )

    logger.info(
        "Challenger promoted to Champion: %s",
        promoted_champion.get("name"),
    )

    return promoted_champion


# ============================================================
# ROLLBACK
# ============================================================

def rollback_to_previous_champion(
    reason: str = "MANUAL_ROLLBACK",
) -> dict[str, Any]:
    """
    Roll back to the previous Champion.
    """

    registry = load_registry()

    current_champion = registry.get(
        "champion"
    )

    previous_champion = registry.get(
        "previous_champion"
    )

    if not isinstance(
        previous_champion,
        dict,
    ):

        raise RuntimeError(
            "No previous Champion is available."
        )

    previous_path = get_model_path(
        previous_champion
    )

    if previous_path is None:

        raise RuntimeError(
            "Previous Champion path is missing."
        )

    if not previous_path.exists():

        raise RuntimeError(
            "Previous Champion file does not exist: "
            f"{previous_path}"
        )

    restored_champion = dict(
        previous_champion
    )

    restored_champion[
        "restored_at"
    ] = utc_now_iso()

    restored_champion[
        "rollback_reason"
    ] = reason

    registry["champion"] = (
        restored_champion
    )

    registry[
        "previous_champion"
    ] = current_champion

    history = registry.setdefault(
        "history",
        [],
    )

    history.append(
        {
            "event": "ROLLBACK",
            "timestamp": utc_now_iso(),
            "reason": reason,
            "restored_champion": (
                restored_champion
            ),
            "replaced_champion": (
                current_champion
            ),
        }
    )

    save_registry(
        registry
    )

    logger.warning(
        "Rolled back Champion to: %s",
        restored_champion.get("name"),
    )

    return restored_champion


# ============================================================
# STATUS
# ============================================================

def get_registry_status() -> dict[str, Any]:
    """
    Return a safe summary of registry status.
    """

    registry = load_registry()

    champion = registry.get(
        "champion"
    )

    challenger = registry.get(
        "challenger"
    )

    previous = registry.get(
        "previous_champion"
    )

    champion_path = get_model_path(
        champion
        if isinstance(champion, dict)
        else None
    )

    return {
        "registry_path": str(
            get_registry_path()
        ),
        "champion": champion,
        "champion_path": (
            str(champion_path)
            if champion_path
            else None
        ),
        "champion_exists": (
            champion_path.exists()
            if champion_path
            else False
        ),
        "challenger": challenger,
        "previous_champion": previous,
        "updated_at": registry.get(
            "updated_at"
        ),
        "history_count": len(
            registry.get(
                "history",
                [],
            )
        ),
    }


# ============================================================
# CLI
# ============================================================

def main() -> int:
    """Display model registry status."""

    import json

    status = get_registry_status()

    print()

    print("=" * 70)

    print("PRODUCTION MODEL REGISTRY")

    print("=" * 70)

    print(
        json.dumps(
            status,
            indent=2,
            default=str,
        )
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

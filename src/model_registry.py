#!/usr/bin/env python3

"""
Model Registry.

Maintains the current production Champion model and Challenger
model metadata.

The registry stores:

    - model names
    - model versions
    - model paths
    - model status
    - promotion history

States
------

CHAMPION
    Current production model.

CHALLENGER
    Candidate model being evaluated.

ARCHIVED
    Previous model retained for historical reference.
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

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger("model_registry")


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


# ============================================================
# REGISTRY PATH
# ============================================================

def get_registry_path() -> Path:
    """Get model registry file path."""

    cfg = load_config()

    if cfg is not None:

        section = getattr(
            cfg,
            "model_registry",
            None,
        )

        values = object_to_dict(
            section
        )

        path_value = values.get(
            "registry_file"
        )

        if path_value:

            path = Path(
                str(path_value)
            )

            if not path.is_absolute():
                path = PROJECT_ROOT / path

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

def default_registry() -> dict[str, Any]:
    """Create an empty model registry."""

    return {
        "champion": None,
        "challenger": None,
        "history": [],
        "updated_at": utc_now_iso(),
    }


# ============================================================
# LOAD / SAVE
# ============================================================

def load_registry() -> dict[str, Any]:
    """Load the persistent model registry."""

    path = get_registry_path()

    if not path.exists():
        return default_registry()

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            registry = json.load(
                file
            )

        if not isinstance(
            registry,
            dict,
        ):
            return default_registry()

        result = default_registry()

        result.update(
            registry
        )

        return result

    except Exception as error:

        logger.error(
            "Could not load model registry: %s",
            error,
        )

        return default_registry()


def save_registry(
    registry: dict[str, Any],
) -> None:
    """Save model registry."""

    path = get_registry_path()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry["updated_at"] = (
        utc_now_iso()
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            registry,
            file,
            indent=2,
            default=str,
        )


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_model_record(
    name: str,
    version: str,
    path: str | None = None,
    metrics: dict[str, Any] | None = None,
    status: str = "CHALLENGER",
) -> dict[str, Any]:
    """Build a standard model record."""

    return {
        "name": str(name),
        "version": str(version),
        "path": path,
        "status": str(status).upper(),
        "metrics": metrics or {},
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }


# ============================================================
# CHAMPION
# ============================================================

def get_champion() -> dict[str, Any] | None:
    """Return current champion model."""

    registry = load_registry()

    champion = registry.get(
        "champion"
    )

    if isinstance(
        champion,
        dict,
    ):
        return champion

    return None


def set_champion(
    name: str,
    version: str,
    path: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Set a model as Champion.

    Existing champion is archived.
    """

    registry = load_registry()

    old_champion = registry.get(
        "champion"
    )

    if isinstance(
        old_champion,
        dict,
    ):

        old_champion["status"] = (
            "ARCHIVED"
        )

        old_champion["archived_at"] = (
            utc_now_iso()
        )

        registry["history"].append(
            old_champion
        )

    champion = build_model_record(
        name=name,
        version=version,
        path=path,
        metrics=metrics,
        status="CHAMPION",
    )

    registry["champion"] = champion

    save_registry(
        registry
    )

    logger.info(
        "Champion set: %s %s",
        name,
        version,
    )

    return champion


# ============================================================
# CHALLENGER
# ============================================================

def get_challenger() -> dict[str, Any] | None:
    """Return current challenger model."""

    registry = load_registry()

    challenger = registry.get(
        "challenger"
    )

    if isinstance(
        challenger,
        dict,
    ):
        return challenger

    return None


def set_challenger(
    name: str,
    version: str,
    path: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register a Challenger model."""

    registry = load_registry()

    challenger = build_model_record(
        name=name,
        version=version,
        path=path,
        metrics=metrics,
        status="CHALLENGER",
    )

    registry["challenger"] = challenger

    save_registry(
        registry
    )

    logger.info(
        "Challenger registered: %s %s",
        name,
        version,
    )

    return challenger


# ============================================================
# METRICS
# ============================================================

def update_model_metrics(
    role: str,
    metrics: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Update Champion or Challenger metrics.
    """

    role = str(
        role
    ).lower()

    if role not in {
        "champion",
        "challenger",
    }:
        raise ValueError(
            "role must be champion or challenger"
        )

    registry = load_registry()

    model = registry.get(
        role
    )

    if not isinstance(
        model,
        dict,
    ):
        return None

    model["metrics"] = metrics
    model["updated_at"] = utc_now_iso()

    registry[role] = model

    save_registry(
        registry
    )

    return model


# ============================================================
# PROMOTION
# ============================================================

def promote_challenger(
    reason: str,
) -> dict[str, Any] | None:
    """
    Promote current Challenger to Champion.

    Existing Champion is archived.
    """

    registry = load_registry()

    challenger = registry.get(
        "challenger"
    )

    if not isinstance(
        challenger,
        dict,
    ):

        logger.warning(
            "No Challenger available "
            "for promotion."
        )

        return None

    old_champion = registry.get(
        "champion"
    )

    if isinstance(
        old_champion,
        dict,
    ):

        old_champion["status"] = (
            "ARCHIVED"
        )

        old_champion["archived_at"] = (
            utc_now_iso()
        )

        old_champion[
            "archive_reason"
        ] = (
            "Replaced by Challenger."
        )

        registry["history"].append(
            old_champion
        )

    challenger["status"] = (
        "CHAMPION"
    )

    challenger["promoted_at"] = (
        utc_now_iso()
    )

    challenger[
        "promotion_reason"
    ] = str(reason)

    challenger["updated_at"] = (
        utc_now_iso()
    )

    registry["champion"] = (
        challenger
    )

    registry["challenger"] = None

    save_registry(
        registry
    )

    logger.warning(
        "CHALLENGER PROMOTED TO CHAMPION | %s",
        reason,
    )

    return challenger


# ============================================================
# STATUS
# ============================================================

def get_registry_status() -> dict[str, Any]:
    """Return registry summary."""

    registry = load_registry()

    champion = registry.get(
        "champion"
    )

    challenger = registry.get(
        "challenger"
    )

    return {
        "champion": champion,
        "challenger": challenger,
        "history_count": len(
            registry.get(
                "history",
                [],
            )
        ),
        "updated_at": registry.get(
            "updated_at"
        ),
    }


# ============================================================
# CLI
# ============================================================

def main() -> int:

    status = get_registry_status()

    print()

    print("=" * 70)

    print("MODEL REGISTRY")

    print("=" * 70)

    champion = status.get(
        "champion"
    )

    challenger = status.get(
        "challenger"
    )

    print()

    print("Champion:")

    print(
        champion
        if champion
        else "None"
    )

    print()

    print("Challenger:")

    print(
        challenger
        if challenger
        else "None"
    )

    print()

    print(
        "Archived models: "
        f"{status.get('history_count')}"
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

#!/usr/bin/env python3

"""
Register a trained ProductionModel as a Challenger.

Expected workflow:

    train_model.py
          ↓
    challenger_model.joblib
          ↓
    register_model.py
          ↓
    model_registry.json
          ↓
    Challenger

Example:

    python -m src.register_model
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


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
# LOGGING
# ============================================================

logger = logging.getLogger(
    "register_model"
)


# ============================================================
# DEFAULT MODEL PATH
# ============================================================

def get_default_model_path() -> Path:
    """
    Return the default path used by train_model.py.
    """

    return (
        PROJECT_ROOT
        / "data"
        / "models"
        / "challenger_model.joblib"
    )


# ============================================================
# LOAD SAVED METADATA
# ============================================================

def load_model_metadata(
    model_path: Path,
) -> dict[str, Any]:
    """
    Load metadata stored inside the trained model payload.

    Expected format:

        {
            "model": ProductionModel,
            "metadata": {...}
        }
    """

    try:

        import joblib

        payload = joblib.load(
            model_path
        )

    except Exception as error:

        logger.warning(
            "Could not load model metadata: %s",
            error,
        )

        return {}

    if not isinstance(
        payload,
        dict,
    ):

        return {}

    metadata = payload.get(
        "metadata",
        {},
    )

    if not isinstance(
        metadata,
        dict,
    ):

        return {}

    return dict(
        metadata
    )


# ============================================================
# REGISTER MODEL
# ============================================================

def register_model(
    model_path: str | Path | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Register a trained model as the current Challenger.
    """

    from src.model_registry import (
        register_challenger,
    )

    # --------------------------------------------------------
    # MODEL PATH
    # --------------------------------------------------------

    if model_path is None:

        path = get_default_model_path()

    else:

        path = Path(
            model_path
        )

        if not path.is_absolute():

            path = (
                PROJECT_ROOT
                / path
            )

    if not path.exists():

        raise FileNotFoundError(
            "Trained model not found: "
            f"{path}"
        )

    # --------------------------------------------------------
    # LOAD TRAINING METADATA
    # --------------------------------------------------------

    metadata = load_model_metadata(
        path
    )

    # --------------------------------------------------------
    # MODEL NAME
    # --------------------------------------------------------

    if model_name is None:

        model_name = str(
            metadata.get(
                "model_version",
                "production-model-v1",
            )
        )

    # --------------------------------------------------------
    # REGISTER CHALLENGER
    # --------------------------------------------------------

    challenger = register_challenger(
        model_name=model_name,
        model_path=path,
        model_type="ProductionModel",
        metadata=metadata,
    )

    logger.info(
        "Model registered as Challenger | "
        "name=%s | path=%s",
        challenger.get(
            "name"
        ),
        path,
    )

    return challenger


# ============================================================
# CLI
# ============================================================

def main() -> int:
    """
    Register a trained model as Challenger.

    Examples:

        python -m src.register_model

        python -m src.register_model \
            --model data/models/my_model.joblib

        python -m src.register_model \
            --name production-model-v2
    """

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Register a trained model "
            "as a production Challenger."
        ),
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Path to the trained model. "
            "Defaults to "
            "data/models/challenger_model.joblib"
        ),
    )

    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Optional model name. "
            "Defaults to the saved "
            "model_version."
        ),
    )

    args = parser.parse_args()

    try:

        challenger = register_model(
            model_path=args.model,
            model_name=args.name,
        )

        print()

        print("=" * 70)
        print("CHALLENGER REGISTERED")
        print("=" * 70)

        print()

        for key, value in (
            challenger.items()
        ):

            print(
                f"{key}: {value}"
            )

        print()

        print(
            "SUCCESS: Model is ready "
            "for evaluation and promotion."
        )

        return 0

    except Exception as error:

        logger.exception(
            "Model registration failed."
        )

        print()

        print(
            f"ERROR: {error}"
        )

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

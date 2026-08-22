#!/usr/bin/env python3

"""
Production Prediction Service.

This module loads the current production Champion model
and performs predictions.

Flow:

    Input
      │
      ▼
    Load Champion
      │
      ▼
    Validate input
      │
      ▼
    model.predict()
      │
      ▼
    Return prediction + metadata
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(
    "predict"
)


# ============================================================
# LOAD PRODUCTION MODEL
# ============================================================

def get_production_model(
    use_cache: bool = True,
) -> tuple[
    Any,
    dict[str, Any],
]:
    """
    Load the current production Champion model.

    Returns:

        model
        metadata
    """

    from src.model_loader import (
        load_champion_model,
    )

    model, metadata = (
        load_champion_model(
            use_cache=use_cache
        )
    )

    return (
        model,
        metadata,
    )


# ============================================================
# PREDICT
# ============================================================

def predict(
    data: Any,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Run a prediction using the current Champion model.

    Parameters:

        data:
            Input data accepted by the model.

        use_cache:
            Whether to reuse a cached model.

    Returns:

        Dictionary containing prediction
        and production model metadata.
    """

    model, metadata = (
        get_production_model(
            use_cache=use_cache
        )
    )

    if not hasattr(
        model,
        "predict",
    ):

        raise RuntimeError(
            "Loaded Champion model does not "
            "provide a predict() method."
        )

    try:

        prediction = model.predict(
            data
        )

    except Exception as error:

        logger.exception(
            "Prediction failed."
        )

        raise RuntimeError(
            "Production prediction failed."
        ) from error

    logger.info(
        "Prediction completed | "
        "model=%s",
        metadata.get(
            "name"
        ),
    )

    return {
        "prediction": prediction,
        "model": {
            "role": metadata.get(
                "role"
            ),
            "name": metadata.get(
                "name"
            ),
            "model_type": metadata.get(
                "model_type"
            ),
            "model_path": metadata.get(
                "model_path"
            ),
            "registered_at": metadata.get(
                "registered_at"
            ),
            "promoted_at": metadata.get(
                "promoted_at"
            ),
        },
    }


# ============================================================
# PREDICT PROBABILITY
# ============================================================

def predict_proba(
    data: Any,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Run probability prediction using
    the current Champion model.

    Requires the model to implement:

        predict_proba()
    """

    model, metadata = (
        get_production_model(
            use_cache=use_cache
        )
    )

    if not hasattr(
        model,
        "predict_proba",
    ):

        raise RuntimeError(
            "Loaded Champion model does not "
            "provide predict_proba()."
        )

    try:

        probabilities = (
            model.predict_proba(
                data
            )
        )

    except Exception as error:

        logger.exception(
            "Probability prediction failed."
        )

        raise RuntimeError(
            "Production probability "
            "prediction failed."
        ) from error

    return {
        "probabilities": probabilities,
        "model": {
            "role": metadata.get(
                "role"
            ),
            "name": metadata.get(
                "name"
            ),
            "model_type": metadata.get(
                "model_type"
            ),
        },
    }


# ============================================================
# SAFE PREDICTION
# ============================================================

def try_predict(
    data: Any,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Safely run a production prediction.

    Returns a status dictionary instead
    of raising an exception.
    """

    try:

        result = predict(
            data=data,
            use_cache=use_cache,
        )

        return {
            "success": True,
            "error": None,
            **result,
        }

    except Exception as error:

        logger.error(
            "Prediction failed: %s",
            error,
        )

        return {
            "success": False,
            "prediction": None,
            "model": None,
            "error": str(
                error
            ),
        }


# ============================================================
# CLI TEST
# ============================================================

def main() -> int:
    """
    Test whether the production model
    can be loaded successfully.
    """

    from src.model_loader import (
        try_load_champion_model,
    )

    model, metadata = (
        try_load_champion_model()
    )

    print()

    print("=" * 70)

    print(
        "PRODUCTION PREDICTION SERVICE"
    )

    print("=" * 70)

    print(
        f"Loaded: "
        f"{metadata.get('loaded')}"
    )

    print(
        f"Model: "
        f"{metadata.get('name')}"
    )

    print(
        f"Path: "
        f"{metadata.get('model_path')}"
    )

    if model is None:

        print()

        print(
            "No production model "
            "is available."
        )

        print(
            f"Error: "
            f"{metadata.get('error')}"
        )

        return 1

    print()

    print(
        "Model loaded successfully."
    )

    print(
        "Use predict(data) from Python "
        "to run predictions."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

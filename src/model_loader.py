#!/usr/bin/env python3

"""
Dynamic Production Model Loader.

This module loads the currently registered Champion model
from the production model registry.

Loading flow:

    Model Registry
          │
          ▼
    Current Champion
          │
          ▼
    Validate model path
          │
          ▼
    Detect model format
          │
          ▼
    Load model
          │
          ▼
    Return production model

Supported formats:

    .pkl
    .pickle
    .joblib

The loader uses:

    joblib
    pickle fallback
"""

from __future__ import annotations

import logging
import pickle
import sys
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
    "model_loader"
)


# ============================================================
# MODEL CACHE
# ============================================================

_MODEL_CACHE: dict[
    str,
    Any,
] = {}


# ============================================================
# LOAD MODEL FILE
# ============================================================

def load_model_file(
    model_path: str | Path,
) -> Any:
    """
    Load a serialized model.

    Supports:

        .joblib
        .pkl
        .pickle
    """

    path = Path(
        model_path
    )

    if not path.exists():

        raise FileNotFoundError(
            "Model file does not exist: "
            f"{path}"
        )

    cache_key = str(
        path.resolve()
    )

    if cache_key in _MODEL_CACHE:

        logger.info(
            "Using cached model: %s",
            path,
        )

        return _MODEL_CACHE[
            cache_key
        ]

    suffix = path.suffix.lower()

    # --------------------------------------------------------
    # JOBLIB
    # --------------------------------------------------------

    if suffix == ".joblib":

        try:

            import joblib

            model = joblib.load(
                path
            )

            _MODEL_CACHE[
                cache_key
            ] = model

            logger.info(
                "Loaded joblib model: %s",
                path,
            )

            return model

        except ImportError as error:

            raise RuntimeError(
                "joblib is required to load "
                "this model."
            ) from error

    # --------------------------------------------------------
    # PICKLE
    # --------------------------------------------------------

    if suffix in {
        ".pkl",
        ".pickle",
    }:

        try:

            with path.open(
                "rb"
            ) as file:

                model = pickle.load(
                    file
                )

        except Exception as pickle_error:

            # Try joblib as fallback.
            try:

                import joblib

                model = joblib.load(
                    path
                )

            except Exception:

                raise RuntimeError(
                    "Could not load model file: "
                    f"{path}"
                ) from pickle_error

        _MODEL_CACHE[
            cache_key
        ] = model

        logger.info(
            "Loaded pickle model: %s",
            path,
        )

        return model

    # --------------------------------------------------------
    # UNKNOWN FORMAT
    # --------------------------------------------------------

    try:

        import joblib

        model = joblib.load(
            path
        )

        _MODEL_CACHE[
            cache_key
        ] = model

        logger.info(
            "Loaded model using joblib fallback: %s",
            path,
        )

        return model

    except Exception:

        pass

    try:

        with path.open(
            "rb"
        ) as file:

            model = pickle.load(
                file
            )

        _MODEL_CACHE[
            cache_key
        ] = model

        logger.info(
            "Loaded model using pickle fallback: %s",
            path,
        )

        return model

    except Exception as error:

        raise RuntimeError(
            "Unsupported or unreadable "
            f"model format: {path}"
        ) from error


# ============================================================
# LOAD CHAMPION
# ============================================================

def load_champion_model(
    use_cache: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """
    Load the currently registered production Champion.

    Returns:

        model
        metadata
    """

    from src.model_registry import (
        get_champion,
        get_champion_path,
    )

    champion = get_champion()

    if champion is None:

        raise RuntimeError(
            "No production Champion is registered."
        )

    model_path = get_champion_path()

    if model_path is None:

        raise RuntimeError(
            "Champion model path is missing."
        )

    if not model_path.exists():

        raise FileNotFoundError(
            "Champion model does not exist: "
            f"{model_path}"
        )

    cache_key = str(
        model_path.resolve()
    )

    if (
        not use_cache
        and cache_key in _MODEL_CACHE
    ):

        del _MODEL_CACHE[
            cache_key
        ]

    model = load_model_file(
        model_path
    )

    metadata = {
        "role": "CHAMPION",
        "name": champion.get(
            "name"
        ),
        "model_path": str(
            model_path
        ),
        "model_type": champion.get(
            "model_type"
        ),
        "registered_at": champion.get(
            "registered_at"
        ),
        "promoted_at": champion.get(
            "promoted_at"
        ),
        "metadata": champion.get(
            "metadata",
            {},
        ),
    }

    logger.info(
        "Production Champion loaded | "
        "name=%s | path=%s",
        metadata.get("name"),
        model_path,
    )

    return (
        model,
        metadata,
    )


# ============================================================
# SAFE LOADER
# ============================================================

def try_load_champion_model() -> tuple[
    Any | None,
    dict[str, Any],
]:
    """
    Safely attempt to load the Champion.

    Returns:

        model or None
        metadata/status
    """

    try:

        model, metadata = (
            load_champion_model()
        )

        metadata["loaded"] = True
        metadata["error"] = None

        return (
            model,
            metadata,
        )

    except Exception as error:

        logger.error(
            "Could not load Champion: %s",
            error,
        )

        return (
            None,
            {
                "role": "CHAMPION",
                "loaded": False,
                "error": str(error),
            },
        )


# ============================================================
# CACHE
# ============================================================

def clear_model_cache() -> None:
    """
    Clear all cached models.

    Useful after model promotion.
    """

    _MODEL_CACHE.clear()

    logger.info(
        "Production model cache cleared."
    )


# ============================================================
# CLI TEST
# ============================================================

def main() -> int:
    """Test Champion loading."""

    model, metadata = (
        try_load_champion_model()
    )

    print()

    print("=" * 70)

    print("PRODUCTION MODEL LOADER")

    print("=" * 70)

    print(
        f"Loaded: {metadata.get('loaded')}"
    )

    print(
        f"Name: {metadata.get('name')}"
    )

    print(
        f"Path: {metadata.get('model_path')}"
    )

    if metadata.get("error"):

        print(
            f"Error: {metadata.get('error')}"
        )

        return 1

    print()

    print(
        "Model type: "
        f"{type(model).__name__}"
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

#!/usr/bin/env python3

"""
Dynamic Production Model Loader.

Loads the currently registered Champion model
from the production model registry.

Supported formats:

    .joblib
    .pkl
    .pickle

Also supports training payloads saved as:

    {
        "model": ProductionModel,
        "metadata": {...}
    }

The public function used by the prediction
pipeline is:

    load_champion_model()
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

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


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
# LOAD SERIALIZED FILE
# ============================================================

def load_model_file(
    model_path: str | Path,
) -> Any:
    """
    Load a serialized model file.

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

    loaded: Any = None

    # --------------------------------------------------------
    # JOBLIB
    # --------------------------------------------------------

    if suffix == ".joblib":

        try:

            import joblib

            loaded = joblib.load(
                path
            )

        except ImportError as error:

            raise RuntimeError(
                "joblib is required "
                "to load this model."
            ) from error

    # --------------------------------------------------------
    # PICKLE
    # --------------------------------------------------------

    elif suffix in {
        ".pkl",
        ".pickle",
    }:

        try:

            with path.open(
                "rb"
            ) as file:

                loaded = pickle.load(
                    file
                )

        except Exception as pickle_error:

            try:

                import joblib

                loaded = joblib.load(
                    path
                )

            except Exception as error:

                raise RuntimeError(
                    "Could not load model file: "
                    f"{path}"
                ) from pickle_error

    # --------------------------------------------------------
    # UNKNOWN FORMAT
    # --------------------------------------------------------

    else:

        try:

            import joblib

            loaded = joblib.load(
                path
            )

        except Exception:

            try:

                with path.open(
                    "rb"
                ) as file:

                    loaded = pickle.load(
                        file
                    )

            except Exception as error:

                raise RuntimeError(
                    "Unsupported or unreadable "
                    f"model format: {path}"
                ) from error

    _MODEL_CACHE[
        cache_key
    ] = loaded

    logger.info(
        "Loaded model file: %s",
        path,
    )

    return loaded


# ============================================================
# EXTRACT MODEL PAYLOAD
# ============================================================

def extract_model_payload(
    loaded: Any,
) -> tuple[
    Any,
    dict[str, Any],
]:
    """
    Extract the actual prediction model.

    Supports either:

        ProductionModel

    or:

        {
            "model": ProductionModel,
            "metadata": {...}
        }
    """

    if isinstance(
        loaded,
        dict,
    ):

        model = loaded.get(
            "model"
        )

        metadata = loaded.get(
            "metadata",
            {},
        )

        if model is None:

            raise RuntimeError(
                "Saved model payload does not "
                "contain a 'model' entry."
            )

        if not isinstance(
            metadata,
            dict,
        ):

            metadata = {}

        return (
            model,
            dict(metadata),
        )

    return (
        loaded,
        {},
    )


# ============================================================
# LOAD CHAMPION
# ============================================================

def load_champion_model(
    use_cache: bool = True,
) -> tuple[
    Any,
    dict[str, Any],
]:
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
            "No production Champion "
            "is registered."
        )

    model_path = get_champion_path()

    if model_path is None:

        raise RuntimeError(
            "Champion model path "
            "is missing."
        )

    model_path = Path(
        model_path
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

    # --------------------------------------------------------
    # LOAD FILE
    # --------------------------------------------------------

    loaded = load_model_file(
        model_path
    )

    # --------------------------------------------------------
    # EXTRACT ACTUAL MODEL
    # --------------------------------------------------------

    model, saved_metadata = (
        extract_model_payload(
            loaded
        )
    )

    # --------------------------------------------------------
    # VALIDATE MODEL
    # --------------------------------------------------------

    predict_function = getattr(
        model,
        "predict",
        None,
    )

    if not callable(
        predict_function
    ):

        raise RuntimeError(
            "Loaded Champion model does not "
            "provide a callable predict() method."
        )

    # --------------------------------------------------------
    # COMBINE METADATA
    # --------------------------------------------------------

    metadata: dict[
        str,
        Any
    ] = {}

    metadata.update(
        saved_metadata
    )

    metadata.update(
        {
            "model_source": "CHAMPION",

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

            "registry_metadata": champion.get(
                "metadata",
                {},
            ),
        }
    )

    metadata.setdefault(
        "model_version",
        getattr(
            model,
            "model_version",
            "unknown",
        ),
    )

    logger.info(
        "Production Champion loaded | "
        "name=%s | path=%s | "
        "model_type=%s",
        metadata.get(
            "name"
        ),
        model_path,
        type(model).__name__,
    )

    return (
        model,
        metadata,
    )


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

def load_production_model() -> tuple[
    Any,
    dict[str, Any],
]:
    """Compatibility alias."""

    return load_champion_model()


def load_active_model() -> tuple[
    Any,
    dict[str, Any],
]:
    """Compatibility alias."""

    return load_champion_model()


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
                "model_source": "NONE",

                "role": "CHAMPION",

                "loaded": False,

                "error": str(
                    error
                ),
            },
        )


# ============================================================
# CACHE MANAGEMENT
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
    """Test Champion model loading."""

    model, metadata = (
        try_load_champion_model()
    )

    print()

    print("=" * 70)

    print(
        "PRODUCTION MODEL LOADER"
    )

    print("=" * 70)

    print()

    print(
        "Loaded:",
        metadata.get(
            "loaded"
        ),
    )

    print(
        "Name:",
        metadata.get(
            "name"
        ),
    )

    print(
        "Path:",
        metadata.get(
            "model_path"
        ),
    )

    print(
        "Source:",
        metadata.get(
            "model_source"
        ),
    )

    if metadata.get(
        "error"
    ):

        print()

        print(
            "Error:",
            metadata.get(
                "error"
            ),
        )

        return 1

    print()

    print(
        "Model type:",
        type(model).__name__,
    )

    print(
        "Model version:",
        metadata.get(
            "model_version"
        ),
    )

    print()

    print(
        "SUCCESS: Champion model "
        "loaded correctly."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

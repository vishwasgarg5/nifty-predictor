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
    Load serialized model
          │
          ▼
    Unwrap training payload if necessary
          │
          ▼
    Return ProductionModel

Supported formats:

    .joblib
    .pkl
    .pickle

Supported saved structures:

    ProductionModel

or:

    {
        "model": ProductionModel,
        "metadata": {...},
    }

The second format is used by train_model.py.
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
    use_cache: bool = True,
) -> Any:
    """
    Load a serialized model file.

    Supports:

        .joblib
        .pkl
        .pickle

    Returns the raw object stored inside
    the serialized file.
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

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    if (
        use_cache
        and cache_key in _MODEL_CACHE
    ):

        logger.info(
            "Using cached model: %s",
            path,
        )

        return _MODEL_CACHE[
            cache_key
        ]

    # --------------------------------------------------------
    # FILE EXTENSION
    # --------------------------------------------------------

    suffix = (
        path.suffix.lower()
    )

    model: Any = None

    # --------------------------------------------------------
    # JOBLIB
    # --------------------------------------------------------

    if suffix == ".joblib":

        try:

            import joblib

            model = joblib.load(
                path
            )

        except ImportError as error:

            raise RuntimeError(
                "joblib is required to "
                "load this model."
            ) from error

        except Exception as error:

            raise RuntimeError(
                "Could not load joblib model: "
                f"{path}"
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

            except Exception as joblib_error:

                raise RuntimeError(
                    "Could not load model file: "
                    f"{path}"
                ) from joblib_error

    # --------------------------------------------------------
    # UNKNOWN FORMAT
    # --------------------------------------------------------

    else:

        # Try joblib first.

        try:

            import joblib

            model = joblib.load(
                path
            )

        except Exception:

            model = None

        # Try pickle.

        if model is None:

            try:

                with path.open(
                    "rb"
                ) as file:

                    model = pickle.load(
                        file
                    )

            except Exception as error:

                raise RuntimeError(
                    "Unsupported or unreadable "
                    f"model format: {path}"
                ) from error

    # --------------------------------------------------------
    # CACHE RESULT
    # --------------------------------------------------------

    if use_cache:

        _MODEL_CACHE[
            cache_key
        ] = model

    logger.info(
        "Loaded model file: %s",
        path,
    )

    return model


# ============================================================
# UNWRAP MODEL PAYLOAD
# ============================================================

def unwrap_model_payload(
    loaded: Any,
) -> tuple[
    Any,
    dict[str, Any],
]:
    """
    Extract the actual model and saved metadata.

    Supports either:

        ProductionModel

    or:

        {
            "model": ProductionModel,
            "metadata": {...},
        }
    """

    # --------------------------------------------------------
    # TRAIN_MODEL.PY FORMAT
    # --------------------------------------------------------

    if isinstance(
        loaded,
        dict,
    ):

        if "model" in loaded:

            model = loaded.get(
                "model"
            )

            metadata = loaded.get(
                "metadata",
                {},
            )

            if not isinstance(
                metadata,
                dict,
            ):

                metadata = {}

            if model is None:

                raise RuntimeError(
                    "Saved model payload contains "
                    "no model."
                )

            return (
                model,
                dict(metadata),
            )

    # --------------------------------------------------------
    # RAW MODEL FORMAT
    # --------------------------------------------------------

    if loaded is None:

        raise RuntimeError(
            "Loaded model is None."
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
    Load the currently registered
    production Champion.

    Returns:

        model
        metadata
    """

    from src.model_registry import (
        get_champion,
        get_champion_path,
    )

    # --------------------------------------------------------
    # GET CHAMPION
    # --------------------------------------------------------

    champion = get_champion()

    if champion is None:

        raise RuntimeError(
            "No production Champion "
            "is registered."
        )

    # --------------------------------------------------------
    # GET MODEL PATH
    # --------------------------------------------------------

    model_path = (
        get_champion_path()
    )

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

    # --------------------------------------------------------
    # LOAD FILE
    # --------------------------------------------------------

    loaded = load_model_file(
        model_path=model_path,
        use_cache=use_cache,
    )

    # --------------------------------------------------------
    # UNWRAP PAYLOAD
    # --------------------------------------------------------

    model, saved_metadata = (
        unwrap_model_payload(
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
            "Loaded Champion does not "
            "provide predict()."
        )

    # --------------------------------------------------------
    # REGISTRY METADATA
    # --------------------------------------------------------

    registry_metadata = (
        champion.get(
            "metadata",
            {},
        )
    )

    if not isinstance(
        registry_metadata,
        dict,
    ):

        registry_metadata = {}

    # --------------------------------------------------------
    # COMBINE METADATA
    # --------------------------------------------------------

    metadata: dict[
        str,
        Any,
    ] = {}

    # Saved metadata comes first.

    if isinstance(
        saved_metadata,
        dict,
    ):

        metadata.update(
            saved_metadata
        )

    # Registry metadata is included separately.

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

            "registry_metadata": (
                registry_metadata
            ),

            "saved_metadata": (
                saved_metadata
            ),

            "loaded": True,

            "error": None,
        }
    )

    # --------------------------------------------------------
    # MODEL VERSION
    # --------------------------------------------------------

    if (
        "model_version"
        not in metadata
    ):

        model_version = getattr(
            model,
            "model_version",
            None,
        )

        if model_version is not None:

            metadata[
                "model_version"
            ] = model_version

    # --------------------------------------------------------
    # MODEL NAME
    # --------------------------------------------------------

    if (
        "model_name"
        not in metadata
    ):

        metadata[
            "model_name"
        ] = type(
            model
        ).__name__

    logger.info(
        "Production Champion loaded | "
        "name=%s | path=%s | "
        "type=%s",
        metadata.get(
            "name"
        ),
        model_path,
        type(
            model
        ).__name__,
    )

    return (
        model,
        metadata,
    )


# ============================================================
# SAFE CHAMPION LOADER
# ============================================================

def try_load_champion_model() -> tuple[
    Any | None,
    dict[str, Any],
]:
    """
    Safely attempt to load
    the production Champion.

    Returns:

        model or None

        metadata/status
    """

    try:

        model, metadata = (
            load_champion_model()
        )

        metadata[
            "loaded"
        ] = True

        metadata[
            "error"
        ] = None

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
# COMPATIBILITY ALIASES
# ============================================================

def load_production_model() -> tuple[
    Any,
    dict[str, Any],
]:
    """
    Compatibility alias for
    load_champion_model().
    """

    return load_champion_model()


def load_active_model() -> tuple[
    Any,
    dict[str, Any],
]:
    """
    Compatibility alias for
    load_champion_model().
    """

    return load_champion_model()


# ============================================================
# CACHE MANAGEMENT
# ============================================================

def clear_model_cache() -> None:
    """
    Clear all cached models.

    Call this after:

        - promoting a new Champion
        - replacing a model file
        - retraining production models
    """

    _MODEL_CACHE.clear()

    logger.info(
        "Production model cache cleared."
    )


def get_cache_size() -> int:
    """Return the number of cached models."""

    return len(
        _MODEL_CACHE
    )


# ============================================================
# CLI TEST
# ============================================================

def main() -> int:
    """
    Test Champion model loading.
    """

    model, metadata = (
        try_load_champion_model()
    )

    print()

    print(
        "=" * 70
    )

    print(
        "PRODUCTION MODEL LOADER"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Loaded:",
        metadata.get(
            "loaded"
        ),
    )

    print(
        "Model source:",
        metadata.get(
            "model_source"
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
        "Version:",
        metadata.get(
            "model_version"
        ),
    )

    if metadata.get(
        "error"
    ):

        print()

        print(
            "ERROR:",
            metadata.get(
                "error"
            ),
        )

        return 1

    print()

    print(
        "Model class:",
        type(
            model
        ).__name__,
    )

    print()

    print(
        "Has predict():",
        callable(
            getattr(
                model,
                "predict",
                None,
            )
        ),
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

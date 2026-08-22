#!/usr/bin/env python3

"""
Production Prediction Pipeline.

This module is the central prediction entry point for the
AI Stock Prediction System.

Pipeline
--------
Universe
    │
    ▼
Load market data
    │
    ▼
Validate data
    │
    ▼
Generate features
    │
    ▼
Load production Champion model
    │
    ├── Champion available
    │       │
    │       ▼
    │   Generate prediction
    │
    └── Champion unavailable
            │
            ▼
    Check fallback policy
            │
            ├── Fallback allowed
            │       │
            ▼
            │   Load existing model
            │
            └── Fallback disabled
                    │
                    ▼
                  Skip / fail

Predictions are returned as a pandas DataFrame.

Primary public entry point:

    run_pipeline()

Compatibility aliases:

    run()
    run_predictions()
"""

from __future__ import annotations

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

logger = logging.getLogger("prediction_pipeline")


# ============================================================
# TIME HELPERS
# ============================================================

def utc_now() -> datetime:
    """Return the current UTC datetime."""

    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return the current UTC timestamp."""

    return utc_now().isoformat()


# ============================================================
# CONFIG HELPERS
# ============================================================

def object_to_dict(
    value: Any,
) -> dict[str, Any]:
    """Convert common configuration objects into dictionaries."""

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

        try:
            return {
                key: item
                for key, item in vars(value).items()
                if not key.startswith("_")
            }

        except Exception:
            pass

    return {}


def load_config() -> Any:
    """Load the project configuration."""

    from src.config import cfg

    return cfg


def get_models_config() -> dict[str, Any]:
    """Get model loading configuration."""

    try:

        cfg = load_config()

        section = getattr(
            cfg,
            "models",
            None,
        )

        values = object_to_dict(
            section
        )

    except Exception as error:

        logger.warning(
            "Could not load models configuration: %s",
            error,
        )

        values = {}

    return {
        "registry_path": values.get(
            "registry_path",
            "data/models/model_registry.json",
        ),
        "require_champion": bool(
            values.get(
                "require_champion",
                False,
            )
        ),
        "allow_fallback_model": bool(
            values.get(
                "allow_fallback_model",
                True,
            )
        ),
        "cache_enabled": bool(
            values.get(
                "cache_enabled",
                True,
            )
        ),
    }


def get_prediction_config() -> dict[str, Any]:
    """Get prediction configuration."""

    try:

        cfg = load_config()

        section = getattr(
            cfg,
            "prediction",
            None,
        )

        values = object_to_dict(
            section
        )

    except Exception:

        values = {}

    top_n = values.get(
        "top_n",
        5,
    )

    try:
        top_n = int(top_n)
    except Exception:
        top_n = 5

    return {
        "enabled": bool(
            values.get(
                "enabled",
                True,
            )
        ),
        "top_n": max(
            1,
            top_n,
        ),
        "minimum_confidence": values.get(
            "minimum_confidence",
            0.0,
        ),
    }


# ============================================================
# DATAFRAME HELPERS
# ============================================================

def ensure_dataframe(
    value: Any,
) -> pd.DataFrame:
    """Convert common output formats into a DataFrame."""

    if value is None:
        return pd.DataFrame()

    if isinstance(
        value,
        pd.DataFrame,
    ):
        return value.copy()

    if isinstance(
        value,
        pd.Series,
    ):
        return value.to_frame().T

    if isinstance(
        value,
        list,
    ):

        try:
            return pd.DataFrame(
                value
            )

        except Exception:
            return pd.DataFrame()

    if isinstance(
        value,
        dict,
    ):

        try:
            return pd.DataFrame(
                [value]
            )

        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Find the first matching DataFrame column."""

    for column in candidates:

        if column in frame.columns:
            return column

    return None


# ============================================================
# UNIVERSE LOADING
# ============================================================

def load_universe() -> list[str]:
    """
    Load the prediction universe.

    Tries common project interfaces from:

        src.universe
        src.universe_filter
    """

    attempts: list[str] = []

    # --------------------------------------------------------
    # src.universe
    # --------------------------------------------------------

    try:

        from src import universe

        for function_name in [
            "get_universe",
            "load_universe",
            "get_symbols",
            "load_symbols",
        ]:

            function = getattr(
                universe,
                function_name,
                None,
            )

            if callable(function):

                logger.info(
                    "Loading universe using "
                    "src.universe.%s()",
                    function_name,
                )

                result = function()

                if isinstance(
                    result,
                    pd.DataFrame,
                ):

                    column = find_column(
                        result,
                        [
                            "symbol",
                            "ticker",
                            "stock",
                        ],
                    )

                    if column:

                        return (
                            result[column]
                            .dropna()
                            .astype(str)
                            .tolist()
                        )

                if isinstance(
                    result,
                    (list, tuple, set),
                ):

                    return [
                        str(item)
                        for item in result
                        if item
                    ]

    except Exception as error:

        attempts.append(
            f"src.universe: {error}"
        )

    # --------------------------------------------------------
    # src.universe_filter
    # --------------------------------------------------------

    try:

        from src import universe_filter

        for function_name in [
            "get_universe",
            "load_universe",
            "get_symbols",
            "filter_universe",
        ]:

            function = getattr(
                universe_filter,
                function_name,
                None,
            )

            if callable(function):

                logger.info(
                    "Loading universe using "
                    "src.universe_filter.%s()",
                    function_name,
                )

                result = function()

                if isinstance(
                    result,
                    pd.DataFrame,
                ):

                    column = find_column(
                        result,
                        [
                            "symbol",
                            "ticker",
                            "stock",
                        ],
                    )

                    if column:

                        return (
                            result[column]
                            .dropna()
                            .astype(str)
                            .tolist()
                        )

                if isinstance(
                    result,
                    (list, tuple, set),
                ):

                    return [
                        str(item)
                        for item in result
                        if item
                    ]

    except Exception as error:

        attempts.append(
            f"src.universe_filter: {error}"
        )

    logger.warning(
        "Could not load prediction universe. "
        "Attempts: %s",
        " | ".join(attempts),
    )

    return []


# ============================================================
# MARKET DATA LOADING
# ============================================================

def load_market_data(
    symbol: str,
) -> pd.DataFrame:
    """
    Load market data for one symbol.

    Tries common interfaces from src.data_loader.
    """

    from src import data_loader

    attempts: list[str] = []

    for function_name in [
        "load_data",
        "get_data",
        "fetch_data",
        "load_market_data",
        "fetch_market_data",
    ]:

        function = getattr(
            data_loader,
            function_name,
            None,
        )

        if not callable(function):
            continue

        try:

            logger.info(
                "Loading market data for %s "
                "using %s()",
                symbol,
                function_name,
            )

            result = function(
                symbol
            )

            frame = ensure_dataframe(
                result
            )

            if not frame.empty:
                return frame

        except TypeError:

            try:

                result = function(
                    symbol=symbol
                )

                frame = ensure_dataframe(
                    result
                )

                if not frame.empty:
                    return frame

            except Exception as error:

                attempts.append(
                    f"{function_name}: {error}"
                )

        except Exception as error:

            attempts.append(
                f"{function_name}: {error}"
            )

    raise RuntimeError(
        f"Could not load market data for {symbol}. "
        + " | ".join(attempts)
    )


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_market_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate market data.

    Uses src.data_validation when available.
    Falls back to basic validation.
    """

    if data is None:
        return pd.DataFrame()

    frame = data.copy()

    try:

        from src import data_validation

        for function_name in [
            "validate_data",
            "validate_market_data",
            "validate",
        ]:

            function = getattr(
                data_validation,
                function_name,
                None,
            )

            if not callable(function):
                continue

            result = function(
                frame
            )

            # Validation functions may return:
            #
            # DataFrame
            # bool
            # tuple(valid, data)
            #
            if isinstance(
                result,
                pd.DataFrame,
            ):

                return result

            if isinstance(
                result,
                tuple,
            ):

                for item in result:

                    if isinstance(
                        item,
                        pd.DataFrame,
                    ):

                        return item

                if result and result[0] is False:
                    return pd.DataFrame()

            if result is False:
                return pd.DataFrame()

            break

    except Exception as error:

        logger.warning(
            "Data validation module failed: %s",
            error,
        )

    return frame.dropna(
        axis=0,
        how="all",
    )


# ============================================================
# FEATURE GENERATION
# ============================================================

def generate_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate model features.

    Tries:

        src.feature_engine
        src.features
    """

    frame = data.copy()

    attempts: list[str] = []

    # --------------------------------------------------------
    # src.feature_engine
    # --------------------------------------------------------

    try:

        from src import feature_engine

        for function_name in [
            "generate_features",
            "create_features",
            "build_features",
            "transform",
        ]:

            function = getattr(
                feature_engine,
                function_name,
                None,
            )

            if not callable(function):
                continue

            logger.info(
                "Generating features using "
                "src.feature_engine.%s()",
                function_name,
            )

            result = function(
                frame
            )

            result_frame = ensure_dataframe(
                result
            )

            if not result_frame.empty:
                return result_frame

    except Exception as error:

        attempts.append(
            f"src.feature_engine: {error}"
        )

    # --------------------------------------------------------
    # src.features
    # --------------------------------------------------------

    try:

        from src import features

        for function_name in [
            "generate_features",
            "create_features",
            "build_features",
        ]:

            function = getattr(
                features,
                function_name,
                None,
            )

            if not callable(function):
                continue

            logger.info(
                "Generating features using "
                "src.features.%s()",
                function_name,
            )

            result = function(
                frame
            )

            result_frame = ensure_dataframe(
                result
            )

            if not result_frame.empty:
                return result_frame

    except Exception as error:

        attempts.append(
            f"src.features: {error}"
        )

    logger.warning(
        "Feature generation fallback used. "
        "Attempts: %s",
        " | ".join(attempts),
    )

    return frame


# ============================================================
# CHAMPION MODEL LOADING
# ============================================================

def load_champion_model() -> tuple[
    Any | None,
    dict[str, Any],
]:
    """
    Load the active production Champion.

    Returns:

        model
        metadata
    """

    try:

        from src import model_loader

        candidates = [
            "load_champion_model",
            "load_production_model",
            "load_active_model",
            "load_model",
        ]

        for function_name in candidates:

            function = getattr(
                model_loader,
                function_name,
                None,
            )

            if not callable(function):
                continue

            try:

                logger.info(
                    "Attempting Champion load using "
                    "src.model_loader.%s()",
                    function_name,
                )

                result = function()

                if isinstance(
                    result,
                    tuple,
                ):

                    model = result[0]

                    metadata: dict[str, Any] = {}

                    if len(result) > 1:

                        metadata = object_to_dict(
                            result[1]
                        )

                    if model is not None:

                        metadata.setdefault(
                            "model_source",
                            "CHAMPION",
                        )

                        return (
                            model,
                            metadata,
                        )

                if isinstance(
                    result,
                    dict,
                ):

                    model = (
                        result.get("model")
                        or result.get("estimator")
                    )

                    metadata = object_to_dict(
                        result
                    )

                    if model is not None:

                        metadata.setdefault(
                            "model_source",
                            "CHAMPION",
                        )

                        return (
                            model,
                            metadata,
                        )

                if result is not None:

                    return (
                        result,
                        {
                            "model_source": (
                                "CHAMPION"
                            ),
                            "model_loader": (
                                function_name
                            ),
                        },
                    )

            except Exception as error:

                logger.warning(
                    "Champion loading attempt failed "
                    "using %s: %s",
                    function_name,
                    error,
                )

    except Exception as error:

        logger.warning(
            "Champion model loader unavailable: %s",
            error,
        )

    return (
        None,
        {
            "model_source": "NONE",
        },
    )


# ============================================================
# FALLBACK MODEL LOADING
# ============================================================

def load_fallback_model(
    symbol: str,
) -> tuple[
    Any | None,
    dict[str, Any],
]:
    """
    Load the existing per-symbol model.

    Tries src.model_store first.
    """

    try:

        from src import model_store

        for function_name in [
            "load_model",
            "get_model",
            "load",
        ]:

            function = getattr(
                model_store,
                function_name,
                None,
            )

            if not callable(function):
                continue

            try:

                result = function(
                    symbol
                )

            except TypeError:

                try:

                    result = function(
                        symbol=symbol
                    )

                except Exception as error:

                    logger.warning(
                        "Fallback model load failed "
                        "for %s: %s",
                        symbol,
                        error,
                    )

                    continue

            except Exception as error:

                logger.warning(
                    "Fallback model load failed "
                    "for %s: %s",
                    symbol,
                    error,
                )

                continue

            if isinstance(
                result,
                tuple,
            ):

                model = result[0]

                metadata = {}

                if len(result) > 1:

                    metadata = object_to_dict(
                        result[1]
                    )

                if model is not None:

                    metadata.setdefault(
                        "model_source",
                        "FALLBACK",
                    )

                    return (
                        model,
                        metadata,
                    )

            if result is not None:

                return (
                    result,
                    {
                        "model_source": (
                            "FALLBACK"
                        ),
                        "symbol": symbol,
                    },
                )

    except Exception as error:

        logger.warning(
            "Fallback model store unavailable: %s",
            error,
        )

    return (
        None,
        {
            "model_source": "NONE",
        },
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_with_model(
    model: Any,
    features: pd.DataFrame,
) -> Any:
    """
    Generate a prediction using a loaded model.
    """

    if model is None:

        raise RuntimeError(
            "Prediction model is None."
        )

    prediction_function = getattr(
        model,
        "predict",
        None,
    )

    if not callable(
        prediction_function
    ):

        raise RuntimeError(
            "Model does not provide predict()."
        )

    return prediction_function(
        features
    )


# ============================================================
# PREDICTION NORMALIZATION
# ============================================================

def normalize_prediction(
    prediction: Any,
) -> dict[str, Any]:
    """
    Convert common prediction outputs into a standard dictionary.
    """

    result: dict[str, Any] = {}

    if prediction is None:
        return result

    if isinstance(
        prediction,
        pd.DataFrame,
    ):

        if prediction.empty:
            return result

        return (
            prediction.iloc[-1]
            .to_dict()
        )

    if isinstance(
        prediction,
        pd.Series,
    ):

        return prediction.to_dict()

    if isinstance(
        prediction,
        dict,
    ):

        return dict(prediction)

    if isinstance(
        prediction,
        (list, tuple),
    ):

        if len(prediction) == 0:
            return result

        first = prediction[0]

        if isinstance(
            first,
            dict,
        ):

            return dict(first)

        if hasattr(
            first,
            "item",
        ):

            try:
                first = first.item()

            except Exception:
                pass

        result[
            "predicted_return"
        ] = first

        return result

    if hasattr(
        prediction,
        "item",
    ):

        try:
            prediction = prediction.item()

        except Exception:
            pass

    result[
        "predicted_return"
    ] = prediction

    return result


# ============================================================
# SINGLE SYMBOL PREDICTION
# ============================================================

def predict_symbol(
    symbol: str,
) -> dict[str, Any] | None:
    """
    Generate a prediction for one symbol.
    """

    model_config = (
        get_models_config()
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    try:

        data = load_market_data(
            symbol
        )

    except Exception as error:

        logger.error(
            "Data loading failed for %s: %s",
            symbol,
            error,
        )

        return None

    # --------------------------------------------------------
    # VALIDATE DATA
    # --------------------------------------------------------

    data = validate_market_data(
        data
    )

    if data.empty:

        logger.warning(
            "No valid data for %s.",
            symbol,
        )

        return None

    # --------------------------------------------------------
    # GENERATE FEATURES
    # --------------------------------------------------------

    try:

        features = generate_features(
            data
        )

    except Exception as error:

        logger.error(
            "Feature generation failed "
            "for %s: %s",
            symbol,
            error,
        )

        return None

    if features.empty:

        logger.warning(
            "No features generated for %s.",
            symbol,
        )

        return None

    # --------------------------------------------------------
    # LOAD CHAMPION
    # --------------------------------------------------------

    champion_model, metadata = (
        load_champion_model()
    )

    model = champion_model

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if model is None:

        require_champion = bool(
            model_config.get(
                "require_champion",
                False,
            )
        )

        allow_fallback = bool(
            model_config.get(
                "allow_fallback_model",
                True,
            )
        )

        if require_champion:

            logger.error(
                "Champion is required but "
                "not available for %s.",
                symbol,
            )

            return None

        if allow_fallback:

            logger.info(
                "Champion unavailable. "
                "Trying fallback model for %s.",
                symbol,
            )

            model, fallback_metadata = (
                load_fallback_model(
                    symbol
                )
            )

            metadata.update(
                fallback_metadata
            )

        if model is None:

            logger.warning(
                "No prediction model available "
                "for %s.",
                symbol,
            )

            return None

    # --------------------------------------------------------
    # USE LATEST FEATURE ROW
    # --------------------------------------------------------

    prediction_features = (
        features.tail(1).copy()
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    try:

        prediction = (
            predict_with_model(
                model,
                prediction_features,
            )
        )

    except Exception as error:

        logger.error(
            "Prediction failed for %s: %s",
            symbol,
            error,
        )

        return None

    # --------------------------------------------------------
    # NORMALIZE OUTPUT
    # --------------------------------------------------------

    result = normalize_prediction(
        prediction
    )

    # --------------------------------------------------------
    # STANDARD METADATA
    # --------------------------------------------------------

    result["symbol"] = symbol

    result.setdefault(
        "prediction_date",
        utc_now_iso(),
    )

    result.setdefault(
        "created_at",
        utc_now_iso(),
    )

    result.setdefault(
        "model_source",
        metadata.get(
            "model_source",
            "UNKNOWN",
        ),
    )

    for key in [
        "model_version",
        "model_name",
        "feature_version",
        "champion_version",
    ]:

        if key in metadata:

            result.setdefault(
                key,
                metadata.get(key),
            )

    return result


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline() -> pd.DataFrame:
    """
    Run the complete production prediction pipeline.

    Returns a DataFrame containing predictions.
    """

    prediction_config = (
        get_prediction_config()
    )

    if not prediction_config.get(
        "enabled",
        True,
    ):

        logger.warning(
            "Prediction pipeline is disabled."
        )

        return pd.DataFrame()

    logger.info(
        "=" * 70
    )

    logger.info(
        "STARTING PRODUCTION "
        "PREDICTION PIPELINE"
    )

    logger.info(
        "=" * 70
    )

    # --------------------------------------------------------
    # LOAD UNIVERSE
    # --------------------------------------------------------

    symbols = load_universe()

    if not symbols:

        logger.warning(
            "Prediction universe is empty."
        )

        return pd.DataFrame()

    logger.info(
        "Loaded %s symbol(s).",
        len(symbols),
    )

    # --------------------------------------------------------
    # GENERATE PREDICTIONS
    # --------------------------------------------------------

    predictions: list[
        dict[str, Any]
    ] = []

    for symbol in symbols:

        try:

            logger.info(
                "Processing symbol: %s",
                symbol,
            )

            prediction = predict_symbol(
                symbol
            )

            if prediction is not None:

                predictions.append(
                    prediction
                )

        except Exception:

            logger.exception(
                "Unexpected failure "
                "processing %s.",
                symbol,
            )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if not predictions:

        logger.warning(
            "No predictions generated."
        )

        return pd.DataFrame()

    result = pd.DataFrame(
        predictions
    )

    # --------------------------------------------------------
    # RANK
    # --------------------------------------------------------

    score_column = find_column(
        result,
        [
            "opportunity_score",
            "quality_score",
            "confidence",
            "predicted_return",
        ],
    )

    if score_column:

        result[
            score_column
        ] = pd.to_numeric(
            result[
                score_column
            ],
            errors="coerce",
        )

        result = result.sort_values(
            by=score_column,
            ascending=False,
            na_position="last",
        )

    result = result.reset_index(
        drop=True
    )

    logger.info(
        "Prediction pipeline completed. "
        "Generated %s prediction(s).",
        len(result),
    )

    return result


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

def run() -> pd.DataFrame:
    """Compatibility alias."""

    return run_pipeline()


def run_predictions() -> pd.DataFrame:
    """Compatibility alias."""

    return run_pipeline()


# ============================================================
# CLI
# ============================================================

def main() -> int:
    """Run the prediction pipeline from the command line."""

    try:

        predictions = run_pipeline()

        print()

        print("=" * 70)

        print(
            "PREDICTION PIPELINE RESULT"
        )

        print("=" * 70)

        print(
            "Predictions generated:",
            len(predictions),
        )

        if not predictions.empty:

            print()

            print(
                predictions.to_string(
                    index=False
                )
            )

        return 0

    except Exception as error:

        logger.error(
            "Prediction pipeline failed: %s",
            error,
        )

        logger.error(
            traceback.format_exc()
        )

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

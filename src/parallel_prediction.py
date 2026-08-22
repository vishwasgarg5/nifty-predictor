#!/usr/bin/env python3

"""
Champion / Challenger Parallel Prediction Engine.

Responsibilities
----------------
1. Load the model registry.
2. Identify the active Champion.
3. Identify active Challenger models.
4. Sort Challengers by newest first.
5. Enforce max_active_challengers.
6. Respect run_champion and run_challengers settings.
7. Run all selected models against the same feature matrix.
8. Return a normalized prediction DataFrame.
9. Allow the existing production pipeline to provide a Champion
   adapter when the Champion architecture is not a single model file.

This module does not rank stocks, select Top 5 opportunities,
or send Telegram messages.

Those responsibilities remain in scripts/morning_job.py.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


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

logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

ACTIVE_CHALLENGER_STATUSES = {
    "CHALLENGER",
    "ACTIVE",
    "EVALUATING",
}

INACTIVE_CHALLENGER_STATUSES = {
    "REJECTED",
    "RETIRED",
    "FAILED",
    "PROMOTED",
    "ARCHIVED",
    "DISABLED",
}


# ============================================================
# RESULT MODEL
# ============================================================

@dataclass
class ModelPredictionResult:
    """Result of a single model prediction."""

    model_name: str

    model_path: str | None

    model_status: str

    success: bool

    predictions: pd.DataFrame

    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable result summary."""

        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "model_status": self.model_status,
            "success": self.success,
            "prediction_count": len(
                self.predictions
            ),
            "error": self.error,
        }


# ============================================================
# PATH HELPERS
# ============================================================

def resolve_project_path(
    value: str | Path,
) -> Path:
    """Resolve a path relative to the project root."""

    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# ============================================================
# CONFIG HELPERS
# ============================================================

def load_parallel_settings() -> dict[str, Any]:
    """
    Load parallel prediction settings from src.config.cfg.

    The engine can still run without configuration and falls back
    to safe defaults.
    """

    defaults = {
        "enabled": True,
        "registry_path": (
            "data/model_registry.json"
        ),
        "run_champion": True,
        "run_challengers": True,
        "champion_name": "current",
        "max_active_challengers": 1,
    }

    try:

        from src.config import cfg

        section = getattr(
            cfg,
            "parallel_prediction",
            None,
        )

        if section is None:
            return defaults

        if isinstance(section, dict):

            defaults.update(
                section
            )

            return defaults

        if hasattr(section, "items"):

            defaults.update(
                dict(section.items())
            )

            return defaults

        if hasattr(section, "__dict__"):

            values = {
                key: value
                for key, value
                in vars(section).items()
                if not key.startswith("_")
            }

            defaults.update(
                values
            )

    except Exception as error:

        logger.debug(
            "Unable to load parallel settings: %s",
            error,
        )

    return defaults


# ============================================================
# REGISTRY
# ============================================================

def empty_registry() -> dict[str, Any]:
    """Return an empty model registry."""

    return {
        "champion": None,
        "challengers": [],
        "history": [],
    }


def load_model_registry(
    registry_path: str | Path,
) -> dict[str, Any]:
    """Load the model registry safely."""

    path = resolve_project_path(
        registry_path
    )

    if not path.exists():

        logger.warning(
            "Model registry does not exist: %s",
            path,
        )

        return empty_registry()

    try:

        with open(
            path,
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
            raise ValueError(
                "Registry root must be an object."
            )

        registry.setdefault(
            "champion",
            None,
        )

        registry.setdefault(
            "challengers",
            [],
        )

        registry.setdefault(
            "history",
            [],
        )

        if not isinstance(
            registry["challengers"],
            list,
        ):

            logger.warning(
                "Invalid challengers registry format."
            )

            registry["challengers"] = []

        return registry

    except Exception as error:

        logger.exception(
            "Failed to load model registry: %s",
            error,
        )

        return empty_registry()


# ============================================================
# TIMESTAMP HELPERS
# ============================================================

def parse_timestamp(
    value: Any,
) -> datetime:
    """
    Convert registry timestamps into datetime.

    Invalid or missing timestamps become datetime.min so that
    malformed entries naturally sort last.
    """

    if value is None:
        return datetime.min

    if isinstance(
        value,
        datetime,
    ):
        return value

    try:

        return datetime.fromisoformat(
            str(value).replace(
                "Z",
                "+00:00",
            )
        )

    except Exception:

        return datetime.min


# ============================================================
# CHAMPION
# ============================================================

def get_active_champion(
    registry: dict[str, Any],
    default_name: str = "current",
) -> dict[str, Any]:
    """Normalize the Champion registry entry."""

    champion = registry.get(
        "champion"
    )

    if isinstance(
        champion,
        dict,
    ):

        return {
            "model_name": str(
                champion.get(
                    "model_name",
                    default_name,
                )
            ),

            "model_path": champion.get(
                "model_path"
            ),

            "status": "CHAMPION",

            "created_at": champion.get(
                "created_at"
            ),
        }

    if isinstance(
        champion,
        str,
    ):

        return {
            "model_name": champion,
            "model_path": None,
            "status": "CHAMPION",
            "created_at": None,
        }

    return {
        "model_name": default_name,
        "model_path": None,
        "status": "CHAMPION",
        "created_at": None,
    }


# ============================================================
# CHALLENGERS
# ============================================================

def get_active_challengers(
    registry: dict[str, Any],
    max_active: int = 1,
) -> list[dict[str, Any]]:
    """
    Return active Challengers.

    Challengers are sorted newest-first using created_at.
    Only max_active models are returned.
    """

    challengers = registry.get(
        "challengers",
        []
    )

    if not isinstance(
        challengers,
        list,
    ):
        return []

    active: list[
        dict[str, Any]
    ] = []

    for challenger in challengers:

        if not isinstance(
            challenger,
            dict,
        ):
            continue

        model_name = challenger.get(
            "model_name"
        )

        if not model_name:
            continue

        status = str(
            challenger.get(
                "status",
                "CHALLENGER",
            )
        ).upper()

        if status in INACTIVE_CHALLENGER_STATUSES:
            continue

        if status not in ACTIVE_CHALLENGER_STATUSES:

            continue

        active.append(
            {
                "model_name": str(
                    model_name
                ),

                "model_path": challenger.get(
                    "model_path"
                ),

                "status": status,

                "created_at": challenger.get(
                    "created_at"
                ),
            }
        )

    active.sort(
        key=lambda item: parse_timestamp(
            item.get(
                "created_at"
            )
        ),
        reverse=True,
    )

    if max_active <= 0:
        return []

    return active[:max_active]


# ============================================================
# MODEL PATH
# ============================================================

def resolve_model_path(
    model_path: str | Path | None,
) -> Path | None:
    """Resolve and validate a model artifact path."""

    if not model_path:
        return None

    path = resolve_project_path(
        model_path
    )

    if not path.exists():

        logger.warning(
            "Model artifact does not exist: %s",
            path,
        )

        return None

    return path


# ============================================================
# MODEL LOADING
# ============================================================

def load_serialized_model(
    model_path: str | Path,
) -> Any:
    """
    Load supported serialized model artifacts.

    Supported:
        .joblib
        .pkl
        .pickle
    """

    path = Path(model_path)

    suffix = path.suffix.lower()

    if suffix == ".joblib":

        try:

            import joblib

            return joblib.load(
                path
            )

        except Exception as error:

            raise RuntimeError(
                "Unable to load joblib model "
                f"{path}: {error}"
            ) from error

    if suffix in (
        ".pkl",
        ".pickle",
    ):

        try:

            with open(
                path,
                "rb",
            ) as file:

                return pickle.load(
                    file
                )

        except Exception as error:

            raise RuntimeError(
                "Unable to load pickle model "
                f"{path}: {error}"
            ) from error

    raise ValueError(
        "Unsupported model format: "
        f"{suffix}"
    )


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_feature_matrix(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Convert the feature DataFrame into a numeric model matrix.

    Metadata and known target columns are excluded.
    """

    if (
        features is None
        or features.empty
    ):
        return (
            pd.DataFrame(),
            [],
        )

    excluded_columns = {

        "symbol",
        "ticker",
        "sector",
        "company",
        "name",

        "date",
        "market_date",
        "prediction_date",
        "created_at",

        "target",
        "target_return",
        "target_direction",

        "actual_return",
        "actual_direction",
        "actual_risk",
    }

    feature_columns = [
        column
        for column in features.columns
        if str(column).lower()
        not in excluded_columns
    ]

    matrix = features[
        feature_columns
    ].copy()

    matrix = matrix.apply(
        pd.to_numeric,
        errors="coerce",
    )

    matrix = matrix.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    matrix = matrix.fillna(
        0.0
    )

    return (
        matrix,
        feature_columns,
    )


# ============================================================
# PREDICTION NORMALIZATION
# ============================================================

def normalize_predictions(
    raw_predictions: Any,
    row_count: int,
) -> np.ndarray:
    """Normalize arbitrary prediction output."""

    if isinstance(
        raw_predictions,
        pd.DataFrame,
    ):

        if raw_predictions.empty:

            return np.full(
                row_count,
                np.nan,
            )

        raw_predictions = (
            raw_predictions.iloc[
                :,
                0,
            ].to_numpy()
        )

    elif isinstance(
        raw_predictions,
        pd.Series,
    ):

        raw_predictions = (
            raw_predictions.to_numpy()
        )

    values = np.asarray(
        raw_predictions
    )

    if values.ndim == 0:

        values = np.full(
            row_count,
            float(values),
        )

    elif values.ndim > 1:

        if values.shape[1] == 1:

            values = values.reshape(
                -1
            )

        else:

            values = values[
                :,
                -1,
            ]

    values = values.astype(
        float,
        copy=False,
    )

    if len(values) != row_count:

        raise ValueError(
            "Prediction length mismatch. "
            f"Expected {row_count}, "
            f"received {len(values)}."
        )

    return values


# ============================================================
# DIRECTION PROBABILITY
# ============================================================

def calculate_direction_probability(
    model: Any,
    matrix: pd.DataFrame,
    predicted_return: np.ndarray,
) -> np.ndarray:
    """Return probability of positive direction."""

    if hasattr(
        model,
        "predict_proba",
    ):

        try:

            probabilities = (
                model.predict_proba(
                    matrix
                )
            )

            probabilities = np.asarray(
                probabilities,
                dtype=float,
            )

            if probabilities.ndim == 2:

                if probabilities.shape[1] >= 2:

                    return probabilities[
                        :,
                        -1,
                    ]

                if probabilities.shape[1] == 1:

                    return probabilities[
                        :,
                        0,
                    ]

        except Exception as error:

            logger.debug(
                "predict_proba failed: %s",
                error,
            )

    return np.where(
        predicted_return >= 0,
        0.60,
        0.40,
    )


# ============================================================
# RISK PREDICTION
# ============================================================

def calculate_risk_prediction(
    features: pd.DataFrame,
    row_count: int,
) -> np.ndarray:
    """
    Estimate risk from available feature columns.

    Used as fallback when the serialized Challenger is not a
    dedicated multi-output risk model.
    """

    risk_columns = [

        "volatility",

        "atr",

        "atr_pct",

        "historical_volatility",

        "realized_volatility",

        "risk_score",
    ]

    for column in risk_columns:

        if column in features.columns:

            values = pd.to_numeric(
                features[column],
                errors="coerce",
            )

            median = values.median()

            if pd.isna(
                median
            ):
                median = 0.0

            values = values.fillna(
                median
            )

            return values.to_numpy(
                dtype=float
            )

    return np.full(
        row_count,
        np.nan,
    )


# ============================================================
# SINGLE MODEL PREDICTION
# ============================================================

def predict_with_model(
    model_name: str,
    model_path: str | Path | None,
    features: pd.DataFrame,
    prediction_date: str | None = None,
    model_status: str = "CHALLENGER",
) -> ModelPredictionResult:
    """Run prediction using one serialized model."""

    try:

        path = resolve_model_path(
            model_path
        )

        if path is None:

            raise FileNotFoundError(
                f"No usable model path for "
                f"'{model_name}'."
            )

        model = load_serialized_model(
            path
        )

        matrix, _ = (
            prepare_feature_matrix(
                features
            )
        )

        if matrix.empty:

            raise ValueError(
                "No usable numeric features."
            )

        if not hasattr(
            model,
            "predict",
        ):

            raise TypeError(
                f"Model '{model_name}' "
                "does not implement predict()."
            )

        raw_predictions = model.predict(
            matrix
        )

        predicted_return = (
            normalize_predictions(
                raw_predictions,
                len(features),
            )
        )

        direction_probability = (
            calculate_direction_probability(
                model=model,
                matrix=matrix,
                predicted_return=predicted_return,
            )
        )

        predicted_direction = np.where(
            direction_probability >= 0.50,
            1,
            -1,
        )

        predicted_risk = (
            calculate_risk_prediction(
                features,
                len(features),
            )
        )

        result = pd.DataFrame(
            {
                "prediction_date": (
                    prediction_date
                    or datetime.now()
                    .date()
                    .isoformat()
                ),

                "model_name": model_name,

                "model_path": str(
                    path
                ),

                "model_status": model_status,

                "predicted_return": (
                    predicted_return
                ),

                "predicted_direction": (
                    predicted_direction
                ),

                "direction_probability": (
                    direction_probability
                ),

                "predicted_risk": (
                    predicted_risk
                ),

                "evaluation_status": (
                    "PENDING"
                ),

                "created_at": (
                    datetime.now()
                    .isoformat()
                ),
            }
        )

        for column in (
            "symbol",
            "ticker",
            "sector",
        ):

            if column in features.columns:

                result[column] = (
                    features[column]
                    .astype(str)
                    .to_numpy()
                )

        return ModelPredictionResult(

            model_name=model_name,

            model_path=str(
                path
            ),

            model_status=model_status,

            success=True,

            predictions=result,

            error=None,
        )

    except Exception as error:

        logger.exception(
            "Prediction failed for %s.",
            model_name,
        )

        return ModelPredictionResult(

            model_name=model_name,

            model_path=(
                str(model_path)
                if model_path
                else None
            ),

            model_status=model_status,

            success=False,

            predictions=pd.DataFrame(),

            error=str(error),
        )


# ============================================================
# CHAMPION ADAPTER
# ============================================================

def normalize_champion_output(
    output: pd.DataFrame,
    champion: dict[str, Any],
    features: pd.DataFrame,
    prediction_date: str | None,
) -> pd.DataFrame:
    """
    Normalize predictions returned by the existing Champion pipeline.
    """

    if output is None:

        return pd.DataFrame()

    if not isinstance(
        output,
        pd.DataFrame,
    ):

        raise TypeError(
            "champion_predictor must return "
            "a pandas DataFrame."
        )

    if output.empty:

        return pd.DataFrame()

    result = output.copy()

    result["model_name"] = (
        champion["model_name"]
    )

    result["model_path"] = (
        champion.get(
            "model_path"
        )
    )

    result["model_status"] = (
        "CHAMPION"
    )

    result["prediction_date"] = (
        prediction_date
        or datetime.now()
        .date()
        .isoformat()
    )

    result["evaluation_status"] = (
        "PENDING"
    )

    result["created_at"] = (
        datetime.now()
        .isoformat()
    )

    for column in (
        "symbol",
        "ticker",
        "sector",
    ):

        if (
            column in features.columns
            and column not in result.columns
        ):

            result[column] = (
                features[column]
                .astype(str)
                .to_numpy()
            )

    return result


# ============================================================
# PARALLEL ENGINE
# ============================================================

def run_parallel_predictions(
    features: pd.DataFrame,
    registry_path: str | Path | None = None,
    champion_predictor: Callable[
        [pd.DataFrame],
        pd.DataFrame,
    ]
    | None = None,
    prediction_date: str | None = None,
    settings: dict[str, Any] | None = None,
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
]:
    """
    Run configured Champion and Challenger predictions.

    The same feature DataFrame is supplied to every selected model.

    Returns
    -------
    predictions:
        Combined normalized predictions.

    summaries:
        Per-model execution summaries.
    """

    if (
        features is None
        or features.empty
    ):

        logger.warning(
            "No features supplied for "
            "parallel prediction."
        )

        return (
            pd.DataFrame(),
            [],
        )

    configuration = (
        load_parallel_settings()
    )

    if settings:

        configuration.update(
            settings
        )

    if not bool(
        configuration.get(
            "enabled",
            True,
        )
    ):

        logger.info(
            "Parallel prediction is disabled."
        )

        return (
            pd.DataFrame(),
            [],
        )

    if registry_path is None:

        registry_path = (
            configuration.get(
                "registry_path",
                "data/model_registry.json",
            )
        )

    registry = load_model_registry(
        registry_path
    )

    champion = get_active_champion(
        registry,
        default_name=str(
            configuration.get(
                "champion_name",
                "current",
            )
        ),
    )

    max_active = int(
        configuration.get(
            "max_active_challengers",
            1,
        )
    )

    challengers = (
        get_active_challengers(
            registry=registry,
            max_active=max_active,
        )
    )

    run_champion = bool(
        configuration.get(
            "run_champion",
            True,
        )
    )

    run_challengers = bool(
        configuration.get(
            "run_challengers",
            True,
        )
    )

    prediction_frames: list[
        pd.DataFrame
    ] = []

    summaries: list[
        dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # CHAMPION
    # --------------------------------------------------------

    if run_champion:

        if champion_predictor is not None:

            try:

                champion_output = (
                    champion_predictor(
                        features.copy()
                    )
                )

                normalized = (
                    normalize_champion_output(
                        output=champion_output,
                        champion=champion,
                        features=features,
                        prediction_date=prediction_date,
                    )
                )

                if not normalized.empty:

                    prediction_frames.append(
                        normalized
                    )

                summaries.append(
                    {
                        "model_name": (
                            champion[
                                "model_name"
                            ]
                        ),

                        "model_path": (
                            champion.get(
                                "model_path"
                            )
                        ),

                        "model_status": (
                            "CHAMPION"
                        ),

                        "success": True,

                        "prediction_count": len(
                            normalized
                        ),

                        "error": None,
                    }
                )

            except Exception as error:

                logger.exception(
                    "Champion prediction failed."
                )

                summaries.append(
                    {
                        "model_name": (
                            champion[
                                "model_name"
                            ]
                        ),

                        "model_path": (
                            champion.get(
                                "model_path"
                            )
                        ),

                        "model_status": (
                            "CHAMPION"
                        ),

                        "success": False,

                        "prediction_count": 0,

                        "error": str(error),
                    }
                )

        elif champion.get(
            "model_path"
        ):

            result = predict_with_model(

                model_name=(
                    champion[
                        "model_name"
                    ]
                ),

                model_path=(
                    champion.get(
                        "model_path"
                    )
                ),

                features=features,

                prediction_date=prediction_date,

                model_status="CHAMPION",
            )

            summaries.append(
                result.to_dict()
            )

            if result.success:

                prediction_frames.append(
                    result.predictions
                )

        else:

            summaries.append(
                {
                    "model_name": (
                        champion[
                            "model_name"
                        ]
                    ),

                    "model_path": None,

                    "model_status": (
                        "CHAMPION"
                    ),

                    "success": False,

                    "prediction_count": 0,

                    "error": (
                        "Champion enabled but no "
                        "predictor or model path "
                        "is available."
                    ),
                }
            )

    # --------------------------------------------------------
    # CHALLENGERS
    # --------------------------------------------------------

    if run_challengers:

        if not challengers:

            logger.info(
                "No active Challengers found."
            )

        for challenger in challengers:

            result = predict_with_model(

                model_name=(
                    challenger[
                        "model_name"
                    ]
                ),

                model_path=(
                    challenger.get(
                        "model_path"
                    )
                ),

                features=features,

                prediction_date=prediction_date,

                model_status="CHALLENGER",
            )

            summaries.append(
                result.to_dict()
            )

            if result.success:

                prediction_frames.append(
                    result.predictions
                )

    # --------------------------------------------------------
    # COMBINE RESULTS
    # --------------------------------------------------------

    if not prediction_frames:

        logger.warning(
            "No parallel predictions were generated."
        )

        return (
            pd.DataFrame(),
            summaries,
        )

    combined = pd.concat(
        prediction_frames,
        ignore_index=True,
        sort=False,
    )

    preferred_columns = [

        "prediction_date",

        "symbol",

        "ticker",

        "sector",

        "model_name",

        "model_status",

        "model_path",

        "predicted_return",

        "predicted_direction",

        "direction_probability",

        "predicted_risk",

        "evaluation_status",

        "created_at",
    ]

    ordered_columns = [
        column
        for column in preferred_columns
        if column in combined.columns
    ]

    remaining_columns = [
        column
        for column in combined.columns
        if column not in ordered_columns
    ]

    combined = combined[
        ordered_columns
        + remaining_columns
    ]

    logger.info(
        "Parallel prediction complete. "
        "Rows=%s Models=%s",
        len(combined),
        combined[
            "model_name"
        ].nunique()
        if "model_name"
        in combined.columns
        else 0,
    )

    return (
        combined,
        summaries,
    )


# ============================================================
# LEDGER APPEND
# ============================================================

def append_parallel_predictions(
    predictions: pd.DataFrame,
    ledger_path: str | Path,
) -> Path:
    """
    Append normalized parallel predictions to a CSV ledger.

    This helper is optional because projects may already have their
    own prediction ledger implementation.
    """

    if (
        predictions is None
        or predictions.empty
    ):

        raise ValueError(
            "No predictions available "
            "to append."
        )

    path = resolve_project_path(
        ledger_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():

        predictions.to_csv(
            path,
            mode="a",
            header=False,
            index=False,
        )

    else:

        predictions.to_csv(
            path,
            index=False,
        )

    logger.info(
        "Appended %s parallel predictions to %s.",
        len(predictions),
        path,
    )

    return path

"""Parallel Champion / Challenger prediction engine.

Runs the production Champion and active Challenger models against
the same feature data and returns a combined prediction DataFrame.

The engine is intentionally model-loader agnostic. It first tries
to use the project's existing prediction pipeline and falls back to
common sklearn/joblib/pickle model artifacts when possible.
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


logger = logging.getLogger(__name__)


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
# RESULT
# ============================================================

@dataclass
class ModelPredictionResult:
    """Result of one model prediction run."""

    model_name: str
    model_path: str | None
    success: bool
    predictions: pd.DataFrame
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable summary."""

        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "success": self.success,
            "prediction_count": len(
                self.predictions
            ),
            "error": self.error,
        }


# ============================================================
# REGISTRY
# ============================================================

def load_model_registry(
    registry_path: str | Path,
) -> dict[str, Any]:
    """Load the model registry safely."""

    path = Path(registry_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():

        logger.warning(
            "Model registry not found: %s",
            path,
        )

        return {
            "champion": None,
            "challengers": [],
            "history": [],
        }

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
                "Model registry must contain "
                "a JSON object."
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

        return registry

    except Exception as error:

        logger.error(
            "Failed to load model registry: %s",
            error,
        )

        return {
            "champion": None,
            "challengers": [],
            "history": [],
        }


def get_active_champion(
    registry: dict[str, Any],
    default_name: str = "current",
) -> dict[str, Any]:
    """Return the active Champion model definition."""

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
        }

    if isinstance(
        champion,
        str,
    ):

        return {
            "model_name": champion,
            "model_path": None,
            "status": "CHAMPION",
        }

    return {
        "model_name": default_name,
        "model_path": None,
        "status": "CHAMPION",
    }


def get_active_challengers(
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return active challenger models."""

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

        status = str(
            challenger.get(
                "status",
                "",
            )
        ).upper()

        if status not in (
            "CHALLENGER",
            "ACTIVE",
            "EVALUATING",
        ):
            continue

        model_name = challenger.get(
            "model_name"
        )

        if not model_name:
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

    return active


# ============================================================
# MODEL LOADING
# ============================================================

def resolve_model_path(
    model_path: str | Path | None,
) -> Path | None:
    """Resolve a model path."""

    if not model_path:
        return None

    path = Path(model_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():

        logger.warning(
            "Model artifact does not exist: %s",
            path,
        )

        return None

    return path


def load_serialized_model(
    model_path: str | Path,
) -> Any:
    """Load joblib or pickle model artifacts."""

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
                f"Unable to load joblib model "
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
                f"Unable to load pickle model "
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
    """Prepare numeric feature matrix.

    Non-numeric metadata columns are excluded automatically.
    """

    if features is None or features.empty:
        return (
            pd.DataFrame(),
            [],
        )

    metadata_columns = {

        "symbol",
        "ticker",
        "date",
        "market_date",
        "prediction_date",
        "created_at",
        "sector",
        "company",
        "name",

        "target",
        "actual_return",
        "actual_direction",
        "actual_risk",
    }

    available_columns = [
        column
        for column in features.columns
        if column.lower()
        not in metadata_columns
    ]

    matrix = features[
        available_columns
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
        available_columns,
    )


# ============================================================
# PREDICTION NORMALIZATION
# ============================================================

def normalize_predictions(
    raw_predictions: Any,
    row_count: int,
) -> np.ndarray:
    """Normalize model output to one value per row."""

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

            # For multi-output classifiers, use the
            # last column as the positive prediction.

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
            "Prediction length mismatch: "
            f"expected {row_count}, "
            f"received {len(values)}."
        )

    return values


def calculate_direction_probability(
    model: Any,
    matrix: pd.DataFrame,
    predicted_return: np.ndarray,
) -> np.ndarray:
    """Calculate probability of positive direction."""

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

    # Fallback conversion from predicted return.

    return np.where(
        predicted_return >= 0,
        0.60,
        0.40,
    )


def calculate_risk_prediction(
    features: pd.DataFrame,
    row_count: int,
) -> np.ndarray:
    """Create a fallback risk estimate from available features."""

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

            values = values.fillna(
                values.median()
            )

            values = values.fillna(
                0.0
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
) -> ModelPredictionResult:
    """Run prediction using one serialized model."""

    try:

        path = resolve_model_path(
            model_path
        )

        if path is None:

            raise FileNotFoundError(
                f"No model path available for "
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
                "No usable feature columns "
                "were available."
            )

        if not hasattr(
            model,
            "predict",
        ):

            raise TypeError(
                f"Model '{model_name}' "
                "does not expose predict()."
            )

        raw_predictions = (
            model.predict(
                matrix
            )
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
                "model_name": model_name,

                "model_path": str(
                    path
                ),

                "prediction_date": (
                    prediction_date
                    or datetime.now()
                    .date()
                    .isoformat()
                ),

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
            }
        )

        # Preserve useful identifiers.

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

        result["evaluation_status"] = (
            "PENDING"
        )

        result["created_at"] = (
            datetime.now()
            .isoformat()
        )

        return ModelPredictionResult(

            model_name=model_name,

            model_path=str(
                path
            ),

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

            success=False,

            predictions=pd.DataFrame(),

            error=str(error),
        )


# ============================================================
# PARALLEL PREDICTION ENGINE
# ============================================================

def run_parallel_predictions(
    features: pd.DataFrame,
    registry_path: str | Path = (
        "data/model_registry.json"
    ),
    champion_predictor: Callable[
        [pd.DataFrame],
        pd.DataFrame,
    ]
    | None = None,
    prediction_date: str | None = None,
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
]:
    """Run Champion and all active Challengers.

    Parameters
    ----------
    features:
        Feature DataFrame containing at least symbol/ticker and
        numeric model features.

    registry_path:
        Model registry JSON path.

    champion_predictor:
        Optional adapter for the existing production prediction
        pipeline. This is useful when the Champion is not stored as
        one simple pickle/joblib artifact.

        It must return a DataFrame with at least:

            predicted_return

        Optional:

            predicted_direction
            direction_probability
            predicted_risk

    prediction_date:
        Market date for the prediction.

    Returns
    -------
    combined_predictions, summaries
    """

    if features is None or features.empty:

        logger.warning(
            "No features supplied for "
            "parallel prediction."
        )

        return (
            pd.DataFrame(),
            [],
        )

    registry = load_model_registry(
        registry_path
    )

    champion = get_active_champion(
        registry
    )

    challengers = (
        get_active_challengers(
            registry
        )
    )

    summaries: list[
        dict[str, Any]
    ] = []

    prediction_frames: list[
        pd.DataFrame
    ] = []

    # --------------------------------------------------------
    # CHAMPION
    # --------------------------------------------------------

    if champion_predictor is not None:

        try:

            champion_output = (
                champion_predictor(
                    features.copy()
                )
            )

            if not isinstance(
                champion_output,
                pd.DataFrame,
            ):

                raise TypeError(
                    "champion_predictor must "
                    "return a DataFrame."
                )

            result = champion_output.copy()

            result["model_name"] = (
                champion["model_name"]
            )

            result["model_path"] = (
                champion.get(
                    "model_path"
                )
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
                    and column
                    not in result.columns
                ):

                    result[column] = (
                        features[column]
                        .astype(str)
                        .to_numpy()
                    )

            prediction_frames.append(
                result
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
                    "success": True,
                    "prediction_count": len(
                        result
                    ),
                    "error": None,
                }
            )

        except Exception as error:

            logger.exception(
                "Champion adapter prediction failed."
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
                    "success": False,
                    "prediction_count": 0,
                    "error": str(error),
                }
            )

    else:

        champion_result = (
            predict_with_model(
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
            )
        )

        summaries.append(
            champion_result.to_dict()
        )

        if champion_result.success:

            prediction_frames.append(
                champion_result.predictions
            )

    # --------------------------------------------------------
    # CHALLENGERS
    # --------------------------------------------------------

    for challenger in challengers:

        challenger_result = (
            predict_with_model(
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
            )
        )

        summaries.append(
            challenger_result.to_dict()
        )

        if challenger_result.success:

            prediction_frames.append(
                challenger_result.predictions
            )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    if not prediction_frames:

        logger.error(
            "All model predictions failed."
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
        if column
        not in ordered_columns
    ]

    combined = combined[
        ordered_columns
        + remaining_columns
    ]

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
    """Append predictions to the prediction ledger."""

    if predictions is None or predictions.empty:

        raise ValueError(
            "No predictions available to append."
        )

    path = Path(
        ledger_path
    )

    if not path.is_absolute():
        path = PROJECT_ROOT / path

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

"""Multi-model ML pipeline.

Combines:

    ReturnModel
    DirectionModel
    RiskModel

Supports:

    - Training
    - Prediction
    - Persistent model storage
    - Loading saved models
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.feature_engine import (
    build_feature_frame,
    feature_columns,
    latest_features,
)

from src.ml_targets import (
    add_ml_targets,
)

from src.return_model import (
    ReturnModel,
)

from src.direction_model import (
    DirectionModel,
)

from src.risk_model import (
    RiskModel,
)

from src.ensemble import (
    build_ensemble,
)

from src.model_store import (
    ModelStore,
)


logger = logging.getLogger(__name__)


MODEL_VERSION = "ensemble-v1"


class MultiModelPipeline:
    """Phase 3 persistent stock prediction pipeline."""

    def __init__(
        self,
        minimum_training_rows: int = 80,
    ) -> None:

        self.features = feature_columns()

        self.minimum_training_rows = (
            minimum_training_rows
        )

        self.return_model = ReturnModel(
            self.features
        )

        self.direction_model = DirectionModel(
            self.features
        )

        self.risk_model = RiskModel(
            self.features
        )

        self.is_fitted = False

        self.training_rows = 0

        self.model_version = MODEL_VERSION

    def prepare_training_data(
        self,
        history: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build leakage-safe features and targets."""

        if (
            history is None
            or history.empty
        ):
            return pd.DataFrame()

        features = build_feature_frame(
            history
        )

        if features is None or features.empty:
            return pd.DataFrame()

        training_data = add_ml_targets(
            features
        )

        return training_data

    def fit(
        self,
        history: pd.DataFrame,
    ) -> "MultiModelPipeline":
        """Train Return, Direction and Risk models."""

        training_data = (
            self.prepare_training_data(
                history
            )
        )

        if training_data.empty:

            raise ValueError(
                "Unable to create training dataset."
            )

        required_columns = (
            self.features
            + [
                "target_return",
                "target_direction",
                "target_risk",
            ]
        )

        usable = training_data.dropna(
            subset=required_columns
        ).copy()

        if (
            len(usable)
            < self.minimum_training_rows
        ):

            raise ValueError(
                "Insufficient training data: "
                f"{len(usable)} rows, "
                f"minimum required "
                f"{self.minimum_training_rows}."
            )

        logger.info(
            "Training %s with %s rows",
            self.model_version,
            len(usable),
        )

        self.return_model.fit(
            usable
        )

        self.direction_model.fit(
            usable
        )

        self.risk_model.fit(
            usable
        )

        self.is_fitted = True

        self.training_rows = len(
            usable
        )

        return self

    def predict(
        self,
        history: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """Generate a multi-model prediction."""

        if not self.is_fitted:

            logger.warning(
                "Pipeline is not fitted."
            )

            return None

        if (
            history is None
            or history.empty
        ):

            return None

        features = latest_features(
            history
        )

        if not features:

            logger.warning(
                "Could not generate latest features."
            )

            return None

        expected_return = (
            self.return_model.predict(
                features
            )
        )

        probability_up = (
            self.direction_model
            .predict_probability(
                features
            )
        )

        expected_risk = (
            self.risk_model.predict(
                features
            )
        )

        prediction = build_ensemble(

            expected_return=(
                expected_return
            ),

            probability_up=(
                probability_up
            ),

            expected_risk=(
                expected_risk
            ),
        )

        prediction[
            "training_rows"
        ] = self.training_rows

        prediction[
            "feature_timestamp"
        ] = features.get(
            "feature_timestamp"
        )

        prediction[
            "feature_version"
        ] = features.get(
            "feature_version"
        )

        prediction[
            "model_version"
        ] = self.model_version

        return prediction

    def save(
        self,
        symbol: str,
        store: ModelStore,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save this trained pipeline."""

        if not self.is_fitted:

            raise ValueError(
                "Cannot save an unfitted pipeline."
            )

        payload = {
            "model_version": self.model_version,
            "minimum_training_rows": (
                self.minimum_training_rows
            ),
            "feature_count": len(
                self.features
            ),
        }

        if metadata:
            payload.update(
                metadata
            )

        store.save(
            symbol=symbol,
            pipeline=self,
            metadata=payload,
        )

    @classmethod
    def load(
        cls,
        symbol: str,
        store: ModelStore,
    ) -> tuple[
        "MultiModelPipeline",
        dict[str, Any],
    ] | None:
        """Load a saved pipeline."""

        result = store.load(
            symbol
        )

        if result is None:

            return None

        pipeline, metadata = result

        if not isinstance(
            pipeline,
            cls,
        ):

            logger.error(
                "Invalid pipeline type for %s: %s",
                symbol,
                type(pipeline),
            )

            return None

        if not getattr(
            pipeline,
            "is_fitted",
            False,
        ):

            logger.error(
                "Loaded pipeline for %s is not fitted.",
                symbol,
            )

            return None

        # Compatibility for older saved models.
        if not hasattr(
            pipeline,
            "model_version",
        ):

            pipeline.model_version = (
                metadata.get(
                    "model_version",
                    MODEL_VERSION,
                )
            )

        if not hasattr(
            pipeline,
            "training_rows",
        ):

            pipeline.training_rows = int(
                metadata.get(
                    "training_rows",
                    0,
                )
            )

        logger.info(
            "Loaded %s for %s "
            "(training rows: %s)",
            pipeline.model_version,
            symbol,
            pipeline.training_rows,
        )

        return (
            pipeline,
            metadata,
        )


def train_and_save(
    symbol: str,
    history: pd.DataFrame,
    store: ModelStore,
    minimum_training_rows: int = 80,
    metadata: dict[str, Any] | None = None,
) -> MultiModelPipeline:
    """Train and persist one stock pipeline."""

    pipeline = MultiModelPipeline(
        minimum_training_rows=(
            minimum_training_rows
        )
    )

    pipeline.fit(
        history
    )

    pipeline.save(
        symbol=symbol,
        store=store,
        metadata=metadata,
    )

    return pipeline


def load_and_predict(
    symbol: str,
    history: pd.DataFrame,
    store: ModelStore,
) -> dict[str, Any] | None:
    """Load a saved model and generate prediction."""

    result = MultiModelPipeline.load(
        symbol=symbol,
        store=store,
    )

    if result is None:

        return None

    pipeline, metadata = result

    prediction = pipeline.predict(
        history
    )

    if prediction is None:

        return None

    prediction["model_saved_at"] = (
        metadata.get(
            "saved_at"
        )
    )

    return prediction


def predict_stock(
    history: pd.DataFrame,
) -> dict[str, Any] | None:
    """Backward-compatible convenience function.

    This trains and predicts immediately.
    Prefer load_and_predict() in production.
    """

    pipeline = MultiModelPipeline()

    pipeline.fit(
        history
    )

    return pipeline.predict(
        history
    )

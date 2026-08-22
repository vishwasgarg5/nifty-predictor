"""Multi-model ML pipeline.

Combines:

    ReturnModel
    DirectionModel
    RiskModel

into one prediction interface.
"""

from __future__ import annotations

import logging

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


logger = logging.getLogger(
    __name__
)


class MultiModelPipeline:
    """Phase 3 stock prediction pipeline."""

    def __init__(
        self,
        minimum_training_rows: int = 80,
    ):

        self.features = (
            feature_columns()
        )

        self.minimum_training_rows = (
            minimum_training_rows
        )

        self.return_model = (
            ReturnModel(
                self.features
            )
        )

        self.direction_model = (
            DirectionModel(
                self.features
            )
        )

        self.risk_model = (
            RiskModel(
                self.features
            )
        )

        self.is_fitted = False

        self.training_rows = 0

    def prepare_training_data(
        self,
        history: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build features and targets."""

        features = (
            build_feature_frame(
                history
            )
        )

        if features.empty:

            return pd.DataFrame()

        training_data = (
            add_ml_targets(
                features
            )
        )

        return training_data

    def fit(
        self,
        history: pd.DataFrame,
    ) -> "MultiModelPipeline":
        """Train all three models."""

        training_data = (
            self.prepare_training_data(
                history
            )
        )

        if training_data.empty:

            raise ValueError(
                "Unable to create "
                "training dataset."
            )

        usable = training_data.dropna(
            subset=(
                self.features
                + [
                    "target_return",
                    "target_direction",
                    "target_risk",
                ]
            )
        )

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
            "Training ReturnModel "
            "with %s rows",
            len(usable),
        )

        self.return_model.fit(
            usable
        )

        logger.info(
            "Training DirectionModel "
            "with %s rows",
            len(usable),
        )

        self.direction_model.fit(
            usable
        )

        logger.info(
            "Training RiskModel "
            "with %s rows",
            len(usable),
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
    ) -> dict | None:
        """Generate a complete multi-model prediction."""

        if not self.is_fitted:

            return None

        features = latest_features(
            history
        )

        if not features:

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

        return prediction


def predict_stock(
    history: pd.DataFrame,
) -> dict | None:
    """Convenience function for one stock.

    This trains the models using available
    history and predicts the latest row.
    """

    pipeline = (
        MultiModelPipeline()
    )

    pipeline.fit(
        history
    )

    return pipeline.predict(
        history
    )

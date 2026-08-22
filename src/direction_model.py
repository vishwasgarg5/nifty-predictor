"""Probability-of-upward-move classification model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)

from src.ml_targets import (
    split_training_data,
)


class DirectionModel:
    """Predict probability of positive return."""

    def __init__(
        self,
        feature_columns: list[str],
    ):

        self.feature_columns = (
            feature_columns
        )

        self.model = (
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42,
            )
        )

        self.is_fitted = False

        self.train_size = 0

    def fit(
        self,
        frame: pd.DataFrame,
    ) -> "DirectionModel":

        x, y = split_training_data(
            frame=frame,
            feature_columns=(
                self.feature_columns
            ),
            target_column=(
                "target_direction"
            ),
        )

        if len(x) < 60:

            raise ValueError(
                "Insufficient training rows "
                f"for DirectionModel: {len(x)}"
            )

        # Direction must contain both classes.
        if y.nunique() < 2:

            raise ValueError(
                "DirectionModel requires "
                "both UP and DOWN samples."
            )

        self.model.fit(
            x,
            y.astype(int),
        )

        self.is_fitted = True

        self.train_size = len(x)

        return self

    def predict_probability(
        self,
        features: dict,
    ) -> float | None:
        """Return probability of positive move."""

        if not self.is_fitted:
            return None

        try:

            x = pd.DataFrame(
                [
                    {
                        column: features.get(
                            column,
                            np.nan,
                        )
                        for column
                        in self.feature_columns
                    }
                ]
            )

            if x.isna().any().any():
                return None

            probabilities = (
                self.model.predict_proba(x)[0]
            )

            classes = (
                self.model.classes_
            )

            if 1 not in classes:

                return None

            positive_index = list(
                classes
            ).index(1)

            return float(
                probabilities[
                    positive_index
                ]
            )

        except Exception:

            return None

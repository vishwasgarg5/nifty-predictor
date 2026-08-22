"""Expected downside / movement risk model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
)

from src.ml_targets import (
    split_training_data,
)


class RiskModel:
    """Predict expected absolute forward move."""

    def __init__(
        self,
        feature_columns: list[str],
    ):

        self.feature_columns = (
            feature_columns
        )

        self.model = (
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=12,
                l2_regularization=1.5,
                random_state=42,
            )
        )

        self.is_fitted = False

        self.train_size = 0

    def fit(
        self,
        frame: pd.DataFrame,
    ) -> "RiskModel":

        x, y = split_training_data(
            frame=frame,
            feature_columns=(
                self.feature_columns
            ),
            target_column=(
                "target_risk"
            ),
        )

        if len(x) < 60:

            raise ValueError(
                "Insufficient training rows "
                f"for RiskModel: {len(x)}"
            )

        self.model.fit(
            x,
            y,
        )

        self.is_fitted = True

        self.train_size = len(x)

        return self

    def predict(
        self,
        features: dict,
    ) -> float | None:
        """Predict expected absolute move."""

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

            prediction = (
                self.model.predict(x)[0]
            )

            # Risk cannot be negative.
            return float(
                max(
                    0.0,
                    prediction,
                )
            )

        except Exception:

            return None

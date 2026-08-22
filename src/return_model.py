"""Expected return regression model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
)

from src.ml_targets import (
    split_training_data,
)


class ReturnModel:
    """Predict next-period expected return."""

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
    ) -> "ReturnModel":

        x, y = split_training_data(
            frame=frame,
            feature_columns=(
                self.feature_columns
            ),
            target_column=(
                "target_return"
            ),
        )

        if len(x) < 60:

            raise ValueError(
                "Insufficient training rows "
                f"for ReturnModel: {len(x)}"
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
        """Predict expected forward return."""

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

            return float(
                prediction
            )

        except Exception:

            return None

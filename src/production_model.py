#!/usr/bin/env python3

"""
Unified Production Model.

Combines:

    - ReturnModel
    - DirectionModel
    - RiskModel
    - Ensemble logic

This provides one standard predict() interface
for prediction_pipeline.py.
"""

from __future__ import annotations

from typing import Any

import math

import pandas as pd

from src.ensemble import build_ensemble


MODEL_VERSION = "production-model-v1"


class ProductionModel:
    """
    Unified production model.

    Combines ReturnModel, DirectionModel,
    and RiskModel into one production model.
    """

    def __init__(
        self,
        return_model: Any,
        direction_model: Any,
        risk_model: Any,
        feature_columns: list[str],
    ) -> None:

        self.return_model = return_model
        self.direction_model = direction_model
        self.risk_model = risk_model

        self.feature_columns = list(
            feature_columns
        )

        self.is_fitted = True

        self.model_version = (
            MODEL_VERSION
        )

    def _prepare_features(
        self,
        features: Any,
    ) -> dict[str, float]:
        """
        Convert DataFrame, Series, or dict
        into a clean feature dictionary.
        """

        if features is None:

            raise ValueError(
                "Features cannot be None."
            )

        if isinstance(
            features,
            pd.DataFrame,
        ):

            if features.empty:

                raise ValueError(
                    "Feature DataFrame is empty."
                )

            source = (
                features.iloc[-1]
                .to_dict()
            )

        elif isinstance(
            features,
            pd.Series,
        ):

            source = features.to_dict()

        elif isinstance(
            features,
            dict,
        ):

            source = dict(
                features
            )

        else:

            raise TypeError(
                "Features must be a DataFrame, "
                "Series, or dictionary."
            )

        prepared: dict[
            str,
            float,
        ] = {}

        for column in self.feature_columns:

            if column not in source:

                raise ValueError(
                    f"Missing feature: {column}"
                )

            try:

                value = float(
                    source[column]
                )

            except (
                TypeError,
                ValueError,
            ) as error:

                raise ValueError(
                    f"Invalid feature: {column}"
                ) from error

            if not math.isfinite(
                value
            ):

                raise ValueError(
                    f"Non-finite feature: {column}"
                )

            prepared[column] = value

        return prepared

    def predict(
        self,
        features: Any,
    ) -> dict[str, Any]:
        """
        Generate a production prediction.

        Compatible with:

            model.predict(features)

        Returns a dictionary containing:

            expected_return
            probability_up
            expected_risk
            opportunity_score
            confidence
            direction
        """

        prepared_features = (
            self._prepare_features(
                features
            )
        )

        # --------------------------------
        # RETURN PREDICTION
        # --------------------------------

        expected_return = (
            self.return_model.predict(
                prepared_features
            )
        )

        # --------------------------------
        # DIRECTION PREDICTION
        # --------------------------------

        probability_up = (
            self.direction_model
            .predict_probability(
                prepared_features
            )
        )

        # --------------------------------
        # RISK PREDICTION
        # --------------------------------

        expected_risk = (
            self.risk_model.predict(
                prepared_features
            )
        )

        # --------------------------------
        # ENSEMBLE
        # --------------------------------

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

        # --------------------------------
        # METADATA
        # --------------------------------

        prediction[
            "model_version"
        ] = self.model_version

        prediction[
            "model_type"
        ] = "ProductionModel"

        prediction[
            "feature_count"
        ] = len(
            self.feature_columns
        )

        return prediction

    def get_metadata(
        self,
    ) -> dict[str, Any]:
        """
        Return metadata describing
        this production model.
        """

        return {

            "model_version": (
                self.model_version
            ),

            "model_type": (
                "ProductionModel"
            ),

            "feature_count": len(
                self.feature_columns
            ),

            "return_model": (
                type(
                    self.return_model
                ).__name__
            ),

            "direction_model": (
                type(
                    self.direction_model
                ).__name__
            ),

            "risk_model": (
                type(
                    self.risk_model
                ).__name__
            ),

        }

    def __repr__(
        self,
    ) -> str:

        return (
            "ProductionModel("
            f"version={self.model_version}, "
            f"features={len(self.feature_columns)}"
            ")"
        )

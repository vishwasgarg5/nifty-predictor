#!/usr/bin/env python3

"""
Unified Production Model.

Combines:

    - ReturnModel
    - DirectionModel
    - RiskModel
    - Ensemble logic

Provides one standard predict() interface for
prediction_pipeline.py.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.ensemble import build_ensemble


MODEL_VERSION = "production-model-v1"


class ProductionModel:
    """
    Unified production model.

    Combines the three ML models into a single
    production prediction interface.
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


    # ========================================================
    # FEATURE PREPARATION
    # ========================================================

    def _prepare_features(
        self,
        features: Any,
    ) -> dict[str, float]:
        """
        Convert DataFrame, Series, or dictionary
        into a clean feature dictionary.
        """

        if features is None:

            raise ValueError(
                "Features cannot be None."
            )

        # ----------------------------------------------------
        # DATAFRAME
        # ----------------------------------------------------

        if isinstance(
            features,
            pd.DataFrame,
        ):

            if features.empty:

                raise ValueError(
                    "Feature DataFrame is empty."
                )

            source = (
                features
                .iloc[-1]
                .to_dict()
            )

        # ----------------------------------------------------
        # SERIES
        # ----------------------------------------------------

        elif isinstance(
            features,
            pd.Series,
        ):

            source = (
                features
                .to_dict()
            )

        # ----------------------------------------------------
        # DICTIONARY
        # ----------------------------------------------------

        elif isinstance(
            features,
            dict,
        ):

            source = dict(
                features
            )

        else:

            raise TypeError(
                "Features must be a "
                "DataFrame, Series, or dictionary."
            )

        prepared: dict[
            str,
            float,
        ] = {}

        # ----------------------------------------------------
        # VALIDATE FEATURES
        # ----------------------------------------------------

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


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        features: Any,
    ) -> dict[str, Any]:
        """
        Generate a complete production prediction.

        Accepts:

            pandas.DataFrame
            pandas.Series
            dict

        Returns:

            expected_return
            probability_up
            expected_risk
            risk_adjusted_return
            opportunity_score
            confidence
            direction
        """

        prepared_features = (
            self._prepare_features(
                features
            )
        )

        # ----------------------------------------------------
        # RETURN MODEL
        # ----------------------------------------------------

        expected_return = (
            self.return_model.predict(
                prepared_features
            )
        )

        # ----------------------------------------------------
        # DIRECTION MODEL
        # ----------------------------------------------------

        probability_up = (
            self.direction_model
            .predict_probability(
                prepared_features
            )
        )

        # ----------------------------------------------------
        # RISK MODEL
        # ----------------------------------------------------

        expected_risk = (
            self.risk_model.predict(
                prepared_features
            )
        )

        # ----------------------------------------------------
        # ENSEMBLE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # MODEL METADATA
        # ----------------------------------------------------

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


    # ========================================================
    # METADATA
    # ========================================================

    def get_metadata(
        self,
    ) -> dict[str, Any]:
        """
        Return production model metadata.
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

            "is_fitted": (
                self.is_fitted
            ),

        }


    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(
        self,
    ) -> str:

        return (
            "ProductionModel("
            f"version={self.model_version}, "
            f"features={len(self.feature_columns)}"
            ")"
        )

# stock_ranker.py
#!/usr/bin/env python3

"""
Production Stock Ranking Engine.

This module:

    1. Loads the current Champion model.
    2. Runs predictions for multiple stocks.
    3. Calculates and collects prediction results.
    4. Ranks stocks by opportunity score.
    5. Returns the Top N opportunities.

Expected flow:

    Stock Universe
          │
          ▼
    Feature Generation
          │
          ▼
    StockRanker
          │
          ▼
    Champion Model
          │
          ▼
    Predictions
          │
          ▼
    Rank by opportunity_score
          │
          ▼
    Top 5 Stocks
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import pandas as pd


logger = logging.getLogger(
    "stock_ranker"
)


# ============================================================
# TIME
# ============================================================

def utc_now_iso() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# VALIDATION
# ============================================================

def validate_top_n(
    top_n: int,
) -> int:
    """Validate the requested number of top stocks."""

    if isinstance(
        top_n,
        bool,
    ):

        raise TypeError(
            "top_n must be an integer."
        )

    try:

        value = int(
            top_n
        )

    except (
        TypeError,
        ValueError,
    ) as error:

        raise TypeError(
            "top_n must be an integer."
        ) from error

    if value <= 0:

        raise ValueError(
            "top_n must be greater than zero."
        )

    return value


def validate_prediction_score(
    prediction: dict[str, Any],
    score_column: str,
) -> float:
    """
    Extract and validate the ranking score
    from a prediction.
    """

    if score_column not in prediction:

        raise ValueError(
            f"Prediction does not contain "
            f"ranking column: {score_column}"
        )

    try:

        score = float(
            prediction[score_column]
        )

    except (
        TypeError,
        ValueError,
    ) as error:

        raise ValueError(
            f"Invalid ranking score: "
            f"{score_column}"
        ) from error

    if not math.isfinite(
        score
    ):

        raise ValueError(
            f"Non-finite ranking score: "
            f"{score_column}"
        )

    return score


# ============================================================
# STOCK RANKER
# ============================================================

class StockRanker:
    """
    Rank multiple stocks using a production model.

    Example:

        ranker = StockRanker()

        result = ranker.rank(
            stock_features={
                "RELIANCE.NS": features_1,
                "TCS.NS": features_2,
                "INFY.NS": features_3,
            },
            top_n=5,
        )

        top_stocks = result["top_stocks"]
    """

    def __init__(
        self,
        model: Any | None = None,
        model_metadata: dict[str, Any] | None = None,
        score_column: str = "opportunity_score",
    ) -> None:
        """
        Initialize the stock ranker.

        If no model is supplied, the current
        production Champion is loaded automatically.
        """

        self.model = model

        self.model_metadata = (
            dict(model_metadata)
            if model_metadata
            else {}
        )

        self.score_column = str(
            score_column
        )

        self.loaded_at: str | None = None

        if self.model is None:

            self._load_champion()

        self._validate_model()


    # ========================================================
    # MODEL LOADING
    # ========================================================

    def _load_champion(
        self,
    ) -> None:
        """Load the currently registered Champion."""

        from src.model_loader import (
            load_champion_model,
        )

        model, metadata = (
            load_champion_model()
        )

        self.model = model

        self.model_metadata = dict(
            metadata
        )

        self.loaded_at = utc_now_iso()

        logger.info(
            "Champion model loaded for "
            "stock ranking."
        )


    def _validate_model(
        self,
    ) -> None:
        """Ensure the model provides predict()."""

        if self.model is None:

            raise RuntimeError(
                "StockRanker requires "
                "a prediction model."
            )

        predict_function = getattr(
            self.model,
            "predict",
            None,
        )

        if not callable(
            predict_function
        ):

            raise TypeError(
                "Ranking model must provide "
                "a callable predict() method."
            )


    # ========================================================
    # MODEL MANAGEMENT
    # ========================================================

    def reload_champion(
        self,
    ) -> None:
        """
        Reload the current Champion model.

        Useful after a Challenger has been promoted.
        """

        self.model = None

        self.model_metadata = {}

        self._load_champion()

        self._validate_model()

        logger.info(
            "Champion model reloaded."
        )


    # ========================================================
    # SINGLE STOCK PREDICTION
    # ========================================================

    def predict_stock(
        self,
        symbol: str,
        features: Any,
    ) -> dict[str, Any]:
        """
        Predict and prepare one stock result.
        """

        normalized_symbol = str(
            symbol
        ).strip()

        if not normalized_symbol:

            raise ValueError(
                "Stock symbol cannot be empty."
            )

        prediction = self.model.predict(
            features
        )

        if not isinstance(
            prediction,
            dict,
        ):

            raise TypeError(
                "Model predict() must return "
                "a dictionary."
            )

        result = dict(
            prediction
        )

        score = validate_prediction_score(
            result,
            self.score_column,
        )

        result["symbol"] = (
            normalized_symbol
        )

        result["ranking_score"] = (
            score
        )

        result["predicted_at"] = (
            utc_now_iso()
        )

        result.setdefault(
            "model_version",
            getattr(
                self.model,
                "model_version",
                "unknown",
            ),
        )

        return result


    # ========================================================
    # PREDICT MULTIPLE STOCKS
    # ========================================================

    def predict_all(
        self,
        stock_features: dict[
            str,
            Any,
        ],
        continue_on_error: bool = True,
    ) -> dict[str, Any]:
        """
        Run predictions for multiple stocks.

        Parameters:

            stock_features:

                {
                    "RELIANCE.NS": features,
                    "TCS.NS": features,
                }

            continue_on_error:

                True:
                    Continue scanning if one
                    stock fails.

                False:
                    Stop immediately when a
                    prediction fails.
        """

        if not isinstance(
            stock_features,
            dict,
        ):

            raise TypeError(
                "stock_features must be "
                "a dictionary."
            )

        if not stock_features:

            raise ValueError(
                "stock_features is empty."
            )

        predictions: list[
            dict[str, Any]
        ] = []

        errors: list[
            dict[str, str]
        ] = []

        for symbol, features in (
            stock_features.items()
        ):

            try:

                prediction = (
                    self.predict_stock(
                        symbol=symbol,
                        features=features,
                    )
                )

                predictions.append(
                    prediction
                )

            except Exception as error:

                logger.exception(
                    "Prediction failed for %s",
                    symbol,
                )

                error_info = {
                    "symbol": str(symbol),
                    "error": str(error),
                }

                errors.append(
                    error_info
                )

                if not continue_on_error:

                    raise

        return {
            "predictions": predictions,
            "errors": errors,
            "total_symbols": len(
                stock_features
            ),
            "successful_predictions": len(
                predictions
            ),
            "failed_predictions": len(
                errors
            ),
        }


    # ========================================================
    # RANK STOCKS
    # ========================================================

    def rank(
        self,
        stock_features: dict[
            str,
            Any,
        ],
        top_n: int = 5,
        continue_on_error: bool = True,
    ) -> dict[str, Any]:
        """
        Predict all stocks and return the Top N.

        Stocks are ranked by:

            opportunity_score

        unless a different score_column was
        provided during initialization.
        """

        top_n = validate_top_n(
            top_n
        )

        prediction_result = (
            self.predict_all(
                stock_features=stock_features,
                continue_on_error=(
                    continue_on_error
                ),
            )
        )

        predictions = (
            prediction_result[
                "predictions"
            ]
        )

        ranked_predictions = sorted(
            predictions,
            key=lambda item: item[
                "ranking_score"
            ],
            reverse=True,
        )

        top_stocks = (
            ranked_predictions[:top_n]
        )

        for index, prediction in enumerate(
            ranked_predictions,
            start=1,
        ):

            prediction["rank"] = index

            prediction["is_top_selection"] = (
                index <= top_n
            )

        return {
            "ranked_at": utc_now_iso(),
            "score_column": (
                self.score_column
            ),
            "requested_top_n": top_n,
            "available_predictions": len(
                ranked_predictions
            ),
            "top_count": len(
                top_stocks
            ),
            "top_stocks": top_stocks,
            "all_ranked_stocks": (
                ranked_predictions
            ),
            "errors": prediction_result[
                "errors"
            ],
            "total_symbols": prediction_result[
                "total_symbols"
            ],
            "successful_predictions": (
                prediction_result[
                    "successful_predictions"
                ]
            ),
            "failed_predictions": (
                prediction_result[
                    "failed_predictions"
                ]
            ),
            "model_metadata": dict(
                self.model_metadata
            ),
        }


    # ========================================================
    # DATAFRAME OUTPUT
    # ========================================================

    def rank_dataframe(
        self,
        stock_features: dict[
            str,
            Any,
        ],
        top_n: int = 5,
        continue_on_error: bool = True,
    ) -> pd.DataFrame:
        """
        Return the ranked stocks as a DataFrame.

        Useful for:

            Telegram formatting
            CSV storage
            dashboards
            evaluation
        """

        result = self.rank(
            stock_features=stock_features,
            top_n=top_n,
            continue_on_error=(
                continue_on_error
            ),
        )

        frame = pd.DataFrame(
            result[
                "all_ranked_stocks"
            ]
        )

        if frame.empty:

            return frame

        priority_columns = [
            "rank",
            "symbol",
            "ranking_score",
            self.score_column,
            "expected_return",
            "probability_up",
            "expected_risk",
            "risk_adjusted_return",
            "confidence",
            "direction",
            "model_version",
            "predicted_at",
            "is_top_selection",
        ]

        existing_priority_columns = [
            column
            for column in priority_columns
            if column in frame.columns
        ]

        remaining_columns = [
            column
            for column in frame.columns
            if column
            not in existing_priority_columns
        ]

        return frame[
            existing_priority_columns
            + remaining_columns
        ]


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def rank_stocks(
    stock_features: dict[
        str,
        Any,
    ],
    top_n: int = 5,
    model: Any | None = None,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    """
    Rank stocks using the supplied model or
    the current production Champion.

    Example:

        result = rank_stocks(
            stock_features=features,
            top_n=5,
        )
    """

    ranker = StockRanker(
        model=model
    )

    return ranker.rank(
        stock_features=stock_features,
        top_n=top_n,
        continue_on_error=(
            continue_on_error
        ),
    )


# ============================================================
# CLI TEST
# ============================================================

def main() -> int:
    """
    Display StockRanker status.

    Actual stock ranking requires prepared
    features and is normally performed by
    prediction_pipeline.py.
    """

    try:

        ranker = StockRanker()

        print()

        print("=" * 70)

        print(
            "PRODUCTION STOCK RANKER"
        )

        print("=" * 70)

        print()

        print(
            "Model:",
            type(
                ranker.model
            ).__name__,
        )

        print(
            "Score column:",
            ranker.score_column,
        )

        print()

        print(
            "Model metadata:"
        )

        for key, value in (
            ranker.model_metadata.items()
        ):

            print(
                f"{key}: {value}"
            )

        print()

        print(
            "SUCCESS: StockRanker "
            "is ready."
        )

        return 0

    except Exception as error:

        logger.exception(
            "StockRanker initialization failed."
        )

        print()

        print(
            f"ERROR: {error}"
        )

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

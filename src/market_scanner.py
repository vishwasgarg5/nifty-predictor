#!/usr/bin/env python3

"""
Market Scanner.

This module scans a universe of stocks, builds the latest
feature set for each symbol, loads the current production
Champion model, generates predictions, ranks opportunities,
and returns the Top N stocks.

Flow:

    Stock Universe
          │
          ▼
    Historical OHLCV Data
          │
          ▼
    Feature Engineering
          │
          ▼
    Champion ProductionModel
          │
          ▼
    Prediction
          │
          ▼
    Validation / Filtering
          │
          ▼
    Opportunity Ranking
          │
          ▼
    Top N Stocks

The module is data-provider agnostic.

A history provider must supply historical OHLCV data as:

    dict[str, pandas.DataFrame]

or the caller can use scan_symbol() directly.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Iterable

import pandas as pd


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(
    "market_scanner"
)


# ============================================================
# TYPES
# ============================================================

HistoryProvider = Callable[
    [str],
    pd.DataFrame,
]


# ============================================================
# REQUIRED OHLCV COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
}


# ============================================================
# MARKET SCANNER
# ============================================================

class MarketScanner:
    """
    Scan multiple stocks using the production Champion model.

    Example:

        scanner = MarketScanner()

        results = scanner.scan(
            symbols=[
                "RELIANCE.NS",
                "TCS.NS",
                "INFY.NS",
            ],
            history_provider=get_history,
            top_n=5,
        )
    """

    def __init__(
        self,
        model: Any | None = None,
        model_metadata: dict[str, Any] | None = None,
        minimum_probability: float = 0.50,
        minimum_confidence: float = 0.0,
        minimum_opportunity_score: float | None = None,
    ) -> None:

        self.model = model

        self.model_metadata = (
            dict(model_metadata)
            if model_metadata
            else {}
        )

        self.minimum_probability = float(
            minimum_probability
        )

        self.minimum_confidence = float(
            minimum_confidence
        )

        self.minimum_opportunity_score = (
            float(minimum_opportunity_score)
            if minimum_opportunity_score
            is not None
            else None
        )

        self.last_results: list[
            dict[str, Any]
        ] = []

        self.last_errors: dict[
            str,
            str,
        ] = {}

    # ========================================================
    # MODEL LOADING
    # ========================================================

    def load_model(
        self,
        force_reload: bool = False,
    ) -> Any:
        """
        Load the current production Champion model.

        The model is loaded only when needed unless
        force_reload=True.
        """

        if (
            self.model is not None
            and not force_reload
        ):

            return self.model

        from src.model_loader import (
            load_champion_model,
        )

        model, metadata = (
            load_champion_model(
                use_cache=not force_reload
            )
        )

        self.model = model

        self.model_metadata = dict(
            metadata
        )

        logger.info(
            "Production model loaded | "
            "name=%s | version=%s",
            self.model_metadata.get(
                "name"
            ),
            self.model_metadata.get(
                "model_version"
            ),
        )

        return self.model

    # ========================================================
    # HISTORY VALIDATION
    # ========================================================

    @staticmethod
    def validate_history(
        history: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate and normalize OHLCV history.
        """

        if history is None:

            raise ValueError(
                "Historical data is None."
            )

        if not isinstance(
            history,
            pd.DataFrame,
        ):

            raise TypeError(
                "Historical data must be "
                "a pandas DataFrame."
            )

        if history.empty:

            raise ValueError(
                "Historical data is empty."
            )

        frame = history.copy()

        missing = (
            REQUIRED_COLUMNS
            - set(frame.columns)
        )

        if missing:

            raise ValueError(
                "Missing OHLCV columns: "
                + ", ".join(
                    sorted(missing)
                )
            )

        for column in REQUIRED_COLUMNS:

            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

        frame = frame.dropna(
            subset=list(
                REQUIRED_COLUMNS
            )
        )

        if frame.empty:

            raise ValueError(
                "Historical data contains "
                "no valid OHLCV rows."
            )

        return frame

    # ========================================================
    # FEATURE BUILDING
    # ========================================================

    @staticmethod
    def build_features(
        history: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build the feature frame used by ProductionModel.
        """

        from src.feature_engine import (
            build_feature_frame,
        )

        frame = (
            MarketScanner.validate_history(
                history
            )
        )

        features = build_feature_frame(
            frame
        )

        if features is None:

            raise RuntimeError(
                "Feature engine returned None."
            )

        if not isinstance(
            features,
            pd.DataFrame,
        ):

            raise TypeError(
                "Feature engine must return "
                "a pandas DataFrame."
            )

        if features.empty:

            raise ValueError(
                "Feature generation returned "
                "an empty DataFrame."
            )

        return features

    # ========================================================
    # NUMERIC VALIDATION
    # ========================================================

    @staticmethod
    def safe_float(
        value: Any,
        default: float | None = None,
    ) -> float | None:
        """
        Convert a value to a finite float.
        """

        try:

            result = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return default

        if not math.isfinite(
            result
        ):

            return default

        return result

    # ========================================================
    # PREDICTION VALIDATION
    # ========================================================

    def validate_prediction(
        self,
        prediction: dict[str, Any],
    ) -> bool:
        """
        Validate a model prediction.

        Requires a finite opportunity score.
        """

        if not isinstance(
            prediction,
            dict,
        ):

            return False

        opportunity_score = (
            self.safe_float(
                prediction.get(
                    "opportunity_score"
                )
            )
        )

        if opportunity_score is None:

            return False

        probability_up = (
            self.safe_float(
                prediction.get(
                    "probability_up"
                )
            )
        )

        if (
            probability_up is not None
            and probability_up
            < self.minimum_probability
        ):

            return False

        confidence = (
            self.safe_float(
                prediction.get(
                    "confidence"
                )
            )
        )

        if (
            confidence is not None
            and confidence
            < self.minimum_confidence
        ):

            return False

        if (
            self.minimum_opportunity_score
            is not None
            and opportunity_score
            < self.minimum_opportunity_score
        ):

            return False

        return True

    # ========================================================
    # SCAN ONE SYMBOL
    # ========================================================

    def scan_symbol(
        self,
        symbol: str,
        history: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Generate a prediction for one stock.

        Returns a normalized result dictionary.
        """

        if not symbol:

            raise ValueError(
                "Symbol cannot be empty."
            )

        model = self.load_model()

        clean_history = (
            self.validate_history(
                history
            )
        )

        feature_frame = (
            self.build_features(
                clean_history
            )
        )

        prediction = model.predict(
            feature_frame
        )

        if not isinstance(
            prediction,
            dict,
        ):

            raise RuntimeError(
                "Production model predict() "
                "must return a dictionary."
            )

        if not self.validate_prediction(
            prediction
        ):

            raise RuntimeError(
                "Prediction did not pass "
                "scanner filters."
            )

        latest = clean_history.iloc[
            -1
        ]

        result: dict[
            str,
            Any
        ] = {
            "symbol": str(symbol),

            "scan_status": "SUCCESS",

            "latest_open": self.safe_float(
                latest["Open"]
            ),

            "latest_high": self.safe_float(
                latest["High"]
            ),

            "latest_low": self.safe_float(
                latest["Low"]
            ),

            "latest_close": self.safe_float(
                latest["Close"]
            ),

            "latest_volume": self.safe_float(
                latest["Volume"]
            ),

            "expected_return": (
                self.safe_float(
                    prediction.get(
                        "expected_return"
                    )
                )
            ),

            "probability_up": (
                self.safe_float(
                    prediction.get(
                        "probability_up"
                    )
                )
            ),

            "expected_risk": (
                self.safe_float(
                    prediction.get(
                        "expected_risk"
                    )
                )
            ),

            "risk_adjusted_return": (
                self.safe_float(
                    prediction.get(
                        "risk_adjusted_return"
                    )
                )
            ),

            "opportunity_score": (
                self.safe_float(
                    prediction.get(
                        "opportunity_score"
                    )
                )
            ),

            "confidence": (
                self.safe_float(
                    prediction.get(
                        "confidence"
                    )
                )
            ),

            "direction": prediction.get(
                "direction"
            ),

            "model_version": prediction.get(
                "model_version",
                self.model_metadata.get(
                    "model_version"
                ),
            ),

            "model_name": (
                self.model_metadata.get(
                    "name"
                )
            ),

            "feature_count": prediction.get(
                "feature_count"
            ),

            "prediction": dict(
                prediction
            ),
        }

        return result

    # ========================================================
    # SCAN MULTIPLE SYMBOLS
    # ========================================================

    def scan(
        self,
        symbols: Iterable[str],
        history_provider: HistoryProvider,
        top_n: int = 5,
        continue_on_error: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Scan multiple symbols and return the Top N.

        Parameters:

            symbols:
                Iterable of stock symbols.

            history_provider:
                Callable:

                    history_provider(symbol)
                        -> pandas.DataFrame

            top_n:
                Number of top opportunities.

            continue_on_error:
                If True, failed symbols are recorded
                and scanning continues.
        """

        if top_n <= 0:

            raise ValueError(
                "top_n must be greater than zero."
            )

        if not callable(
            history_provider
        ):

            raise TypeError(
                "history_provider must be callable."
            )

        self.last_results = []

        self.last_errors = {}

        self.load_model()

        for raw_symbol in symbols:

            symbol = str(
                raw_symbol
            ).strip()

            if not symbol:

                continue

            try:

                history = history_provider(
                    symbol
                )

                result = self.scan_symbol(
                    symbol=symbol,
                    history=history,
                )

                self.last_results.append(
                    result
                )

                logger.info(
                    "Scan success | "
                    "symbol=%s | score=%s",
                    symbol,
                    result.get(
                        "opportunity_score"
                    ),
                )

            except Exception as error:

                self.last_errors[
                    symbol
                ] = str(error)

                logger.warning(
                    "Scan failed | "
                    "symbol=%s | error=%s",
                    symbol,
                    error,
                )

                if not continue_on_error:

                    raise

        ranked = self.rank_results(
            self.last_results
        )

        return ranked[:top_n]

    # ========================================================
    # SCAN PRELOADED HISTORIES
    # ========================================================

    def scan_histories(
        self,
        histories: dict[
            str,
            pd.DataFrame,
        ],
        top_n: int = 5,
        continue_on_error: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Scan preloaded historical data.

        Example:

            histories = {
                "RELIANCE.NS": reliance_df,
                "TCS.NS": tcs_df,
            }
        """

        if not isinstance(
            histories,
            dict,
        ):

            raise TypeError(
                "histories must be a dictionary."
            )

        def provider(
            symbol: str,
        ) -> pd.DataFrame:

            if symbol not in histories:

                raise KeyError(
                    f"No history available "
                    f"for {symbol}"
                )

            return histories[
                symbol
            ]

        return self.scan(
            symbols=histories.keys(),
            history_provider=provider,
            top_n=top_n,
            continue_on_error=continue_on_error,
        )

    # ========================================================
    # RANK RESULTS
    # ========================================================

    @staticmethod
    def rank_results(
        results: list[
            dict[str, Any]
        ],
    ) -> list[
        dict[str, Any]
    ]:
        """
        Rank successful results.

        Primary ranking:

            opportunity_score descending

        Secondary ranking:

            confidence descending

        Third ranking:

            probability_up descending
        """

        def ranking_key(
            result: dict[str, Any],
        ) -> tuple[
            float,
            float,
            float,
        ]:

            score = (
                MarketScanner.safe_float(
                    result.get(
                        "opportunity_score"
                    ),
                    default=float(
                        "-inf"
                    ),
                )
            )

            confidence = (
                MarketScanner.safe_float(
                    result.get(
                        "confidence"
                    ),
                    default=0.0,
                )
            )

            probability = (
                MarketScanner.safe_float(
                    result.get(
                        "probability_up"
                    ),
                    default=0.0,
                )
            )

            return (
                score
                if score is not None
                else float("-inf"),

                confidence
                if confidence is not None
                else 0.0,

                probability
                if probability is not None
                else 0.0,
            )

        ranked = sorted(
            results,
            key=ranking_key,
            reverse=True,
        )

        output: list[
            dict[str, Any]
        ] = []

        for rank, result in enumerate(
            ranked,
            start=1,
        ):

            item = dict(
                result
            )

            item["rank"] = rank

            output.append(
                item
            )

        return output

    # ========================================================
    # SCAN SUMMARY
    # ========================================================

    def get_scan_summary(
        self,
    ) -> dict[str, Any]:
        """
        Return a summary of the most recent scan.
        """

        return {
            "successful_symbols": len(
                self.last_results
            ),

            "failed_symbols": len(
                self.last_errors
            ),

            "errors": dict(
                self.last_errors
            ),

            "model_name": (
                self.model_metadata.get(
                    "name"
                )
            ),

            "model_version": (
                self.model_metadata.get(
                    "model_version",
                    getattr(
                        self.model,
                        "model_version",
                        None,
                    ),
                )
            ),
        }


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def scan_market(
    symbols: Iterable[str],
    history_provider: HistoryProvider,
    top_n: int = 5,
    minimum_probability: float = 0.50,
    minimum_confidence: float = 0.0,
    minimum_opportunity_score: (
        float | None
    ) = None,
) -> list[
    dict[str, Any]
]:
    """
    Convenience function for market scanning.
    """

    scanner = MarketScanner(
        minimum_probability=(
            minimum_probability
        ),
        minimum_confidence=(
            minimum_confidence
        ),
        minimum_opportunity_score=(
            minimum_opportunity_score
        ),
    )

    return scanner.scan(
        symbols=symbols,
        history_provider=history_provider,
        top_n=top_n,
    )


# ============================================================
# CLI DEMO
# ============================================================

def main() -> int:
    """
    Demonstrate scanner usage.

    A real history provider must be connected
    before using this in production.
    """

    print()

    print("=" * 70)

    print("MARKET SCANNER")

    print("=" * 70)

    print()

    print(
        "MarketScanner is ready."
    )

    print()

    print(
        "Usage:"
    )

    print()

    print(
        "    scanner = MarketScanner()"
    )

    print()

    print(
        "    results = scanner.scan("
    )

    print(
        "        symbols=symbols,"
    )

    print(
        "        history_provider="
        "your_history_provider,"
    )

    print(
        "        top_n=5,"
    )

    print(
        "    )"
    )

    print()

    return 0


if __name__ == "__main__"

    raise SystemExit(
        main()
    )

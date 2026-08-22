# prediction_store.py
#!/usr/bin/env python3

"""
Prediction Storage.

This module stores production predictions so they can
later be compared with actual market results.

Typical flow:

    Morning Prediction
            │
            ▼
    StockRanker
            │
            ▼
    Top 5 Predictions
            │
            ▼
    PredictionStore
            │
            ▼
    data/predictions/YYYY-MM-DD.json
            │
            ▼
    Later:
            │
            ▼
    Load Prediction
            │
            ▼
    Fetch Actual OHLC
            │
            ▼
    Evaluation Engine

The store supports:

    - Saving prediction runs
    - Loading predictions by date
    - Listing prediction dates
    - Updating actual results
    - Atomic file writes
    - Symbol lookup
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(
    "prediction_store"
)


# ============================================================
# VERSION
# ============================================================

PREDICTION_STORE_VERSION = "1.0"


# ============================================================
# TIME HELPERS
# ============================================================

def utc_now_iso() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def normalize_prediction_date(
    value: str | date | datetime | None,
) -> str:
    """
    Convert a date value into YYYY-MM-DD format.
    """

    if value is None:

        return datetime.now(
            timezone.utc
        ).date().isoformat()

    if isinstance(
        value,
        datetime,
    ):

        return value.date().isoformat()

    if isinstance(
        value,
        date,
    ):

        return value.isoformat()

    text = str(
        value
    ).strip()

    if not text:

        raise ValueError(
            "Prediction date cannot be empty."
        )

    try:

        parsed = datetime.fromisoformat(
            text
        )

        return parsed.date().isoformat()

    except ValueError:

        pass

    try:

        parsed_date = date.fromisoformat(
            text
        )

        return parsed_date.isoformat()

    except ValueError as error:

        raise ValueError(
            "Invalid prediction date: "
            f"{value}"
        ) from error


# ============================================================
# JSON HELPERS
# ============================================================

def make_json_safe(
    value: Any,
) -> Any:
    """
    Convert common Python and pandas values into
    JSON-safe values.
    """

    if value is None:

        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):

        return value

    if isinstance(
        value,
        (
            datetime,
            date,
        ),
    ):

        return value.isoformat()

    if hasattr(
        value,
        "item",
    ):

        try:

            return make_json_safe(
                value.item()
            )

        except Exception:

            pass

    if hasattr(
        value,
        "isoformat",
    ):

        try:

            return value.isoformat()

        except Exception:

            pass

    if isinstance(
        value,
        dict,
    ):

        return {
            str(key): make_json_safe(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):

        return [
            make_json_safe(
                item
            )
            for item in value
        ]

    return str(
        value
    )


# ============================================================
# PREDICTION STORE
# ============================================================

class PredictionStore:
    """
    Persistent storage for production predictions.

    Default structure:

        data/
            predictions/
                2026-08-22.json
                2026-08-23.json
    """

    def __init__(
        self,
        base_path: str | Path | None = None,
    ) -> None:

        if base_path is None:

            base_path = (
                PROJECT_ROOT
                / "data"
                / "predictions"
            )

        self.base_path = Path(
            base_path
        )

        if not self.base_path.is_absolute():

            self.base_path = (
                PROJECT_ROOT
                / self.base_path
            )

        self.base_path.mkdir(
            parents=True,
            exist_ok=True,
        )


    # ========================================================
    # PATH HELPERS
    # ========================================================

    def get_prediction_path(
        self,
        prediction_date: (
            str
            | date
            | datetime
            | None
        ) = None,
    ) -> Path:
        """
        Return the prediction file path for a date.
        """

        normalized_date = (
            normalize_prediction_date(
                prediction_date
            )
        )

        return (
            self.base_path
            / f"{normalized_date}.json"
        )


    def exists(
        self,
        prediction_date: (
            str
            | date
            | datetime
            | None
        ) = None,
    ) -> bool:
        """Check whether predictions exist."""

        return (
            self.get_prediction_path(
                prediction_date
            ).exists()
        )


    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        predictions: dict[str, Any],
        prediction_date: (
            str
            | date
            | datetime
            | None
        ) = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Save a complete prediction run.

        Example:

            store.save(
                predictions=result,
                prediction_date="2026-08-22",
            )
        """

        if not isinstance(
            predictions,
            dict,
        ):

            raise TypeError(
                "predictions must be "
                "a dictionary."
            )

        normalized_date = (
            normalize_prediction_date(
                prediction_date
            )
        )

        output_path = (
            self.get_prediction_path(
                normalized_date
            )
        )

        if (
            output_path.exists()
            and not overwrite
        ):

            raise FileExistsError(
                "Prediction file already exists: "
                f"{output_path}"
            )

        payload = {
            "store_version": (
                PREDICTION_STORE_VERSION
            ),
            "prediction_date": (
                normalized_date
            ),
            "saved_at": (
                utc_now_iso()
            ),
            "evaluated": False,
            "predictions": (
                make_json_safe(
                    predictions
                )
            ),
        }

        temporary_path: Path | None = None

        try:

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json.tmp",
                dir=self.base_path,
                encoding="utf-8",
                delete=False,
            ) as temporary_file:

                temporary_path = Path(
                    temporary_file.name
                )

                json.dump(
                    payload,
                    temporary_file,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )

            os.replace(
                temporary_path,
                output_path,
            )

            logger.info(
                "Saved predictions | "
                "date=%s | path=%s",
                normalized_date,
                output_path,
            )

            return output_path

        except Exception:

            logger.exception(
                "Failed to save predictions "
                "for %s",
                normalized_date,
            )

            raise

        finally:

            if (
                temporary_path is not None
                and temporary_path.exists()
            ):

                try:

                    temporary_path.unlink()

                except OSError:

                    pass


    # ========================================================
    # LOAD
    # ========================================================

    def load(
        self,
        prediction_date: (
            str
            | date
            | datetime
            | None
        ) = None,
    ) -> dict[str, Any] | None:
        """
        Load predictions for a date.
        """

        normalized_date = (
            normalize_prediction_date(
                prediction_date
            )
        )

        path = self.get_prediction_path(
            normalized_date
        )

        if not path.exists():

            return None

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                payload = json.load(
                    file
                )

            if not isinstance(
                payload,
                dict,
            ):

                raise RuntimeError(
                    "Prediction file must contain "
                    "a JSON object."
                )

            return payload

        except Exception:

            logger.exception(
                "Failed to load predictions "
                "for %s",
                normalized_date,
            )

            raise


    # ========================================================
    # LOAD PREDICTION RESULTS
    # ========================================================

    def load_predictions(
        self,
        prediction_date: (
            str
            | date
            | datetime
            | None
        ) = None,
    ) -> dict[str, Any] | None:
        """
        Load only the stored prediction result.
        """

        payload = self.load(
            prediction_date
        )

        if payload is None:

            return None

        predictions = payload.get(
            "predictions"
        )

        if not isinstance(
            predictions,
            dict,
        ):

            raise RuntimeError(
                "Stored predictions are invalid."
            )

        return predictions


    # ========================================================
    # LIST DATES
    # ========================================================

    def list_prediction_dates(
        self,
    ) -> list[str]:
        """
        Return all stored prediction dates.
        """

        if not self.base_path.exists():

            return []

        dates: list[str] = []

        for path in self.base_path.glob(
            "*.json"
        ):

            try:

                parsed = date.fromisoformat(
                    path.stem
                )

                dates.append(
                    parsed.isoformat()
                )

            except ValueError:

                continue

        return sorted(
            dates
        )


    # ========================================================
    # GET STOCK PREDICTION
    # ========================================================

    def get_stock_prediction(
        self,
        symbol: str,
        prediction_date: (
            str
            | date
            | datetime
            | None
        ) = None,
    ) -> dict[str, Any] | None:
        """
        Find one stock prediction by symbol.
        """

        predictions = (
            self.load_predictions(
                prediction_date
            )
        )

        if predictions is None:

            return None

        normalized_symbol = str(
            symbol
        ).strip()

        if not normalized_symbol:

            raise ValueError(
                "Symbol cannot be empty."
            )

        candidates = (
            predictions.get(
                "all_ranked_stocks"
            )
            or predictions.get(
                "top_stocks"
            )
            or []
        )

        if not isinstance(
            candidates,
            list,
        ):

            return None

        for prediction in candidates:

            if not isinstance(
                prediction,
                dict,
            ):

                continue

            if str(
                prediction.get(
                    "symbol",
                    "",
                )
            ) == normalized_symbol:

                return dict(
                    prediction
                )

        return None


    # ========================================================
    # GET TOP STOCKS
    # ========================================================

    def get_top_stocks(
        self,
        prediction_date: (
            str
            | date
            | datetime
            | None
        ) = None,
    ) -> list[dict[str, Any]]:
        """
        Return the Top selected stocks.
        """

        predictions = (
            self.load_predictions(
                prediction_date
            )
        )

        if predictions is None:

            return []

        top_stocks = predictions.get(
            "top_stocks",
            [],
        )

        if not isinstance(
            top_stocks,
            list,
        ):

            return []

        return [
            dict(item)
            for item in top_stocks
            if isinstance(
                item,
                dict,
            )
        ]


    # ========================================================
    # UPDATE ACTUAL RESULT
    # ========================================================

    def update_actual_result(
        self,
        symbol: str,
        actual_result: dict[str, Any],
        prediction_date: (
            str
            | date
            | datetime
            | None
        ) = None,
    ) -> bool:
        """
        Attach actual market results to
        a stored stock prediction.

        Example:

            store.update_actual_result(
                symbol="RELIANCE.NS",
                actual_result={
                    "Open": 1400.0,
                    "High": 1420.0,
                    "Low": 1385.0,
                    "Close": 1415.0,
                },
                prediction_date="2026-08-22",
            )
        """

        if not isinstance(
            actual_result,
            dict,
        ):

            raise TypeError(
                "actual_result must be "
                "a dictionary."
            )

        normalized_date = (
            normalize_prediction_date(
                prediction_date
            )
        )

        payload = self.load(
            normalized_date
        )

        if payload is None:

            raise FileNotFoundError(
                "No predictions found for "
                f"{normalized_date}"
            )

        prediction_result = payload.get(
            "predictions"
        )

        if not isinstance(
            prediction_result,
            dict,
        ):

            raise RuntimeError(
                "Stored prediction result "
                "is invalid."
            )

        candidates = (
            prediction_result.get(
                "all_ranked_stocks"
            )
            or prediction_result.get(
                "top_stocks"
            )
            or []
        )

        if not isinstance(
            candidates,
            list,
        ):

            raise RuntimeError(
                "Stored stock predictions "
                "are invalid."
            )

        normalized_symbol = str(
            symbol
        ).strip()

        updated = False

        for prediction in candidates:

            if not isinstance(
                prediction,
                dict,
            ):

                continue

            if str(
                prediction.get(
                    "symbol",
                    "",
                )
            ) != normalized_symbol:

                continue

            prediction["actual_result"] = (
                make_json_safe(
                    actual_result
                )
            )

            prediction["actual_updated_at"] = (
                utc_now_iso()
            )

            updated = True

            break

        if not updated:

            return False

        payload["saved_at"] = (
            utc_now_iso()
        )

        self._save_payload(
            payload=payload,
            prediction_date=normalized_date,
        )

        logger.info(
            "Actual result updated | "
            "symbol=%s | date=%s",
            normalized_symbol,
            normalized_date,
        )

        return True


    # ========================================================
    # MARK EVALUATED
    # ========================================================

    def mark_evaluated(
        self,
        prediction_date: (
            str
            | date
            | datetime
            | None
        ) = None,
        evaluation: dict[str, Any] | None = None,
    ) -> None:
        """
        Mark a prediction run as evaluated.
        """

        normalized_date = (
            normalize_prediction_date(
                prediction_date
            )
        )

        payload = self.load(
            normalized_date
        )

        if payload is None:

            raise FileNotFoundError(
                "No predictions found for "
                f"{normalized_date}"
            )

        payload["evaluated"] = True

        payload["evaluated_at"] = (
            utc_now_iso()
        )

        if evaluation is not None:

            payload["evaluation"] = (
                make_json_safe(
                    evaluation
                )
            )

        self._save_payload(
            payload=payload,
            prediction_date=normalized_date,
        )

        logger.info(
            "Predictions marked as evaluated: %s",
            normalized_date,
        )


    # ========================================================
    # INTERNAL SAVE
    # ========================================================

    def _save_payload(
        self,
        payload: dict[str, Any],
        prediction_date: str,
    ) -> None:
        """
        Atomically save an existing payload.
        """

        output_path = (
            self.get_prediction_path(
                prediction_date
            )
        )

        temporary_path: Path | None = None

        try:

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json.tmp",
                dir=self.base_path,
                encoding="utf-8",
                delete=False,
            ) as temporary_file:

                temporary_path = Path(
                    temporary_file.name
                )

                json.dump(
                    make_json_safe(
                        payload
                    ),
                    temporary_file,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )

            os.replace(
                temporary_path,
                output_path,
            )

        finally:

            if (
                temporary_path is not None
                and temporary_path.exists()
            ):

                try:

                    temporary_path.unlink()

                except OSError:

                    pass


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def save_predictions(
    predictions: dict[str, Any],
    prediction_date: (
        str
        | date
        | datetime
        | None
    ) = None,
    overwrite: bool = False,
) -> Path:
    """Save predictions using the default store."""

    store = PredictionStore()

    return store.save(
        predictions=predictions,
        prediction_date=prediction_date,
        overwrite=overwrite,
    )


def load_predictions(
    prediction_date: (
        str
        | date
        | datetime
        | None
    ) = None,
) -> dict[str, Any] | None:
    """Load predictions using the default store."""

    store = PredictionStore()

    return store.load_predictions(
        prediction_date
    )


# ============================================================
# CLI TEST
# ============================================================

def main() -> int:
    """Display prediction store information."""

    store = PredictionStore()

    print()

    print("=" * 70)

    print(
        "PREDICTION STORE"
    )

    print("=" * 70)

    print()

    print(
        "Base path:",
        store.base_path,
    )

    dates = (
        store.list_prediction_dates()
    )

    print()

    print(
        "Stored prediction dates:",
        len(dates),
    )

    for prediction_date in dates:

        print(
            "-",
            prediction_date,
        )

    print()

    print(
        "SUCCESS: PredictionStore "
        "is ready."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

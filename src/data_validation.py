"""Validation helpers for predictions and actual market data."""

from __future__ import annotations

import math


def validate_prediction(
    symbol: str,
    prediction: dict,
) -> tuple[bool, dict]:
    """Validate predicted OHLC values."""

    required = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    values = {}

    for key in required:

        try:
            value = float(
                prediction[key]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            return False, {
                "reason": (
                    f"invalid_{key.lower()}"
                )
            }

        if (
            not math.isfinite(value)
            or value <= 0
        ):

            return False, {
                "reason": (
                    f"non_positive_{key.lower()}"
                )
            }

        values[key] = value

    # High must be greater than or equal to
    # every other OHLC value.
    if values["High"] < max(
        values["Open"],
        values["Close"],
        values["Low"],
    ):

        return False, {
            "reason": "high_inconsistent"
        }

    # Low must be lower than or equal to
    # every other OHLC value.
    if values["Low"] > min(
        values["Open"],
        values["Close"],
        values["High"],
    ):

        return False, {
            "reason": "low_inconsistent"
        }

    return True, {
        "data_quality_score": 1.0
    }


def validate_actual(
    actual: dict,
) -> bool:
    """Validate actual OHLC data."""

    valid, _ = validate_prediction(
        "actual",
        actual,
    )

    return valid

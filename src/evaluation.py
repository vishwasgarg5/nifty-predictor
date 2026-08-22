"""Model evaluation utilities."""

from __future__ import annotations

import pandas as pd


def metrics_from_ledger(
    df: pd.DataFrame,
) -> dict:
    """Calculate metrics only from evaluated records."""

    if df.empty:

        return {
            "n": 0,
            "mae": None,
            "mape": None,
            "directional_accuracy": None,
        }

    evaluated = df[
        df["evaluation_status"]
        == "evaluated"
    ].copy()

    if evaluated.empty:

        return {
            "n": 0,
            "mae": None,
            "mape": None,
            "directional_accuracy": None,
        }

    absolute_error = pd.to_numeric(
        evaluated["abs_error"],
        errors="coerce",
    )

    absolute_error_pct = pd.to_numeric(
        evaluated["abs_error_pct"],
        errors="coerce",
    )

    direction_correct = pd.to_numeric(
        evaluated["direction_correct"],
        errors="coerce",
    )

    return {
        "n": int(len(evaluated)),

        "mae": float(
            absolute_error.mean()
        ),

        "mape": float(
            absolute_error_pct.mean()
        ),

        "directional_accuracy": float(
            direction_correct.mean()
            * 100
        ),
    }

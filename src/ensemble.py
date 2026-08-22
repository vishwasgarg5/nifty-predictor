"""Ensemble logic for Return, Direction and Risk models."""

from __future__ import annotations

import math


MODEL_VERSION = "ensemble-v1"


def _clip(
    value: float,
    minimum: float,
    maximum: float,
) -> float:

    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def build_ensemble(
    expected_return: float | None,
    probability_up: float | None,
    expected_risk: float | None,
) -> dict:
    """Combine independent model outputs.

    Higher return and higher UP probability
    increase opportunity.

    Higher expected risk reduces opportunity.
    """

    if expected_return is None:

        expected_return = 0.0

    if probability_up is None:

        probability_up = 0.5

    if expected_risk is None:

        expected_risk = 0.05

    expected_return = float(
        expected_return
    )

    probability_up = _clip(
        float(probability_up),
        0.0,
        1.0,
    )

    expected_risk = max(
        0.0001,
        float(expected_risk),
    )

    # Return score.
    #
    # +/- 5% expected daily movement maps
    # approximately to the full score range.
    return_score = _clip(
        0.5
        + (
            expected_return
            / 0.10
        ),
        0.0,
        1.0,
    )

    # Risk-adjusted expected return.
    risk_adjusted_return = (
        expected_return
        / expected_risk
    )

    risk_adjusted_score = _clip(
        0.5
        + (
            risk_adjusted_return
            / 4.0
        ),
        0.0,
        1.0,
    )

    # Opportunity combines the three views.
    opportunity_score = (

        0.40
        * return_score

        + 0.35
        * probability_up

        + 0.25
        * risk_adjusted_score
    )

    opportunity_score = _clip(
        opportunity_score,
        0.0,
        1.0,
    )

    # Agreement between return model and
    # direction model.
    if expected_return > 0:

        return_direction = 1

    elif expected_return < 0:

        return_direction = -1

    else:

        return_direction = 0

    probability_direction = (

        1
        if probability_up >= 0.5
        else -1
    )

    agreement = (

        1.0
        if (
            return_direction
            == probability_direction
        )
        else 0.5
    )

    # Direction certainty.
    direction_certainty = abs(
        probability_up - 0.5
    ) * 2.0

    # Lower predicted risk means higher
    # confidence, but the risk component
    # is intentionally capped.
    risk_confidence = 1.0 - _clip(
        expected_risk / 0.10,
        0.0,
        1.0,
    )

    confidence = (

        0.45
        * direction_certainty

        + 0.30
        * agreement

        + 0.25
        * risk_confidence
    )

    confidence = _clip(
        confidence,
        0.0,
        1.0,
    )

    if probability_up >= 0.55:

        direction = "UP"

    elif probability_up <= 0.45:

        direction = "DOWN"

    else:

        direction = "NEUTRAL"

    return {

        "expected_return": round(
            expected_return,
            6,
        ),

        "probability_up": round(
            probability_up,
            6,
        ),

        "expected_risk": round(
            expected_risk,
            6,
        ),

        "risk_adjusted_return": round(
            risk_adjusted_return,
            6,
        ),

        "opportunity_score": round(
            opportunity_score,
            6,
        ),

        "confidence": round(
            confidence,
            6,
        ),

        "agreement": round(
            agreement,
            6,
        ),

        "direction": direction,

        "model_version": MODEL_VERSION,
    }

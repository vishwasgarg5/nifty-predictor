"""Market regime detection.

The regime is intentionally simple and transparent:

    BULLISH
    NEUTRAL
    BEARISH

It uses index trend, medium-term momentum and volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_last(series: pd.Series):
    """Return the latest valid value."""

    if series is None:
        return None

    clean = series.dropna()

    if clean.empty:
        return None

    return float(clean.iloc[-1])


def detect_market_regime(
    history: pd.DataFrame,
) -> dict:
    """Classify the market regime.

    Expected columns:

        Open
        High
        Low
        Close
        Volume
    """

    if history is None or history.empty:
        return {
            "regime": "UNKNOWN",
            "score": 0.0,
            "confidence": 0.0,
            "reason": "no_data",
        }

    df = history.copy()

    if "Close" not in df.columns:
        return {
            "regime": "UNKNOWN",
            "score": 0.0,
            "confidence": 0.0,
            "reason": "missing_close",
        }

    close = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    close = close.dropna()

    if len(close) < 60:
        return {
            "regime": "UNKNOWN",
            "score": 0.0,
            "confidence": 0.0,
            "reason": "insufficient_history",
        }

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    returns = close.pct_change()

    momentum_20 = (
        close / close.shift(20)
    ) - 1

    volatility_20 = (
        returns
        .rolling(20)
        .std()
        * np.sqrt(252)
    )

    latest_close = _safe_last(close)
    latest_sma20 = _safe_last(sma20)
    latest_sma50 = _safe_last(sma50)
    latest_momentum = _safe_last(
        momentum_20
    )
    latest_volatility = _safe_last(
        volatility_20
    )

    if None in (
        latest_close,
        latest_sma20,
        latest_sma50,
        latest_momentum,
    ):
        return {
            "regime": "UNKNOWN",
            "score": 0.0,
            "confidence": 0.0,
            "reason": "invalid_indicators",
        }

    score = 0.0

    # Price trend.
    if latest_close > latest_sma20:
        score += 1.0
    else:
        score -= 1.0

    if latest_sma20 > latest_sma50:
        score += 1.0
    else:
        score -= 1.0

    # Medium-term momentum.
    if latest_momentum > 0.02:
        score += 1.0
    elif latest_momentum < -0.02:
        score -= 1.0

    # High volatility reduces regime confidence.
    volatility_penalty = 0.0

    if (
        latest_volatility is not None
        and latest_volatility > 0.35
    ):
        volatility_penalty = 0.5

    if score >= 2.0:
        regime = "BULLISH"

    elif score <= -2.0:
        regime = "BEARISH"

    else:
        regime = "NEUTRAL"

    confidence = min(
        1.0,
        (abs(score) / 3.0)
        * (1.0 - volatility_penalty),
    )

    return {
        "regime": regime,
        "score": round(score, 4),
        "confidence": round(
            max(0.0, confidence),
            4,
        ),
        "close": latest_close,
        "sma20": latest_sma20,
        "sma50": latest_sma50,
        "momentum_20": latest_momentum,
        "volatility_20": latest_volatility,
    }

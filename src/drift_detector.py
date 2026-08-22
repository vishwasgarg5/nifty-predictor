"""Model performance drift detection.

Phase 4 architecture:

    Prediction Ledger
            │
            ▼
    Historical Performance
            │
            ├── Return MAE
            ├── Direction Accuracy
            ├── Brier Score
            └── Risk MAE
                    │
                    ▼
             Recent Performance
                    │
                    ▼
               Compare Windows
                    │
                    ▼
              Drift Detection
                    │
            ┌───────┴────────┐
            ▼                ▼
        STABLE             DRIFT
                              │
                              ▼
                    Retrain / Challenger
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class DriftResult:
    """Result of model performance drift analysis."""

    drift_detected: bool

    severity: str

    score: float

    reasons: list[str]

    historical_metrics: dict[str, float]

    recent_metrics: dict[str, float]

    metric_changes: dict[str, float]

    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""

        return {

            "drift_detected": self.drift_detected,

            "severity": self.severity,

            "score": self.score,

            "reasons": self.reasons,

            "historical_metrics": (
                self.historical_metrics
            ),

            "recent_metrics": (
                self.recent_metrics
            ),

            "metric_changes": (
                self.metric_changes
            ),

            "recommendation": (
                self.recommendation
            ),
        }


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_CONFIG = {

    # Number of recent evaluated predictions
    # used to detect short-term deterioration.
    "recent_window": 20,

    # Minimum number of older observations
    # required for baseline comparison.
    "minimum_history": 30,

    # Return MAE increase threshold.
    "return_mae_threshold": 0.30,

    # Direction accuracy decrease threshold.
    "direction_accuracy_threshold": 0.10,

    # Brier score increase threshold.
    "brier_score_threshold": 0.20,

    # Risk MAE increase threshold.
    "risk_mae_threshold": 0.30,

    # Drift score thresholds.
    "warning_threshold": 1.0,

    "drift_threshold": 2.0,

    "critical_threshold": 3.0,
}


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_mean(
    values: pd.Series,
) -> float | None:
    """Calculate a safe numeric mean."""

    if values is None:

        return None

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:

        return None

    return float(
        numeric.mean()
    )


def safe_relative_change(
    historical: float | None,
    recent: float | None,
) -> float | None:
    """Calculate relative change safely.

    Positive values mean the recent metric
    is larger than the historical metric.

    Example:

        historical = 0.10
        recent = 0.15

        result = +0.50

    which means a 50% increase.
    """

    if historical is None:

        return None

    if recent is None:

        return None

    if not np.isfinite(
        historical
    ):

        return None

    if not np.isfinite(
        recent
    ):

        return None

    denominator = abs(
        historical
    )

    if denominator < 1e-12:

        # If baseline is effectively zero,
        # use absolute change instead.
        return float(
            recent - historical
        )

    return float(
        (recent - historical)
        / denominator
    )


def safe_absolute_change(
    historical: float | None,
    recent: float | None,
) -> float | None:
    """Calculate absolute metric change."""

    if historical is None:

        return None

    if recent is None:

        return None

    if not np.isfinite(
        historical
    ):

        return None

    if not np.isfinite(
        recent
    ):

        return None

    return float(
        recent - historical
    )


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_performance_metrics(
    frame: pd.DataFrame,
) -> dict[str, float]:
    """Calculate model performance metrics.

    Expected ledger columns:

        return_absolute_error
        direction_correct
        brier_score
        risk_absolute_error
    """

    if frame is None or frame.empty:

        return {}

    metrics: dict[str, float] = {}

    # --------------------------------------------------------
    # RETURN MAE
    # --------------------------------------------------------

    if (
        "return_absolute_error"
        in frame.columns
    ):

        value = safe_mean(
            frame[
                "return_absolute_error"
            ]
        )

        if value is not None:

            metrics[
                "return_mae"
            ] = value

    # --------------------------------------------------------
    # DIRECTION ACCURACY
    # --------------------------------------------------------

    if (
        "direction_correct"
        in frame.columns
    ):

        value = safe_mean(
            frame[
                "direction_correct"
            ]
        )

        if value is not None:

            metrics[
                "direction_accuracy"
            ] = value

    # --------------------------------------------------------
    # PROBABILITY CALIBRATION
    # --------------------------------------------------------

    if (
        "brier_score"
        in frame.columns
    ):

        value = safe_mean(
            frame[
                "brier_score"
            ]
        )

        if value is not None:

            metrics[
                "brier_score"
            ] = value

    # --------------------------------------------------------
    # RISK MAE
    # --------------------------------------------------------

    if (
        "risk_absolute_error"
        in frame.columns
    ):

        value = safe_mean(
            frame[
                "risk_absolute_error"
            ]
        )

        if value is not None:

            metrics[
                "risk_mae"
            ] = value

    return metrics


# ============================================================
# DATA PREPARATION
# ============================================================

def get_evaluated_records(
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    """Return evaluated prediction records only."""

    if ledger is None:

        return pd.DataFrame()

    if ledger.empty:

        return pd.DataFrame()

    frame = ledger.copy()

    if (
        "evaluation_status"
        in frame.columns
    ):

        status = (
            frame[
                "evaluation_status"
            ]
            .astype(str)
            .str.upper()
        )

        frame = frame.loc[
            status == "EVALUATED"
        ].copy()

    if frame.empty:

        return frame

    # Sort chronologically where possible.
    date_column = None

    for candidate in (

        "market_date",

        "date",

        "evaluated_at",

    ):

        if candidate in frame.columns:

            date_column = candidate

            break

    if date_column:

        frame[
            "_drift_sort_date"
        ] = pd.to_datetime(
            frame[
                date_column
            ],
            errors="coerce",
        )

        frame = frame.sort_values(

            by="_drift_sort_date",

            ascending=True,

            na_position="last",
        )

    return frame.reset_index(
        drop=True
    )


# ============================================================
# DRIFT SCORING
# ============================================================

def detect_performance_drift(
    ledger: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> DriftResult:
    """Detect performance drift.

    The function compares:

        Historical baseline
                vs
        Most recent prediction window

    Metrics where LOWER is better:

        return_mae
        brier_score
        risk_mae

    Metric where HIGHER is better:

        direction_accuracy
    """

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    settings = (
        DEFAULT_CONFIG.copy()
    )

    if config:

        settings.update(
            config
        )

    recent_window = int(
        settings[
            "recent_window"
        ]
    )

    minimum_history = int(
        settings[
            "minimum_history"
        ]
    )

    # --------------------------------------------------------
    # EVALUATED DATA
    # --------------------------------------------------------

    evaluated = (
        get_evaluated_records(
            ledger
        )
    )

    total_records = len(
        evaluated
    )

    required_records = (
        recent_window
        + minimum_history
    )

    if (
        total_records
        < required_records
    ):

        return DriftResult(

            drift_detected=False,

            severity="INSUFFICIENT_DATA",

            score=0.0,

            reasons=[
                (
                    "Insufficient evaluated "
                    "predictions for drift analysis: "
                    f"{total_records} available, "
                    f"{required_records} required."
                )
            ],

            historical_metrics={},

            recent_metrics={},

            metric_changes={},

            recommendation=(
                "Continue collecting prediction "
                "performance data."
            ),
        )

    # --------------------------------------------------------
    # SPLIT WINDOWS
    # --------------------------------------------------------

    recent = (
        evaluated.iloc[
            -recent_window:
        ]
        .copy()
    )

    historical = (
        evaluated.iloc[
            :-recent_window
        ]
        .copy()
    )

    # Keep only the most recent
    # historical baseline window if
    # the dataset becomes very large.

    maximum_baseline = (
        max(
            minimum_history,
            recent_window * 5,
        )
    )

    if len(
        historical
    ) > maximum_baseline:

        historical = (
            historical.iloc[
                -maximum_baseline:
            ]
            .copy()
        )

    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    historical_metrics = (
        calculate_performance_metrics(
            historical
        )
    )

    recent_metrics = (
        calculate_performance_metrics(
            recent
        )
    )

    metric_changes: dict[
        str,
        float
    ] = {}

    reasons: list[
        str
    ] = []

    drift_score = 0.0

    # --------------------------------------------------------
    # RETURN MAE
    # LOWER IS BETTER
    # --------------------------------------------------------

    historical_return_mae = (
        historical_metrics.get(
            "return_mae"
        )
    )

    recent_return_mae = (
        recent_metrics.get(
            "return_mae"
        )
    )

    return_mae_change = (
        safe_relative_change(

            historical_return_mae,

            recent_return_mae,
        )
    )

    if return_mae_change is not None:

        metric_changes[
            "return_mae_change"
        ] = return_mae_change

        threshold = float(
            settings[
                "return_mae_threshold"
            ]
        )

        if (
            return_mae_change
            > threshold
        ):

            severity_ratio = (
                return_mae_change
                / threshold
            )

            drift_score += min(
                2.0,
                severity_ratio,
            )

            reasons.append(

                "Return MAE increased by "
                f"{return_mae_change:.1%} "
                f"(threshold: {threshold:.1%})."
            )

    # --------------------------------------------------------
    # DIRECTION ACCURACY
    # HIGHER IS BETTER
    # --------------------------------------------------------

    historical_accuracy = (
        historical_metrics.get(
            "direction_accuracy"
        )
    )

    recent_accuracy = (
        recent_metrics.get(
            "direction_accuracy"
        )
    )

    direction_change = (
        safe_absolute_change(

            historical_accuracy,

            recent_accuracy,
        )
    )

    if direction_change is not None:

        metric_changes[
            "direction_accuracy_change"
        ] = direction_change

        threshold = float(
            settings[
                "direction_accuracy_threshold"
            ]
        )

        if (
            direction_change
            < -threshold
        ):

            severity_ratio = (
                abs(
                    direction_change
                )
                / threshold
            )

            drift_score += min(
                2.0,
                severity_ratio,
            )

            reasons.append(

                "Direction accuracy dropped by "
                f"{abs(direction_change):.1%} "
                f"(threshold: {threshold:.1%})."
            )

    # --------------------------------------------------------
    # BRIER SCORE
    # LOWER IS BETTER
    # --------------------------------------------------------

    historical_brier = (
        historical_metrics.get(
            "brier_score"
        )
    )

    recent_brier = (
        recent_metrics.get(
            "brier_score"
        )
    )

    brier_change = (
        safe_relative_change(

            historical_brier,

            recent_brier,
        )
    )

    if brier_change is not None:

        metric_changes[
            "brier_score_change"
        ] = brier_change

        threshold = float(
            settings[
                "brier_score_threshold"
            ]
        )

        if (
            brier_change
            > threshold
        ):

            severity_ratio = (
                brier_change
                / threshold
            )

            drift_score += min(
                2.0,
                severity_ratio,
            )

            reasons.append(

                "Brier score increased by "
                f"{brier_change:.1%} "
                f"(threshold: {threshold:.1%})."
            )

    # --------------------------------------------------------
    # RISK MAE
    # LOWER IS BETTER
    # --------------------------------------------------------

    historical_risk_mae = (
        historical_metrics.get(
            "risk_mae"
        )
    )

    recent_risk_mae = (
        recent_metrics.get(
            "risk_mae"
        )
    )

    risk_mae_change = (
        safe_relative_change(

            historical_risk_mae,

            recent_risk_mae,
        )
    )

    if risk_mae_change is not None:

        metric_changes[
            "risk_mae_change"
        ] = risk_mae_change

        threshold = float(
            settings[
                "risk_mae_threshold"
            ]
        )

        if (
            risk_mae_change
            > threshold
        ):

            severity_ratio = (
                risk_mae_change
                / threshold
            )

            drift_score += min(
                2.0,
                severity_ratio,
            )

            reasons.append(

                "Risk MAE increased by "
                f"{risk_mae_change:.1%} "
                f"(threshold: {threshold:.1%})."
            )

    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    warning_threshold = float(
        settings[
            "warning_threshold"
        ]
    )

    drift_threshold = float(
        settings[
            "drift_threshold"
        ]
    )

    critical_threshold = float(
        settings[
            "critical_threshold"
        ]
    )

    if (
        drift_score
        >= critical_threshold
    ):

        severity = "CRITICAL"

        drift_detected = True

        recommendation = (

            "Performance has deteriorated "
            "significantly. Retrain the model "
            "and activate challenger evaluation."
        )

    elif (
        drift_score
        >= drift_threshold
    ):

        severity = "DRIFT"

        drift_detected = True

        recommendation = (

            "Performance drift detected. "
            "Schedule model retraining and "
            "compare the challenger model "
            "against the current champion."
        )

    elif (
        drift_score
        >= warning_threshold
    ):

        severity = "WARNING"

        drift_detected = False

        recommendation = (

            "Performance deterioration detected. "
            "Monitor the next evaluation window "
            "before replacing the champion."
        )

    else:

        severity = "STABLE"

        drift_detected = False

        recommendation = (

            "Model performance is stable. "
            "Continue monitoring."
        )

    # --------------------------------------------------------
    # NO DRIFT REASONS
    # --------------------------------------------------------

    if not reasons:

        reasons.append(

            "No metric exceeded its "
            "configured deterioration threshold."
        )

    return DriftResult(

        drift_detected=drift_detected,

        severity=severity,

        score=round(
            float(
                drift_score
            ),
            4,
        ),

        reasons=reasons,

        historical_metrics=(
            historical_metrics
        ),

        recent_metrics=(
            recent_metrics
        ),

        metric_changes=(
            metric_changes
        ),

        recommendation=(
            recommendation
        ),
    )


# ============================================================
# SIMPLE CONVENIENCE FUNCTION
# ============================================================

def analyze_drift(
    ledger: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze drift and return a dictionary.

    Example:

        result = analyze_drift(ledger)

        if result["drift_detected"]:
            print("Retraining recommended")
    """

    result = (
        detect_performance_drift(

            ledger=ledger,

            config=config,
        )
    )

    return result.to_dict()

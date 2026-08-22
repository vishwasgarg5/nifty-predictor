"""Champion / Challenger model comparison.

Compares the current production model (Champion) against a
new candidate model (Challenger) using evaluated predictions.

Metrics:

    Lower is better:
        - Return MAE
        - Brier Score
        - Risk MAE

    Higher is better:
        - Direction Accuracy

The challenger is promoted only when:

    1. Enough evaluated predictions exist.
    2. The challenger performs better overall.
    3. The configured minimum improvement is achieved.
    4. Auto-promotion is enabled, or the result is returned
       for manual approval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_CONFIG = {
    "enabled": True,
    "champion": "current",
    "challenger": "challenger_v1",
    "minimum_evaluations": 30,
    "comparison_window": 50,
    "minimum_improvement": 0.02,
    "evaluate_return_mae": True,
    "evaluate_direction_accuracy": True,
    "evaluate_brier_score": True,
    "evaluate_risk_mae": True,
    "auto_promote": False,
    "require_manual_approval": True,
}


# ============================================================
# RESULT
# ============================================================

@dataclass
class ComparisonResult:
    """Champion / Challenger comparison result."""

    champion_name: str
    challenger_name: str

    champion_records: int
    challenger_records: int

    comparison_window: int

    champion_metrics: dict[str, float]
    challenger_metrics: dict[str, float]

    metric_improvements: dict[str, float]

    champion_score: float
    challenger_score: float

    overall_improvement: float

    winner: str

    promotion_recommended: bool
    auto_promoted: bool

    reasons: list[str]
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""

        return {
            "champion_name": self.champion_name,
            "challenger_name": self.challenger_name,

            "champion_records": self.champion_records,
            "challenger_records": self.challenger_records,

            "comparison_window": self.comparison_window,

            "champion_metrics": self.champion_metrics,
            "challenger_metrics": self.challenger_metrics,

            "metric_improvements": self.metric_improvements,

            "champion_score": self.champion_score,
            "challenger_score": self.challenger_score,

            "overall_improvement": self.overall_improvement,

            "winner": self.winner,

            "promotion_recommended": (
                self.promotion_recommended
            ),

            "auto_promoted": self.auto_promoted,

            "reasons": self.reasons,

            "recommendation": self.recommendation,
        }


# ============================================================
# HELPERS
# ============================================================

def safe_mean(
    series: pd.Series | None,
) -> float | None:
    """Return numeric mean safely."""

    if series is None:
        return None

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    value = float(values.mean())

    if not np.isfinite(value):
        return None

    return value


def get_evaluated_records(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return evaluated records only."""

    if frame is None or frame.empty:
        return pd.DataFrame()

    result = frame.copy()

    if "evaluation_status" in result.columns:

        status = (
            result["evaluation_status"]
            .astype(str)
            .str.upper()
        )

        result = result.loc[
            status == "EVALUATED"
        ].copy()

    # Sort by date when possible.
    date_column = None

    for candidate in (
        "market_date",
        "date",
        "evaluated_at",
        "prediction_date",
        "created_at",
    ):

        if candidate in result.columns:
            date_column = candidate
            break

    if date_column:

        result["_comparison_date"] = (
            pd.to_datetime(
                result[date_column],
                errors="coerce",
            )
        )

        result = result.sort_values(
            "_comparison_date",
            ascending=True,
            na_position="last",
        )

    return result.reset_index(
        drop=True
    )


def calculate_metrics(
    frame: pd.DataFrame,
) -> dict[str, float]:
    """Calculate supported performance metrics."""

    if frame is None or frame.empty:
        return {}

    metrics: dict[str, float] = {}

    if "return_absolute_error" in frame.columns:

        value = safe_mean(
            frame["return_absolute_error"]
        )

        if value is not None:
            metrics["return_mae"] = value

    if "direction_correct" in frame.columns:

        value = safe_mean(
            frame["direction_correct"]
        )

        if value is not None:
            metrics["direction_accuracy"] = value

    if "brier_score" in frame.columns:

        value = safe_mean(
            frame["brier_score"]
        )

        if value is not None:
            metrics["brier_score"] = value

    if "risk_absolute_error" in frame.columns:

        value = safe_mean(
            frame["risk_absolute_error"]
        )

        if value is not None:
            metrics["risk_mae"] = value

    return metrics


# ============================================================
# MODEL FILTERING
# ============================================================

def filter_model_records(
    ledger: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Filter ledger records belonging to a model.

    Supported model identifier columns:

        model_name
        model_version
        model
        model_id
    """

    if ledger is None or ledger.empty:
        return pd.DataFrame()

    frame = get_evaluated_records(
        ledger
    )

    if frame.empty:
        return frame

    model_column = None

    for candidate in (
        "model_name",
        "model_version",
        "model",
        "model_id",
    ):

        if candidate in frame.columns:
            model_column = candidate
            break

    if model_column is None:
        return pd.DataFrame()

    values = (
        frame[model_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    target = str(
        model_name
    ).strip().lower()

    return frame.loc[
        values == target
    ].copy()


# ============================================================
# METRIC IMPROVEMENT
# ============================================================

def calculate_improvement(
    champion_value: float,
    challenger_value: float,
    higher_is_better: bool,
) -> float | None:
    """Calculate normalized challenger improvement.

    Positive:
        Challenger improved.

    Negative:
        Challenger deteriorated.

    Example for lower-is-better metric:

        Champion MAE = 0.10
        Challenger MAE = 0.08

        Improvement = +0.20
    """

    if champion_value is None:
        return None

    if challenger_value is None:
        return None

    if not np.isfinite(champion_value):
        return None

    if not np.isfinite(challenger_value):
        return None

    denominator = abs(
        champion_value
    )

    if denominator < 1e-12:

        if higher_is_better:
            return challenger_value - champion_value

        return champion_value - challenger_value

    if higher_is_better:

        return (
            challenger_value
            - champion_value
        ) / denominator

    return (
        champion_value
        - challenger_value
    ) / denominator


# ============================================================
# METRIC CONFIGURATION
# ============================================================

def get_active_metrics(
    settings: dict[str, Any],
) -> list[tuple[str, bool]]:
    """Return enabled metrics.

    Tuple format:

        (
            metric_name,
            higher_is_better,
        )
    """

    metrics: list[
        tuple[str, bool]
    ] = []

    if settings.get(
        "evaluate_return_mae",
        True,
    ):

        metrics.append(
            (
                "return_mae",
                False,
            )
        )

    if settings.get(
        "evaluate_direction_accuracy",
        True,
    ):

        metrics.append(
            (
                "direction_accuracy",
                True,
            )
        )

    if settings.get(
        "evaluate_brier_score",
        True,
    ):

        metrics.append(
            (
                "brier_score",
                False,
            )
        )

    if settings.get(
        "evaluate_risk_mae",
        True,
    ):

        metrics.append(
            (
                "risk_mae",
                False,
            )
        )

    return metrics


# ============================================================
# COMPARISON
# ============================================================

def compare_models(
    ledger: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> ComparisonResult:
    """Compare Champion and Challenger performance."""

    settings = DEFAULT_CONFIG.copy()

    if config:
        settings.update(config)

    champion_name = str(
        settings["champion"]
    )

    challenger_name = str(
        settings["challenger"]
    )

    minimum_evaluations = int(
        settings["minimum_evaluations"]
    )

    comparison_window = int(
        settings["comparison_window"]
    )

    minimum_improvement = float(
        settings["minimum_improvement"]
    )

    # --------------------------------------------------------
    # FILTER MODEL RECORDS
    # --------------------------------------------------------

    champion_records = (
        filter_model_records(
            ledger=ledger,
            model_name=champion_name,
        )
    )

    challenger_records = (
        filter_model_records(
            ledger=ledger,
            model_name=challenger_name,
        )
    )

    # --------------------------------------------------------
    # LIMIT COMPARISON WINDOW
    # --------------------------------------------------------

    if len(
        champion_records
    ) > comparison_window:

        champion_records = (
            champion_records.iloc[
                -comparison_window:
            ].copy()
        )

    if len(
        challenger_records
    ) > comparison_window:

        challenger_records = (
            challenger_records.iloc[
                -comparison_window:
            ].copy()
        )

    champion_count = len(
        champion_records
    )

    challenger_count = len(
        challenger_records
    )

    # --------------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------------

    if (
        champion_count
        < minimum_evaluations
        or challenger_count
        < minimum_evaluations
    ):

        reasons = []

        if (
            champion_count
            < minimum_evaluations
        ):

            reasons.append(
                "Champion has insufficient "
                f"evaluations: {champion_count} "
                f"available, "
                f"{minimum_evaluations} required."
            )

        if (
            challenger_count
            < minimum_evaluations
        ):

            reasons.append(
                "Challenger has insufficient "
                f"evaluations: {challenger_count} "
                f"available, "
                f"{minimum_evaluations} required."
            )

        return ComparisonResult(

            champion_name=champion_name,

            challenger_name=challenger_name,

            champion_records=champion_count,

            challenger_records=challenger_count,

            comparison_window=comparison_window,

            champion_metrics={},

            challenger_metrics={},

            metric_improvements={},

            champion_score=0.0,

            challenger_score=0.0,

            overall_improvement=0.0,

            winner="INSUFFICIENT_DATA",

            promotion_recommended=False,

            auto_promoted=False,

            reasons=reasons,

            recommendation=(
                "Continue collecting evaluated "
                "predictions before comparing models."
            ),
        )

    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    champion_metrics = (
        calculate_metrics(
            champion_records
        )
    )

    challenger_metrics = (
        calculate_metrics(
            challenger_records
        )
    )

    active_metrics = (
        get_active_metrics(
            settings
        )
    )

    metric_improvements: dict[
        str,
        float
    ] = {}

    valid_improvements: list[
        float
    ] = []

    reasons: list[
        str
    ] = []

    champion_score_components: list[
        float
    ] = []

    challenger_score_components: list[
        float
    ] = []

    # --------------------------------------------------------
    # COMPARE EACH METRIC
    # --------------------------------------------------------

    for (
        metric,
        higher_is_better,
    ) in active_metrics:

        champion_value = (
            champion_metrics.get(
                metric
            )
        )

        challenger_value = (
            challenger_metrics.get(
                metric
            )
        )

        if (
            champion_value is None
            or challenger_value is None
        ):

            reasons.append(
                f"Metric unavailable: {metric}."
            )

            continue

        improvement = (
            calculate_improvement(

                champion_value,

                challenger_value,

                higher_is_better,
            )
        )

        if improvement is None:
            continue

        metric_improvements[
            metric
        ] = float(
            improvement
        )

        valid_improvements.append(
            float(
                improvement
            )
        )

        if higher_is_better:

            champion_score_components.append(
                champion_value
            )

            challenger_score_components.append(
                challenger_value
            )

        else:

            # Invert error metrics so
            # lower error produces a
            # larger internal score.

            champion_score_components.append(
                1.0
                / (
                    1.0
                    + abs(champion_value)
                )
            )

            challenger_score_components.append(
                1.0
                / (
                    1.0
                    + abs(challenger_value)
                )
            )

        if improvement > 0:

            reasons.append(
                f"Challenger improved "
                f"{metric} by "
                f"{improvement:.2%}."
            )

        elif improvement < 0:

            reasons.append(
                f"Challenger deteriorated "
                f"{metric} by "
                f"{abs(improvement):.2%}."
            )

        else:

            reasons.append(
                f"No change in {metric}."
            )

    # --------------------------------------------------------
    # OVERALL SCORES
    # --------------------------------------------------------

    if champion_score_components:

        champion_score = float(
            np.mean(
                champion_score_components
            )
        )

    else:

        champion_score = 0.0

    if challenger_score_components:

        challenger_score = float(
            np.mean(
                challenger_score_components
            )
        )

    else:

        challenger_score = 0.0

    if valid_improvements:

        overall_improvement = float(
            np.mean(
                valid_improvements
            )
        )

    else:

        overall_improvement = 0.0

    # --------------------------------------------------------
    # DECIDE WINNER
    # --------------------------------------------------------

    if not valid_improvements:

        winner = "NO_COMPARISON"

        promotion_recommended = False

        recommendation = (
            "No compatible metrics were available "
            "for model comparison."
        )

    elif (
        overall_improvement
        >= minimum_improvement
    ):

        winner = "CHALLENGER"

        promotion_recommended = True

        recommendation = (
            "Challenger outperformed the champion "
            f"by {overall_improvement:.2%}. "
            "Promotion is recommended."
        )

    elif overall_improvement > 0:

        winner = "CHALLENGER_SLIGHTLY"

        promotion_recommended = False

        recommendation = (
            "Challenger performed slightly better, "
            "but did not meet the configured "
            "minimum improvement threshold."
        )

    elif overall_improvement < 0:

        winner = "CHAMPION"

        promotion_recommended = False

        recommendation = (
            "Champion remains superior. "
            "Do not promote the challenger."
        )

    else:

        winner = "TIE"

        promotion_recommended = False

        recommendation = (
            "Both models performed similarly. "
            "Continue collecting evaluations."
        )

    # --------------------------------------------------------
    # AUTO PROMOTION
    # --------------------------------------------------------

    auto_promoted = False

    if (
        promotion_recommended
        and bool(
            settings.get(
                "auto_promote",
                False,
            )
        )
        and not bool(
            settings.get(
                "require_manual_approval",
                True,
            )
        )
    ):

        auto_promoted = True

        recommendation = (
            "Challenger meets the promotion "
            "criteria and is eligible for "
            "automatic promotion."
        )

    return ComparisonResult(

        champion_name=champion_name,

        challenger_name=challenger_name,

        champion_records=champion_count,

        challenger_records=challenger_count,

        comparison_window=comparison_window,

        champion_metrics=champion_metrics,

        challenger_metrics=challenger_metrics,

        metric_improvements=metric_improvements,

        champion_score=round(
            champion_score,
            6,
        ),

        challenger_score=round(
            challenger_score,
            6,
        ),

        overall_improvement=round(
            overall_improvement,
            6,
        ),

        winner=winner,

        promotion_recommended=(
            promotion_recommended
        ),

        auto_promoted=auto_promoted,

        reasons=reasons,

        recommendation=recommendation,
    )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def analyze_champion_challenger(
    ledger: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare models and return a dictionary."""

    result = compare_models(
        ledger=ledger,
        config=config,
    )

    return result.to_dict()

#!/usr/bin/env python3

"""
Champion / Challenger Evaluation and Promotion Engine.

Responsibilities
----------------
1. Load evaluated predictions from the prediction ledger.
2. Identify the current Champion and Challenger models.
3. Compare models on common evaluated predictions.
4. Calculate:
   - Return MAE
   - Direction Accuracy
   - Brier Score
   - Risk MAE
5. Apply a promotion gate.
6. Promote a Challenger only when it has enough evidence and
   outperforms the Champion.
7. Update data/model_registry.json.

This module does not train models or generate predictions.
"""

from __future__ import annotations

import json
import logging
import sys

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

DEFAULT_CONFIG = {
    "enabled": True,
    "champion": "current",
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


def load_champion_challenger_config() -> dict[str, Any]:
    """Load champion/challenger settings with safe defaults."""

    config = DEFAULT_CONFIG.copy()

    try:
        from src.config import cfg

        section = getattr(cfg, "champion_challenger", None)

        if section is None:
            return config

        if isinstance(section, dict):
            config.update(section)
            return config

        if hasattr(section, "items"):
            config.update(dict(section.items()))
            return config

        if hasattr(section, "__dict__"):
            config.update(
                {
                    key: value
                    for key, value in vars(section).items()
                    if not key.startswith("_")
                }
            )

    except Exception as error:
        logger.debug(
            "Could not load champion/challenger config: %s",
            error,
        )

    return config


# ============================================================
# PATHS
# ============================================================

def resolve_project_path(value: str | Path) -> Path:
    """Resolve paths relative to the repository root."""

    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ModelMetrics:
    model_name: str
    sample_count: int
    return_mae: float | None
    direction_accuracy: float | None
    brier_score: float | None
    risk_mae: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromotionDecision:
    challenger_name: str
    champion_name: str
    eligible: bool
    better_than_champion: bool
    improvement_score: float
    minimum_improvement: float
    sample_count: int
    reasons: list[str]
    promoted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# REGISTRY
# ============================================================

def empty_registry() -> dict[str, Any]:
    return {
        "champion": {
            "model_name": "current",
            "model_path": None,
            "status": "CHAMPION",
        },
        "challengers": [],
        "history": [],
    }


def load_model_registry(
    registry_path: str | Path = "data/model_registry.json",
) -> dict[str, Any]:
    """Load the model registry."""

    path = resolve_project_path(registry_path)

    if not path.exists():
        return empty_registry()

    try:
        with open(path, "r", encoding="utf-8") as file:
            registry = json.load(file)

        if not isinstance(registry, dict):
            return empty_registry()

        registry.setdefault("champion", None)
        registry.setdefault("challengers", [])
        registry.setdefault("history", [])

        return registry

    except Exception as error:
        logger.error(
            "Unable to load model registry: %s",
            error,
        )

        return empty_registry()


def save_model_registry(
    registry: dict[str, Any],
    registry_path: str | Path = "data/model_registry.json",
) -> Path:
    """Atomically save the model registry."""

    path = resolve_project_path(registry_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(".tmp")

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            registry,
            file,
            indent=2,
            default=str,
        )

    temporary_path.replace(path)

    return path


# ============================================================
# LEDGER
# ============================================================

def load_evaluated_predictions(
    ledger_path: str | Path = "data/ledger/predictions.csv",
) -> pd.DataFrame:
    """
    Load only predictions that have actual outcomes available.
    """

    path = resolve_project_path(ledger_path)

    if not path.exists():
        logger.warning("Prediction ledger not found: %s", path)
        return pd.DataFrame()

    try:
        frame = pd.read_csv(path)
    except Exception as error:
        logger.error("Unable to read prediction ledger: %s", error)
        return pd.DataFrame()

    if frame.empty:
        return frame

    if "model_name" not in frame.columns:
        frame["model_name"] = "current"

    if "evaluation_status" in frame.columns:
        evaluated = frame[
            frame["evaluation_status"]
            .astype(str)
            .str.upper()
            .eq("EVALUATED")
        ].copy()
    else:
        evaluated = frame.copy()

    actual_columns = [
        "actual_return",
        "actual_direction",
        "actual_close",
    ]

    available_actual_columns = [
        column
        for column in actual_columns
        if column in evaluated.columns
    ]

    if not available_actual_columns:
        logger.warning(
            "Ledger contains no recognized actual outcome columns."
        )
        return pd.DataFrame()

    return evaluated.reset_index(drop=True)


# ============================================================
# COLUMN HELPERS
# ============================================================

def first_existing_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def numeric_series(
    frame: pd.DataFrame,
    column: str | None,
) -> pd.Series:
    if column is None:
        return pd.Series(np.nan, index=frame.index)

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


# ============================================================
# METRICS
# ============================================================

def calculate_model_metrics(
    predictions: pd.DataFrame,
    model_name: str,
) -> ModelMetrics:
    """Calculate evaluation metrics for one model."""

    frame = predictions[
        predictions["model_name"]
        .astype(str)
        .eq(str(model_name))
    ].copy()

    sample_count = len(frame)

    # Return MAE
    predicted_return_column = first_existing_column(
        frame,
        [
            "predicted_return",
            "expected_return",
        ],
    )

    actual_return_column = first_existing_column(
        frame,
        [
            "actual_return",
            "realized_return",
        ],
    )

    predicted_return = numeric_series(
        frame,
        predicted_return_column,
    )

    actual_return = numeric_series(
        frame,
        actual_return_column,
    )

    valid_return = (
        predicted_return.notna()
        & actual_return.notna()
    )

    return_mae = None

    if valid_return.any():
        return_mae = float(
            np.mean(
                np.abs(
                    predicted_return[valid_return]
                    - actual_return[valid_return]
                )
            )
        )

    # Direction Accuracy
    predicted_direction_column = first_existing_column(
        frame,
        [
            "predicted_direction",
            "direction",
        ],
    )

    actual_direction_column = first_existing_column(
        frame,
        [
            "actual_direction",
        ],
    )

    direction_accuracy = None

    if (
        predicted_direction_column is not None
        and actual_direction_column is not None
    ):
        predicted_direction = (
            frame[predicted_direction_column]
            .astype(str)
            .str.upper()
        )

        actual_direction = (
            frame[actual_direction_column]
            .astype(str)
            .str.upper()
        )

        direction_map = {
            "1": "UP",
            "1.0": "UP",
            "TRUE": "UP",
            "-1": "DOWN",
            "-1.0": "DOWN",
            "FALSE": "DOWN",
        }

        predicted_direction = predicted_direction.replace(
            direction_map
        )

        actual_direction = actual_direction.replace(
            direction_map
        )

        valid_direction = (
            predicted_direction.isin(["UP", "DOWN"])
            & actual_direction.isin(["UP", "DOWN"])
        )

        if valid_direction.any():
            direction_accuracy = float(
                (
                    predicted_direction[valid_direction]
                    == actual_direction[valid_direction]
                ).mean()
            )

    # Brier Score
    probability_column = first_existing_column(
        frame,
        [
            "direction_probability",
            "probability_up",
            "confidence",
        ],
    )

    brier_score = None

    if (
        probability_column is not None
        and actual_direction_column is not None
    ):
        probability = numeric_series(
            frame,
            probability_column,
        )

        actual_direction = (
            frame[actual_direction_column]
            .astype(str)
            .str.upper()
        )

        actual_binary = actual_direction.map(
            {
                "UP": 1.0,
                "DOWN": 0.0,
                "1": 1.0,
                "1.0": 1.0,
                "-1": 0.0,
                "-1.0": 0.0,
            }
        )

        valid_probability = (
            probability.notna()
            & actual_binary.notna()
        )

        if valid_probability.any():
            probability = probability.clip(0.0, 1.0)

            brier_score = float(
                np.mean(
                    (
                        probability[valid_probability]
                        - actual_binary[valid_probability]
                    ) ** 2
                )
            )

    # Risk MAE
    predicted_risk_column = first_existing_column(
        frame,
        [
            "predicted_risk",
            "expected_risk",
        ],
    )

    actual_risk_column = first_existing_column(
        frame,
        [
            "actual_risk",
            "realized_risk",
        ],
    )

    predicted_risk = numeric_series(
        frame,
        predicted_risk_column,
    )

    actual_risk = numeric_series(
        frame,
        actual_risk_column,
    )

    valid_risk = (
        predicted_risk.notna()
        & actual_risk.notna()
    )

    risk_mae = None

    if valid_risk.any():
        risk_mae = float(
            np.mean(
                np.abs(
                    predicted_risk[valid_risk]
                    - actual_risk[valid_risk]
                )
            )
        )

    return ModelMetrics(
        model_name=str(model_name),
        sample_count=sample_count,
        return_mae=return_mae,
        direction_accuracy=direction_accuracy,
        brier_score=brier_score,
        risk_mae=risk_mae,
    )


# ============================================================
# COMMON COMPARISON DATA
# ============================================================

def get_comparable_predictions(
    predictions: pd.DataFrame,
    champion_name: str,
    challenger_name: str,
    comparison_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Restrict comparison to the most recent evaluated predictions.

    If symbol/date keys exist for both models, only common prediction
    opportunities are compared.
    """

    champion = predictions[
        predictions["model_name"]
        .astype(str)
        .eq(str(champion_name))
    ].copy()

    challenger = predictions[
        predictions["model_name"]
        .astype(str)
        .eq(str(challenger_name))
    ].copy()

    if champion.empty or challenger.empty:
        return champion, challenger

    date_column = first_existing_column(
        predictions,
        [
            "prediction_date",
            "market_date",
            "date",
        ],
    )

    symbol_column = first_existing_column(
        predictions,
        [
            "symbol",
            "ticker",
        ],
    )

    key_columns = [
        column
        for column in [date_column, symbol_column]
        if column is not None
    ]

    if key_columns:
        champion_keys = champion[key_columns].drop_duplicates()
        challenger_keys = challenger[key_columns].drop_duplicates()

        common_keys = champion_keys.merge(
            challenger_keys,
            on=key_columns,
            how="inner",
        )

        champion = champion.merge(
            common_keys,
            on=key_columns,
            how="inner",
        )

        challenger = challenger.merge(
            common_keys,
            on=key_columns,
            how="inner",
        )

    if date_column is not None:
        champion = champion.sort_values(
            date_column
        ).tail(comparison_window)

        challenger = challenger.sort_values(
            date_column
        ).tail(comparison_window)

    else:
        champion = champion.tail(comparison_window)
        challenger = challenger.tail(comparison_window)

    return champion.reset_index(drop=True), challenger.reset_index(drop=True)


# ============================================================
# IMPROVEMENT SCORING
# ============================================================

def calculate_improvement_score(
    champion: ModelMetrics,
    challenger: ModelMetrics,
    config: dict[str, Any],
) -> tuple[float, list[str]]:
    """
    Calculate average normalized Challenger improvement.

    Positive score = Challenger better.
    Negative score = Champion better.
    """

    improvements: list[float] = []
    reasons: list[str] = []

    if (
        config.get("evaluate_return_mae", True)
        and champion.return_mae is not None
        and challenger.return_mae is not None
    ):
        baseline = max(abs(champion.return_mae), 1e-12)

        improvement = (
            champion.return_mae
            - challenger.return_mae
        ) / baseline

        improvements.append(improvement)

        reasons.append(
            "Return MAE improvement: "
            f"{improvement:.2%}"
        )

    if (
        config.get("evaluate_direction_accuracy", True)
        and champion.direction_accuracy is not None
        and challenger.direction_accuracy is not None
    ):
        baseline = max(
            abs(champion.direction_accuracy),
            1e-12,
        )

        improvement = (
            challenger.direction_accuracy
            - champion.direction_accuracy
        ) / baseline

        improvements.append(improvement)

        reasons.append(
            "Direction accuracy improvement: "
            f"{improvement:.2%}"
        )

    if (
        config.get("evaluate_brier_score", True)
        and champion.brier_score is not None
        and challenger.brier_score is not None
    ):
        baseline = max(
            abs(champion.brier_score),
            1e-12,
        )

        improvement = (
            champion.brier_score
            - challenger.brier_score
        ) / baseline

        improvements.append(improvement)

        reasons.append(
            "Brier score improvement: "
            f"{improvement:.2%}"
        )

    if (
        config.get("evaluate_risk_mae", True)
        and champion.risk_mae is not None
        and challenger.risk_mae is not None
    ):
        baseline = max(
            abs(champion.risk_mae),
            1e-12,
        )

        improvement = (
            champion.risk_mae
            - challenger.risk_mae
        ) / baseline

        improvements.append(improvement)

        reasons.append(
            "Risk MAE improvement: "
            f"{improvement:.2%}"
        )

    if not improvements:
        return (
            0.0,
            ["No comparable metrics available."],
        )

    score = float(np.mean(improvements))

    return score, reasons


# ============================================================
# PROMOTION DECISION
# ============================================================

def evaluate_challenger(
    predictions: pd.DataFrame,
    champion_name: str,
    challenger_name: str,
    config: dict[str, Any] | None = None,
) -> tuple[
    ModelMetrics,
    ModelMetrics,
    PromotionDecision,
]:
    """Evaluate one Challenger against the Champion."""

    settings = load_champion_challenger_config()

    if config:
        settings.update(config)

    comparison_window = int(
        settings.get("comparison_window", 50)
    )

    minimum_evaluations = int(
        settings.get("minimum_evaluations", 30)
    )

    minimum_improvement = float(
        settings.get("minimum_improvement", 0.02)
    )

    champion_frame, challenger_frame = (
        get_comparable_predictions(
            predictions=predictions,
            champion_name=champion_name,
            challenger_name=challenger_name,
            comparison_window=comparison_window,
        )
    )

    champion_metrics = calculate_model_metrics(
        champion_frame,
        champion_name,
    )

    challenger_metrics = calculate_model_metrics(
        challenger_frame,
        challenger_name,
    )

    sample_count = min(
        champion_metrics.sample_count,
        challenger_metrics.sample_count,
    )

    improvement_score, reasons = (
        calculate_improvement_score(
            champion=champion_metrics,
            challenger=challenger_metrics,
            config=settings,
        )
    )

    eligible = sample_count >= minimum_evaluations

    if not eligible:
        reasons.append(
            "Insufficient evaluations: "
            f"{sample_count}/{minimum_evaluations}"
        )

    better_than_champion = (
        improvement_score >= minimum_improvement
    )

    if better_than_champion:
        reasons.append(
            "Challenger exceeds minimum "
            f"improvement of {minimum_improvement:.2%}."
        )
    else:
        reasons.append(
            "Challenger does not exceed minimum "
            f"improvement of {minimum_improvement:.2%}."
        )

    decision = PromotionDecision(
        challenger_name=str(challenger_name),
        champion_name=str(champion_name),
        eligible=eligible,
        better_than_champion=better_than_champion,
        improvement_score=improvement_score,
        minimum_improvement=minimum_improvement,
        sample_count=sample_count,
        reasons=reasons,
        promoted=False,
    )

    return (
        champion_metrics,
        challenger_metrics,
        decision,
    )


# ============================================================
# PROMOTION
# ============================================================

def promote_challenger(
    challenger_name: str,
    registry_path: str | Path = "data/model_registry.json",
    decision: PromotionDecision | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Promote a Challenger to Champion.

    Unless force=True, a positive eligible promotion decision
    is required.
    """

    if not force:
        if decision is None:
            raise ValueError(
                "Promotion decision is required."
            )

        if not decision.eligible:
            raise ValueError(
                "Challenger is not eligible for promotion."
            )

        if not decision.better_than_champion:
            raise ValueError(
                "Challenger did not outperform Champion."
            )

    registry = load_model_registry(registry_path)

    current_champion = registry.get("champion")

    challenger_index = None
    challenger_entry = None

    for index, entry in enumerate(
        registry.get("challengers", [])
    ):
        if (
            str(entry.get("model_name"))
            == str(challenger_name)
        ):
            challenger_index = index
            challenger_entry = entry
            break

    if challenger_entry is None:
        raise ValueError(
            f"Challenger not found: {challenger_name}"
        )

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    new_champion = dict(challenger_entry)

    new_champion["status"] = "CHAMPION"
    new_champion["promoted_at"] = timestamp

    registry["champion"] = new_champion

    registry["challengers"].pop(
        challenger_index
    )

    if current_champion:
        previous = dict(current_champion)

        previous["status"] = "RETIRED"
        previous["retired_at"] = timestamp

        registry.setdefault(
            "history",
            [],
        ).append(previous)

    registry.setdefault(
        "history",
        [],
    ).append(
        {
            "event": "PROMOTION",
            "timestamp": timestamp,
            "from": (
                current_champion.get("model_name")
                if isinstance(
                    current_champion,
                    dict,
                )
                else current_champion
            ),
            "to": challenger_name,
            "decision": (
                decision.to_dict()
                if decision is not None
                else None
            ),
        }
    )

    save_model_registry(
        registry,
        registry_path,
    )

    logger.warning(
        "CHALLENGER PROMOTED: %s -> CHAMPION",
        challenger_name,
    )

    return registry


# ============================================================
# FULL COMPARISON
# ============================================================

def compare_all_challengers(
    ledger_path: str | Path = "data/ledger/predictions.csv",
    registry_path: str | Path = "data/model_registry.json",
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Evaluate every active Challenger against the current Champion.
    """

    settings = load_champion_challenger_config()

    if config:
        settings.update(config)

    if not bool(settings.get("enabled", True)):
        logger.info(
            "Champion/Challenger evaluation disabled."
        )
        return []

    predictions = load_evaluated_predictions(
        ledger_path
    )

    if predictions.empty:
        logger.info(
            "No evaluated predictions available."
        )
        return []

    registry = load_model_registry(
        registry_path
    )

    champion_entry = registry.get("champion")

    if isinstance(champion_entry, dict):
        champion_name = str(
            champion_entry.get(
                "model_name",
                settings.get("champion", "current"),
            )
        )
    else:
        champion_name = str(
            champion_entry
            or settings.get("champion", "current")
        )

    results: list[dict[str, Any]] = []

    for challenger in registry.get(
        "challengers",
        [],
    ):
        status = str(
            challenger.get(
                "status",
                "CHALLENGER",
            )
        ).upper()

        if status not in {
            "CHALLENGER",
            "ACTIVE",
            "EVALUATING",
        }:
            continue

        challenger_name = challenger.get(
            "model_name"
        )

        if not challenger_name:
            continue

        champion_metrics, challenger_metrics, decision = (
            evaluate_challenger(
                predictions=predictions,
                champion_name=champion_name,
                challenger_name=str(challenger_name),
                config=settings,
            )
        )

        result = {
            "champion_metrics": (
                champion_metrics.to_dict()
            ),
            "challenger_metrics": (
                challenger_metrics.to_dict()
            ),
            "decision": (
                decision.to_dict()
            ),
        }

        results.append(result)

        logger.info(
            "Comparison | Champion=%s Challenger=%s "
            "Eligible=%s Better=%s Improvement=%.2f%%",
            champion_name,
            challenger_name,
            decision.eligible,
            decision.better_than_champion,
            decision.improvement_score * 100,
        )

    return results


# ============================================================
# AUTO PROMOTION
# ============================================================

def evaluate_and_maybe_promote(
    ledger_path: str | Path = "data/ledger/predictions.csv",
    registry_path: str | Path = "data/model_registry.json",
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Compare Challengers and optionally promote them.

    Promotion obeys:
        auto_promote
        require_manual_approval
    """

    settings = load_champion_challenger_config()

    if config:
        settings.update(config)

    results = compare_all_challengers(
        ledger_path=ledger_path,
        registry_path=registry_path,
        config=settings,
    )

    auto_promote = bool(
        settings.get("auto_promote", False)
    )

    require_manual_approval = bool(
        settings.get(
            "require_manual_approval",
            True,
        )
    )

    if (
        not auto_promote
        or require_manual_approval
    ):
        return results

    for result in results:
        decision_data = result["decision"]

        if not (
            decision_data["eligible"]
            and decision_data["better_than_champion"]
        ):
            continue

        decision = PromotionDecision(
            **decision_data
        )

        promote_challenger(
            challenger_name=decision.challenger_name,
            registry_path=registry_path,
            decision=decision,
        )

        decision.promoted = True
        result["decision"] = decision.to_dict()

    return results


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    results = evaluate_and_maybe_promote()

    print(
        json.dumps(
            results,
            indent=2,
            default=str,
        )
    )

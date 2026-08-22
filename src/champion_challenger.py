#!/usr/bin/env python3

"""
Champion / Challenger Evaluation.

Compares the current production Champion model against
a Challenger model.

Promotion only happens when the Challenger demonstrates
sufficient improvement according to configured thresholds.

Metrics considered:

    - direction_accuracy
    - mae
    - rmse
    - average_return_error

Higher is better:

    direction_accuracy

Lower is better:

    mae
    rmse
    average_return_error
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "champion_challenger"
)


def utc_now_iso() -> str:

    return datetime.now(
        timezone.utc
    ).isoformat()


def object_to_dict(
    value: Any,
) -> dict[str, Any]:

    if value is None:
        return {}

    if isinstance(value, dict):
        return dict(value)

    if hasattr(value, "items"):

        try:
            return dict(value.items())
        except Exception:
            pass

    if hasattr(value, "__dict__"):

        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    return {}


def load_config() -> Any:

    try:

        from src.config import cfg

        return cfg

    except Exception:

        return None


def get_config() -> dict[str, Any]:

    defaults = {
        "enabled": True,
        "minimum_accuracy_improvement": 0.01,
        "minimum_error_improvement": 0.01,
        "required_wins": 2,
    }

    cfg = load_config()

    if cfg is None:
        return defaults

    section = getattr(
        cfg,
        "champion_challenger",
        None,
    )

    values = object_to_dict(
        section
    )

    result = defaults.copy()

    for key in defaults:

        if key in values:

            result[key] = values[key]

    try:

        result[
            "minimum_accuracy_improvement"
        ] = float(
            result[
                "minimum_accuracy_improvement"
            ]
        )

    except Exception:

        result[
            "minimum_accuracy_improvement"
        ] = 0.01

    try:

        result[
            "minimum_error_improvement"
        ] = float(
            result[
                "minimum_error_improvement"
            ]
        )

    except Exception:

        result[
            "minimum_error_improvement"
        ] = 0.01

    try:

        result[
            "required_wins"
        ] = max(
            1,
            int(
                result[
                    "required_wins"
                ]
            ),
        )

    except Exception:

        result[
            "required_wins"
        ] = 2

    return result


def safe_float(
    value: Any,
) -> float | None:

    try:

        if value is None:
            return None

        return float(value)

    except Exception:

        return None


def compare_metrics(
    champion_metrics: dict[str, Any],
    challenger_metrics: dict[str, Any],
) -> dict[str, Any]:

    config = get_config()

    results: dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "comparisons": {},
        "challenger_wins": 0,
        "champion_wins": 0,
        "draws": 0,
    }

    # --------------------------------------------------------
    # DIRECTION ACCURACY
    # Higher is better.
    # --------------------------------------------------------

    champion_accuracy = safe_float(
        champion_metrics.get(
            "direction_accuracy"
        )
    )

    challenger_accuracy = safe_float(
        challenger_metrics.get(
            "direction_accuracy"
        )
    )

    if (
        champion_accuracy is not None
        and challenger_accuracy is not None
    ):

        improvement = (
            challenger_accuracy
            - champion_accuracy
        )

        winner = "DRAW"

        if (
            improvement
            >= config[
                "minimum_accuracy_improvement"
            ]
        ):

            winner = "CHALLENGER"

            results[
                "challenger_wins"
            ] += 1

        elif improvement < 0:

            winner = "CHAMPION"

            results[
                "champion_wins"
            ] += 1

        else:

            results[
                "draws"
            ] += 1

        results[
            "comparisons"
        ]["direction_accuracy"] = {
            "champion": champion_accuracy,
            "challenger": challenger_accuracy,
            "improvement": improvement,
            "winner": winner,
        }

    # --------------------------------------------------------
    # ERROR METRICS
    # Lower is better.
    # --------------------------------------------------------

    for metric in [
        "mae",
        "rmse",
        "average_return_error",
    ]:

        champion_value = safe_float(
            champion_metrics.get(
                metric
            )
        )

        challenger_value = safe_float(
            challenger_metrics.get(
                metric
            )
        )

        if (
            champion_value is None
            or challenger_value is None
        ):
            continue

        improvement = (
            champion_value
            - challenger_value
        )

        winner = "DRAW"

        if (
            improvement
            >= config[
                "minimum_error_improvement"
            ]
        ):

            winner = "CHALLENGER"

            results[
                "challenger_wins"
            ] += 1

        elif improvement < 0:

            winner = "CHAMPION"

            results[
                "champion_wins"
            ] += 1

        else:

            results[
                "draws"
            ] += 1

        results[
            "comparisons"
        ][metric] = {
            "champion": champion_value,
            "challenger": challenger_value,
            "improvement": improvement,
            "winner": winner,
        }

    return results


def evaluate_promotion() -> dict[str, Any]:
    """
    Evaluate Challenger against Champion.
    """

    from src.model_registry import (
        get_champion,
        get_challenger,
        promote_challenger,
    )

    config = get_config()

    result: dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "status": "UNKNOWN",
        "promoted": False,
        "reason": None,
    }

    if not config.get(
        "enabled",
        True,
    ):

        result["status"] = "DISABLED"

        result["reason"] = (
            "Champion/challenger system disabled."
        )

        return result

    champion = get_champion()

    challenger = get_challenger()

    if champion is None:

        result["status"] = (
            "NO_CHAMPION"
        )

        result["reason"] = (
            "No Champion model registered."
        )

        return result

    if challenger is None:

        result["status"] = (
            "NO_CHALLENGER"
        )

        result["reason"] = (
            "No Challenger model registered."
        )

        return result

    champion_metrics = (
        champion.get(
            "metrics",
            {},
        )
    )

    challenger_metrics = (
        challenger.get(
            "metrics",
            {},
        )
    )

    comparison = compare_metrics(
        champion_metrics,
        challenger_metrics,
    )

    result["comparison"] = (
        comparison
    )

    challenger_wins = comparison.get(
        "challenger_wins",
        0,
    )

    champion_wins = comparison.get(
        "champion_wins",
        0,
    )

    required_wins = config[
        "required_wins"
    ]

    if (
        challenger_wins
        >= required_wins
        and challenger_wins
        > champion_wins
    ):

        reason = (
            "Challenger outperformed Champion. "
            f"Wins={challenger_wins}, "
            f"Champion wins={champion_wins}."
        )

        promoted = promote_challenger(
            reason=reason
        )

        result["status"] = (
            "PROMOTED"
        )

        result["promoted"] = True

        result["reason"] = reason

        result["new_champion"] = (
            promoted
        )

        logger.warning(
            "%s",
            reason,
        )

    else:

        result["status"] = (
            "CHAMPION_RETAINED"
        )

        result["reason"] = (
            "Challenger did not meet "
            "promotion criteria. "
            f"Challenger wins={challenger_wins}, "
            f"Champion wins={champion_wins}, "
            f"Required wins={required_wins}."
        )

        logger.info(
            "%s",
            result["reason"],
        )

    return result


def main() -> int:

    result = evaluate_promotion()

    print()

    print("=" * 70)

    print("CHAMPION / CHALLENGER EVALUATION")

    print("=" * 70)

    print(
        f"Status: "
        f"{result.get('status')}"
    )

    print(
        f"Promoted: "
        f"{result.get('promoted')}"
    )

    print(
        f"Reason: "
        f"{result.get('reason')}"
    )

    comparison = result.get(
        "comparison"
    )

    if comparison:

        print()

        print(
            f"Challenger wins: "
            f"{comparison.get('challenger_wins')}"
        )

        print(
            f"Champion wins: "
            f"{comparison.get('champion_wins')}"
        )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

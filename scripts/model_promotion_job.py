#!/usr/bin/env python3

"""
Model Promotion Job.

This job integrates:

1. Shadow prediction evaluation
2. Champion / Challenger comparison
3. Promotion eligibility checking
4. Safe automatic model promotion

The promotion engine itself remains responsible for
backups, registry updates, history, and rollback safety.

This script is designed to be executed by:

    GitHub Actions
    Cron
    Manual CLI execution
"""

from __future__ import annotations

import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LOGGING
# ============================================================

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
    "model_promotion_job"
)


# ============================================================
# TIME
# ============================================================

def utc_now_iso() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# MAIN JOB
# ============================================================

def run_model_promotion_job(
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Run the complete model promotion workflow.

    Steps
    -----
    1. Load shadow predictions.
    2. Compare Champion and Challenger.
    3. Check promotion eligibility.
    4. Run promotion engine.
    """

    result: dict[str, Any] = {
        "started_at": utc_now_iso(),
        "finished_at": None,
        "status": "STARTED",
        "dry_run": dry_run,
        "promoted": False,
        "evaluation": {},
        "promotion": {},
        "error": None,
    }

    try:

        logger.info(
            "=" * 70
        )

        logger.info(
            "STARTING MODEL PROMOTION JOB"
        )

        logger.info(
            "=" * 70
        )

        # ----------------------------------------------------
        # STEP 1: LOAD SHADOW DATA
        # ----------------------------------------------------

        from src.model_promotion import (
            load_shadow_predictions,
        )

        logger.info(
            "Step 1: Loading shadow predictions."
        )

        predictions = (
            load_shadow_predictions()
        )

        result[
            "shadow_prediction_count"
        ] = len(predictions)

        if predictions.empty:

            result["status"] = (
                "NO_SHADOW_DATA"
            )

            logger.warning(
                "No shadow prediction data available."
            )

            return result

        # ----------------------------------------------------
        # STEP 2: EVALUATE CHAMPION / CHALLENGER
        # ----------------------------------------------------

        from src.champion_challenger_evaluation import (
            compare_models,
        )

        logger.info(
            "Step 2: Comparing Champion and Challenger."
        )

        evaluation = (
            compare_models(
                predictions
            )
        )

        result["evaluation"] = evaluation

        promotion_evaluation = (
            evaluation.get(
                "promotion",
                {},
            )
        )

        result[
            "promotion_eligible"
        ] = bool(
            promotion_evaluation.get(
                "eligible",
                False,
            )
        )

        result[
            "promotion_reasons"
        ] = (
            promotion_evaluation.get(
                "reasons",
                [],
            )
        )

        # ----------------------------------------------------
        # STEP 3: RUN PROMOTION ENGINE
        # ----------------------------------------------------

        from src.model_promotion import (
            promote_challenger,
        )

        logger.info(
            "Step 3: Running promotion engine."
        )

        promotion_result = (
            promote_challenger(
                dry_run=dry_run
            )
        )

        result["promotion"] = (
            promotion_result
        )

        result["promoted"] = bool(
            promotion_result.get(
                "promoted",
                False,
            )
        )

        result["status"] = (
            promotion_result.get(
                "status",
                "UNKNOWN",
            )
        )

        return result

    except Exception as error:

        logger.exception(
            "Model promotion job failed."
        )

        result["status"] = (
            "FAILED"
        )

        result["error"] = str(
            error
        )

        result["traceback"] = (
            traceback.format_exc()
        )

        return result

    finally:

        result["finished_at"] = (
            utc_now_iso()
        )

        logger.info(
            "=" * 70
        )

        logger.info(
            "MODEL PROMOTION JOB FINISHED | STATUS=%s",
            result.get("status"),
        )

        logger.info(
            "=" * 70
        )


# ============================================================
# CLI
# ============================================================

def main() -> int:
    """CLI entry point."""

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run Champion / Challenger "
            "model promotion workflow."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Evaluate promotion without "
            "changing the model registry."
        ),
    )

    args = parser.parse_args()

    result = (
        run_model_promotion_job(
            dry_run=args.dry_run
        )
    )

    print()

    print("=" * 70)

    print("MODEL PROMOTION JOB RESULT")

    print("=" * 70)

    print(
        f"Status: {result.get('status')}"
    )

    print(
        "Shadow predictions: "
        f"{result.get('shadow_prediction_count', 0)}"
    )

    print(
        "Promotion eligible: "
        f"{result.get('promotion_eligible')}"
    )

    print(
        f"Promoted: {result.get('promoted')}"
    )

    reasons = result.get(
        "promotion_reasons",
        [],
    )

    if reasons:

        print()

        print("Promotion reasons:")

        for reason in reasons:

            print(f"- {reason}")

    if result.get("error"):

        print()

        print(
            f"Error: {result.get('error')}"
        )

    return (
        0
        if result.get("status")
        not in {
            "FAILED",
            "PROMOTION_FAILED",
        }
        else 1
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

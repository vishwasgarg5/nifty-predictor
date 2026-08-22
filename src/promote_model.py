#!/usr/bin/env python3

"""
Promote the currently registered Challenger model
to the production Champion.

Workflow:

    Challenger
        ↓
    promote_model.py
        ↓
    Previous Champion
        ↓
    New Champion
        ↓
    Clear model cache

Example:

    python -m src.promote_model

    python -m src.promote_model \
        --reason "Passed validation tests"
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(
    "promote_model"
)


# ============================================================
# PROMOTE CHALLENGER
# ============================================================

def promote_model(
    reason: str = "MANUAL_PROMOTION",
) -> dict[str, Any]:
    """
    Promote the registered Challenger to Champion.

    Steps:

        1. Validate Challenger exists.
        2. Validate model file exists.
        3. Promote Challenger.
        4. Preserve old Champion.
        5. Clear production model cache.

    Returns:

        The new Champion record.
    """

    from src.model_registry import (
        get_challenger,
        get_model_path,
        promote_registered_challenger,
    )

    # --------------------------------------------------------
    # VALIDATE CHALLENGER
    # --------------------------------------------------------

    challenger = get_challenger()

    if challenger is None:

        raise RuntimeError(
            "No Challenger model is registered."
        )

    challenger_path = get_model_path(
        challenger
    )

    if challenger_path is None:

        raise RuntimeError(
            "Challenger model path is missing."
        )

    if not challenger_path.exists():

        raise FileNotFoundError(
            "Challenger model does not exist: "
            f"{challenger_path}"
        )

    # --------------------------------------------------------
    # PROMOTE
    # --------------------------------------------------------

    champion = (
        promote_registered_challenger(
            reason=reason
        )
    )

    # --------------------------------------------------------
    # CLEAR MODEL CACHE
    # --------------------------------------------------------

    try:

        from src.model_loader import (
            clear_model_cache,
        )

        clear_model_cache()

    except Exception as error:

        # Promotion already succeeded.
        # Cache clearing failure should not undo it.

        logger.warning(
            "Champion promoted, but model cache "
            "could not be cleared: %s",
            error,
        )

    logger.info(
        "Challenger promoted successfully | "
        "name=%s | reason=%s",
        champion.get(
            "name"
        ),
        reason,
    )

    return champion


# ============================================================
# CLI
# ============================================================

def main() -> int:
    """
    Promote the registered Challenger.

    Example:

        python -m src.promote_model

        python -m src.promote_model \
            --reason "Validation passed"
    """

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Promote the registered Challenger "
            "model to production Champion."
        ),
    )

    parser.add_argument(
        "--reason",
        default="MANUAL_PROMOTION",
        help=(
            "Reason recorded in the "
            "model registry history."
        ),
    )

    args = parser.parse_args()

    try:

        champion = promote_model(
            reason=args.reason
        )

        print()

        print("=" * 70)
        print("MODEL PROMOTED TO CHAMPION")
        print("=" * 70)

        print()

        for key, value in champion.items():

            print(
                f"{key}: {value}"
            )

        print()

        print(
            "SUCCESS: The model is now "
            "the production Champion."
        )

        return 0

    except Exception as error:

        logger.exception(
            "Model promotion failed."
        )

        print()

        print(
            f"ERROR: {error}"
        )

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

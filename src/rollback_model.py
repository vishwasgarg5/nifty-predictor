# rollback_model.py
#!/usr/bin/env python3

"""
Roll back the production Champion model.

Restores the previous Champion stored in the
production model registry.

Workflow:

    Current Champion
          ↓
    rollback_model.py
          ↓
    Previous Champion restored
          ↓
    Model cache cleared

Example:

    python -m src.rollback_model

    python -m src.rollback_model \
        --reason "Production validation failed"
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
    "rollback_model"
)


# ============================================================
# ROLLBACK
# ============================================================

def rollback_model(
    reason: str = "MANUAL_ROLLBACK",
) -> dict[str, Any]:
    """
    Restore the previous production Champion.

    Steps:

        1. Check that a previous Champion exists.
        2. Validate its model file.
        3. Restore it as Champion.
        4. Move the current Champion to previous_champion.
        5. Clear the production model cache.

    Returns:

        The restored Champion record.
    """

    from src.model_registry import (
        get_model_path,
        load_registry,
        rollback_to_previous_champion,
    )

    # --------------------------------------------------------
    # VALIDATE PREVIOUS CHAMPION
    # --------------------------------------------------------

    registry = load_registry()

    previous_champion = registry.get(
        "previous_champion"
    )

    if not isinstance(
        previous_champion,
        dict,
    ):

        raise RuntimeError(
            "No previous Champion is available "
            "for rollback."
        )

    previous_path = get_model_path(
        previous_champion
    )

    if previous_path is None:

        raise RuntimeError(
            "Previous Champion model path "
            "is missing."
        )

    if not previous_path.exists():

        raise FileNotFoundError(
            "Previous Champion model "
            "does not exist: "
            f"{previous_path}"
        )

    # --------------------------------------------------------
    # ROLLBACK
    # --------------------------------------------------------

    champion = (
        rollback_to_previous_champion(
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

        logger.warning(
            "Rollback succeeded, but model "
            "cache could not be cleared: %s",
            error,
        )

    logger.warning(
        "Production rollback completed | "
        "restored=%s | reason=%s",
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
    Roll back to the previous Champion.

    Examples:

        python -m src.rollback_model

        python -m src.rollback_model \
            --reason "Prediction pipeline failure"
    """

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Restore the previous production "
            "Champion model."
        ),
    )

    parser.add_argument(
        "--reason",
        default="MANUAL_ROLLBACK",
        help=(
            "Reason recorded in the "
            "model registry history."
        ),
    )

    args = parser.parse_args()

    try:

        champion = rollback_model(
            reason=args.reason
        )

        print()

        print("=" * 70)
        print("MODEL ROLLBACK COMPLETE")
        print("=" * 70)

        print()

        for key, value in champion.items():

            print(
                f"{key}: {value}"
            )

        print()

        print(
            "SUCCESS: Previous Champion "
            "has been restored."
        )

        return 0

    except Exception as error:

        logger.exception(
            "Model rollback failed."
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

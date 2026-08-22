#!/usr/bin/env python3

"""Train and persist Phase 3 ML models.

Pipeline:

    Stock Universe
          │
          ▼
    Historical Data
          │
          ▼
    MultiModelPipeline
      ├── Return Model
      ├── Direction Model
      └── Risk Model
          │
          ▼
      Validation
          │
          ▼
       ModelStore
          │
          ▼
    data/models/{SYMBOL}/

This script should be run separately from the
morning prediction job.

Example:

    python scripts/train_models.py
"""

from __future__ import annotations

import sys
import logging
import traceback

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.config import cfg

from src.universe import (
    get_universe_symbols,
)

from src.data_loader import (
    download_history,
)

from src.model_store import (
    ModelStore,
)

from src.ml_pipeline import (
    train_and_save,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    __name__
)


# ============================================================
# CONFIG HELPERS
# ============================================================

def get_model_store_path() -> Path:
    """Resolve model storage directory."""

    try:

        paths = getattr(
            cfg,
            "paths",
            None,
        )

        configured_path = getattr(
            paths,
            "models",
            None,
        )

        if configured_path:

            return Path(
                configured_path
            )

    except Exception:

        pass

    return (
        PROJECT_ROOT
        / "data"
        / "models"
    )


def get_history_period() -> str:
    """Return training history period."""

    try:

        ml_config = getattr(
            cfg,
            "ml",
            None,
        )

        period = getattr(
            ml_config,
            "training_period",
            None,
        )

        if period:

            return str(
                period
            )

    except Exception:

        pass

    return "2y"


def get_minimum_training_rows() -> int:
    """Return minimum rows required for training."""

    try:

        ml_config = getattr(
            cfg,
            "ml",
            None,
        )

        value = getattr(
            ml_config,
            "minimum_training_rows",
            None,
        )

        if value is not None:

            return int(
                value
            )

    except Exception:

        pass

    return 80


def get_training_limit() -> int | None:
    """Optional limit for testing training runs."""

    try:

        ml_config = getattr(
            cfg,
            "ml",
            None,
        )

        value = getattr(
            ml_config,
            "training_limit",
            None,
        )

        if value:

            return int(
                value
            )

    except Exception:

        pass

    return None


# ============================================================
# TRAIN ONE SYMBOL
# ============================================================

def train_symbol(
    symbol: str,
    store: ModelStore,
    period: str,
    minimum_training_rows: int,
) -> dict:
    """Train and save one stock model."""

    started_at = datetime.now(
        timezone.utc
    )

    result = {

        "symbol": symbol,

        "status": "FAILED",

        "training_rows": 0,

        "history_rows": 0,

        "error": None,

        "started_at": (
            started_at.isoformat()
        ),

        "completed_at": None,
    }

    try:

        logger.info(
            "[TRAIN] %s",
            symbol,
        )

        # ----------------------------------------------------
        # DOWNLOAD HISTORY
        # ----------------------------------------------------

        history = download_history(
            symbol,
            period=period,
        )

        if (
            history is None
            or history.empty
        ):

            raise ValueError(
                "No historical data returned."
            )

        result[
            "history_rows"
        ] = len(
            history
        )

        if len(history) < minimum_training_rows:

            raise ValueError(

                "Insufficient raw history: "

                f"{len(history)} rows "

                f"(minimum {minimum_training_rows})"
            )


        # ----------------------------------------------------
        # TRAIN + SAVE
        # ----------------------------------------------------

        metadata = {

            "symbol": symbol,

            "training_period": period,

            "history_rows": len(
                history
            ),

            "trained_at": datetime.now(
                timezone.utc
            ).isoformat(),

            "pipeline_type": (
                "MultiModelPipeline"
            ),
        }

        pipeline = train_and_save(

            symbol=symbol,

            history=history,

            store=store,

            minimum_training_rows=(
                minimum_training_rows
            ),

            metadata=metadata,
        )

        result[
            "training_rows"
        ] = pipeline.training_rows

        result[
            "status"
        ] = "SUCCESS"

        logger.info(

            "[SUCCESS] %s | "
            "history=%s | "
            "training=%s",

            symbol,

            len(history),

            pipeline.training_rows,
        )


    except Exception as error:

        result[
            "error"
        ] = str(error)

        logger.error(
            "[FAILED] %s | %s",
            symbol,
            error,
        )

        logger.debug(
            traceback.format_exc()
        )


    finally:

        result[
            "completed_at"
        ] = datetime.now(
            timezone.utc
        ).isoformat()

    return result


# ============================================================
# SAVE TRAINING REPORT
# ============================================================

def save_training_report(
    results: list[dict],
) -> Path:
    """Save training results as CSV."""

    reports_dir = (
        PROJECT_ROOT
        / "data"
        / "reports"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_path = (
        reports_dir
        / f"training_{timestamp}.csv"
    )

    frame = pd.DataFrame(
        results
    )

    if not frame.empty:

        frame.to_csv(
            report_path,
            index=False,
        )

    logger.info(
        "Training report saved: %s",
        report_path,
    )

    return report_path


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    """Train models for the configured universe."""

    started_at = datetime.now()

    logger.info(
        "=" * 70
    )

    logger.info(
        "PHASE 3C MODEL TRAINING STARTED"
    )

    logger.info(
        "=" * 70
    )


    # --------------------------------------------------------
    # LOAD CONFIG
    # --------------------------------------------------------

    store_path = (
        get_model_store_path()
    )

    history_period = (
        get_history_period()
    )

    minimum_training_rows = (
        get_minimum_training_rows()
    )

    training_limit = (
        get_training_limit()
    )

    store = ModelStore(
        base_path=store_path
    )

    logger.info(
        "Model store: %s",
        store_path,
    )

    logger.info(
        "History period: %s",
        history_period,
    )

    logger.info(
        "Minimum training rows: %s",
        minimum_training_rows,
    )


    # --------------------------------------------------------
    # LOAD UNIVERSE
    # --------------------------------------------------------

    try:

        symbols = (
            get_universe_symbols()
        )

    except Exception as error:

        logger.error(
            "Failed to load universe: %s",
            error,
        )

        return 1


    if not symbols:

        logger.error(
            "Universe is empty."
        )

        return 1


    # Remove duplicates while preserving order.
    symbols = list(
        dict.fromkeys(
            symbols
        )
    )


    # --------------------------------------------------------
    # OPTIONAL LIMIT
    # --------------------------------------------------------

    if training_limit:

        symbols = symbols[
            :training_limit
        ]

        logger.warning(
            "Training limit enabled: %s",
            training_limit,
        )


    logger.info(
        "Symbols to train: %s",
        len(symbols),
    )


    # --------------------------------------------------------
    # TRAIN MODELS
    # --------------------------------------------------------

    results: list[dict] = []

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):

        logger.info(
            "[%s/%s] Processing %s",
            index,
            len(symbols),
            symbol,
        )

        result = train_symbol(

            symbol=symbol,

            store=store,

            period=history_period,

            minimum_training_rows=(
                minimum_training_rows
            ),
        )

        results.append(
            result
        )


    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report_path = (
        save_training_report(
            results
        )
    )

    successful = sum(

        1

        for result in results

        if result[
            "status"
        ] == "SUCCESS"
    )

    failed = len(
        results
    ) - successful

    elapsed = int(

        (
            datetime.now()
            - started_at
        ).total_seconds()
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        "PHASE 3C MODEL TRAINING COMPLETED"
    )

    logger.info(
        "Successful: %s",
        successful,
    )

    logger.info(
        "Failed: %s",
        failed,
    )

    logger.info(
        "Total: %s",
        len(results),
    )

    logger.info(
        "Elapsed: %ss",
        elapsed,
    )

    logger.info(
        "Report: %s",
        report_path,
    )

    logger.info(
        "=" * 70
    )


    # Non-zero exit only if every model failed.
    if successful == 0:

        return 1

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

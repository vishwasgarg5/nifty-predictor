import logging

from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

from src.config import cfg
from src.telegram_utils import send_telegram


logger = logging.getLogger(
    __name__
)


def detect_drift() -> dict:
    """Detect model drift using only evaluated ledger records."""

    ledger_path = Path(
        getattr(
            cfg.paths,
            "ledger_file",
            "data/predictions/prediction_ledger.csv",
        )
    )

    if not ledger_path.exists():

        return {
            "status": "no_data"
        }

    try:

        df = pd.read_csv(
            ledger_path
        )

    except Exception as error:

        logger.warning(
            "Could not read ledger: %s",
            error,
        )

        return {
            "status": "read_error"
        }

    required_columns = {
        "market_date",
        "evaluation_status",
        "abs_error_pct",
        "direction_correct",
    }

    if not required_columns.issubset(
        df.columns
    ):

        return {
            "status": "invalid_ledger"
        }

    # Only completed evaluations count.
    df = df[
        df["evaluation_status"]
        == "evaluated"
    ].copy()

    if df.empty:

        return {
            "status": "no_evaluated_data"
        }

    df["market_date"] = (
        pd.to_datetime(
            df["market_date"],
            errors="coerce",
        )
    )

    cutoff = (
        datetime.now()
        - timedelta(
            days=cfg.drift.lookback_days
        )
    )

    recent = df[
        df["market_date"] >= cutoff
    ].copy()

    recent = recent.dropna(
        subset=[
            "abs_error_pct",
            "direction_correct",
        ]
    )

    if len(recent) < 5:

        return {
            "status": "insufficient",
            "n": int(len(recent)),
        }

    mape = float(
        pd.to_numeric(
            recent["abs_error_pct"],
            errors="coerce",
        ).mean()
    )

    directional_accuracy = float(
        pd.to_numeric(
            recent["direction_correct"],
            errors="coerce",
        ).mean()
        * 100
    )

    reasons = []

    if (
        mape
        > cfg.drift.mape_threshold
    ):

        reasons.append(
            f"MAPE high: "
            f"{mape:.2f}% > "
            f"{cfg.drift.mape_threshold}%"
        )

    if (
        directional_accuracy
        < cfg.drift.dir_acc_threshold
    ):

        reasons.append(
            f"Directional accuracy low: "
            f"{directional_accuracy:.1f}% < "
            f"{cfg.drift.dir_acc_threshold}%"
        )

    status = (
        "drift"
        if reasons
        else "ok"
    )

    result = {
        "status": status,

        "mape": round(
            mape,
            3,
        ),

        "dir_acc": round(
            directional_accuracy,
            1,
        ),

        "reasons": reasons,

        "n": int(
            len(recent)
        ),
    }

    # -----------------------------
    # SAVE DRIFT HISTORY
    # -----------------------------

    drift_path = Path(
        cfg.paths.drift_file
    )

    drift_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    drift_row = pd.DataFrame(
        [
            {
                "date": (
                    datetime.now()
                    .strftime(
                        "%Y-%m-%d"
                    )
                ),

                "mape": result["mape"],

                "dir_acc": (
                    result["dir_acc"]
                ),

                "status": (
                    result["status"]
                ),

                "n": result["n"],
            }
        ]
    )

    if drift_path.exists():

        old = pd.read_csv(
            drift_path
        )

        today = (
            datetime.now()
            .strftime(
                "%Y-%m-%d"
            )
        )

        old = old[
            old["date"]
            .astype(str)
            != today
        ]

        drift_row = pd.concat(
            [
                old,
                drift_row,
            ],
            ignore_index=True,
        )

    drift_row.to_csv(
        drift_path,
        index=False,
    )

    # -----------------------------
    # ALERT
    # -----------------------------

    if reasons:

        message = (
            "⚠️ *Model Drift Detected*\n"
            f"MAPE: `{result['mape']}%`\n"
            f"Directional Acc: "
            f"`{result['dir_acc']}%`\n"
            f"Samples: `{result['n']}`\n"
            f"Reasons: "
            f"{', '.join(reasons)}"
        )

        send_telegram(
            message
        )

        logger.warning(
            "MODEL DRIFT: %s",
            result,
        )

    else:

        logger.info(
            "Drift check OK: %s",
            result,
        )

    return result

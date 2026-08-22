"""Prediction ledger.

This module stores one prediction per:

    market_date + symbol + model_version

The ledger prevents duplicate records and keeps predictions pending until
actual market data is available for evaluation.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import cfg


LEDGER_COLUMNS = [
    "prediction_id",
    "prediction_timestamp",
    "market_date",
    "symbol",
    "model_version",
    "feature_version",

    "current_close",

    "predicted_open",
    "predicted_high",
    "predicted_low",
    "predicted_close",
    "predicted_return",
    "predicted_direction",

    "probability_up",
    "predicted_risk",
    "confidence",
    "opportunity_score",

    "market_regime",
    "sector",
    "data_quality_score",

    "actual_open",
    "actual_high",
    "actual_low",
    "actual_close",

    "actual_return",
    "actual_direction",

    "abs_error",
    "abs_error_pct",
    "direction_correct",

    "evaluation_status",
    "actual_source",
    "evaluated_timestamp",
]


def _path() -> Path:
    return Path(
        getattr(
            cfg.paths,
            "ledger_file",
            "data/predictions/prediction_ledger.csv",
        )
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prediction_id(
    market_date: str,
    symbol: str,
    model_version: str,
) -> str:
    """Create a stable prediction identity."""

    raw = (
        f"{market_date}|{symbol}|{model_version}"
    ).encode()

    return hashlib.sha256(raw).hexdigest()[:24]


def _read() -> pd.DataFrame:
    """Read the ledger and guarantee all expected columns exist."""

    path = _path()

    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    for column in LEDGER_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    return df[LEDGER_COLUMNS]


def _write(df: pd.DataFrame) -> None:
    """Write the ledger safely."""

    path = _path()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )


def record_predictions(
    records: list[dict],
) -> pd.DataFrame:
    """Store predictions without creating duplicates.

    Re-running the morning workflow for the same:

        market_date + symbol + model_version

    replaces the pending record instead of appending duplicates.
    """

    existing = _read()

    rows = []

    for record in records:

        market_date = str(record["market_date"])
        symbol = str(record["symbol"])

        model_version = str(
            record.get("model_version")
            or getattr(
                cfg.model,
                "version",
                "ohlc-v1",
            )
        )

        feature_version = str(
            record.get("feature_version")
            or getattr(
                cfg,
                "feature_version",
                "features-v1",
            )
        )

        record_id = (
            record.get("prediction_id")
            or prediction_id(
                market_date,
                symbol,
                model_version,
            )
        )

        row = {
            column: record.get(
                column,
                pd.NA,
            )
            for column in LEDGER_COLUMNS
        }

        row["prediction_id"] = record_id
        row["prediction_timestamp"] = (
            record.get("prediction_timestamp")
            or _now()
        )

        row["market_date"] = market_date
        row["symbol"] = symbol

        row["model_version"] = model_version
        row["feature_version"] = feature_version

        row["evaluation_status"] = "pending"

        rows.append(row)

    if not rows:
        return existing

    incoming = pd.DataFrame(
        rows,
        columns=LEDGER_COLUMNS,
    )

    # Remove existing records with the same identity.
    existing = existing[
        ~existing["prediction_id"].isin(
            incoming["prediction_id"]
        )
    ]

    combined = pd.concat(
        [
            existing,
            incoming,
        ],
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=["prediction_id"],
        keep="last",
    )

    _write(combined)

    return incoming


def pending_for_date(
    market_date: str,
) -> pd.DataFrame:
    """Return predictions waiting for evaluation."""

    df = _read()

    return df[
        (
            df["market_date"]
            .astype(str)
            == str(market_date)
        )
        &
        (
            df["evaluation_status"]
            == "pending"
        )
    ].copy()


def evaluate_prediction(
    prediction_id_value: str,
    actual: dict,
    evaluated_timestamp: str | None = None,
) -> dict | None:
    """Attach actual OHLC data and evaluation metrics."""

    df = _read()

    mask = (
        df["prediction_id"]
        .astype(str)
        == str(prediction_id_value)
    )

    if not mask.any():
        return None

    index = df.index[mask][0]

    row = df.loc[index]

    actual_close = float(actual["Close"])

    current_close = row.get("current_close")

    try:
        previous_close = float(current_close)
    except (
        TypeError,
        ValueError,
    ):
        previous_close = actual_close

    if previous_close <= 0:
        previous_close = actual_close

    predicted_close = float(
        row["predicted_close"]
    )

    actual_return = (
        actual_close / previous_close
    ) - 1

    predicted_return = row.get(
        "predicted_return"
    )

    if pd.isna(predicted_return):

        predicted_return = (
            predicted_close / previous_close
        ) - 1

    else:

        predicted_return = float(
            predicted_return
        )

    if predicted_return > 0:
        predicted_direction = 1

    elif predicted_return < 0:
        predicted_direction = -1

    else:
        predicted_direction = 0

    if actual_return > 0:
        actual_direction = 1

    elif actual_return < 0:
        actual_direction = -1

    else:
        actual_direction = 0

    absolute_error = abs(
        actual_close - predicted_close
    )

    absolute_error_pct = (
        absolute_error
        / abs(predicted_close)
        * 100
        if predicted_close != 0
        else pd.NA
    )

    updates = {
        "actual_open": actual.get("Open"),
        "actual_high": actual.get("High"),
        "actual_low": actual.get("Low"),
        "actual_close": actual_close,

        "actual_return": actual_return,
        "actual_direction": actual_direction,

        "abs_error": absolute_error,
        "abs_error_pct": absolute_error_pct,

        "direction_correct": int(
            predicted_direction
            == actual_direction
        ),

        "evaluation_status": "evaluated",

        "actual_source": actual.get(
            "source",
            "unknown",
        ),

        "evaluated_timestamp": (
            evaluated_timestamp
            or _now()
        ),
    }

    for key, value in updates.items():

        df.at[index, key] = value

    _write(df)

    return {
        **row.to_dict(),
        **updates,
    }

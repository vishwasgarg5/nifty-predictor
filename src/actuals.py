#!/usr/bin/env python3

"""
Actual Market Outcome Resolver.

This module evaluates past predictions by fetching market prices
and calculating the actual outcome.

Pipeline
--------
Prediction Ledger
       │
       ▼
Find prediction symbol
       │
       ▼
Fetch historical market prices
       │
       ▼
Determine prediction entry price
       │
       ▼
Determine evaluation price
       │
       ▼
Calculate actual return
       │
       ├── actual_return
       ├── actual_direction
       └── evaluation_status
       │
       ▼
Return evaluated DataFrame

The evaluation job calls:

    resolve_actual_outcomes(predictions)

Supported data source:
    yfinance / Yahoo Finance
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


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

logger = logging.getLogger("actuals")


# ============================================================
# CONFIG
# ============================================================

def object_to_dict(value: Any) -> dict[str, Any]:
    """Convert a config object into a dictionary."""

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
    """Load project configuration."""

    try:
        from src.config import cfg
        return cfg

    except Exception as error:

        logger.warning(
            "Could not load config: %s",
            error,
        )

        return None


def get_evaluation_config() -> dict[str, Any]:
    """
    Get evaluation settings.

    Supported config sections:

        evaluation:
            horizon_days: 1
            price_source: yahoo

    Defaults are used when unavailable.
    """

    cfg = load_config()

    if cfg is None:
        return {
            "horizon_days": 1,
            "price_source": "yahoo",
        }

    section = getattr(
        cfg,
        "evaluation",
        None,
    )

    values = object_to_dict(section)

    horizon_days = values.get(
        "horizon_days",
        1,
    )

    try:
        horizon_days = int(horizon_days)
    except Exception:
        horizon_days = 1

    return {
        "horizon_days": max(
            1,
            horizon_days,
        ),
        "price_source": values.get(
            "price_source",
            "yahoo",
        ),
    }


# ============================================================
# DATE HELPERS
# ============================================================

def utc_now() -> datetime:
    """Return current UTC datetime."""

    return datetime.now(
        timezone.utc
    )


def parse_datetime(
    value: Any,
) -> datetime | None:
    """
    Parse a date/datetime value safely.
    """

    if value is None:
        return None

    if isinstance(
        value,
        datetime,
    ):

        if value.tzinfo is None:

            return value.replace(
                tzinfo=timezone.utc
            )

        return value.astimezone(
            timezone.utc
        )

    try:

        parsed = pd.to_datetime(
            value,
            utc=True,
            errors="coerce",
        )

        if pd.isna(parsed):
            return None

        return parsed.to_pydatetime()

    except Exception:

        return None


# ============================================================
# SYMBOL HELPERS
# ============================================================

def normalize_symbol(
    symbol: Any,
) -> str:
    """
    Normalize a symbol for Yahoo Finance.

    Examples:

        RELIANCE     -> RELIANCE.NS
        TCS          -> TCS.NS
        INFY.NS      -> INFY.NS
        ^NSEI        -> ^NSEI

    Existing exchange suffixes are preserved.
    """

    if symbol is None:
        return ""

    value = str(symbol).strip().upper()

    if not value:
        return ""

    if value.startswith("^"):
        return value

    if "." in value:
        return value

    return f"{value}.NS"


# ============================================================
# PRICE FETCHING
# ============================================================

def fetch_historical_prices(
    symbol: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Fetch historical prices from Yahoo Finance.

    Returns a DataFrame containing:

        Open
        High
        Low
        Close
        Volume
    """

    try:

        import yfinance as yf

    except ImportError as error:

        raise RuntimeError(
            "yfinance is required for "
            "actual outcome evaluation."
        ) from error

    ticker = yf.Ticker(symbol)

    start_date = (
        start.date()
        - timedelta(days=7)
    )

    end_date = (
        end.date()
        + timedelta(days=7)
    )

    logger.info(
        "Fetching prices for %s | %s -> %s",
        symbol,
        start_date,
        end_date,
    )

    frame = ticker.history(
        start=start_date,
        end=end_date,
        auto_adjust=False,
    )

    if frame is None or frame.empty:

        return pd.DataFrame()

    frame = frame.copy()

    frame.index = pd.to_datetime(
        frame.index,
        utc=True,
        errors="coerce",
    )

    frame = frame.dropna(
        axis=0,
        how="all",
    )

    return frame


# ============================================================
# PRICE SELECTION
# ============================================================

def get_price_column(
    frame: pd.DataFrame,
) -> str | None:
    """
    Select the preferred price column.
    """

    for column in [
        "Close",
        "Adj Close",
    ]:

        if column in frame.columns:
            return column

    return None


def select_price_on_or_after(
    prices: pd.DataFrame,
    target_time: datetime,
) -> tuple[float | None, datetime | None]:
    """
    Get the first available market close
    on or after the target datetime.
    """

    if prices.empty:
        return None, None

    price_column = get_price_column(
        prices
    )

    if price_column is None:
        return None, None

    target_timestamp = pd.Timestamp(
        target_time
    )

    eligible = prices.loc[
        prices.index >= target_timestamp
    ]

    if eligible.empty:
        return None, None

    row = eligible.iloc[0]

    value = row.get(
        price_column
    )

    if pd.isna(value):
        return None, None

    timestamp = eligible.index[0]

    return (
        float(value),
        timestamp.to_pydatetime(),
    )


def select_price_on_or_before(
    prices: pd.DataFrame,
    target_time: datetime,
) -> tuple[float | None, datetime | None]:
    """
    Get the last available market close
    on or before the target datetime.
    """

    if prices.empty:
        return None, None

    price_column = get_price_column(
        prices
    )

    if price_column is None:
        return None, None

    target_timestamp = pd.Timestamp(
        target_time
    )

    eligible = prices.loc[
        prices.index <= target_timestamp
    ]

    if eligible.empty:
        return None, None

    row = eligible.iloc[-1]

    value = row.get(
        price_column
    )

    if pd.isna(value):
        return None, None

    timestamp = eligible.index[-1]

    return (
        float(value),
        timestamp.to_pydatetime(),
    )


# ============================================================
# PREDICTION HELPERS
# ============================================================

def find_first_value(
    row: pd.Series,
    candidates: list[str],
) -> Any:
    """Return the first available value."""

    for column in candidates:

        if column not in row.index:
            continue

        value = row.get(column)

        if pd.notna(value):

            return value

    return None


def get_prediction_timestamp(
    row: pd.Series,
) -> datetime | None:
    """
    Get prediction timestamp.
    """

    value = find_first_value(
        row,
        [
            "prediction_date",
            "created_at",
            "timestamp",
            "date",
        ],
    )

    return parse_datetime(
        value
    )


def get_prediction_symbol(
    row: pd.Series,
) -> str:
    """Get stock symbol."""

    value = find_first_value(
        row,
        [
            "symbol",
            "ticker",
            "stock",
        ],
    )

    return normalize_symbol(
        value
    )


def get_entry_price(
    row: pd.Series,
) -> float | None:
    """
    Get the recorded prediction entry price.

    If unavailable, historical market data
    will be used.
    """

    value = find_first_value(
        row,
        [
            "entry_price",
            "prediction_price",
            "current_price",
            "close",
            "price",
        ],
    )

    if value is None:
        return None

    try:

        value = float(value)

        if value <= 0:
            return None

        return value

    except Exception:

        return None


# ============================================================
# RETURN / DIRECTION
# ============================================================

def calculate_actual_return(
    entry_price: float,
    exit_price: float,
) -> float:
    """
    Calculate percentage return.

        (exit - entry) / entry * 100
    """

    if entry_price <= 0:
        raise ValueError(
            "Entry price must be greater than zero."
        )

    return (
        (exit_price - entry_price)
        / entry_price
        * 100.0
    )


def determine_direction(
    actual_return: float,
) -> str:
    """
    Convert actual return into direction.
    """

    if actual_return > 0:
        return "UP"

    if actual_return < 0:
        return "DOWN"

    return "FLAT"


# ============================================================
# SINGLE PREDICTION EVALUATION
# ============================================================

def evaluate_single_prediction(
    row: pd.Series,
    horizon_days: int,
) -> dict[str, Any]:
    """
    Evaluate a single prediction.
    """

    result: dict[str, Any] = {
        "actual_return": None,
        "actual_direction": None,
        "actual_risk": None,
        "evaluation_status": "WAITING",
        "evaluation_timestamp": None,
        "entry_price": None,
        "exit_price": None,
    }

    # --------------------------------------------------------
    # SYMBOL
    # --------------------------------------------------------

    symbol = get_prediction_symbol(
        row
    )

    if not symbol:

        result["evaluation_status"] = (
            "INVALID"
        )

        result["evaluation_error"] = (
            "Missing symbol."
        )

        return result

    # --------------------------------------------------------
    # PREDICTION TIME
    # --------------------------------------------------------

    prediction_time = (
        get_prediction_timestamp(
            row
        )
    )

    if prediction_time is None:

        result["evaluation_status"] = (
            "INVALID"
        )

        result["evaluation_error"] = (
            "Missing prediction timestamp."
        )

        return result

    # --------------------------------------------------------
    # TARGET TIME
    # --------------------------------------------------------

    target_time = (
        prediction_time
        + timedelta(
            days=horizon_days
        )
    )

    # Do not evaluate before horizon.
    if utc_now() < target_time:

        result["evaluation_status"] = (
            "WAITING"
        )

        result["evaluation_error"] = (
            "Evaluation horizon not reached."
        )

        return result

    # --------------------------------------------------------
    # FETCH MARKET DATA
    # --------------------------------------------------------

    try:

        prices = (
            fetch_historical_prices(
                symbol=symbol,
                start=prediction_time,
                end=target_time,
            )
        )

    except Exception as error:

        result["evaluation_status"] = (
            "WAITING"
        )

        result["evaluation_error"] = (
            f"Price fetch failed: {error}"
        )

        return result

    if prices.empty:

        result["evaluation_status"] = (
            "WAITING"
        )

        result["evaluation_error"] = (
            "No market price data available."
        )

        return result

    # --------------------------------------------------------
    # ENTRY PRICE
    # --------------------------------------------------------

    entry_price = get_entry_price(
        row
    )

    entry_timestamp = None

    if entry_price is None:

        (
            entry_price,
            entry_timestamp,
        ) = select_price_on_or_after(
            prices,
            prediction_time,
        )

    if entry_price is None:

        result["evaluation_status"] = (
            "WAITING"
        )

        result["evaluation_error"] = (
            "Could not determine entry price."
        )

        return result

    # --------------------------------------------------------
    # EXIT PRICE
    # --------------------------------------------------------

    (
        exit_price,
        exit_timestamp,
    ) = select_price_on_or_after(
        prices,
        target_time,
    )

    # If target market session is not available,
    # do not evaluate yet.
    if exit_price is None:

        result["evaluation_status"] = (
            "WAITING"
        )

        result["evaluation_error"] = (
            "Evaluation market close not available."
        )

        return result

    # --------------------------------------------------------
    # ACTUAL RETURN
    # --------------------------------------------------------

    try:

        actual_return = (
            calculate_actual_return(
                entry_price,
                exit_price,
            )
        )

    except Exception as error:

        result["evaluation_status"] = (
            "INVALID"
        )

        result["evaluation_error"] = (
            f"Return calculation failed: {error}"
        )

        return result

    # --------------------------------------------------------
    # ACTUAL RISK
    # --------------------------------------------------------

    actual_risk = calculate_actual_risk(
        prices=prices,
        entry_price=entry_price,
        prediction_time=prediction_time,
        target_time=target_time,
    )

    result.update(
        {
            "actual_return": actual_return,
            "actual_direction": (
                determine_direction(
                    actual_return
                )
            ),
            "actual_risk": actual_risk,
            "evaluation_status": (
                "EVALUATED"
            ),
            "evaluation_timestamp": (
                utc_now().isoformat()
            ),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_timestamp": (
                entry_timestamp.isoformat()
                if entry_timestamp
                else None
            ),
            "exit_timestamp": (
                exit_timestamp.isoformat()
                if exit_timestamp
                else None
            ),
            "evaluation_error": None,
        }
    )

    return result


# ============================================================
# ACTUAL RISK
# ============================================================

def calculate_actual_risk(
    prices: pd.DataFrame,
    entry_price: float,
    prediction_time: datetime,
    target_time: datetime,
) -> float | None:
    """
    Calculate realised risk during the
    prediction horizon.

    Uses the maximum absolute percentage move
    from the entry price.
    """

    if prices.empty:
        return None

    if entry_price <= 0:
        return None

    price_column = get_price_column(
        prices
    )

    if price_column is None:
        return None

    start_timestamp = pd.Timestamp(
        prediction_time
    )

    end_timestamp = pd.Timestamp(
        target_time
    )

    period = prices.loc[
        (
            prices.index >= start_timestamp
        )
        &
        (
            prices.index <= end_timestamp
        )
    ]

    if period.empty:
        return None

    values = pd.to_numeric(
        period[price_column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    percentage_moves = (
        (
            values - entry_price
        )
        .abs()
        / entry_price
        * 100.0
    )

    if percentage_moves.empty:
        return None

    return float(
        percentage_moves.max()
    )


# ============================================================
# BULK EVALUATION
# ============================================================

def resolve_actual_outcomes(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resolve actual outcomes for all pending predictions.

    This is the main function called by:

        scripts/evaluation_job.py
    """

    if predictions is None:

        return pd.DataFrame()

    if predictions.empty:

        return predictions.copy()

    config = get_evaluation_config()

    horizon_days = config.get(
        "horizon_days",
        1,
    )

    logger.info(
        "Evaluating %s prediction(s) "
        "with horizon=%s day(s).",
        len(predictions),
        horizon_days,
    )

    results = predictions.copy()

    # Ensure evaluation columns exist.
    for column in [
        "actual_return",
        "actual_direction",
        "actual_risk",
        "evaluation_status",
        "evaluation_timestamp",
        "evaluation_error",
        "exit_price",
        "exit_timestamp",
    ]:

        if column not in results.columns:

            results[column] = pd.NA

    evaluated_count = 0
    waiting_count = 0
    invalid_count = 0

    for index, row in predictions.iterrows():

        logger.info(
            "Evaluating prediction index=%s",
            index,
        )

        outcome = (
            evaluate_single_prediction(
                row=row,
                horizon_days=horizon_days,
            )
        )

        for column, value in outcome.items():

            if column not in results.columns:

                results[column] = pd.NA

            results.at[
                index,
                column,
            ] = value

        status = str(
            outcome.get(
                "evaluation_status",
                "WAITING",
            )
        ).upper()

        if status == "EVALUATED":

            evaluated_count += 1

        elif status == "INVALID":

            invalid_count += 1

        else:

            waiting_count += 1

    logger.info(
        "Actual outcome resolution complete | "
        "evaluated=%s | waiting=%s | invalid=%s",
        evaluated_count,
        waiting_count,
        invalid_count,
    )

    return results


# ============================================================
# ALIAS
# ============================================================

def fetch_actuals(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compatibility alias.
    """

    return resolve_actual_outcomes(
        predictions
    )


# ============================================================
# CLI TEST
# ============================================================

def main() -> int:
    """
    Test the actual outcome resolver.
    """

    sample = pd.DataFrame(
        [
            {
                "symbol": "RELIANCE",
                "prediction_date": (
                    (
                        utc_now()
                        - timedelta(
                            days=3
                        )
                    )
                    .isoformat()
                ),
                "predicted_return": 1.5,
                "predicted_direction": "UP",
            }
        ]
    )

    result = (
        resolve_actual_outcomes(
            sample
        )
    )

    print()

    print("=" * 70)

    print("ACTUAL OUTCOME TEST")

    print("=" * 70)

    print(
        result.to_string(
            index=False
        )
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

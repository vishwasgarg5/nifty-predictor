#!/usr/bin/env python3

"""
Actual Market Outcome Resolver.

This module evaluates past predictions by fetching historical
market prices and calculating actual outcomes.

Evaluation flow
---------------
Prediction Ledger
       │
       ▼
Read prediction_id / symbol / market_date
       │
       ▼
Determine prediction trading session
       │
       ▼
Calculate evaluation target date
       │
       ▼
Wait for required market session to complete
       │
       ▼
Fetch historical market prices
       │
       ▼
Determine entry price
       │
       ▼
Determine evaluation price
       │
       ▼
Calculate actual outcome
       │
       ├── actual_return
       ├── actual_direction
       ├── actual_risk
       └── evaluation_status
       │
       ▼
Return evaluated DataFrame

Important
---------
The resolver does not evaluate predictions merely because a
calendar-day horizon has passed.

A prediction is evaluated only when the required target trading
session has historical close data available.

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
# CONFIG HELPERS
# ============================================================

def object_to_dict(
    value: Any,
) -> dict[str, Any]:
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

    Supported configuration:

        evaluation:
            horizon_days: 1
            price_source: yahoo

    horizon_days means the number of calendar days after the
    prediction market date at which evaluation becomes eligible.
    The actual exit price is selected from the first available
    trading session on or after that target date.
    """

    cfg = load_config()

    defaults = {
        "horizon_days": 1,
        "price_source": "yahoo",
    }

    if cfg is None:
        return defaults

    section = getattr(
        cfg,
        "evaluation",
        None,
    )

    values = object_to_dict(
        section
    )

    horizon_days = values.get(
        "horizon_days",
        defaults["horizon_days"],
    )

    try:

        horizon_days = int(
            horizon_days
        )

    except Exception:

        horizon_days = defaults[
            "horizon_days"
        ]

    return {
        "horizon_days": max(
            1,
            horizon_days,
        ),
        "price_source": values.get(
            "price_source",
            defaults["price_source"],
        ),
    }


# ============================================================
# TIME HELPERS
# ============================================================

def utc_now() -> datetime:
    """Return current UTC datetime."""

    return datetime.now(
        timezone.utc
    )


def utc_now_iso() -> str:
    """Return current UTC time as ISO text."""

    return utc_now().isoformat()


def parse_datetime(
    value: Any,
) -> datetime | None:
    """
    Parse a date or datetime value safely.
    """

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

    except Exception:
        pass

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


def parse_market_date(
    value: Any,
) -> datetime | None:
    """
    Parse a market date.

    The returned value is normalized to midnight UTC.

    Examples:

        2026-08-22
        2026-08-22T04:30:00+00:00
    """

    parsed = parse_datetime(
        value
    )

    if parsed is None:
        return None

    return datetime(
        year=parsed.year,
        month=parsed.month,
        day=parsed.day,
        tzinfo=timezone.utc,
    )


def normalize_datetime_to_date(
    value: datetime,
) -> datetime:
    """Normalize a datetime to midnight UTC."""

    return datetime(
        year=value.year,
        month=value.month,
        day=value.day,
        tzinfo=timezone.utc,
    )


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

    try:

        if pd.isna(symbol):
            return ""

    except Exception:
        pass

    value = str(
        symbol
    ).strip().upper()

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

    The date range includes a buffer before and after the
    requested period so the resolver can handle weekends,
    holidays, and missing sessions.

    Returns a DataFrame containing historical OHLCV data.
    """

    try:

        import yfinance as yf

    except ImportError as error:

        raise RuntimeError(
            "yfinance is required for "
            "actual outcome evaluation."
        ) from error

    ticker = yf.Ticker(
        symbol
    )

    start_date = (
        normalize_datetime_to_date(
            start
        )
        - timedelta(days=10)
    ).date()

    end_date = (
        normalize_datetime_to_date(
            end
        )
        + timedelta(days=10)
    ).date()

    logger.info(
        "Fetching prices | symbol=%s | "
        "start=%s | end=%s",
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

    frame = frame.loc[
        ~frame.index.isna()
    ]

    frame = frame.dropna(
        axis=0,
        how="all",
    )

    frame = frame.sort_index()

    return frame


# ============================================================
# PRICE COLUMN
# ============================================================

def get_price_column(
    frame: pd.DataFrame,
) -> str | None:
    """
    Select the preferred closing price column.
    """

    for column in [
        "Close",
        "Adj Close",
    ]:

        if column in frame.columns:
            return column

    return None


# ============================================================
# TRADING DATE HELPERS
# ============================================================

def get_trading_dates(
    prices: pd.DataFrame,
) -> list[datetime]:
    """
    Return unique available trading dates.

    Dates are normalized to midnight UTC.
    """

    if prices.empty:
        return []

    dates: list[datetime] = []

    seen: set[str] = set()

    for timestamp in prices.index:

        try:

            value = pd.Timestamp(
                timestamp
            )

            if value.tzinfo is None:

                value = value.tz_localize(
                    "UTC"
                )

            else:

                value = value.tz_convert(
                    "UTC"
                )

            normalized = datetime(
                year=value.year,
                month=value.month,
                day=value.day,
                tzinfo=timezone.utc,
            )

            key = normalized.strftime(
                "%Y-%m-%d"
            )

            if key not in seen:

                seen.add(key)

                dates.append(
                    normalized
                )

        except Exception:
            continue

    return sorted(
        dates
    )


def select_first_trading_date_on_or_after(
    prices: pd.DataFrame,
    target_date: datetime,
) -> datetime | None:
    """
    Return the first available trading date on or after
    target_date.
    """

    normalized_target = (
        normalize_datetime_to_date(
            target_date
        )
    )

    for trading_date in get_trading_dates(
        prices
    ):

        if trading_date >= normalized_target:

            return trading_date

    return None


def select_last_trading_date_on_or_before(
    prices: pd.DataFrame,
    target_date: datetime,
) -> datetime | None:
    """
    Return the last available trading date on or before
    target_date.
    """

    normalized_target = (
        normalize_datetime_to_date(
            target_date
        )
    )

    selected = None

    for trading_date in get_trading_dates(
        prices
    ):

        if trading_date <= normalized_target:

            selected = trading_date

        else:

            break

    return selected


# ============================================================
# PRICE SELECTION
# ============================================================

def select_price_for_trading_date(
    prices: pd.DataFrame,
    trading_date: datetime,
) -> tuple[float | None, datetime | None]:
    """
    Select the closing price for a specific trading date.
    """

    if prices.empty:
        return None, None

    price_column = get_price_column(
        prices
    )

    if price_column is None:
        return None, None

    normalized_date = (
        normalize_datetime_to_date(
            trading_date
        )
    )

    target_day = normalized_date.date()

    selected_rows = prices.loc[
        prices.index.date == target_day
    ]

    if selected_rows.empty:
        return None, None

    row = selected_rows.iloc[-1]

    value = pd.to_numeric(
        row.get(
            price_column
        ),
        errors="coerce",
    )

    if pd.isna(value):
        return None, None

    timestamp = selected_rows.index[-1]

    return (
        float(value),
        timestamp.to_pydatetime(),
    )


def select_price_on_or_after(
    prices: pd.DataFrame,
    target_time: datetime,
) -> tuple[float | None, datetime | None]:
    """
    Select the first available trading session close
    on or after the target date.
    """

    target_date = (
        normalize_datetime_to_date(
            target_time
        )
    )

    trading_date = (
        select_first_trading_date_on_or_after(
            prices,
            target_date,
        )
    )

    if trading_date is None:
        return None, None

    return select_price_for_trading_date(
        prices,
        trading_date,
    )


def select_price_on_or_before(
    prices: pd.DataFrame,
    target_time: datetime,
) -> tuple[float | None, datetime | None]:
    """
    Select the last available trading session close
    on or before the target date.
    """

    target_date = (
        normalize_datetime_to_date(
            target_time
        )
    )

    trading_date = (
        select_last_trading_date_on_or_before(
            prices,
            target_date,
        )
    )

    if trading_date is None:
        return None, None

    return select_price_for_trading_date(
        prices,
        trading_date,
    )


# ============================================================
# PREDICTION HELPERS
# ============================================================

def find_first_value(
    row: pd.Series,
    candidates: list[str],
) -> Any:
    """
    Return the first non-empty value from candidate columns.
    """

    for column in candidates:

        if column not in row.index:
            continue

        value = row.get(
            column
        )

        try:

            if pd.isna(value):
                continue

        except Exception:
            pass

        if isinstance(
            value,
            str,
        ) and not value.strip():

            continue

        return value

    return None


def get_prediction_id(
    row: pd.Series,
) -> str | None:
    """Get prediction_id when available."""

    value = find_first_value(
        row,
        [
            "prediction_id",
        ],
    )

    if value is None:
        return None

    result = str(
        value
    ).strip()

    return result or None


def get_prediction_timestamp(
    row: pd.Series,
) -> datetime | None:
    """
    Get the original prediction timestamp.
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


def get_prediction_market_date(
    row: pd.Series,
) -> datetime | None:
    """
    Determine the market date for the prediction.

    Priority:

        1. market_date
        2. prediction_date
        3. created_at
        4. timestamp
        5. date
    """

    value = find_first_value(
        row,
        [
            "market_date",
            "prediction_date",
            "created_at",
            "timestamp",
            "date",
        ],
    )

    return parse_market_date(
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

    If unavailable, historical market data is used.
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

        numeric = float(
            value
        )

        if numeric <= 0:
            return None

        return numeric

    except Exception:

        return None


# ============================================================
# EVALUATION DATE
# ============================================================

def calculate_evaluation_target_date(
    prediction_market_date: datetime,
    horizon_days: int,
) -> datetime:
    """
    Calculate the earliest date eligible for evaluation.

    Example:

        prediction market date = Monday
        horizon_days = 1

        target = Tuesday

    If the target is a weekend or holiday, the resolver uses
    the first actual trading session on or after this date.
    """

    return (
        normalize_datetime_to_date(
            prediction_market_date
        )
        + timedelta(
            days=max(
                1,
                int(horizon_days),
            )
        )
    )


def target_horizon_reached(
    target_date: datetime,
) -> bool:
    """
    Check whether the target calendar date has been reached.

    This does not itself mean the market close is available.
    Historical price availability is checked separately.
    """

    now_date = (
        normalize_datetime_to_date(
            utc_now()
        )
    )

    normalized_target = (
        normalize_datetime_to_date(
            target_date
        )
    )

    return now_date >= normalized_target


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
        (
            exit_price
            - entry_price
        )
        / entry_price
        * 100.0
    )


def determine_direction(
    actual_return: float,
) -> str:
    """Convert actual return into direction."""

    if actual_return > 0:
        return "UP"

    if actual_return < 0:
        return "DOWN"

    return "FLAT"


# ============================================================
# ACTUAL RISK
# ============================================================

def calculate_actual_risk(
    prices: pd.DataFrame,
    entry_price: float,
    entry_date: datetime,
    exit_date: datetime,
) -> float | None:
    """
    Calculate realised risk during the evaluation period.

    Risk is the maximum absolute percentage movement from
    entry_price between the entry and exit trading dates.
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

    start_date = (
        normalize_datetime_to_date(
            entry_date
        ).date()
    )

    end_date = (
        normalize_datetime_to_date(
            exit_date
        ).date()
    )

    period = prices.loc[
        (
            prices.index.date >= start_date
        )
        &
        (
            prices.index.date <= end_date
        )
    ]

    if period.empty:
        return None

    values = pd.to_numeric(
        period[
            price_column
        ],
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    percentage_moves = (
        (
            values
            - entry_price
        ).abs()
        / entry_price
        * 100.0
    )

    if percentage_moves.empty:
        return None

    return float(
        percentage_moves.max()
    )


# ============================================================
# SINGLE PREDICTION EVALUATION
# ============================================================

def evaluate_single_prediction(
    row: pd.Series,
    horizon_days: int,
) -> dict[str, Any]:
    """
    Evaluate a single prediction.

    Evaluation rules:

    1. Determine symbol.
    2. Determine prediction market date.
    3. Calculate evaluation target date.
    4. Wait until target date is reached.
    5. Fetch historical prices.
    6. Find actual entry trading session.
    7. Find first trading session on or after target date.
    8. Evaluate only when both prices are available.
    """

    prediction_id = get_prediction_id(
        row
    )

    result: dict[str, Any] = {
        "prediction_id": prediction_id,
        "actual_return": None,
        "actual_direction": None,
        "actual_risk": None,
        "evaluation_status": "WAITING",
        "evaluation_timestamp": None,
        "evaluation_target_date": None,
        "entry_price": None,
        "exit_price": None,
        "entry_timestamp": None,
        "exit_timestamp": None,
        "evaluation_error": None,
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
    # PREDICTION MARKET DATE
    # --------------------------------------------------------

    prediction_market_date = (
        get_prediction_market_date(
            row
        )
    )

    if prediction_market_date is None:

        result["evaluation_status"] = (
            "INVALID"
        )

        result["evaluation_error"] = (
            "Missing prediction market date."
        )

        return result

    # --------------------------------------------------------
    # EVALUATION TARGET DATE
    # --------------------------------------------------------

    target_date = (
        calculate_evaluation_target_date(
            prediction_market_date,
            horizon_days,
        )
    )

    result[
        "evaluation_target_date"
    ] = target_date.strftime(
        "%Y-%m-%d"
    )

    # --------------------------------------------------------
    # WAIT FOR HORIZON
    # --------------------------------------------------------

    if not target_horizon_reached(
        target_date
    ):

        result["evaluation_status"] = (
            "WAITING"
        )

        result["evaluation_error"] = (
            "Evaluation target date has not "
            "been reached."
        )

        return result

    # --------------------------------------------------------
    # FETCH MARKET DATA
    # --------------------------------------------------------

    try:

        prices = (
            fetch_historical_prices(
                symbol=symbol,
                start=prediction_market_date,
                end=target_date,
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

    entry_timestamp: datetime | None = None

    entry_trading_date = (
        select_first_trading_date_on_or_after(
            prices,
            prediction_market_date,
        )
    )

    if entry_trading_date is None:

        result["evaluation_status"] = (
            "WAITING"
        )

        result["evaluation_error"] = (
            "Prediction trading session is "
            "not available."
        )

        return result

    if entry_price is None:

        (
            entry_price,
            entry_timestamp,
        ) = select_price_for_trading_date(
            prices,
            entry_trading_date,
        )

    else:

        entry_timestamp = (
            entry_trading_date
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
    # EXIT TRADING SESSION
    # --------------------------------------------------------

    exit_trading_date = (
        select_first_trading_date_on_or_after(
            prices,
            target_date,
        )
    )

    if exit_trading_date is None:

        result["evaluation_status"] = (
            "WAITING"
        )

        result["evaluation_error"] = (
            "Evaluation trading session close "
            "is not available yet."
        )

        return result

    (
        exit_price,
        exit_timestamp,
    ) = select_price_for_trading_date(
        prices,
        exit_trading_date,
    )

    if exit_price is None:

        result["evaluation_status"] = (
            "WAITING"
        )

        result["evaluation_error"] = (
            "Could not determine evaluation "
            "exit price."
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

    actual_risk = (
        calculate_actual_risk(
            prices=prices,
            entry_price=entry_price,
            entry_date=entry_trading_date,
            exit_date=exit_trading_date,
        )
    )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

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
                utc_now_iso()
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
# BULK EVALUATION
# ============================================================

def resolve_actual_outcomes(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resolve actual outcomes for pending predictions.

    This function preserves the existing prediction rows and
    adds or updates evaluation fields.

    It is compatible with:

        scripts/evaluation_job.py
    """

    if predictions is None:

        return pd.DataFrame()

    if predictions.empty:

        return predictions.copy()

    config = get_evaluation_config()

    horizon_days = int(
        config.get(
            "horizon_days",
            1,
        )
    )

    logger.info(
        "Evaluating %s prediction(s) | "
        "horizon=%s day(s)",
        len(predictions),
        horizon_days,
    )

    results = predictions.copy()

    required_columns = [
        "actual_return",
        "actual_direction",
        "actual_risk",
        "evaluation_status",
        "evaluation_timestamp",
        "evaluation_target_date",
        "evaluation_error",
        "entry_price",
        "entry_timestamp",
        "exit_price",
        "exit_timestamp",
    ]

    for column in required_columns:

        if column not in results.columns:

            results[column] = pd.NA

    evaluated_count = 0
    waiting_count = 0
    invalid_count = 0

    for index, row in predictions.iterrows():

        prediction_id = get_prediction_id(
            row
        )

        logger.info(
            "Evaluating prediction | "
            "index=%s | prediction_id=%s",
            index,
            prediction_id,
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
                "prediction_id": (
                    "test_prediction_001"
                ),
                "symbol": "RELIANCE",
                "market_date": (
                    utc_now()
                    - timedelta(days=3)
                ).strftime(
                    "%Y-%m-%d"
                ),
                "prediction_date": (
                    utc_now()
                    - timedelta(days=3)
                ).isoformat(),
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

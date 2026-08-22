"""Universe filtering.

The full universe is evaluated first for basic data availability and
liquidity. The resulting candidates are ranked by average traded value,
not alphabetical symbol order.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.data_loader import download_history


logger = logging.getLogger(__name__)


def _get_candidate_metrics(
    symbol: str,
    lookback_days: int = 60,
) -> dict | None:
    """Calculate basic liquidity metrics for one symbol."""

    try:
        history = download_history(
            symbol,
            period=f"{lookback_days + 30}d",
        )

        if history is None or history.empty:
            return None

        required = {
            "Close",
            "Volume",
        }

        if not required.issubset(
            history.columns
        ):
            return None

        df = history.copy()

        df["Close"] = pd.to_numeric(
            df["Close"],
            errors="coerce",
        )

        df["Volume"] = pd.to_numeric(
            df["Volume"],
            errors="coerce",
        )

        df = df.dropna(
            subset=[
                "Close",
                "Volume",
            ]
        )

        if len(df) < min(
            20,
            lookback_days,
        ):
            return None

        df = df.tail(
            lookback_days
        )

        avg_volume = float(
            df["Volume"].mean()
        )

        avg_price = float(
            df["Close"].mean()
        )

        avg_traded_value = (
            avg_volume
            * avg_price
        )

        latest_close = float(
            df["Close"].iloc[-1]
        )

        return {
            "symbol": symbol,
            "avg_volume": avg_volume,
            "avg_price": avg_price,
            "avg_traded_value": (
                avg_traded_value
            ),
            "latest_close": latest_close,
            "history_days": int(
                len(df)
            ),
        }

    except Exception as error:

        logger.debug(
            "Universe filter failed for %s: %s",
            symbol,
            error,
        )

        return None


def liquidity_prefilter(
    symbols: list[str],
    top_n: int = 250,
    min_avg_volume: float = 100_000,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """Rank the complete universe by liquidity.

    Unlike the old ``symbols[:220]`` logic, every supplied symbol gets a
    chance to qualify.
    """

    rows = []

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):

        metrics = _get_candidate_metrics(
            symbol=symbol,
            lookback_days=lookback_days,
        )

        if metrics is None:
            continue

        if (
            metrics["avg_volume"]
            < min_avg_volume
        ):
            continue

        rows.append(
            metrics
        )

        if index % 25 == 0:

            logger.info(
                "Liquidity scan progress: %s/%s",
                index,
                len(symbols),
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "avg_volume",
                "avg_price",
                "avg_traded_value",
                "latest_close",
                "history_days",
            ]
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "avg_traded_value",
        ascending=False,
    )

    return (
        result
        .head(top_n)
        .reset_index(drop=True)
    )


def select_liquid_symbols(
    symbols: list[str],
    config,
) -> list[str]:
    """Return liquid symbols using project configuration."""

    stage1_limit = int(
        getattr(
            config.scoring,
            "stage1_limit",
            250,
        )
    )

    min_avg_volume = float(
        getattr(
            config.scoring,
            "min_avg_volume",
            100_000,
        )
    )

    lookback_days = int(
        getattr(
            config,
            "liquidity_lookback_days",
            60,
        )
    )

    filtered = liquidity_prefilter(
        symbols=symbols,
        top_n=stage1_limit,
        min_avg_volume=min_avg_volume,
        lookback_days=lookback_days,
    )

    return filtered[
        "symbol"
    ].tolist()

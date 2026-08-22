"""Stock candidate scoring."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import cfg
from src.data_loader import download_history
from src.features import calculate_features
from src.universe_filter import (
    select_liquid_symbols,
)


logger = logging.getLogger(
    __name__
)


def score_stock(
    symbol: str,
) -> dict | None:
    """Score one stock using technical and fundamental signals."""

    try:

        history = download_history(
            symbol,
            period=f"{cfg.lookback_days + 60}d",
        )

        if (
            history is None
            or len(history)
            < cfg.min_history_days
        ):

            return None

        features = calculate_features(
            history
        )

        if not features:
            return None

        technical_score = 0.0

        rsi = features.get("rsi")

        if rsi is not None:

            if (
                rsi
                <= cfg.scoring.rsi_oversold
            ):
                technical_score += 2.0

            elif rsi < 55:
                technical_score += 1.0

        if features.get(
            "above_sma20",
            False,
        ):
            technical_score += 1.5

        if features.get(
            "above_sma50",
            False,
        ):
            technical_score += 1.5

        volume_ratio = features.get(
            "volume_ratio",
            0,
        )

        if (
            volume_ratio
            >= cfg.scoring.volume_spike
        ):
            technical_score += 1.0

        atr_pct = features.get(
            "atr_pct"
        )

        if (
            atr_pct is not None
            and atr_pct
            > cfg.scoring.max_atr_pct / 100
        ):
            technical_score -= 1.0

        # Fundamental score remains optional.
        # Missing API data should not automatically
        # be treated as a negative score.
        fundamental_score = 0.0
        fundamental_available = 0

        pe = features.get("pe")
        pb = features.get("pb")
        roe = features.get("roe")

        if pe is not None:

            fundamental_available += 1

            if (
                0 < pe
                <= cfg.scoring.pe_max
            ):
                fundamental_score += 1.0

        if pb is not None:

            fundamental_available += 1

            if (
                0 < pb
                <= cfg.scoring.pb_max
            ):
                fundamental_score += 1.0

        if roe is not None:

            fundamental_available += 1

            if roe >= cfg.scoring.roe_min:
                fundamental_score += 1.0

        if fundamental_available == 0:
            fundamental_score = 1.0

        total_score = (
            cfg.scoring.weights.technical
            * technical_score
            +
            cfg.scoring.weights.fundamental
            * fundamental_score
        )

        return {
            "symbol": symbol,

            "score": float(
                total_score
            ),

            "technical_score": float(
                technical_score
            ),

            "fundamental_score": float(
                fundamental_score
            ),

            **features,
        }

    except Exception as error:

        logger.debug(
            "Scoring failed for %s: %s",
            symbol,
            error,
        )

        return None


def select_top5(
    symbols: list[str],
    top_n: int | None = None,
) -> pd.DataFrame:
    """Select top candidates.

    Pipeline:

        Full Universe
            ->
        Liquidity Prefilter
            ->
        Feature/Score Evaluation
            ->
        Ranked Candidates

    Every supplied symbol gets a chance during
    the liquidity prefilter. There is no alphabetical truncation.
    """

    if top_n is None:
        top_n = cfg.top_n

    if not symbols:
        return pd.DataFrame()

    logger.info(
        "Starting liquidity prefilter "
        "for %s symbols",
        len(symbols),
    )

    liquid_symbols = (
        select_liquid_symbols(
            symbols,
            cfg,
        )
    )

    if not liquid_symbols:

        logger.warning(
            "No liquid symbols passed prefilter"
        )

        return pd.DataFrame()

    logger.info(
        "Scoring %s liquid candidates",
        len(liquid_symbols),
    )

    results = []

    for index, symbol in enumerate(
        liquid_symbols,
        start=1,
    ):

        result = score_stock(
            symbol
        )

        if result:
            results.append(
                result
            )

        if index % 25 == 0:

            logger.info(
                "Scoring progress: %s/%s",
                index,
                len(liquid_symbols),
            )

    if not results:

        return pd.DataFrame()

    ranked = pd.DataFrame(
        results
    )

    ranked = ranked.sort_values(
        "score",
        ascending=False,
    )

    return (
        ranked
        .head(top_n)
        .reset_index(drop=True)
    )

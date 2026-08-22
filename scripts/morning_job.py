#!/usr/bin/env python3

"""Morning prediction pipeline.

Pipeline:

    Trading Day Check
            ↓
    Full Stock Universe
            ↓
    Liquidity Prefilter
            ↓
    Market Regime Detection
            ↓
    Candidate Scoring
            ↓
    Feature Engine
            ↓
    OHLC Prediction
            ↓
    Prediction Validation
            ↓
    Prediction Ledger
            ↓
    Telegram
"""

from __future__ import annotations

import sys
import logging
import traceback

from pathlib import Path
from datetime import datetime

import pandas as pd


# --------------------------------------------
# PROJECT PATH
# --------------------------------------------

sys.path.insert(
    0,
    str(
        Path(__file__)
        .resolve()
        .parent
        .parent
    ),
)


# --------------------------------------------
# PROJECT IMPORTS
# --------------------------------------------

from src.config import cfg

from src.holidays import is_trading_day

from src.universe import get_universe_symbols

from src.scoring import select_top5

from src.model import OHLCPredictor

from src.sentiment import (
    get_sentiment_engine,
)

from src.indexes import predict_indexes

from src.ipo import (
    ipo_watchlist_telegram_lines,
)

from src.ipo_gmp import (
    build_ipo_desk_message,
)

from src.telegram_utils import (
    send_telegram,
)

from src.ledger import (
    record_predictions,
)

from src.data_loader import (
    download_history,
)

from src.data_validation import (
    validate_prediction,
)

from src.market_regime import (
    detect_market_regime,
)

from src.feature_engine import (
    latest_features,
    feature_quality_score,
)


# --------------------------------------------
# LOGGING
# --------------------------------------------

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


# ============================================
# HELPERS
# ============================================

def _uni_label() -> str:
    """Return a readable universe label."""

    try:

        if (
            hasattr(
                cfg,
                "universes",
            )
            and getattr(
                cfg.universes,
                "primary",
                None,
            )
        ):

            return "+".join(
                str(value).upper()
                for value
                in cfg.universes.primary
            )

    except Exception:

        pass

    return "NIFTY"


def get_current_close(
    symbol: str,
) -> float | None:
    """Get the latest available closing price."""

    try:

        history = download_history(
            symbol,
            period="10d",
        )

        if (
            history is None
            or history.empty
            or "Close"
            not in history.columns
        ):

            return None

        close = (
            pd.to_numeric(
                history["Close"],
                errors="coerce",
            )
            .dropna()
        )

        if close.empty:

            return None

        return float(
            close.iloc[-1]
        )

    except Exception as error:

        logger.warning(
            "Could not get current close "
            "for %s: %s",
            symbol,
            error,
        )

        return None


def get_market_regime() -> dict:
    """Download NIFTY history and detect regime."""

    try:

        regime_config = getattr(
            cfg,
            "market_regime",
            None,
        )

        symbol = (
            getattr(
                regime_config,
                "symbol",
                "^NSEI",
            )
            if regime_config
            else "^NSEI"
        )

        history = download_history(
            symbol,
            period="1y",
        )

        regime = detect_market_regime(
            history
        )

        logger.info(
            "Market regime: %s | "
            "score: %.2f | "
            "confidence: %.2f",
            regime.get(
                "regime",
                "UNKNOWN",
            ),
            regime.get(
                "score",
                0.0,
            ),
            regime.get(
                "confidence",
                0.0,
            ),
        )

        return regime

    except Exception as error:

        logger.warning(
            "Market regime detection failed: %s",
            error,
        )

        return {
            "regime": "UNKNOWN",
            "score": 0.0,
            "confidence": 0.0,
        }


def build_stock_features(
    symbol: str,
) -> dict:
    """Build Phase 2 features for a stock."""

    try:

        history = download_history(
            symbol,
            period="1y",
        )

        features = latest_features(
            history
        )

        quality = feature_quality_score(
            features
        )

        logger.info(
            "%s feature quality: %.2f",
            symbol,
            quality,
        )

        return {
            "features": features,
            "quality": quality,
        }

    except Exception as error:

        logger.warning(
            "Feature generation failed for "
            "%s: %s",
            symbol,
            error,
        )

        return {
            "features": None,
            "quality": 0.0,
        }


def direction_from_return(
    value: float | None,
) -> int | None:
    """Convert return to direction."""

    if value is None:

        return None

    try:

        value = float(value)

    except (
        TypeError,
        ValueError,
    ):

        return None

    if value > 0:

        return 1

    if value < 0:

        return -1

    return 0


# ============================================
# MAIN PIPELINE
# ============================================

def main():

    start = datetime.now()

    today = start.strftime(
        "%Y-%m-%d"
    )

    logger.info(
        "=" * 60
    )

    logger.info(
        "MORNING JOB STARTED: %s",
        today,
    )

    logger.info(
        "=" * 60
    )


    # ----------------------------------------
    # TRADING DAY CHECK
    # ----------------------------------------

    if not is_trading_day():

        logger.info(
            "Not a trading day."
        )

        try:

            if (
                getattr(
                    cfg,
                    "ipo_desk",
                    None,
                )
                and cfg.ipo_desk.enabled
            ):

                send_telegram(
                    build_ipo_desk_message(
                        cfg.ipo_desk.max_rows
                    )
                )

        except Exception as error:

            logger.warning(
                "IPO desk failed: %s",
                error,
            )

        return


    try:

        # ====================================
        # STEP 1
        # FULL STOCK UNIVERSE
        # ====================================

        logger.info(
            "Loading stock universe..."
        )

        symbols = (
            get_universe_symbols()
        )

        logger.info(
            "Universe size: %s",
            len(symbols),
        )

        if not symbols:

            send_telegram(
                "⚠️ Morning job: "
                "stock universe is empty."
            )

            return


        # ====================================
        # STEP 2
        # MARKET REGIME
        # ====================================

        market_regime = (
            get_market_regime()
        )


        # ====================================
        # STEP 3
        # LIQUIDITY PREFILTER +
        # CANDIDATE SCORING
        # ====================================

        logger.info(
            "Selecting Top %s candidates...",
            cfg.top_n,
        )

        top5 = select_top5(
            symbols,
            cfg.top_n,
        )

        if top5 is None or top5.empty:

            logger.warning(
                "No stocks selected."
            )

            send_telegram(
                f"⚠️ No stocks selected "
                f"for `{today}`"
            )

            return

        logger.info(
            "Selected stocks: %s",
            ", ".join(
                top5["symbol"].tolist()
            ),
        )


        # ====================================
        # STEP 4
        # FEATURE ENGINE
        # ====================================

        logger.info(
            "Building feature metadata..."
        )

        feature_metadata = {}

        minimum_quality = float(
            getattr(
                getattr(
                    cfg,
                    "features",
                    None,
                ),
                "min_quality_score",
                0.85,
            )
        )

        for _, row in top5.iterrows():

            symbol = row["symbol"]

            metadata = (
                build_stock_features(
                    symbol
                )
            )

            feature_metadata[
                symbol
            ] = metadata

            if (
                metadata["quality"]
                < minimum_quality
            ):

                logger.warning(
                    "%s feature quality "
                    "below threshold: %.2f < %.2f",
                    symbol,
                    metadata["quality"],
                    minimum_quality,
                )


        # ====================================
        # STEP 5
        # OHLC PREDICTIONS
        # ====================================

        logger.info(
            "Generating OHLC predictions..."
        )

        predictions = {}

        for _, row in top5.iterrows():

            symbol = row["symbol"]

            try:

                predictor = (
                    OHLCPredictor(
                        symbol
                    )
                )

                prediction = (
                    predictor.predict_next()
                )

                if prediction:

                    predictions[
                        symbol
                    ] = prediction

                    logger.info(
                        "Prediction generated: %s",
                        symbol,
                    )

                else:

                    logger.warning(
                        "No prediction: %s",
                        symbol,
                    )

            except Exception as error:

                logger.warning(
                    "Prediction failed for "
                    "%s: %s",
                    symbol,
                    error,
                )


        if not predictions:

            send_telegram(
                f"⚠️ No valid predictions "
                f"for `{today}`"
            )

            return


        # ====================================
        # STEP 6
        # SENTIMENT
        # ====================================

        sentiments = {}

        try:

            logger.info(
                "Running sentiment analysis..."
            )

            engine = (
                get_sentiment_engine()
            )

            for symbol in predictions:

                try:

                    sentiments[symbol] = (
                        engine.analyze_stock(
                            symbol,
                            cfg.sentiment.max_articles,
                        )
                    )

                except Exception as error:

                    logger.warning(
                        "Sentiment failed for "
                        "%s: %s",
                        symbol,
                        error,
                    )

        except Exception as error:

            logger.warning(
                "Sentiment engine failed: %s",
                error,
            )


        # ====================================
        # STEP 7
        # DAILY PREDICTION FILE
        # ====================================

        prediction_dir = Path(
            cfg.paths.predictions
        )

        prediction_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        records = []

        for _, row in top5.iterrows():

            symbol = row["symbol"]

            prediction = predictions.get(
                symbol
            )

            if not prediction:

                continue

            score = row.get(
                "score",
                0.0,
            )

            record = {
                "date": today,

                "symbol": symbol,

                "score": float(score),

                "market_regime": (
                    market_regime.get(
                        "regime",
                        "UNKNOWN",
                    )
                ),

                "feature_quality": (
                    feature_metadata
                    .get(symbol, {})
                    .get(
                        "quality",
                        0.0,
                    )
                ),

                **prediction,
            }

            records.append(
                record
            )


        if not records:

            send_telegram(
                f"⚠️ Prediction records empty "
                f"for `{today}`"
            )

            return


        prediction_file = (
            prediction_dir
            / f"{today}.csv"
        )

        pd.DataFrame(
            records
        ).to_csv(
            prediction_file,
            index=False,
        )

        logger.info(
            "Saved predictions: %s",
            prediction_file,
        )


        # ====================================
        # STEP 8
        # PREDICTION LEDGER
        # ====================================

        logger.info(
            "Recording prediction ledger..."
        )

        ledger_records = []

        for record in records:

            symbol = record[
                "symbol"
            ]

            valid, validation = (
                validate_prediction(
                    symbol,
                    record,
                )
            )

            if not valid:

                logger.warning(
                    "Invalid prediction %s: %s",
                    symbol,
                    validation,
                )

                continue

            predicted_close = float(
                record["Close"]
            )

            current_close = (
                get_current_close(
                    symbol
                )
            )

            predicted_return = None

            if (
                current_close is not None
                and current_close > 0
            ):

                predicted_return = (
                    predicted_close
                    / current_close
                ) - 1

            predicted_direction = (
                direction_from_return(
                    predicted_return
                )
            )

            metadata = (
                feature_metadata
                .get(symbol, {})
            )

            ledger_record = {

                "market_date": today,

                "symbol": symbol,

                "current_close": (
                    current_close
                ),

                "predicted_open": float(
                    record["Open"]
                ),

                "predicted_high": float(
                    record["High"]
                ),

                "predicted_low": float(
                    record["Low"]
                ),

                "predicted_close": (
                    predicted_close
                ),

                "predicted_return": (
                    predicted_return
                ),

                "predicted_direction": (
                    predicted_direction
                ),

                "market_regime": (
                    market_regime.get(
                        "regime",
                        "UNKNOWN",
                    )
                ),

                "data_quality_score": (
                    metadata.get(
                        "quality",
                        0.0,
                    )
                ),

                "feature_version": (
                    metadata
                    .get(
                        "features",
                        {},
                    )
                    .get(
                        "feature_version",
                        getattr(
                            cfg,
                            "feature_version",
                            "features-v2",
                        ),
                    )
                    if metadata.get(
                        "features"
                    )
                    else getattr(
                        cfg,
                        "feature_version",
                        "features-v2",
                    )
                ),

                "opportunity_score": (
                    record.get(
                        "score",
                        0.0,
                    )
                ),
            }

            ledger_records.append(
                ledger_record
            )


        if ledger_records:

            record_predictions(
                ledger_records
            )

            logger.info(
                "Recorded %s predictions "
                "in ledger.",
                len(ledger_records),
            )

        else:

            logger.warning(
                "No valid records for ledger."
            )


        # ====================================
        # STEP 9
        # INDEX PREDICTIONS
        # ====================================

        index_predictions = []

        try:

            if (
                getattr(
                    cfg,
                    "indexes",
                    None,
                )
                and cfg.indexes.enabled
            ):

                logger.info(
                    "Generating index predictions..."
                )

                index_predictions = (
                    predict_indexes()
                )

                if index_predictions:

                    index_file = (
                        prediction_dir
                        / (
                            f"{today}"
                            "_indexes.csv"
                        )
                    )

                    pd.DataFrame(
                        index_predictions
                    ).to_csv(
                        index_file,
                        index=False,
                    )

        except Exception as error:

            logger.warning(
                "Index predictions failed: %s",
                error,
            )


        # ====================================
        # STEP 10
        # TELEGRAM REPORT
        # ====================================

        lines = [

            "*TOP STOCK PREDICTIONS*",

            (
                f"Date: `{today}` | "
                f"`{_uni_label()}`"
            ),

            "",

            (
                f"*Market Regime:* "
                f"`{market_regime.get('regime', 'UNKNOWN')}` "
                f"({market_regime.get('confidence', 0):.0%})"
            ),

            "",

            "```",

            (
                f"{'Stock':<12} "
                f"{'Open':>9} "
                f"{'High':>9} "
                f"{'Low':>9} "
                f"{'Close':>9}"
            ),

            "-" * 52,
        ]


        for _, row in top5.iterrows():

            symbol = row[
                "symbol"
            ]

            prediction = (
                predictions.get(
                    symbol
                )
            )

            if not prediction:

                continue

            name = symbol.replace(
                ".NS",
                "",
            )

            lines.append(

                f"{name:<12} "

                f"{float(prediction['Open']):>9.2f} "

                f"{float(prediction['High']):>9.2f} "

                f"{float(prediction['Low']):>9.2f} "

                f"{float(prediction['Close']):>9.2f}"

            )


        lines.append(
            "```"
        )


        # ------------------------------------
        # FEATURE QUALITY
        # ------------------------------------

        lines += [

            "",

            "*Feature Quality*",

        ]

        for _, row in top5.iterrows():

            symbol = row[
                "symbol"
            ]

            quality = (
                feature_metadata
                .get(symbol, {})
                .get(
                    "quality",
                    0.0,
                )
            )

            if quality >= minimum_quality:

                emoji = "🟢"

            elif quality >= 0.60:

                emoji = "🟡"

            else:

                emoji = "🔴"

            lines.append(

                f"{emoji} "

                f"{symbol.replace('.NS', '')}: "

                f"`{quality:.0%}`"

            )


        # ------------------------------------
        # INDEX PREDICTIONS
        # ------------------------------------

        if index_predictions:

            lines += [

                "",

                "*INDEX PREDICTIONS*",

                "",

                "```",

                (
                    f"{'Index':<12} "
                    f"{'Open':>9} "
                    f"{'High':>9} "
                    f"{'Low':>9} "
                    f"{'Close':>9}"
                ),

                "-" * 52,
            ]

            for prediction in index_predictions:

                lines.append(

                    f"{prediction['name']:<12} "

                    f"{float(prediction['Open']):>9.2f} "

                    f"{float(prediction['High']):>9.2f} "

                    f"{float(prediction['Low']):>9.2f} "

                    f"{float(prediction['Close']):>9.2f}"

                )

            lines.append(
                "```"
            )


        # ------------------------------------
        # SENTIMENT
        # ------------------------------------

        if sentiments:

            lines += [

                "",

                "*SENTIMENT*",

            ]

            for _, row in top5.iterrows():

                symbol = row[
                    "symbol"
                ]

                sentiment = (
                    sentiments.get(
                        symbol
                    )
                )

                if not sentiment:

                    continue

                try:

                    if (
                        not sentiment.article_count
                    ):

                        continue

                    score = (
                        sentiment.overall_score
                    )

                    if score >= 0.15:

                        emoji = "🟢"

                    elif score <= -0.15:

                        emoji = "🔴"

                    else:

                        emoji = "⚪"

                    lines.append(

                        f"{emoji} "

                        f"{symbol.replace('.NS', '')}: "

                        f"`{score:+.2f}` "

                        f"({sentiment.overall_label})"

                    )

                except Exception:

                    continue


        # ------------------------------------
        # IPO WATCHLIST
        # ------------------------------------

        try:

            lines += (
                [""]
                + ipo_watchlist_telegram_lines()
            )

        except Exception as error:

            logger.warning(
                "IPO watchlist failed: %s",
                error,
            )


        # ------------------------------------
        # EXECUTION TIME
        # ------------------------------------

        elapsed = int(
            (
                datetime.now()
                - start
            ).total_seconds()
        )

        lines += [

            "",

            (
                f"_Finished in "
                f"{elapsed}s_"
            ),
        ]


        # ====================================
        # SEND TELEGRAM
        # ====================================

        send_telegram(
            "\n".join(lines)
        )


        # ====================================
        # IPO DESK
        # ====================================

        try:

            if (
                getattr(
                    cfg,
                    "ipo_desk",
                    None,
                )
                and cfg.ipo_desk.enabled
            ):

                send_telegram(

                    build_ipo_desk_message(

                        int(
                            getattr(
                                cfg.ipo_desk,
                                "max_rows",
                                10,
                            )
                        )

                    )

                )

        except Exception as error:

            logger.warning(
                "IPO desk message failed: %s",
                error,
            )


        logger.info(
            "=" * 60
        )

        logger.info(
            "MORNING JOB COMPLETED"
        )

        logger.info(
            "=" * 60
        )


    except Exception as error:

        logger.error(
            traceback.format_exc()
        )

        try:

            send_telegram(

                "❌ *Morning Job Failed*\n"

                f"Date: `{today}`\n"

                f"```{str(error)[:700]}```"

            )

        except Exception:

            pass


if __name__ == "__main__":
    main()

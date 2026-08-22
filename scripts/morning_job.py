#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__)
        .resolve()
        .parent
        .parent
    ),
)

import logging
import traceback
from datetime import datetime

import pandas as pd

from src.config import cfg
from src.holidays import is_trading_day
from src.universe import get_universe_symbols
from src.scoring import select_top5
from src.model import OHLCPredictor
from src.sentiment import get_sentiment_engine
from src.indexes import predict_indexes
from src.ipo import ipo_watchlist_telegram_lines
from src.ipo_gmp import build_ipo_desk_message
from src.telegram_utils import send_telegram

from src.ledger import record_predictions
from src.data_loader import download_history
from src.data_validation import validate_prediction


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


def _uni_label():

    if (
        hasattr(cfg, "universes")
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

    return "NIFTY"


def get_current_close(
    symbol: str,
) -> float | None:
    """Get the latest available close."""

    try:

        history = download_history(
            symbol,
            period="10d",
        )

        if (
            history is None
            or history.empty
        ):

            return None

        close = (
            history["Close"]
            .dropna()
            .iloc[-1]
        )

        return float(close)

    except Exception as error:

        logger.warning(
            "Could not get current close "
            "for %s: %s",
            symbol,
            error,
        )

        return None


def main():

    start = datetime.now()

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    logger.info(
        "Morning job started: %s",
        today,
    )

    if not is_trading_day():

        logger.info(
            "Not a trading day."
        )

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

        return

    try:

        # -----------------------------
        # LOAD STOCK UNIVERSE
        # -----------------------------

        symbols = (
            get_universe_symbols()
        )

        logger.info(
            "Universe size: %s",
            len(symbols),
        )

        # -----------------------------
        # SELECT TOP STOCKS
        # -----------------------------

        top5 = select_top5(
            symbols,
            cfg.top_n,
        )

        if top5.empty:

            send_telegram(
                f"⚠️ No stocks selected "
                f"`{today}`"
            )

            return

        # -----------------------------
        # RUN PREDICTIONS
        # -----------------------------

        predictions = {}

        for _, row in top5.iterrows():

            symbol = row["symbol"]

            prediction = (
                OHLCPredictor(
                    symbol
                )
                .predict_next()
            )

            if prediction:

                predictions[
                    symbol
                ] = prediction

        if not predictions:

            send_telegram(
                f"⚠️ No valid predictions "
                f"`{today}`"
            )

            return

        # -----------------------------
        # SENTIMENT
        # -----------------------------

        sentiments = {}

        try:

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

        # -----------------------------
        # SAVE DAILY PREDICTION FILE
        # -----------------------------

        prediction_dir = Path(
            cfg.paths.predictions
        )

        prediction_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        records = []

        for symbol, prediction in (
            predictions.items()
        ):

            score = top5.loc[
                top5.symbol == symbol,
                "score",
            ]

            records.append(
                {
                    "date": today,
                    "symbol": symbol,

                    **prediction,

                    "score": (
                        float(score.values[0])
                        if len(score)
                        else 0.0
                    ),
                }
            )

        pd.DataFrame(
            records
        ).to_csv(
            prediction_dir
            / f"{today}.csv",
            index=False,
        )

        # -----------------------------
        # PREDICTION LEDGER
        # -----------------------------

        ledger_records = []

        for record in records:

            symbol = record["symbol"]

            valid, metadata = (
                validate_prediction(
                    symbol,
                    record,
                )
            )

            if not valid:

                logger.warning(
                    "Skipping invalid "
                    "prediction %s: %s",
                    symbol,
                    metadata,
                )

                continue

            predicted_close = float(
                record["Close"]
            )

            current_close = (
                get_current_close(symbol)
            )

            predicted_return = None
            predicted_direction = None

            if (
                current_close is not None
                and current_close > 0
            ):

                predicted_return = (
                    predicted_close
                    / current_close
                ) - 1

                if predicted_return > 0:

                    predicted_direction = 1

                elif predicted_return < 0:

                    predicted_direction = -1

                else:

                    predicted_direction = 0

            ledger_records.append(
                {
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

                    "confidence": None,

                    "opportunity_score": (
                        record.get("score")
                    ),

                    "data_quality_score": (
                        metadata[
                            "data_quality_score"
                        ]
                    ),
                }
            )

        if ledger_records:

            record_predictions(
                ledger_records
            )

            logger.info(
                "Recorded %s predictions "
                "in prediction ledger",
                len(ledger_records),
            )

        # -----------------------------
        # INDEX PREDICTIONS
        # -----------------------------

        index_predictions = (
            predict_indexes()
        )

        if index_predictions:

            pd.DataFrame(
                index_predictions
            ).to_csv(
                prediction_dir
                / f"{today}_indexes.csv",
                index=False,
            )

        # -----------------------------
        # TELEGRAM MESSAGE
        # -----------------------------

        lines = [
            "*TOP 5 STOCK PREDICTIONS*",
            (
                f"Date: `{today}` | "
                f"`{_uni_label()}`"
            ),
            "",
            "```",
            (
                f"{'Stock':<12} "
                f"{'Open':>8} "
                f"{'High':>8} "
                f"{'Low':>8} "
                f"{'Close':>8}"
            ),
            "-" * 48,
        ]

        for _, row in top5.iterrows():

            symbol = row["symbol"]

            prediction = predictions.get(
                symbol
            )

            if not prediction:

                continue

            name = symbol.replace(
                ".NS",
                "",
            )

            lines.append(
                f"{name:<12} "
                f"{prediction['Open']:>8.2f} "
                f"{prediction['High']:>8.2f} "
                f"{prediction['Low']:>8.2f} "
                f"{prediction['Close']:>8.2f}"
            )

        lines.append(
            "```"
        )

        # -----------------------------
        # INDEXES
        # -----------------------------

        if index_predictions:

            lines += [
                "",
                "*INDEX PREDICTIONS*",
                "```",
                (
                    f"{'Index':<12} "
                    f"{'Open':>8} "
                    f"{'High':>8} "
                    f"{'Low':>8} "
                    f"{'Close':>8}"
                ),
                "-" * 48,
            ]

            for prediction in (
                index_predictions
            ):

                lines.append(
                    f"{prediction['name']:<12} "
                    f"{prediction['Open']:>8.2f} "
                    f"{prediction['High']:>8.2f} "
                    f"{prediction['Low']:>8.2f} "
                    f"{prediction['Close']:>8.2f}"
                )

            lines.append(
                "```"
            )

        # -----------------------------
        # SENTIMENT OUTPUT
        # -----------------------------

        lines += [
            "",
            "*Sentiment*",
        ]

        for _, row in top5.iterrows():

            symbol = row["symbol"]

            sentiment = sentiments.get(
                symbol
            )

            if (
                sentiment
                and sentiment.article_count
            ):

                if (
                    sentiment.overall_score
                    >= 0.15
                ):

                    emoji = "🟢"

                elif (
                    sentiment.overall_score
                    <= -0.15
                ):

                    emoji = "🔴"

                else:

                    emoji = "⚪"

                lines.append(
                    f"{emoji} "
                    f"{symbol.replace('.NS', '')}: "
                    f"`{sentiment.overall_score:+.2f}` "
                    f"({sentiment.overall_label})"
                )

        # -----------------------------
        # IPO WATCHLIST
        # -----------------------------

        lines += (
            [""]
            + ipo_watchlist_telegram_lines()
        )

        elapsed = (
            datetime.now()
            - start
        ).seconds

        lines.append(
            f"\n_Finished in {elapsed}s_"
        )

        send_telegram(
            "\n".join(lines)
        )

        # -----------------------------
        # IPO DESK
        # -----------------------------

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

        logger.info(
            "Morning job completed."
        )

    except Exception as error:

        logger.error(
            traceback.format_exc()
        )

        send_telegram(
            "❌ Morning failed\n"
            f"```{str(error)[:700]}```"
        )


if __name__ == "__main__":
    main()

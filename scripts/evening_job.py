#!/usr/bin/env python3

"""Evaluate predictions using the prediction ledger."""

import sys
import logging
import traceback

from pathlib import Path
from datetime import datetime

import pandas as pd


sys.path.insert(
    0,
    str(
        Path(__file__)
        .resolve()
        .parent
        .parent
    ),
)


from src.config import cfg
from src.holidays import is_trading_day
from src.data_loader import get_actual_ohlc
from src.drift import detect_drift
from src.ledger import (
    pending_for_date,
    evaluate_prediction,
)
from src.data_validation import validate_actual
from src.evaluation import metrics_from_ledger
from src.telegram_utils import send_telegram


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


def main():

    start = datetime.now()

    today = start.strftime(
        "%Y-%m-%d"
    )

    if not is_trading_day():

        logger.info(
            "Not a trading day."
        )

        return

    try:

        # -----------------------------
        # GET PENDING PREDICTIONS
        # -----------------------------

        predictions = (
            pending_for_date(today)
        )

        if predictions.empty:

            send_telegram(
                "⚠️ Evening: No pending "
                f"ledger predictions for `{today}`"
            )

            return

        evaluated = []

        # -----------------------------
        # FETCH ACTUAL MARKET DATA
        # -----------------------------

        for _, row in (
            predictions.iterrows()
        ):

            symbol = row["symbol"]

            actual = get_actual_ohlc(
                symbol
            )

            if (
                actual is None
                or not validate_actual(actual)
            ):

                logger.warning(
                    "Invalid/no actual data: %s",
                    symbol,
                )

                continue

            result = evaluate_prediction(
                row["prediction_id"],
                actual,
            )

            if result:

                evaluated.append(
                    result
                )

        if not evaluated:

            send_telegram(
                "⚠️ Evening: Could not "
                "evaluate ledger predictions "
                f"on `{today}`"
            )

            return

        # -----------------------------
        # LOAD LEDGER
        # -----------------------------

        ledger_path = Path(
            cfg.paths.ledger_file
        )

        ledger = pd.read_csv(
            ledger_path
        )

        today_records = ledger[
            ledger["market_date"]
            .astype(str)
            == today
        ].copy()

        # -----------------------------
        # CALCULATE METRICS
        # -----------------------------

        metrics = metrics_from_ledger(
            today_records
        )

        metrics_path = Path(
            cfg.paths.metrics_file
        )

        metrics_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metric_row = pd.DataFrame(
            [
                {
                    "date": today,
                    **metrics,
                }
            ]
        )

        # Replace today's metrics if they
        # already exist.
        if metrics_path.exists():

            old_metrics = pd.read_csv(
                metrics_path
            )

            old_metrics = old_metrics[
                old_metrics["date"]
                .astype(str)
                != today
            ]

            metric_row = pd.concat(
                [
                    old_metrics,
                    metric_row,
                ],
                ignore_index=True,
            )

        metric_row.to_csv(
            metrics_path,
            index=False,
        )

        # -----------------------------
        # TELEGRAM REPORT
        # -----------------------------

        lines = [
            "*ACTUAL vs PREDICTED*",
            f"Date: `{today}`",
            "",
            "```",
            (
                f"{'Stock':<12} "
                f"{'Pred C':>9} "
                f"{'Actual':>9} "
                f"{'Err%':>7}"
            ),
            "-" * 43,
        ]

        for result in evaluated:

            symbol = result[
                "symbol"
            ].replace(
                ".NS",
                "",
            )

            lines.append(
                f"{symbol:<12} "
                f"{float(result['predicted_close']):>9.2f} "
                f"{float(result['actual_close']):>9.2f} "
                f"{float(result['abs_error_pct']):>6.2f}%"
            )

        lines.append(
            "```"
        )

        lines += [
            "",
            "*Summary*",
            (
                f"• Stocks: "
                f"`{metrics['n']}`"
            ),
            (
                f"• MAE: "
                f"`{metrics['mae']:.2f}`"
            ),
            (
                f"• MAPE: "
                f"`{metrics['mape']:.2f}%`"
            ),
            (
                f"• Directional Acc: "
                f"`{metrics['directional_accuracy']:.1f}%`"
            ),
        ]

        elapsed = (
            datetime.now()
            - start
        ).seconds

        lines.append(
            f"_Done in {elapsed}s_"
        )

        send_telegram(
            "\n".join(lines)
        )

        # -----------------------------
        # DRIFT DETECTION
        # -----------------------------

        detect_drift()

        logger.info(
            "Evening evaluation completed."
        )

    except Exception as error:

        logger.error(
            traceback.format_exc()
        )

        send_telegram(
            "❌ Evening Failed\n"
            f"`{today}`\n"
            f"```{str(error)[:700]}```"
        )


if __name__ == "__main__":
    main()

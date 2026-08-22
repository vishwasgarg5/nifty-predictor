#!/usr/bin/env python3

"""
Telegram Reporting.

This module sends production stock prediction reports to Telegram.

Supports:

    1. Morning Top-N prediction report.
    2. Next-day predicted OHLC table.
    3. Sentiment, fundamental, and index summaries.
    4. Evening predicted-vs-actual report.
    5. Accuracy metrics.
    6. Retraining recommendation.

The module is intentionally independent from the prediction
pipeline. It receives already prepared prediction and evaluation
data and formats it for Telegram.

Environment variables:

    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID

Optional:

    TELEGRAM_PARSE_MODE=HTML
    TELEGRAM_TIMEOUT=20
"""

from __future__ import annotations

import html
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Iterable

import requests


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(
    "telegram_reporter"
)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_PARSE_MODE = "HTML"

DEFAULT_TIMEOUT = 20

TELEGRAM_MAX_MESSAGE_LENGTH = 4096


# ============================================================
# HELPERS
# ============================================================

def utc_now_iso() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def get_env(
    name: str,
    default: str | None = None,
) -> str | None:
    """Read a non-empty environment variable."""

    value = os.getenv(
        name
    )

    if value is None:

        return default

    value = value.strip()

    if not value:

        return default

    return value


def safe_float(
    value: Any,
    default: float | None = None,
) -> float | None:
    """Convert a value to a finite float."""

    if value is None:

        return default

    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return default

    if not math.isfinite(
        result
    ):

        return default

    return result


def format_number(
    value: Any,
    decimals: int = 2,
    default: str = "-",
) -> str:
    """Format a numeric value safely."""

    number = safe_float(
        value
    )

    if number is None:

        return default

    return (
        f"{number:,.{decimals}f}"
    )


def format_percent(
    value: Any,
    decimals: int = 2,
    default: str = "-",
) -> str:
    """
    Format a percentage.

    Accepts either:

        0.025 -> 2.50%
        2.5   -> 2.50% only when value is explicitly
                 passed through format_percent_points().
    """

    number = safe_float(
        value
    )

    if number is None:

        return default

    return (
        f"{number * 100:.{decimals}f}%"
    )


def format_percent_points(
    value: Any,
    decimals: int = 2,
    default: str = "-",
) -> str:
    """Format an already percentage-based value."""

    number = safe_float(
        value
    )

    if number is None:

        return default

    return (
        f"{number:.{decimals}f}%"
    )


def safe_text(
    value: Any,
    default: str = "-",
) -> str:
    """Convert a value to safe Telegram HTML text."""

    if value is None:

        return default

    text = str(
        value
    ).strip()

    if not text:

        return default

    return html.escape(
        text
    )


def direction_icon(
    direction: Any,
) -> str:
    """Return an icon for a prediction direction."""

    text = str(
        direction or ""
    ).strip().upper()

    if text in {
        "UP",
        "BUY",
        "BULLISH",
        "LONG",
    }:

        return "🟢"

    if text in {
        "DOWN",
        "SELL",
        "BEARISH",
        "SHORT",
    }:

        return "🔴"

    return "🟡"


def risk_icon(
    risk: Any,
) -> str:
    """Return an icon based on risk level."""

    value = safe_float(
        risk
    )

    if value is None:

        return "⚪"

    if value < 0.02:

        return "🟢"

    if value < 0.05:

        return "🟡"

    return "🔴"


# ============================================================
# TELEGRAM CLIENT
# ============================================================

class TelegramReporter:
    """
    Send stock prediction reports to Telegram.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        timeout: int | float | None = None,
        parse_mode: str | None = None,
        enabled: bool = True,
    ) -> None:

        self.bot_token = (
            bot_token
            or get_env(
                "TELEGRAM_BOT_TOKEN"
            )
        )

        self.chat_id = (
            chat_id
            or get_env(
                "TELEGRAM_CHAT_ID"
            )
        )

        self.timeout = float(
            timeout
            if timeout is not None
            else get_env(
                "TELEGRAM_TIMEOUT",
                str(DEFAULT_TIMEOUT),
            )
        )

        self.parse_mode = (
            parse_mode
            or get_env(
                "TELEGRAM_PARSE_MODE",
                DEFAULT_PARSE_MODE,
            )
        )

        self.enabled = bool(
            enabled
        )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    @property
    def is_configured(
        self,
    ) -> bool:
        """Return whether Telegram credentials are configured."""

        return bool(
            self.bot_token
            and self.chat_id
        )

    def get_status(
        self,
    ) -> dict[str, Any]:
        """Return reporter status."""

        return {
            "enabled": self.enabled,
            "configured": self.is_configured,
            "chat_id_configured": bool(
                self.chat_id
            ),
            "bot_token_configured": bool(
                self.bot_token
            ),
            "parse_mode": self.parse_mode,
            "timeout": self.timeout,
        }

    # ========================================================
    # MESSAGE DELIVERY
    # ========================================================

    def _get_api_url(
        self,
    ) -> str:

        if not self.bot_token:

            raise RuntimeError(
                "Telegram bot token is not configured."
            )

        return (
            "https://api.telegram.org/bot"
            f"{self.bot_token}/sendMessage"
        )

    def _split_message(
        self,
        text: str,
    ) -> list[str]:
        """
        Split a long message into Telegram-safe chunks.
        """

        if len(
            text
        ) <= TELEGRAM_MAX_MESSAGE_LENGTH:

            return [
                text
            ]

        chunks: list[str] = []

        current = ""

        for line in text.splitlines(
            keepends=True
        ):

            if (
                len(current)
                + len(line)
                <= TELEGRAM_MAX_MESSAGE_LENGTH
            ):

                current += line

                continue

            if current:

                chunks.append(
                    current
                )

                current = ""

            while (
                len(line)
                > TELEGRAM_MAX_MESSAGE_LENGTH
            ):

                chunks.append(
                    line[
                        :TELEGRAM_MAX_MESSAGE_LENGTH
                    ]
                )

                line = line[
                    TELEGRAM_MAX_MESSAGE_LENGTH:
                ]

            current = line

        if current:

            chunks.append(
                current
            )

        return chunks

    def send_message(
        self,
        text: str,
        disable_notification: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Send text to Telegram.

        Returns the Telegram API response for each message chunk.
        """

        if not self.enabled:

            logger.info(
                "Telegram reporting is disabled."
            )

            return []

        if not self.is_configured:

            raise RuntimeError(
                "Telegram is not configured. "
                "Set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID."
            )

        responses: list[
            dict[str, Any]
        ] = []

        url = self._get_api_url()

        chunks = self._split_message(
            str(text)
        )

        for chunk in chunks:

            payload = {
                "chat_id": self.chat_id,
                "text": chunk,
                "parse_mode": self.parse_mode,
                "disable_notification": (
                    disable_notification
                ),
            }

            try:

                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )

                response.raise_for_status()

                data = response.json()

            except requests.RequestException:

                logger.exception(
                    "Failed to send Telegram message."
                )

                raise

            if not data.get(
                "ok",
                False,
            ):

                raise RuntimeError(
                    "Telegram API returned an error: "
                    f"{data}"
                )

            responses.append(
                data
            )

        logger.info(
            "Sent %s Telegram message chunk(s).",
            len(responses),
        )

        return responses

    # ========================================================
    # PREDICTION NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_predictions(
        predictions: Iterable[
            dict[str, Any]
        ],
    ) -> list[
        dict[str, Any]
    ]:
        """
        Normalize prediction records and sort by opportunity score.

        Expected common fields:

            symbol
            expected_return
            probability_up
            expected_risk
            risk_adjusted_return
            opportunity_score
            confidence
            direction

        Optional predicted OHLC fields:

            predicted_open
            predicted_high
            predicted_low
            predicted_close
        """

        records: list[
            dict[str, Any]
        ] = []

        for item in predictions:

            if not isinstance(
                item,
                dict,
            ):

                continue

            record = dict(
                item
            )

            record.setdefault(
                "symbol",
                "UNKNOWN",
            )

            score = safe_float(
                record.get(
                    "opportunity_score"
                ),
                default=float("-inf"),
            )

            record["_sort_score"] = score

            records.append(
                record
            )

        records.sort(
            key=lambda item: (
                item.get(
                    "_sort_score",
                    float("-inf"),
                ),
                safe_float(
                    item.get(
                        "confidence"
                    ),
                    0.0,
                )
                or 0.0,
            ),
            reverse=True,
        )

        for record in records:

            record.pop(
                "_sort_score",
                None,
            )

        return records

    # ========================================================
    # MORNING REPORT
    # ========================================================

    def build_morning_report(
        self,
        predictions: Iterable[
            dict[str, Any]
        ],
        top_n: int = 5,
        report_date: str | None = None,
        sentiment_summary: dict[
            str,
            Any,
        ] | None = None,
        fundamental_summary: dict[
            str,
            Any,
        ] | None = None,
        index_summary: dict[
            str,
            Any,
        ] | None = None,
    ) -> str:
        """
        Build the morning Top-N stock prediction report.
        """

        records = self.normalize_predictions(
            predictions
        )

        selected = records[
            :max(
                int(top_n),
                1,
            )
        ]

        date_text = (
            report_date
            or datetime.now().strftime(
                "%Y-%m-%d"
            )
        )

        lines = [
            "📈 <b>DAILY STOCK PREDICTIONS</b>",
            "",
            f"📅 Date: <b>{safe_text(date_text)}</b>",
            f"🎯 Selected Stocks: <b>{len(selected)}</b>",
            "",
            "<pre>",
            (
                "SYMBOL      OPEN      HIGH"
                "       LOW     CLOSE"
            ),
            "----------------------------------------",
        ]

        for item in selected:

            symbol = safe_text(
                item.get(
                    "symbol"
                )
            )

            predicted_open = format_number(
                item.get(
                    "predicted_open",
                    item.get("open"),
                )
            )

            predicted_high = format_number(
                item.get(
                    "predicted_high",
                    item.get("high"),
                )
            )

            predicted_low = format_number(
                item.get(
                    "predicted_low",
                    item.get("low"),
                )
            )

            predicted_close = format_number(
                item.get(
                    "predicted_close",
                    item.get(
                        "expected_close"
                    ),
                )
            )

            lines.append(
                f"{symbol:<10} "
                f"{predicted_open:>8} "
                f"{predicted_high:>8} "
                f"{predicted_low:>8} "
                f"{predicted_close:>8}"
            )

        lines.extend(
            [
                "</pre>",
                "",
                "📊 <b>MODEL SIGNALS</b>",
            ]
        )

        for rank, item in enumerate(
            selected,
            start=1,
        ):

            symbol = safe_text(
                item.get(
                    "symbol"
                )
            )

            direction = item.get(
                "direction",
                "NEUTRAL",
            )

            icon = direction_icon(
                direction
            )

            expected_return = format_percent(
                item.get(
                    "expected_return"
                )
            )

            probability_up = format_percent(
                item.get(
                    "probability_up"
                )
            )

            risk = format_percent(
                item.get(
                    "expected_risk"
                )
            )

            confidence = format_percent(
                item.get(
                    "confidence"
                )
            )

            score = format_number(
                item.get(
                    "opportunity_score"
                ),
                decimals=4,
            )

            lines.extend(
                [
                    "",
                    (
                        f"<b>#{rank} {symbol}</b> "
                        f"{icon} "
                        f"<b>{safe_text(direction)}</b>"
                    ),
                    (
                        f"Expected Return: "
                        f"<b>{expected_return}</b>"
                    ),
                    (
                        f"Probability Up: "
                        f"<b>{probability_up}</b>"
                    ),
                    (
                        f"Risk: "
                        f"{risk_icon(item.get('expected_risk'))} "
                        f"<b>{risk}</b>"
                    ),
                    (
                        f"Confidence: "
                        f"<b>{confidence}</b>"
                    ),
                    (
                        f"Opportunity Score: "
                        f"<b>{score}</b>"
                    ),
                ]
            )

        lines.extend(
            self._build_analysis_sections(
                sentiment_summary=(
                    sentiment_summary
                ),
                fundamental_summary=(
                    fundamental_summary
                ),
                index_summary=(
                    index_summary
                ),
            )
        )

        lines.extend(
            [
                "",
                "⚠️ <i>Model-generated predictions are "
                "analytical signals, not guaranteed outcomes "
                "or investment advice.</i>",
                "",
                (
                    "Generated: "
                    f"<code>{utc_now_iso()}</code>"
                ),
            ]
        )

        return "\n".join(
            lines
        )

    def send_morning_report(
        self,
        predictions: Iterable[
            dict[str, Any]
        ],
        top_n: int = 5,
        report_date: str | None = None,
        sentiment_summary: dict[
            str,
            Any,
        ] | None = None,
        fundamental_summary: dict[
            str,
            Any,
        ] | None = None,
        index_summary: dict[
            str,
            Any,
        ] | None = None,
    ) -> list[
        dict[str, Any]
    ]:
        """Build and send the morning prediction report."""

        report = self.build_morning_report(
            predictions=predictions,
            top_n=top_n,
            report_date=report_date,
            sentiment_summary=(
                sentiment_summary
            ),
            fundamental_summary=(
                fundamental_summary
            ),
            index_summary=(
                index_summary
            ),
        )

        return self.send_message(
            report
        )

    # ========================================================
    # ANALYSIS SECTIONS
    # ========================================================

    def _build_analysis_sections(
        self,
        sentiment_summary: dict[
            str,
            Any,
        ] | None,
        fundamental_summary: dict[
            str,
            Any,
        ] | None,
        index_summary: dict[
            str,
            Any,
        ] | None,
    ) -> list[str]:
        """
        Build sentiment, fundamental, and market-index sections.
        """

        lines: list[str] = []

        # ----------------------------------------------------
        # SENTIMENT
        # ----------------------------------------------------

        if sentiment_summary:

            sentiment = safe_text(
                sentiment_summary.get(
                    "sentiment",
                    sentiment_summary.get(
                        "overall_sentiment"
                    ),
                )
            )

            score = format_number(
                sentiment_summary.get(
                    "score",
                    sentiment_summary.get(
                        "sentiment_score"
                    ),
                ),
                decimals=4,
            )

            confidence = format_percent(
                sentiment_summary.get(
                    "confidence"
                )
            )

            article_count = (
                sentiment_summary.get(
                    "article_count",
                    sentiment_summary.get(
                        "news_count",
                        "-",
                    ),
                )
            )

            lines.extend(
                [
                    "",
                    "📰 <b>SENTIMENT ANALYSIS</b>",
                    (
                        f"Overall: "
                        f"<b>{sentiment}</b>"
                    ),
                    (
                        f"Score: "
                        f"<b>{score}</b>"
                    ),
                    (
                        f"Confidence: "
                        f"<b>{confidence}</b>"
                    ),
                    (
                        f"Sources: "
                        f"<b>{safe_text(article_count)}</b>"
                    ),
                ]
            )

        # ----------------------------------------------------
        # FUNDAMENTALS
        # ----------------------------------------------------

        if fundamental_summary:

            valuation = safe_text(
                fundamental_summary.get(
                    "valuation",
                    fundamental_summary.get(
                        "rating"
                    ),
                )
            )

            quality = safe_text(
                fundamental_summary.get(
                    "quality"
                )
            )

            growth = safe_text(
                fundamental_summary.get(
                    "growth"
                )
            )

            fundamental_score = (
                format_number(
                    fundamental_summary.get(
                        "score",
                        fundamental_summary.get(
                            "fundamental_score"
                        ),
                    ),
                    decimals=4,
                )
            )

            lines.extend(
                [
                    "",
                    "🏢 <b>FUNDAMENTAL ANALYSIS</b>",
                    (
                        f"Valuation: "
                        f"<b>{valuation}</b>"
                    ),
                    (
                        f"Quality: "
                        f"<b>{quality}</b>"
                    ),
                    (
                        f"Growth: "
                        f"<b>{growth}</b>"
                    ),
                    (
                        f"Score: "
                        f"<b>{fundamental_score}</b>"
                    ),
                ]
            )

        # ----------------------------------------------------
        # MARKET INDEXES
        # ----------------------------------------------------

        if index_summary:

            lines.extend(
                [
                    "",
                    "🌍 <b>MARKET INDEX SUMMARY</b>",
                ]
            )

            for name, value in (
                index_summary.items()
            ):

                if isinstance(
                    value,
                    dict,
                ):

                    direction = (
                        value.get(
                            "direction",
                            "NEUTRAL",
                        )
                    )

                    change = (
                        value.get(
                            "change_percent",
                            value.get(
                                "return"
                            ),
                        )
                    )

                    if (
                        change is not None
                        and abs(
                            safe_float(
                                change,
                                0.0,
                            )
                            or 0.0
                        ) <= 1
                    ):

                        change_text = (
                            format_percent(
                                change
                            )
                        )

                    else:

                        change_text = (
                            format_percent_points(
                                change
                            )
                        )

                    lines.append(
                        (
                            f"{direction_icon(direction)} "
                            f"<b>{safe_text(name)}</b>: "
                            f"{safe_text(direction)} "
                            f"({change_text})"
                        )
                    )

                else:

                    lines.append(
                        (
                            f"• <b>{safe_text(name)}</b>: "
                            f"{safe_text(value)}"
                        )
                    )

        return lines

    # ========================================================
    # EVENING EVALUATION REPORT
    # ========================================================

    def build_evening_report(
        self,
        comparisons: Iterable[
            dict[str, Any]
        ],
        metrics: dict[
            str,
            Any
        ] | None = None,
        retraining_status: dict[
            str,
            Any
        ] | None = None,
        report_date: str | None = None,
    ) -> str:
        """
        Build predicted-vs-actual Telegram report.

        Comparison fields can include:

            symbol

            predicted_open
            predicted_high
            predicted_low
            predicted_close

            actual_open
            actual_high
            actual_low
            actual_close

            predicted_direction
            actual_direction
        """

        records = [
            dict(item)
            for item in comparisons
            if isinstance(
                item,
                dict,
            )
        ]

        date_text = (
            report_date
            or datetime.now().strftime(
                "%Y-%m-%d"
            )
        )

        lines = [
            "📊 <b>EVENING MODEL EVALUATION</b>",
            "",
            f"📅 Date: <b>{safe_text(date_text)}</b>",
            f"🔎 Stocks Evaluated: <b>{len(records)}</b>",
            "",
            "<pre>",
            (
                "SYMBOL      P.CLOSE    A.CLOSE"
                "      ERROR"
            ),
            "----------------------------------------",
        ]

        for item in records:

            symbol = safe_text(
                item.get(
                    "symbol"
                )
            )

            predicted_close = safe_float(
                item.get(
                    "predicted_close",
                    item.get(
                        "expected_close"
                    ),
                )
            )

            actual_close = safe_float(
                item.get(
                    "actual_close"
                )
            )

            if (
                predicted_close is not None
                and actual_close is not None
            ):

                error = (
                    actual_close
                    - predicted_close
                )

                error_text = (
                    f"{error:+.2f}"
                )

            else:

                error_text = "-"

            lines.append(
                f"{symbol:<10} "
                f"{format_number(predicted_close):>8} "
                f"{format_number(actual_close):>8} "
                f"{error_text:>10}"
            )

        lines.extend(
            [
                "</pre>",
                "",
                "🎯 <b>PERFORMANCE METRICS</b>",
            ]
        )

        if metrics:

            metric_order = [
                (
                    "direction_accuracy",
                    "Direction Accuracy",
                    "percent",
                ),
                (
                    "close_mae",
                    "Close MAE",
                    "number",
                ),
                (
                    "close_rmse",
                    "Close RMSE",
                    "number",
                ),
                (
                    "open_mae",
                    "Open MAE",
                    "number",
                ),
                (
                    "high_mae",
                    "High MAE",
                    "number",
                ),
                (
                    "low_mae",
                    "Low MAE",
                    "number",
                ),
                (
                    "mean_absolute_error",
                    "Mean Absolute Error",
                    "number",
                ),
                (
                    "rmse",
                    "RMSE",
                    "number",
                ),
                (
                    "sample_count",
                    "Samples",
                    "integer",
                ),
            ]

            used_keys: set[str] = set()

            for (
                key,
                label,
                metric_type,
            ) in metric_order:

                if key not in metrics:

                    continue

                used_keys.add(
                    key
                )

                value = metrics.get(
                    key
                )

                if metric_type == "percent":

                    value_text = (
                        format_percent(
                            value
                        )
                    )

                elif metric_type == "integer":

                    value_text = safe_text(
                        value
                    )

                else:

                    value_text = (
                        format_number(
                            value,
                            decimals=4,
                        )
                    )

                lines.append(
                    f"• {label}: "
                    f"<b>{value_text}</b>"
                )

            for key, value in metrics.items():

                if key in used_keys:

                    continue

                lines.append(
                    (
                        f"• {safe_text(key)}: "
                        f"<b>{safe_text(value)}</b>"
                    )
                )

        else:

            lines.append(
                "• No performance metrics available."
            )

        # ----------------------------------------------------
        # RETRAINING STATUS
        # ----------------------------------------------------

        lines.extend(
            [
                "",
                "🔄 <b>RETRAINING STATUS</b>",
            ]
        )

        if retraining_status:

            should_retrain = bool(
                retraining_status.get(
                    "should_retrain",
                    False,
                )
            )

            if should_retrain:

                lines.append(
                    "🔴 <b>RETRAINING RECOMMENDED</b>"
                )

            else:

                lines.append(
                    "🟢 <b>MODEL PERFORMANCE ACCEPTABLE</b>"
                )

            reason = retraining_status.get(
                "reason"
            )

            if reason:

                lines.append(
                    (
                        "Reason: "
                        f"<b>{safe_text(reason)}</b>"
                    )
                )

            triggered_rules = (
                retraining_status.get(
                    "triggered_rules"
                )
            )

            if triggered_rules:

                if isinstance(
                    triggered_rules,
                    (
                        list,
                        tuple,
                    ),
                ):

                    rule_text = (
                        ", ".join(
                            str(rule)
                            for rule in triggered_rules
                        )
                    )

                else:

                    rule_text = str(
                        triggered_rules
                    )

                lines.append(
                    (
                        "Triggered Rules: "
                        f"<b>{safe_text(rule_text)}</b>"
                    )
                )

            candidate_status = (
                retraining_status.get(
                    "candidate_status"
                )
            )

            if candidate_status:

                lines.append(
                    (
                        "Candidate: "
                        f"<b>{safe_text(candidate_status)}</b>"
                    )
                )

        else:

            lines.append(
                "⚪ Retraining status unavailable."
            )

        lines.extend(
            [
                "",
                (
                    "Generated: "
                    f"<code>{utc_now_iso()}</code>"
                ),
            ]
        )

        return "\n".join(
            lines
        )

    def send_evening_report(
        self,
        comparisons: Iterable[
            dict[str, Any]
        ],
        metrics: dict[
            str,
            Any
        ] | None = None,
        retraining_status: dict[
            str,
            Any
        ] | None = None,
        report_date: str | None = None,
    ) -> list[
        dict[str, Any]
    ]:
        """Build and send the evening evaluation report."""

        report = self.build_evening_report(
            comparisons=comparisons,
            metrics=metrics,
            retraining_status=(
                retraining_status
            ),
            report_date=report_date,
        )

        return self.send_message(
            report
        )


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_reporter() -> TelegramReporter:
    """Create a TelegramReporter using environment variables."""

    return TelegramReporter()


def send_morning_predictions(
    predictions: Iterable[
        dict[str, Any]
    ],
    top_n: int = 5,
    report_date: str | None = None,
    sentiment_summary: dict[
        str,
        Any,
    ] | None = None,
    fundamental_summary: dict[
        str,
        Any,
    ] | None = None,
    index_summary: dict[
        str,
        Any,
    ] | None = None,
) -> list[
    dict[str, Any]
]:
    """Convenience function for morning reports."""

    reporter = get_reporter()

    return reporter.send_morning_report(
        predictions=predictions,
        top_n=top_n,
        report_date=report_date,
        sentiment_summary=(
            sentiment_summary
        ),
        fundamental_summary=(
            fundamental_summary
        ),
        index_summary=index_summary,
    )


def send_evening_evaluation(
    comparisons: Iterable[
        dict[str, Any]
    ],
    metrics: dict[
        str,
        Any
    ] | None = None,
    retraining_status: dict[
        str,
        Any
    ] | None = None,
    report_date: str | None = None,
) -> list[
    dict[str, Any]
]:
    """Convenience function for evening evaluation reports."""

    reporter = get_reporter()

    return reporter.send_evening_report(
        comparisons=comparisons,
        metrics=metrics,
        retraining_status=(
            retraining_status
        ),
        report_date=report_date,
    )


# ============================================================
# CLI TEST
# ============================================================

def main() -> int:
    """
    Send a sample Telegram report.

    Run:

        python -m src.telegram_reporter
    """

    reporter = TelegramReporter()

    print()

    print("=" * 70)
    print("TELEGRAM REPORTER")
    print("=" * 70)

    status = reporter.get_status()

    for key, value in status.items():

        print(
            f"{key}: {value}"
        )

    if not reporter.enabled:

        print(
            "\nTelegram reporting is disabled."
        )

        return 0

    if not reporter.is_configured:

        print(
            "\nERROR: Set TELEGRAM_BOT_TOKEN "
            "and TELEGRAM_CHAT_ID."
        )

        return 1

    sample_predictions = [
        {
            "symbol": "RELIANCE",
            "predicted_open": 3000.0,
            "predicted_high": 3050.0,
            "predicted_low": 2970.0,
            "predicted_close": 3035.0,
            "expected_return": 0.012,
            "probability_up": 0.68,
            "expected_risk": 0.018,
            "confidence": 0.81,
            "opportunity_score": 0.7421,
            "direction": "UP",
        },
        {
            "symbol": "TCS",
            "predicted_open": 4100.0,
            "predicted_high": 4170.0,
            "predicted_low": 4075.0,
            "predicted_close": 4150.0,
            "expected_return": 0.009,
            "probability_up": 0.63,
            "expected_risk": 0.016,
            "confidence": 0.77,
            "opportunity_score": 0.6812,
            "direction": "UP",
        },
    ]

    responses = (
        reporter.send_morning_report(
            predictions=sample_predictions,
            top_n=5,
        )
    )

    print(
        f"\nSUCCESS: Sent {len(responses)} "
        "Telegram message(s)."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

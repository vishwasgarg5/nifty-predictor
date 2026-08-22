#!/usr/bin/env python3

"""Morning prediction pipeline.

Phase 3 architecture:

    NSE / Yahoo Data
            │
            ▼
      Data Validation
            │
            ▼
      Market Regime
            │
            ▼
    Technical / Fundamental
            │
            ▼
      Candidate Scoring
            │
            ▼
       Feature Engine
            │
            ▼
       Multi-Model ML
       ├── Return ML
       ├── Direction ML
       └── Risk ML
            │
            ▼
         Ensemble
            │
            ▼
   Opportunity + Confidence
            │
            ▼
        Quality Gate
            │
            ▼
          Top 5
            │
            ▼
         Telegram
            │
            ▼
     Prediction Ledger
"""

from __future__ import annotations

import sys
import logging
import traceback

from pathlib import Path
from datetime import datetime

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.config import cfg

from src.holidays import is_trading_day

from src.universe import get_universe_symbols

from src.scoring import select_top5

from src.model import OHLCPredictor

from src.ml_pipeline import (
    MultiModelPipeline,
)

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


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ============================================================
# HELPERS
# ============================================================

def _uni_label() -> str:
    """Return readable universe label."""

    try:

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

    except Exception:

        pass

    return "NIFTY"


def get_current_close(
    symbol: str,
) -> float | None:
    """Get latest available close."""

    try:

        history = download_history(
            symbol,
            period="10d",
        )

        if (
            history is None
            or history.empty
            or "Close" not in history.columns
        ):

            return None

        close = pd.to_numeric(
            history["Close"],
            errors="coerce",
        ).dropna()

        if close.empty:

            return None

        return float(
            close.iloc[-1]
        )

    except Exception as error:

        logger.warning(
            "Could not get close for %s: %s",
            symbol,
            error,
        )

        return None


def get_market_regime() -> dict:
    """Detect current market regime."""

    try:

        regime_config = getattr(
            cfg,
            "market_regime",
            None,
        )

        symbol = "^NSEI"

        if regime_config:

            symbol = getattr(
                regime_config,
                "symbol",
                "^NSEI",
            )

        history = download_history(
            symbol,
            period="1y",
        )

        regime = detect_market_regime(
            history
        )

        logger.info(
            "Market regime: %s | score %.2f",
            regime.get(
                "regime",
                "UNKNOWN",
            ),
            regime.get(
                "score",
                0.0,
            ),
        )

        return regime

    except Exception as error:

        logger.warning(
            "Market regime failed: %s",
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
    """Build Phase 2 feature metadata."""

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

        return {
            "features": features,
            "quality": float(quality),
        }

    except Exception as error:

        logger.warning(
            "Feature generation failed for %s: %s",
            symbol,
            error,
        )

        return {
            "features": None,
            "quality": 0.0,
        }


def get_ml_prediction(
    symbol: str,
) -> dict | None:
    """Train and run Phase 3 multi-model prediction.

    Current implementation trains using the
    stock's historical data.

    Future Phase 4 can replace this with
    persisted models and scheduled retraining.
    """

    try:

        history = download_history(
            symbol,
            period="2y",
        )

        if (
            history is None
            or history.empty
        ):

            logger.warning(
                "No ML history for %s",
                symbol,
            )

            return None

        pipeline = MultiModelPipeline()

        pipeline.fit(
            history
        )

        prediction = pipeline.predict(
            history
        )

        if prediction:

            logger.info(
                "%s ML | return=%+.4f | "
                "P(up)=%.2f | risk=%.4f | "
                "opp=%.2f | conf=%.2f",
                symbol,
                prediction.get(
                    "expected_return",
                    0.0,
                ),
                prediction.get(
                    "probability_up",
                    0.5,
                ),
                prediction.get(
                    "expected_risk",
                    0.0,
                ),
                prediction.get(
                    "opportunity_score",
                    0.0,
                ),
                prediction.get(
                    "confidence",
                    0.0,
                ),
            )

        return prediction

    except Exception as error:

        logger.warning(
            "ML prediction failed for %s: %s",
            symbol,
            error,
        )

        return None


def calculate_final_score(
    technical_score: float,
    opportunity_score: float,
    confidence: float,
    feature_quality: float,
) -> float:
    """Calculate final Phase 3 ranking score.

    We combine:

    - Existing technical/fundamental score
    - ML opportunity score
    - ML confidence
    - Feature data quality
    """

    technical_score = float(
        technical_score
        if technical_score is not None
        else 0.0
    )

    opportunity_score = float(
        opportunity_score
        if opportunity_score is not None
        else 0.0
    )

    confidence = float(
        confidence
        if confidence is not None
        else 0.0
    )

    feature_quality = float(
        feature_quality
        if feature_quality is not None
        else 0.0
    )

    # Normalize legacy score if it is
    # on a 0-100 scale.
    if technical_score > 1.0:

        technical_score = (
            technical_score / 100.0
        )

    technical_score = max(
        0.0,
        min(
            1.0,
            technical_score,
        ),
    )

    opportunity_score = max(
        0.0,
        min(
            1.0,
            opportunity_score,
        ),
    )

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    feature_quality = max(
        0.0,
        min(
            1.0,
            feature_quality,
        ),
    )

    final_score = (

        0.25
        * technical_score

        + 0.40
        * opportunity_score

        + 0.20
        * confidence

        + 0.15
        * feature_quality
    )

    return round(
        final_score,
        6,
    )


def apply_market_regime_adjustment(
    score: float,
    prediction: dict,
    regime: dict,
) -> float:
    """Adjust final score using market regime."""

    if not prediction:

        return score

    regime_name = str(
        regime.get(
            "regime",
            "UNKNOWN",
        )
    ).upper()

    direction = str(
        prediction.get(
            "direction",
            "NEUTRAL",
        )
    ).upper()

    adjusted = float(score)

    # Bullish market favors UP signals.
    if regime_name in (
        "BULLISH",
        "STRONG_BULLISH",
    ):

        if direction == "UP":

            adjusted *= 1.05

        elif direction == "DOWN":

            adjusted *= 0.90

    # Bearish market penalizes long signals.
    elif regime_name in (
        "BEARISH",
        "STRONG_BEARISH",
    ):

        if direction == "UP":

            adjusted *= 0.90

        elif direction == "DOWN":

            adjusted *= 1.05

    return round(
        max(
            0.0,
            min(
                1.0,
                adjusted,
            ),
        ),
        6,
    )


def quality_gate(
    candidate: dict,
    minimum_feature_quality: float,
) -> bool:
    """Final quality gate before Top 5."""

    prediction = candidate.get(
        "ml_prediction"
    )

    if not prediction:

        return False

    feature_quality = float(
        candidate.get(
            "feature_quality",
            0.0,
        )
    )

    confidence = float(
        prediction.get(
            "confidence",
            0.0,
        )
    )

    opportunity_score = float(
        prediction.get(
            "opportunity_score",
            0.0,
        )
    )

    # Required feature quality.
    if (
        feature_quality
        < minimum_feature_quality
    ):

        return False

    # Reject very weak ML confidence.
    if confidence < 0.20:

        return False

    # Reject very poor opportunity.
    if opportunity_score < 0.20:

        return False

    return True


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    start = datetime.now()

    today = start.strftime(
        "%Y-%m-%d"
    )

    logger.info("=" * 70)

    logger.info(
        "PHASE 3 MORNING JOB STARTED: %s",
        today,
    )

    logger.info("=" * 70)


    # --------------------------------------------------------
    # TRADING DAY CHECK
    # --------------------------------------------------------

    if not is_trading_day():

        logger.info(
            "Not a trading day."
        )

        return


    try:

        # ====================================================
        # STEP 1 — LOAD UNIVERSE
        # ====================================================

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
                "⚠️ Stock universe is empty."
            )

            return


        # ====================================================
        # STEP 2 — MARKET REGIME
        # ====================================================

        market_regime = (
            get_market_regime()
        )


        # ====================================================
        # STEP 3 — INITIAL CANDIDATE SCORING
        # ====================================================

        logger.info(
            "Running candidate selection..."
        )

        initial_limit = int(
            getattr(
                getattr(
                    cfg,
                    "scoring",
                    None,
                ),
                "prefilter_top",
                40,
            )
        )

        candidates = select_top5(
            symbols,
            initial_limit,
        )

        if (
            candidates is None
            or candidates.empty
        ):

            send_telegram(
                f"⚠️ No candidates for `{today}`"
            )

            return

        logger.info(
            "Initial candidates: %s",
            len(candidates),
        )


        # ====================================================
        # STEP 4 — FEATURE ENGINE
        # ====================================================

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

        enriched_candidates = []

        for _, row in candidates.iterrows():

            symbol = row["symbol"]

            metadata = (
                build_stock_features(
                    symbol
                )
            )

            quality = metadata[
                "quality"
            ]

            if quality < minimum_quality:

                logger.info(
                    "%s rejected by feature "
                    "quality: %.2f < %.2f",
                    symbol,
                    quality,
                    minimum_quality,
                )

                continue

            enriched_candidates.append(
                {
                    "symbol": symbol,

                    "technical_score": float(
                        row.get(
                            "score",
                            0.0,
                        )
                    ),

                    "feature_quality": quality,

                    "features": metadata.get(
                        "features"
                    ),
                }
            )


        if not enriched_candidates:

            send_telegram(
                f"⚠️ All candidates failed "
                f"feature quality for `{today}`"
            )

            return


        logger.info(
            "Feature quality survivors: %s",
            len(enriched_candidates),
        )


        # ====================================================
        # STEP 5 — PHASE 3 MULTI-MODEL ML
        # ====================================================

        ml_candidates = []

        for candidate in enriched_candidates:

            symbol = candidate[
                "symbol"
            ]

            prediction = (
                get_ml_prediction(
                    symbol
                )
            )

            if not prediction:

                continue

            candidate[
                "ml_prediction"
            ] = prediction

            candidate[
                "opportunity_score"
            ] = float(
                prediction.get(
                    "opportunity_score",
                    0.0,
                )
            )

            candidate[
                "confidence"
            ] = float(
                prediction.get(
                    "confidence",
                    0.0,
                )
            )

            base_score = (
                calculate_final_score(

                    technical_score=(
                        candidate[
                            "technical_score"
                        ]
                    ),

                    opportunity_score=(
                        candidate[
                            "opportunity_score"
                        ]
                    ),

                    confidence=(
                        candidate[
                            "confidence"
                        ]
                    ),

                    feature_quality=(
                        candidate[
                            "feature_quality"
                        ]
                    ),
                )
            )

            candidate[
                "final_score"
            ] = apply_market_regime_adjustment(

                score=base_score,

                prediction=prediction,

                regime=market_regime,
            )

            # Final quality gate.
            if not quality_gate(
                candidate,
                minimum_quality,
            ):

                logger.info(
                    "%s rejected by "
                    "quality gate",
                    symbol,
                )

                continue

            ml_candidates.append(
                candidate
            )


        if not ml_candidates:

            send_telegram(
                f"⚠️ No ML candidates survived "
                f"quality gate for `{today}`"
            )

            return


        # ====================================================
        # STEP 6 — FINAL RANKING / TOP 5
        # ====================================================

        ml_candidates.sort(
            key=lambda item: item[
                "final_score"
            ],
            reverse=True,
        )

        top_n = int(
            getattr(
                cfg,
                "top_n",
                5,
            )
        )

        final_candidates = (
            ml_candidates[:top_n]
        )

        logger.info(
            "FINAL TOP %s: %s",
            len(final_candidates),
            ", ".join(
                item["symbol"]
                for item
                in final_candidates
            ),
        )


        # ====================================================
        # STEP 7 — OHLC PREDICTIONS
        # ====================================================

        records = []

        for candidate in final_candidates:

            symbol = candidate[
                "symbol"
            ]

            try:

                predictor = OHLCPredictor(
                    symbol
                )

                ohlc = (
                    predictor.predict_next()
                )

                if not ohlc:

                    logger.warning(
                        "No OHLC prediction: %s",
                        symbol,
                    )

                    continue

                ml = candidate[
                    "ml_prediction"
                ]

                record = {

                    "date": today,

                    "symbol": symbol,

                    # Existing score
                    "technical_score": (
                        candidate[
                            "technical_score"
                        ]
                    ),

                    # Phase 2
                    "feature_quality": (
                        candidate[
                            "feature_quality"
                        ]
                    ),

                    # Phase 3
                    "expected_return": (
                        ml.get(
                            "expected_return"
                        )
                    ),

                    "probability_up": (
                        ml.get(
                            "probability_up"
                        )
                    ),

                    "expected_risk": (
                        ml.get(
                            "expected_risk"
                        )
                    ),

                    "risk_adjusted_return": (
                        ml.get(
                            "risk_adjusted_return"
                        )
                    ),

                    "opportunity_score": (
                        ml.get(
                            "opportunity_score"
                        )
                    ),

                    "confidence": (
                        ml.get(
                            "confidence"
                        )
                    ),

                    "direction": (
                        ml.get(
                            "direction"
                        )
                    ),

                    "model_version": (
                        ml.get(
                            "model_version"
                        )
                    ),

                    "training_rows": (
                        ml.get(
                            "training_rows"
                        )
                    ),

                    "feature_version": (
                        ml.get(
                            "feature_version"
                        )
                    ),

                    "market_regime": (
                        market_regime.get(
                            "regime",
                            "UNKNOWN",
                        )
                    ),

                    "final_score": (
                        candidate[
                            "final_score"
                        ]
                    ),

                    **ohlc,
                }

                records.append(
                    record
                )

            except Exception as error:

                logger.warning(
                    "OHLC prediction failed "
                    "for %s: %s",
                    symbol,
                    error,
                )


        if not records:

            send_telegram(
                f"⚠️ No final predictions "
                f"for `{today}`"
            )

            return


        # ====================================================
        # STEP 8 — SAVE DAILY PREDICTIONS
        # ====================================================

        prediction_dir = Path(
            cfg.paths.predictions
        )

        prediction_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

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


        # ====================================================
        # STEP 9 — PREDICTION LEDGER
        # ====================================================

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
                    "Prediction validation "
                    "failed for %s: %s",
                    symbol,
                    validation,
                )

                continue

            current_close = (
                get_current_close(
                    symbol
                )
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

                "predicted_close": float(
                    record["Close"]
                ),

                # Phase 3 outputs
                "expected_return": (
                    record.get(
                        "expected_return"
                    )
                ),

                "probability_up": (
                    record.get(
                        "probability_up"
                    )
                ),

                "expected_risk": (
                    record.get(
                        "expected_risk"
                    )
                ),

                "direction": (
                    record.get(
                        "direction"
                    )
                ),

                "confidence": (
                    record.get(
                        "confidence"
                    )
                ),

                "opportunity_score": (
                    record.get(
                        "opportunity_score"
                    )
                ),

                "final_score": (
                    record.get(
                        "final_score"
                    )
                ),

                "market_regime": (
                    record.get(
                        "market_regime"
                    )
                ),

                "data_quality_score": (
                    record.get(
                        "feature_quality"
                    )
                ),

                "feature_version": (
                    record.get(
                        "feature_version"
                    )
                ),

                "model_version": (
                    record.get(
                        "model_version"
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
                "Ledger records: %s",
                len(ledger_records),
            )


        # ====================================================
        # STEP 10 — SENTIMENT
        # ====================================================

        sentiments = {}

        try:

            engine = (
                get_sentiment_engine()
            )

            for record in records:

                symbol = record[
                    "symbol"
                ]

                try:

                    sentiments[symbol] = (
                        engine.analyze_stock(
                            symbol,
                            cfg.sentiment.max_articles,
                        )
                    )

                except Exception as error:

                    logger.warning(
                        "Sentiment failed for %s: %s",
                        symbol,
                        error,
                    )

        except Exception as error:

            logger.warning(
                "Sentiment engine failed: %s",
                error,
            )


        # ====================================================
        # STEP 11 — INDEX PREDICTIONS
        # ====================================================

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

                index_predictions = (
                    predict_indexes()
                )

        except Exception as error:

            logger.warning(
                "Index predictions failed: %s",
                error,
            )


        # ====================================================
        # STEP 12 — TELEGRAM REPORT
        # ====================================================

        lines = [

            "🚀 *PHASE 3 AI STOCK PREDICTIONS*",

            (
                f"Date: `{today}` | "
                f"`{_uni_label()}`"
            ),

            "",

            (
                "*Market Regime:* "
                f"`{market_regime.get('regime', 'UNKNOWN')}`"
            ),

            "",

            "*TOP OPPORTUNITIES*",

            "",

            "```",

            (
                f"{'Stock':<11} "
                f"{'Dir':<5} "
                f"{'PUp':>5} "
                f"{'Ret%':>7} "
                f"{'Risk%':>7} "
                f"{'Score':>6}"
            ),

            "-" * 52,
        ]


        for record in records:

            symbol = record[
                "symbol"
            ].replace(
                ".NS",
                "",
            )

            direction = str(
                record.get(
                    "direction",
                    "N",
                )
            )

            probability_up = float(
                record.get(
                    "probability_up",
                    0.0,
                )
            )

            expected_return = float(
                record.get(
                    "expected_return",
                    0.0,
                )
            )

            expected_risk = float(
                record.get(
                    "expected_risk",
                    0.0,
                )
            )

            final_score = float(
                record.get(
                    "final_score",
                    0.0,
                )
            )

            lines.append(

                f"{symbol:<11} "

                f"{direction:<5} "

                f"{probability_up:>5.0%} "

                f"{expected_return:>+7.2%} "

                f"{expected_risk:>7.2%} "

                f"{final_score:>6.2f}"

            )


        lines.append(
            "```"
        )


        # ----------------------------------------------------
        # OHLC SECTION
        # ----------------------------------------------------

        lines += [

            "",

            "*OHLC PREDICTIONS*",

            "",

            "```",

            (
                f"{'Stock':<11} "
                f"{'Open':>9} "
                f"{'High':>9} "
                f"{'Low':>9} "
                f"{'Close':>9}"
            ),

            "-" * 52,
        ]


        for record in records:

            symbol = record[
                "symbol"
            ].replace(
                ".NS",
                "",
            )

            lines.append(

                f"{symbol:<11} "

                f"{float(record['Open']):>9.2f} "

                f"{float(record['High']):>9.2f} "

                f"{float(record['Low']):>9.2f} "

                f"{float(record['Close']):>9.2f}"

            )


        lines.append(
            "```"
        )


        # ----------------------------------------------------
        # MODEL CONFIDENCE
        # ----------------------------------------------------

        lines += [

            "",

            "*MODEL CONFIDENCE*",

        ]


        for record in records:

            symbol = record[
                "symbol"
            ].replace(
                ".NS",
                "",
            )

            confidence = float(
                record.get(
                    "confidence",
                    0.0,
                )
            )

            if confidence >= 0.75:

                emoji = "🟢"

            elif confidence >= 0.50:

                emoji = "🟡"

            else:

                emoji = "🔴"

            lines.append(

                f"{emoji} "

                f"{symbol}: "

                f"`{confidence:.0%}` "

                f"| Feature: "

                f"`{float(record.get('feature_quality', 0)):.0%}`"

            )


        # ----------------------------------------------------
        # SENTIMENT
        # ----------------------------------------------------

        if sentiments:

            lines += [

                "",

                "*NEWS SENTIMENT*",

            ]

            for record in records:

                symbol = record[
                    "symbol"
                ]

                sentiment = sentiments.get(
                    symbol
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


        # ----------------------------------------------------
        # IPO WATCHLIST
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # EXECUTION TIME
        # ----------------------------------------------------

        elapsed = int(
            (
                datetime.now()
                - start
            ).total_seconds()
        )

        lines += [

            "",

            (
                f"_Completed in "
                f"{elapsed}s_"
            ),
        ]


        # ====================================================
        # SEND TELEGRAM
        # ====================================================

        send_telegram(
            "\n".join(lines)
        )


        # ====================================================
        # COMPLETED
        # ====================================================

        logger.info("=" * 70)

        logger.info(
            "PHASE 3 MORNING JOB COMPLETED"
        )

        logger.info("=" * 70)


    except Exception as error:

        logger.error(
            traceback.format_exc()
        )

        try:

            send_telegram(

                "❌ *Phase 3 Morning Job Failed*\n"

                f"Date: `{today}`\n"

                f"```{str(error)[:700]}```"

            )

        except Exception:

            pass


if __name__ == "__main__":
    main()

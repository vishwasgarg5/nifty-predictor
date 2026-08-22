#!/usr/bin/env python3

"""Morning prediction pipeline.

Phase 3C architecture:

    NSE / Yahoo / News
            │
            ▼
      Data Validation
            │
            ▼
      Market Regime
            │
            ▼
 Technical / Fundamental Scoring
            │
            ▼
      Candidate Prefilter
            │
            ▼
       Feature Engine
            │
            ▼
     Load Saved ML Models
      ├── Return Model
      ├── Direction Model
      └── Risk Model
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

from datetime import datetime
from pathlib import Path

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

from src.holidays import (
    is_trading_day,
)

from src.universe import (
    get_universe_symbols,
)

from src.scoring import (
    select_top5,
)

from src.model import (
    OHLCPredictor,
)

from src.ml_pipeline import (
    load_and_predict,
)

from src.model_store import (
    ModelStore,
)

from src.sentiment import (
    get_sentiment_engine,
)

from src.indexes import (
    predict_indexes,
)

from src.ipo import (
    ipo_watchlist_telegram_lines,
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

logger = logging.getLogger(
    __name__
)


# ============================================================
# CONFIG HELPERS
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


def get_model_store() -> ModelStore:
    """Create the persistent model store."""

    try:

        paths = getattr(
            cfg,
            "paths",
            None,
        )

        model_path = getattr(
            paths,
            "models",
            None,
        )

        if model_path:

            return ModelStore(
                base_path=model_path
            )

    except Exception:

        pass

    return ModelStore(
        base_path=(
            PROJECT_ROOT
            / "data"
            / "models"
        )
    )


def get_prefilter_limit() -> int:
    """Return the number of candidates before ML filtering."""

    try:

        scoring = getattr(
            cfg,
            "scoring",
            None,
        )

        value = getattr(
            scoring,
            "prefilter_top",
            None,
        )

        if value:

            return int(
                value
            )

    except Exception:

        pass

    return 40


def get_top_n() -> int:
    """Return final number of predictions."""

    try:

        value = getattr(
            cfg,
            "top_n",
            None,
        )

        if value:

            return int(
                value
            )

    except Exception:

        pass

    return 5


def get_minimum_feature_quality() -> float:
    """Return minimum acceptable feature quality."""

    try:

        features = getattr(
            cfg,
            "features",
            None,
        )

        value = getattr(
            features,
            "min_quality_score",
            None,
        )

        if value is not None:

            return float(
                value
            )

    except Exception:

        pass

    return 0.85


# ============================================================
# DATA HELPERS
# ============================================================

def get_current_close(
    symbol: str,
) -> float | None:
    """Get latest available closing price."""

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


# ============================================================
# MARKET REGIME
# ============================================================

def get_market_regime() -> dict:
    """Detect current market regime."""

    try:

        regime_config = getattr(
            cfg,
            "market_regime",
            None,
        )

        symbol = getattr(
            regime_config,
            "symbol",
            "^NSEI",
        )

        history = download_history(
            symbol,
            period="1y",
        )

        if (
            history is None
            or history.empty
        ):

            raise ValueError(
                "No market index history."
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
            float(
                regime.get(
                    "score",
                    0.0,
                )
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


# ============================================================
# FEATURE ENGINE
# ============================================================

def build_stock_features(
    symbol: str,
) -> dict:
    """Build latest features and calculate feature quality."""

    try:

        history = download_history(
            symbol,
            period="1y",
        )

        if (
            history is None
            or history.empty
        ):

            return {
                "features": None,
                "quality": 0.0,
            }

        features = latest_features(
            history
        )

        if not features:

            return {
                "features": None,
                "quality": 0.0,
            }

        quality = feature_quality_score(
            features
        )

        return {
            "features": features,
            "quality": float(
                quality
            ),
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


# ============================================================
# SAVED MODEL INFERENCE
# ============================================================

def get_ml_prediction(
    symbol: str,
    store: ModelStore,
) -> dict | None:
    """Load saved model and generate prediction."""

    try:

        if not store.exists(
            symbol
        ):

            logger.warning(
                "No saved model found for %s",
                symbol,
            )

            return None

        history = download_history(
            symbol,
            period="1y",
        )

        if (
            history is None
            or history.empty
        ):

            logger.warning(
                "No prediction history for %s",
                symbol,
            )

            return None

        prediction = load_and_predict(

            symbol=symbol,

            history=history,

            store=store,
        )

        if prediction:

            logger.info(

                "%s ML INFERENCE | "
                "return=%+.4f | "
                "P(up)=%.2f | "
                "risk=%.4f | "
                "opp=%.2f | "
                "conf=%.2f",

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
            "ML inference failed for %s: %s",
            symbol,
            error,
        )

        return None


# ============================================================
# SCORING
# ============================================================

def calculate_final_score(
    technical_score: float,
    opportunity_score: float,
    confidence: float,
    feature_quality: float,
) -> float:
    """Combine traditional and ML scores."""

    def normalize(
        value: float,
    ) -> float:

        value = float(
            value
            if value is not None
            else 0.0
        )

        if value > 1.0:

            value = value / 100.0

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    technical_score = normalize(
        technical_score
    )

    opportunity_score = normalize(
        opportunity_score
    )

    confidence = normalize(
        confidence
    )

    feature_quality = normalize(
        feature_quality
    )

    score = (

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
        score,
        6,
    )


def apply_market_regime_adjustment(
    score: float,
    prediction: dict,
    regime: dict,
) -> float:
    """Adjust score based on market regime."""

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

    adjusted = float(
        score
    )

    if regime_name in (
        "BULLISH",
        "STRONG_BULLISH",
    ):

        if direction == "UP":

            adjusted *= 1.05

        elif direction == "DOWN":

            adjusted *= 0.90

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


# ============================================================
# QUALITY GATE
# ============================================================

def quality_gate(
    candidate: dict,
    minimum_feature_quality: float,
) -> bool:
    """Reject weak candidates before final ranking."""

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

    if (
        feature_quality
        < minimum_feature_quality
    ):

        return False

    if confidence < 0.20:

        return False

    if opportunity_score < 0.20:

        return False

    return True


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    start = datetime.now()

    today = start.strftime(
        "%Y-%m-%d"
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        "PHASE 3C MORNING JOB STARTED: %s",
        today,
    )

    logger.info(
        "=" * 70
    )


    # ========================================================
    # TRADING DAY CHECK
    # ========================================================

    if not is_trading_day():

        logger.info(
            "Not a trading day."
        )

        return 0


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

        if not symbols:

            logger.error(
                "Stock universe is empty."
            )

            return 1

        logger.info(
            "Universe size: %s",
            len(symbols),
        )


        # ====================================================
        # STEP 2 — MARKET REGIME
        # ====================================================

        market_regime = (
            get_market_regime()
        )


        # ====================================================
        # STEP 3 — MODEL STORE
        # ====================================================

        model_store = (
            get_model_store()
        )

        saved_symbols = (
            model_store.list_symbols()
        )

        logger.info(
            "Model store: %s",
            model_store.base_path,
        )

        logger.info(
            "Saved models available: %s",
            len(saved_symbols),
        )

        if not saved_symbols:

            logger.warning(
                "No saved models found. "
                "Run scripts/train_models.py first."
            )

            send_telegram(
                "⚠️ *No trained ML models found.*\n"
                "Run `scripts/train_models.py` first."
            )

            return 1


        # ====================================================
        # STEP 4 — INITIAL CANDIDATE SELECTION
        # ====================================================

        prefilter_limit = (
            get_prefilter_limit()
        )

        logger.info(
            "Running initial candidate scoring..."
        )

        candidates = select_top5(
            symbols,
            prefilter_limit,
        )

        if (
            candidates is None
            or candidates.empty
        ):

            logger.warning(
                "No candidates found."
            )

            send_telegram(
                f"⚠️ No candidates for `{today}`"
            )

            return 0


        logger.info(
            "Initial candidates: %s",
            len(candidates),
        )


        # ====================================================
        # STEP 5 — FEATURE QUALITY FILTER
        # ====================================================

        minimum_quality = (
            get_minimum_feature_quality()
        )

        enriched_candidates: list[
            dict
        ] = []

        for _, row in candidates.iterrows():

            symbol = row.get(
                "symbol"
            )

            if not symbol:

                continue

            # Only evaluate stocks with
            # trained models.
            if not model_store.exists(
                symbol
            ):

                logger.info(
                    "%s skipped: no saved model",
                    symbol,
                )

                continue

            metadata = (
                build_stock_features(
                    symbol
                )
            )

            quality = metadata.get(
                "quality",
                0.0,
            )

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

                    "feature_quality": (
                        quality
                    ),

                    "features": (
                        metadata.get(
                            "features"
                        )
                    ),
                }
            )


        if not enriched_candidates:

            logger.warning(
                "No candidates passed feature quality."
            )

            send_telegram(
                f"⚠️ All candidates failed "
                f"feature quality for `{today}`"
            )

            return 0


        logger.info(
            "Feature quality survivors: %s",
            len(enriched_candidates),
        )


        # ====================================================
        # STEP 6 — SAVED MODEL INFERENCE
        # ====================================================

        ml_candidates: list[
            dict
        ] = []

        for candidate in enriched_candidates:

            symbol = candidate[
                "symbol"
            ]

            prediction = (
                get_ml_prediction(

                    symbol=symbol,

                    store=model_store,
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

            logger.warning(
                "No ML candidates survived."
            )

            send_telegram(
                f"⚠️ No ML candidates survived "
                f"quality gate for `{today}`"
            )

            return 0


        # ====================================================
        # STEP 7 — FINAL RANKING
        # ====================================================

        ml_candidates.sort(

            key=lambda item: item[
                "final_score"
            ],

            reverse=True,
        )

        top_n = get_top_n()

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
        # STEP 8 — OHLC PREDICTIONS
        # ====================================================

        records: list[
            dict
        ] = []

        for candidate in final_candidates:

            symbol = candidate[
                "symbol"
            ]

            try:

                predictor = (
                    OHLCPredictor(
                        symbol
                    )
                )

                ohlc = (
                    predictor.predict_next()
                )

                if not ohlc:

                    logger.warning(
                        "No OHLC prediction for %s",
                        symbol,
                    )

                    continue

                ml = candidate[
                    "ml_prediction"
                ]

                record = {

                    "date": today,

                    "symbol": symbol,

                    # Traditional score
                    "technical_score": (
                        candidate[
                            "technical_score"
                        ]
                    ),

                    # Feature quality
                    "feature_quality": (
                        candidate[
                            "feature_quality"
                        ]
                    ),

                    # Multi-model outputs
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

                    "model_saved_at": (
                        ml.get(
                            "model_saved_at"
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

                    # OHLC prediction
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

            logger.warning(
                "No final predictions created."
            )

            send_telegram(
                f"⚠️ No final predictions "
                f"for `{today}`"
            )

            return 0


        # ====================================================
        # STEP 9 — SAVE DAILY PREDICTIONS
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
        # STEP 10 — PREDICTION LEDGER
        # ====================================================

        ledger_records: list[
            dict
        ] = []

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
                    "Prediction validation failed "
                    "for %s: %s",
                    symbol,
                    validation,
                )

                continue

            current_close = (
                get_current_close(
                    symbol
                )
            )

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

                    "predicted_close": float(
                        record["Close"]
                    ),

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
            )


        if ledger_records:

            record_predictions(
                ledger_records
            )

            logger.info(
                "Ledger records saved: %s",
                len(ledger_records),
            )


        # ====================================================
        # STEP 11 — NEWS SENTIMENT
        # ====================================================

        sentiments: dict = {}

        try:

            engine = (
                get_sentiment_engine()
            )

            max_articles = getattr(
                cfg.sentiment,
                "max_articles",
                10,
            )

            for record in records:

                symbol = record[
                    "symbol"
                ]

                try:

                    sentiments[symbol] = (
                        engine.analyze_stock(
                            symbol,
                            max_articles,
                        )
                    )

                except Exception as error:

                    logger.warning(
                        "Sentiment failed "
                        "for %s: %s",
                        symbol,
                        error,
                    )

        except Exception as error:

            logger.warning(
                "Sentiment engine failed: %s",
                error,
            )


        # ====================================================
        # STEP 12 — INDEX PREDICTIONS
        # ====================================================

        try:

            if (
                getattr(
                    cfg,
                    "indexes",
                    None,
                )
                and getattr(
                    cfg.indexes,
                    "enabled",
                    False,
                )
            ):

                predict_indexes()

        except Exception as error:

            logger.warning(
                "Index prediction failed: %s",
                error,
            )


        # ====================================================
        # STEP 13 — TELEGRAM REPORT
        # ====================================================

        lines = [

            "🚀 *AI STOCK PREDICTIONS*",

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

            symbol = str(
                record["symbol"]
            ).replace(
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


        # ====================================================
        # OHLC SECTION
        # ====================================================

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

            symbol = str(
                record["symbol"]
            ).replace(
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


        # ====================================================
        # MODEL CONFIDENCE
        # ====================================================

        lines += [

            "",

            "*MODEL CONFIDENCE*",

        ]


        for record in records:

            symbol = str(
                record["symbol"]
            ).replace(
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

            feature_quality = float(
                record.get(
                    "feature_quality",
                    0.0,
                )
            )

            lines.append(

                f"{emoji} {symbol}: "
                f"`{confidence:.0%}` | "
                f"Feature: `{feature_quality:.0%}`"
            )


        # ====================================================
        # SENTIMENT SECTION
        # ====================================================

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

                    if not sentiment.article_count:

                        continue

                    score = float(
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
                        f"{str(symbol).replace('.NS', '')}: "
                        f"`{score:+.2f}` "
                        f"({sentiment.overall_label})"
                    )

                except Exception:

                    continue


        # ====================================================
        # IPO WATCHLIST
        # ====================================================

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


        # ====================================================
        # COMPLETION TIME
        # ====================================================

        elapsed = int(
            (
                datetime.now()
                - start
            ).total_seconds()
        )

        lines += [

            "",

            f"_Completed in {elapsed}s_",
        ]


        # ====================================================
        # SEND TELEGRAM
        # ====================================================

        send_telegram(
            "\n".join(
                lines
            )
        )


        logger.info(
            "=" * 70
        )

        logger.info(
            "PHASE 3C MORNING JOB COMPLETED"
        )

        logger.info(
            "=" * 70
        )

        return 0


    except Exception as error:

        logger.error(
            traceback.format_exc()
        )

        try:

            send_telegram(

                "❌ *Phase 3C Morning Job Failed*\n"

                f"Date: `{today}`\n"

                f"```{str(error)[:700]}```"
            )

        except Exception:

            pass

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

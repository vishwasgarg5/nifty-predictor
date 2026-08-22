# src/model.py

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from src.config import cfg
from src.data_loader import download_history
from src.features import create_features


logger = logging.getLogger(__name__)


class OHLCPredictor:
    TARGET_COLS = [
        "target_o",
        "target_h",
        "target_l",
        "target_c",
    ]

    def __init__(self, symbol: str):
        self.symbol = symbol

        clean_symbol = (
            symbol.replace(".NS", "")
            .replace(".BO", "")
            .replace("/", "_")
            .replace("\\", "_")
        )

        self.model_path = (
            Path(cfg.paths.models) /
            f"{clean_symbol}.joblib"
        )

        self.params_path = (
            Path(cfg.paths.models) /
            f"{clean_symbol}_params.joblib"
        )

        self.model = None

        self.best_params = {
            "n_estimators": 120,
            "max_depth": 4,
            "learning_rate": 0.05,
        }

    # ================================================================
    # BUILD MODEL
    # ================================================================

    def _build(self, params=None):

        p = params or self.best_params

        base_model = XGBRegressor(
            n_estimators=int(p["n_estimators"]),
            max_depth=int(p["max_depth"]),
            learning_rate=float(p["learning_rate"]),
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            n_jobs=2,
            random_state=42,
        )

        return MultiOutputRegressor(base_model)

    # ================================================================
    # LOAD MODEL
    # ================================================================

    def load(self):

        if not self.model_path.exists():
            logger.info(
                f"{self.symbol}: model not found"
            )
            return False

        try:

            self.model = joblib.load(
                self.model_path
            )

            if self.params_path.exists():
                self.best_params = joblib.load(
                    self.params_path
                )

            logger.info(
                f"{self.symbol}: model loaded"
            )

            return True

        except Exception as e:

            logger.error(
                f"{self.symbol}: model loading failed: {e}"
            )

            self.model = None

            return False

    # ================================================================
    # SAMPLE WEIGHTS
    # ================================================================

    def _weights(self, n):

        if n <= 0:
            return np.array([])

        return np.linspace(
            0.6,
            1.4,
            n
        )

    # ================================================================
    # HYPERPARAMETER TUNING
    # ================================================================

    def tune_hyperparameters(self, X, y):

        grid = [
            {
                "n_estimators": 80,
                "max_depth": 3,
                "learning_rate": 0.08,
            },
            {
                "n_estimators": 120,
                "max_depth": 4,
                "learning_rate": 0.05,
            },
            {
                "n_estimators": 160,
                "max_depth": 5,
                "learning_rate": 0.04,
            },
        ]

        if len(X) < 50:
            return self.best_params

        tscv = TimeSeriesSplit(
            n_splits=3
        )

        best_score = float("inf")
        best_params = self.best_params

        for params in grid:

            scores = []

            try:

                for train_idx, test_idx in tscv.split(X):

                    model = self._build(params)

                    model.fit(
                        X.iloc[train_idx],
                        y.iloc[train_idx],
                    )

                    prediction = model.predict(
                        X.iloc[test_idx]
                    )

                    actual_close = (
                        y.iloc[test_idx]["target_c"]
                    )

                    predicted_close = (
                        prediction[:, 3]
                    )

                    score = mean_absolute_percentage_error(
                        actual_close,
                        predicted_close,
                    )

                    scores.append(score)

                if not scores:
                    continue

                avg_score = float(
                    np.mean(scores)
                )

                logger.info(
                    f"{self.symbol}: "
                    f"{params} -> "
                    f"Close MAPE={avg_score:.6f}"
                )

                if avg_score < best_score:

                    best_score = avg_score
                    best_params = params

            except Exception as e:

                logger.warning(
                    f"{self.symbol}: "
                    f"tuning failed: {e}"
                )

        return best_params

    # ================================================================
    # TRAIN
    # ================================================================

    def train(
        self,
        use_error_weights=True,
        do_tune=False,
    ):

        try:

            hist = download_history(
                self.symbol,
                period=f"{cfg.lookback_days}d",
            )

            if hist is None or hist.empty:

                logger.warning(
                    f"{self.symbol}: "
                    f"no historical data"
                )

                return False

            if len(hist) < cfg.min_history_days:

                logger.warning(
                    f"{self.symbol}: "
                    f"insufficient history: "
                    f"{len(hist)} rows"
                )

                return False

            # --------------------------------------------------------
            # TRAINING FEATURES
            # --------------------------------------------------------

            feat = create_features(hist)

            if feat is None or feat.empty:

                logger.warning(
                    f"{self.symbol}: "
                    f"feature dataframe empty"
                )

                return False

            # --------------------------------------------------------
            # TARGET CHECK
            # --------------------------------------------------------

            missing_targets = [
                col
                for col in self.TARGET_COLS
                if col not in feat.columns
            ]

            if missing_targets:

                logger.error(
                    f"{self.symbol}: "
                    f"missing targets: "
                    f"{missing_targets}"
                )

                return False

            # --------------------------------------------------------
            # FEATURES
            # --------------------------------------------------------

            feature_cols = [
                col
                for col in feat.columns
                if col not in self.TARGET_COLS
            ]

            X = feat[
                feature_cols
            ].copy()

            y = feat[
                self.TARGET_COLS
            ].copy()

            # --------------------------------------------------------
            # NUMERIC CLEANUP
            # --------------------------------------------------------

            X = X.apply(
                pd.to_numeric,
                errors="coerce",
            )

            y = y.apply(
                pd.to_numeric,
                errors="coerce",
            )

            valid = (
                X.notna().all(axis=1)
                &
                y.notna().all(axis=1)
            )

            X = X.loc[
                valid
            ].reset_index(drop=True)

            y = y.loc[
                valid
            ].reset_index(drop=True)

            if len(X) < 40:

                logger.warning(
                    f"{self.symbol}: "
                    f"not enough clean rows: "
                    f"{len(X)}"
                )

                return False

            # --------------------------------------------------------
            # OPTIONAL TUNING
            # --------------------------------------------------------

            if do_tune:

                self.best_params = (
                    self.tune_hyperparameters(
                        X,
                        y
                    )
                )

                self.params_path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                joblib.dump(
                    self.best_params,
                    self.params_path
                )

            # --------------------------------------------------------
            # MODEL
            # --------------------------------------------------------

            self.model = self._build(
                self.best_params
            )

            # --------------------------------------------------------
            # WEIGHTS
            # --------------------------------------------------------

            if use_error_weights:

                weights = self._weights(
                    len(X)
                )

            else:

                weights = np.ones(
                    len(X)
                )

            # --------------------------------------------------------
            # TRAIN
            # --------------------------------------------------------

            self.model.fit(
                X,
                y,
                sample_weight=weights,
            )

            # --------------------------------------------------------
            # SAVE
            # --------------------------------------------------------

            self.model_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            joblib.dump(
                self.model,
                self.model_path,
            )

            logger.info(
                f"{self.symbol}: "
                f"model trained successfully "
                f"using {len(X)} rows"
            )

            return True

        except Exception as e:

            logger.exception(
                f"{self.symbol}: "
                f"training failed: {e}"
            )

            self.model = None

            return False

    # ================================================================
    # LATEST FEATURE ROW
    # ================================================================

    def _latest_features(self, hist):

        if hist is None or hist.empty:
            return None, None

        df = hist.copy()

        # Handle yfinance MultiIndex columns.
        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        missing = [
            col
            for col in required
            if col not in df.columns
        ]

        if missing:

            logger.error(
                f"{self.symbol}: "
                f"missing columns: {missing}"
            )

            return None, None

        df = df[
            required
        ].copy()

        df = df.apply(
            pd.to_numeric,
            errors="coerce",
        )

        df = df.dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close",
            ]
        )

        if len(df) < 60:

            logger.warning(
                f"{self.symbol}: "
                f"not enough rows: {len(df)}"
            )

            return None, None

        # ------------------------------------------------------------
        # IMPORTANT
        #
        # We create features WITHOUT creating the next-day target.
        #
        # This preserves the latest trading-day row.
        # ------------------------------------------------------------

        try:

            from ta.trend import (
                SMAIndicator,
                EMAIndicator,
                MACD,
            )

            from ta.momentum import (
                RSIIndicator,
            )

            from ta.volatility import (
                BollingerBands,
                AverageTrueRange,
            )

            from ta.volume import (
                OnBalanceVolumeIndicator,
            )

            df["SMA20"] = (
                SMAIndicator(
                    close=df["Close"],
                    window=20,
                ).sma_indicator()
            )

            df["EMA20"] = (
                EMAIndicator(
                    close=df["Close"],
                    window=20,
                ).ema_indicator()
            )

            df["RSI"] = (
                RSIIndicator(
                    close=df["Close"],
                    window=14,
                ).rsi()
            )

            df["MACD"] = (
                MACD(
                    close=df["Close"]
                ).macd()
            )

            bb = BollingerBands(
                close=df["Close"],
                window=20,
            )

            df["BB_H"] = (
                bb.bollinger_hband()
            )

            df["BB_L"] = (
                bb.bollinger_lband()
            )

            df["ATR"] = (
                AverageTrueRange(
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    window=14,
                ).average_true_range()
            )

            df["OBV"] = (
                OnBalanceVolumeIndicator(
                    close=df["Close"],
                    volume=df["Volume"],
                ).on_balance_volume()
            )

            df["Close_Lag1"] = (
                df["Close"].shift(1)
            )

            df["Close_Lag2"] = (
                df["Close"].shift(2)
            )

            df["Close_Lag3"] = (
                df["Close"].shift(3)
            )

            df["Daily_Return"] = (
                df["Close"].pct_change()
            )

            df["Volatility"] = (
                df["Daily_Return"]
                .rolling(10)
                .std()
            )

            # --------------------------------------------------------
            # GET LATEST ROW
            # --------------------------------------------------------

            latest = df.iloc[[-1]].copy()

            latest_date = df.index[-1]

            # --------------------------------------------------------
            # GET EXACT TRAINING FEATURE COLUMNS
            # --------------------------------------------------------

            if self.model is not None:

                try:

                    estimators = (
                        self.model.estimators_
                    )

                    if estimators:

                        expected_cols = (
                            getattr(
                                estimators[0],
                                "feature_names_in_",
                                None,
                            )
                        )

                        if expected_cols is not None:

                            expected_cols = list(
                                expected_cols
                            )

                            missing_features = [
                                c
                                for c in expected_cols
                                if c not in latest.columns
                            ]

                            if missing_features:

                                logger.error(
                                    f"{self.symbol}: "
                                    f"missing model features: "
                                    f"{missing_features}"
                                )

                                return None, None

                            latest = latest[
                                expected_cols
                            ]

                except Exception as e:

                    logger.debug(
                        f"{self.symbol}: "
                        f"could not read model feature names: "
                        f"{e}"
                    )

            else:

                # If model has not been loaded yet,
                # load it before prediction.
                self.load()

                if self.model is not None:

                    try:

                        expected_cols = (
                            self.model.estimators_[0]
                            .feature_names_in_
                        )

                        expected_cols = list(
                            expected_cols
                        )

                        latest = latest[
                            expected_cols
                        ]

                    except Exception:
                        pass

            # --------------------------------------------------------
            # CLEAN
            # --------------------------------------------------------

            latest = latest.apply(
                pd.to_numeric,
                errors="coerce",
            )

            if latest.isna().any().any():

                bad = latest.columns[
                    latest.isna().any()
                ].tolist()

                logger.error(
                    f"{self.symbol}: "
                    f"NaN in latest features: "
                    f"{bad}"
                )

                return None, None

            return (
                latest,
                latest_date,
            )

        except Exception as e:

            logger.exception(
                f"{self.symbol}: "
                f"latest feature creation failed: {e}"
            )

            return None, None

    # ================================================================
    # PREDICT
    # ================================================================

    def predict_next(self):

        try:

            # --------------------------------------------------------
            # GET LATEST MARKET DATA
            # --------------------------------------------------------

            hist = download_history(
                self.symbol,
                period="6mo",
            )

            if hist is None or hist.empty:

                logger.warning(
                    f"{self.symbol}: "
                    f"no data available"
                )

                return None

            # --------------------------------------------------------
            # LOAD MODEL FIRST
            # --------------------------------------------------------

            if self.model is None:

                loaded = self.load()

                if not loaded:

                    logger.info(
                        f"{self.symbol}: "
                        f"training model before prediction"
                    )

                    trained = self.train()

                    if not trained:

                        logger.error(
                            f"{self.symbol}: "
                            f"training failed"
                        )

                        return None

            # --------------------------------------------------------
            # BUILD LATEST FEATURES
            #
            # IMPORTANT:
            # This uses the latest actual trading day.
            # --------------------------------------------------------

            latest, latest_date = (
                self._latest_features(
                    hist
                )
            )

            if latest is None:

                logger.error(
                    f"{self.symbol}: "
                    f"could not create latest features"
                )

                return None

            # --------------------------------------------------------
            # PREDICT
            # --------------------------------------------------------

            prediction = (
                self.model.predict(
                    latest
                )
            )

            if prediction is None:
                return None

            prediction = np.asarray(
                prediction
            )

            if prediction.ndim != 2:
                return None

            if prediction.shape[1] != 4:

                logger.error(
                    f"{self.symbol}: "
                    f"unexpected prediction shape: "
                    f"{prediction.shape}"
                )

                return None

            pred_open = float(
                prediction[0, 0]
            )

            pred_high = float(
                prediction[0, 1]
            )

            pred_low = float(
                prediction[0, 2]
            )

            pred_close = float(
                prediction[0, 3]
            )

            # --------------------------------------------------------
            # SANITY CHECK
            # --------------------------------------------------------

            values = [
                pred_open,
                pred_high,
                pred_low,
                pred_close,
            ]

            if not all(
                np.isfinite(v)
                for v in values
            ):

                logger.error(
                    f"{self.symbol}: "
                    f"invalid prediction values"
                )

                return None

            # High must be highest.
            pred_high = max(
                pred_high,
                pred_open,
                pred_close,
            )

            # Low must be lowest.
            pred_low = min(
                pred_low,
                pred_open,
                pred_close,
            )

            # --------------------------------------------------------
            # CURRENT PRICE
            # --------------------------------------------------------

            current_close = float(
                hist["Close"].iloc[-1]
            )

            expected_return = (
                (
                    pred_close
                    - current_close
                )
                / current_close
                * 100
            )

            # --------------------------------------------------------
            # RESULT
            # --------------------------------------------------------

            result = {

                "symbol": self.symbol,

                "latest_date": str(
                    latest_date
                ),

                "current_close": round(
                    current_close,
                    2,
                ),

                "Open": round(
                    pred_open,
                    2,
                ),

                "High": round(
                    pred_high,
                    2,
                ),

                "Low": round(
                    pred_low,
                    2,
                ),

                "Close": round(
                    pred_close,
                    2,
                ),

                "expected_return": round(
                    expected_return,
                    4,
                ),

            }

            logger.info(
                f"{self.symbol}: "
                f"Latest={latest_date} "
                f"Current={current_close:.2f} "
                f"Prediction="
                f"O:{pred_open:.2f} "
                f"H:{pred_high:.2f} "
                f"L:{pred_low:.2f} "
                f"C:{pred_close:.2f} "
                f"Return:{expected_return:.2f}%"
            )

            return result

        except Exception as e:

            logger.exception(
                f"{self.symbol}: "
                f"prediction failed: {e}"
            )

            return None

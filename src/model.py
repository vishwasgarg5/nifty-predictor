# src/model.py
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from src.config import cfg
from src.features import create_features
from src.data_loader import download_history

logger = logging.getLogger(__name__)


class OHLCPredictor:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model_path = Path(cfg.paths.models) / f"{symbol.replace('.NS', '')}.joblib"
        self.model = None

    def _build_model(self):
        return MultiOutputRegressor(
            XGBRegressor(
                n_estimators=cfg.model.n_estimators,
                max_depth=cfg.model.max_depth,
                learning_rate=cfg.model.learning_rate,
                subsample=0.85,
                colsample_bytree=0.85,
                n_jobs=2,
                random_state=42
            )
        )

    def load(self) -> bool:
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                return True
            except Exception as e:
                logger.warning(f"Failed to load model for {self.symbol}: {e}")
        return False

    def _get_sample_weights(self, n_samples: int) -> np.ndarray:
        """
        Create sample weights:
        - Recent days get higher weight
        - Optionally boost days where past errors were high
        """
        # Base: linear recency weight
        weights = np.linspace(0.6, 1.4, n_samples)

        # Optional: boost using error history
        errors_file = Path(cfg.paths.errors_file)
        if errors_file.exists():
            try:
                err_df = pd.read_csv(errors_file)
                symbol_errors = err_df[err_df["symbol"] == self.symbol]
                if not symbol_errors.empty:
                    # Higher weight when model was previously wrong
                    avg_error = symbol_errors["abs_error_pct"].mean()
                    boost = min(1.0 + (avg_error / 10), 1.8)  # cap boost
                    weights = weights * boost
            except Exception:
                pass

        return weights

    def train(self, use_error_weights: bool = True) -> bool:
        """
        Train / retrain the model.
        """
        hist = download_history(self.symbol, period=f"{cfg.lookback_days}d")
        if hist is None or len(hist) < cfg.min_history_days:
            logger.warning(f"{self.symbol}: Not enough data to train")
            return False

        feat = create_features(hist)
        if feat is None or feat.empty or len(feat) < 40:
            logger.warning(f"{self.symbol}: Feature creation failed")
            return False

        feature_cols = [c for c in feat.columns if not c.startswith("target_")]
        X = feat[feature_cols]
        y = feat[["target_o", "target_h", "target_l", "target_c"]]

        self.model = self._build_model()

        if use_error_weights:
            sample_weights = self._get_sample_weights(len(X))
            self.model.fit(X, y, sample_weight=sample_weights)
        else:
            # Simple recency weighting
            weights = np.linspace(0.5, 1.5, len(X))
            self.model.fit(X, y, sample_weight=weights)

        # Save model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"{self.symbol}: Model trained & saved → {self.model_path.name}")
        return True

    def predict_next(self) -> dict | None:
        hist = download_history(self.symbol, period="6mo")
        if hist is None:
            return None

        feat = create_features(hist)
        if feat is None or feat.empty:
            return None

        feature_cols = [c for c in feat.columns if not c.startswith("target_")]
        latest = feat[feature_cols].iloc[[-1]]

        if self.model is None:
            if not self.load():
                success = self.train()
                if not success:
                    return None

        try:
            pred = self.model.predict(latest)[0]
            return {
                "Open": round(float(pred[0]), 2),
                "High": round(float(pred[1]), 2),
                "Low": round(float(pred[2]), 2),
                "Close": round(float(pred[3]), 2)
            }
        except Exception as e:
            logger.error(f"{self.symbol}: Prediction failed → {e}")
            return None

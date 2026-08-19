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
            self.model = joblib.load(self.model_path)
            return True
        return False

    def train(self, sample_weights: np.ndarray = None):
        hist = download_history(self.symbol, period=f"{cfg.lookback_days}d")
        if hist is None or len(hist) < cfg.min_history_days:
            logger.warning(f"Not enough data for {self.symbol}")
            return False

        feat = create_features(hist)
        feature_cols = [c for c in feat.columns if not c.startswith("target_")]
        X = feat[feature_cols]
        y = feat[["target_o", "target_h", "target_l", "target_c"]]

        self.model = self._build_model()

        if sample_weights is not None and len(sample_weights) == len(X):
            self.model.fit(X, y, sample_weight=sample_weights)
        else:
            # Simple recency weighting
            weights = np.linspace(0.5, 1.5, len(X))
            self.model.fit(X, y, sample_weight=weights)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        return True

    def predict_next(self) -> dict | None:
        hist = download_history(self.symbol, period="6mo")
        if hist is None:
            return None

        feat = create_features(hist)
        feature_cols = [c for c in feat.columns if not c.startswith("target_")]
        latest = feat[feature_cols].iloc[[-1]]

        if self.model is None:
            if not self.load():
                self.train()

        pred = self.model.predict(latest)[0]
        return {
            "Open": round(float(pred[0]), 2),
            "High": round(float(pred[1]), 2),
            "Low": round(float(pred[2]), 2),
            "Close": round(float(pred[3]), 2)
        }

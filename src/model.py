import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor

from src.config import cfg
from src.features import create_features
from src.data_loader import download_history

logger = logging.getLogger(__name__)

class OHLCPredictor:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model_path = Path(cfg.paths.models) / f"{symbol.replace('.NS','')}.joblib"
        self.params_path = Path(cfg.paths.models) / f"{symbol.replace('.NS','')}_params.joblib"
        self.model = None
        self.best_params = {
            "n_estimators": cfg.model.n_estimators,
            "max_depth": cfg.model.max_depth,
            "learning_rate": cfg.model.learning_rate,
        }

    def _build(self, params=None):
        p = params or self.best_params
        return MultiOutputRegressor(
            XGBRegressor(
                n_estimators=int(p["n_estimators"]),
                max_depth=int(p["max_depth"]),
                learning_rate=float(p["learning_rate"]),
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
                if self.params_path.exists():
                    self.best_params = joblib.load(self.params_path)
                return True
            except Exception as e:
                logger.warning(f"load failed {self.symbol}: {e}")
        return False

    def _weights(self, n):
        w = np.linspace(0.6, 1.4, n)
        errf = Path(cfg.paths.errors_file)
        if errf.exists():
            try:
                ed = pd.read_csv(errf)
                se = ed[ed["symbol"] == self.symbol]
                if not se.empty:
                    boost = min(1.0 + se["abs_error_pct"].mean() / 12, 1.7)
                    w = w * boost
            except Exception:
                pass
        return w

    def tune_hyperparameters(self, X, y) -> dict:
        """Lightweight grid search (fast for Actions)."""
        grid = [
            {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.08},
            {"n_estimators": 120, "max_depth": 4, "learning_rate": 0.05},
            {"n_estimators": 160, "max_depth": 5, "learning_rate": 0.04},
            {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.06},
        ]
        tscv = TimeSeriesSplit(n_splits=3)
        best_score = 1e9
        best = self.best_params
        for params in grid:
            scores = []
            for tr, te in tscv.split(X):
                m = self._build(params)
                m.fit(X.iloc[tr], y.iloc[tr])
                pred = m.predict(X.iloc[te])
                # MAPE on Close (column 3)
                mape = mean_absolute_percentage_error(y.iloc[te].iloc[:, 3], pred[:, 3])
                scores.append(mape)
            avg = float(np.mean(scores))
            if avg < best_score:
                best_score = avg
                best = params
        logger.info(f"{self.symbol} best params {best} (MAPE {best_score:.4f})")
        return best

    def train(self, use_error_weights=True, do_tune=False) -> bool:
        hist = download_history(self.symbol, period=f"{cfg.lookback_days}d")
        if hist is None or len(hist) < cfg.min_history_days:
            return False
        feat = create_features(hist)
        if feat is None or feat.empty or len(feat) < 40:
            return False
        feature_cols = [c for c in feat.columns if not c.startswith("target_")]
        X = feat[feature_cols]
        y = feat[["target_o", "target_h", "target_l", "target_c"]]

        if do_tune and cfg.tuning.enabled:
            self.best_params = self.tune_hyperparameters(X, y)
            joblib.dump(self.best_params, self.params_path)

        self.model = self._build()
        w = self._weights(len(X)) if use_error_weights else np.linspace(0.5, 1.5, len(X))
        self.model.fit(X, y, sample_weight=w)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"{self.symbol} model saved")
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
        if self.model is None and not self.load():
            if not self.train():
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
            logger.error(f"predict failed {self.symbol}: {e}")
            return None

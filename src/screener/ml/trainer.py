from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from screener.ml.features import FEATURE_NAMES
from screener.ml.labels import get_all_labels


MODEL_PATH = Path(__file__).resolve().parents[3] / "data" / "preference_model.pkl"
MIN_SAMPLES = 30


class PreferenceModel:
    def __init__(self):
        self.model = None
        self.feature_names = FEATURE_NAMES
        self.is_trained = False
        self.metrics: dict[str, Any] = {}

    def train(self) -> dict[str, Any]:
        """Train on all labeled data (1-5 star ratings). Uses regression."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        labels_data = get_all_labels()
        if len(labels_data) < MIN_SAMPLES:
            return {
                "error": f"Need at least {MIN_SAMPLES} labels, have {len(labels_data)}",
                "n_samples": len(labels_data),
            }

        rows = []
        targets = []
        for entry in labels_data:
            features = entry.get("features", {})
            if not features:
                continue
            row = [features.get(name, 0.0) for name in self.feature_names]
            rows.append(row)
            targets.append(entry["label"])

        if len(rows) < MIN_SAMPLES:
            return {
                "error": f"Only {len(rows)} entries have features (need {MIN_SAMPLES})",
                "n_samples": len(rows),
            }

        X = np.array(rows, dtype=float)
        y = np.array(targets, dtype=float)
        X = np.nan_to_num(X, nan=0.0)

        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
        )
        self.model.fit(X, y)

        # Cross-validation (R2 score)
        cv_scores = cross_val_score(self.model, X, y, cv=min(5, len(rows)), scoring="r2")

        importances = dict(zip(self.feature_names, self.model.feature_importances_))
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

        self.is_trained = True
        self.metrics = {
            "r2_score": round(float(cv_scores.mean()), 3),
            "r2_std": round(float(cv_scores.std()), 3),
            "n_samples": len(rows),
            "avg_rating": round(float(y.mean()), 2),
            "feature_importance": importances,
        }

        self.save()
        return self.metrics

    def predict(self, features: dict[str, float]) -> float:
        """Predict a 1-5 star rating for this setup. Returns 0-1 normalized."""
        if not self.is_trained or self.model is None:
            return 0.5
        row = [features.get(name, 0.0) for name in self.feature_names]
        X = np.array([row], dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        pred = float(self.model.predict(X)[0])
        # Normalize 1-5 to 0-1
        return max(0.0, min(1.0, (pred - 1) / 4))

    def save(self) -> None:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "metrics": self.metrics,
            }, f)

    def load(self) -> bool:
        if not MODEL_PATH.exists():
            return False
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.feature_names = data.get("feature_names", FEATURE_NAMES)
            self.metrics = data.get("metrics", {})
            self.is_trained = True
            return True
        except Exception:
            return False

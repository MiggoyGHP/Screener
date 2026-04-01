from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
        """Train on all labeled data. Returns metrics dict."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        labels_data = get_all_labels()
        if len(labels_data) < MIN_SAMPLES:
            return {
                "error": f"Need at least {MIN_SAMPLES} labels, have {len(labels_data)}",
                "n_samples": len(labels_data),
            }

        # Build feature matrix
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
        y = np.array(targets, dtype=int)

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)

        # Train
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
        )
        self.model.fit(X, y)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=min(5, len(set(y))), scoring="accuracy")

        # Feature importance
        importances = dict(zip(self.feature_names, self.model.feature_importances_))
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

        self.is_trained = True
        self.metrics = {
            "accuracy": round(float(cv_scores.mean()) * 100, 1),
            "accuracy_std": round(float(cv_scores.std()) * 100, 1),
            "n_samples": len(rows),
            "n_good": int(sum(y)),
            "n_bad": int(len(y) - sum(y)),
            "feature_importance": importances,
        }

        self.save()
        return self.metrics

    def predict_proba(self, features: dict[str, float]) -> float:
        """Return probability that this setup is 'good' (0.0 to 1.0)."""
        if not self.is_trained or self.model is None:
            return 0.5
        row = [features.get(name, 0.0) for name in self.feature_names]
        X = np.array([row], dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        proba = self.model.predict_proba(X)
        # Find index of class 1
        classes = list(self.model.classes_)
        if 1 in classes:
            return float(proba[0][classes.index(1)])
        return 0.5

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

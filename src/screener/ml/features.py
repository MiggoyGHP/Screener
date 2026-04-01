from __future__ import annotations

from typing import Any

import pandas as pd

from screener.patterns.base import PatternResult


FEATURE_NAMES = [
    "pattern_score",
    "rs_rank",
    "rs_new_high",
    "rs_trending_up",
    "relative_volume",
    "atr_pct",
    "distance_to_pivot_pct",
    # VCP-specific
    "num_contractions",
    "base_depth",
    "volume_declining",
    # Reset-specific
    "pullback_depth_pct",
    # Coil-specific
    "box_range_pct",
    "box_days",
    "atr_ratio",
    "coiling",
    "volume_dry",
    # Reversal-specific
    "decline_pct",
    "trendline_broken",
    "stage_transition",
    "volume_confirmed",
]


def extract_features(
    result: PatternResult,
    indicators: dict[str, Any],
) -> dict[str, float]:
    """Extract a feature vector from a PatternResult + indicators for ML training."""
    meta = result.metadata
    features: dict[str, float] = {}

    # Universal features
    features["pattern_score"] = result.score
    features["rs_rank"] = indicators.get("rs_rank", 50)
    features["rs_new_high"] = float(indicators.get("rs_new_high", False))
    features["rs_trending_up"] = float(indicators.get("rs_trending_up", False))

    rvol = indicators.get("relative_volume")
    if rvol is not None and isinstance(rvol, pd.Series) and len(rvol) > 0:
        features["relative_volume"] = float(rvol.iloc[-1])
    else:
        features["relative_volume"] = 1.0

    atr_pct = indicators.get("atr_pct_14")
    if atr_pct is not None and isinstance(atr_pct, pd.Series) and len(atr_pct) > 0:
        features["atr_pct"] = float(atr_pct.iloc[-1])
    else:
        features["atr_pct"] = 0.0

    # Distance to pivot
    if result.pivot_price and result.pivot_price > 0:
        current = indicators.get("current_price", result.pivot_price)
        features["distance_to_pivot_pct"] = abs(result.pivot_price - current) / result.pivot_price * 100
    else:
        features["distance_to_pivot_pct"] = 0.0

    # Pattern-specific (default to 0 for non-applicable patterns)
    features["num_contractions"] = float(meta.get("num_contractions", 0))
    features["base_depth"] = float(meta.get("base_depth", 0))
    features["volume_declining"] = float(meta.get("volume_declining", False))
    features["pullback_depth_pct"] = float(meta.get("pullback_depth_pct", 0))
    features["box_range_pct"] = float(meta.get("box_range_pct", 0))
    features["box_days"] = float(meta.get("box_days", 0))
    features["atr_ratio"] = float(meta.get("atr_ratio", 0))
    features["coiling"] = float(meta.get("coiling", False))
    features["volume_dry"] = float(meta.get("volume_dry", False))
    features["decline_pct"] = abs(float(meta.get("decline_pct", 0)))
    features["trendline_broken"] = float(meta.get("trendline_broken", False))
    features["stage_transition"] = float(meta.get("stage_transition", False))
    features["volume_confirmed"] = float(meta.get("volume_confirmed", False))

    return features

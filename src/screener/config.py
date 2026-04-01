from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class UniverseConfig(BaseModel):
    min_price: float = 15.0
    min_avg_volume: int = 1_000_000
    avg_volume_period: int = 10
    min_market_cap: int = 500_000_000
    min_history_days: int = 252


class VCPConfig(BaseModel):
    enabled: bool = True
    lookback_days: int = 180
    min_contractions: int = 2
    max_contractions: int = 5
    contraction_decay: float = 0.65
    volume_decay: float = 0.85
    min_base_depth: float = 10.0
    max_base_depth: float = 50.0
    pivot_proximity: float = 0.03


class ReversalConfig(BaseModel):
    enabled: bool = True
    min_downtrend_days: int = 60
    decline_threshold: float = -20.0
    volume_expansion: float = 1.5
    ma_reclaim_window: int = 10
    min_score: float = 40.0


class ResetConfig(BaseModel):
    enabled: bool = True
    ma_targets: list[str] = ["ema_10", "ema_20", "sma_50"]
    touch_tolerance: float = 0.015
    pullback_vol_ratio: float = 0.7
    bounce_vol_ratio: float = 1.2
    max_pullback_days: int = 20
    min_uptrend_days: int = 50


class CoilConfig(BaseModel):
    enabled: bool = True
    min_box_days: int = 10
    max_box_days: int = 45
    atr_contraction: float = 0.6
    box_range_max: float = 15.0
    box_range_min: float = 2.0
    volume_dry_pct: float = 0.7


class PatternsConfig(BaseModel):
    vcp: VCPConfig = VCPConfig()
    reversal: ReversalConfig = ReversalConfig()
    reset: ResetConfig = ResetConfig()
    coil: CoilConfig = CoilConfig()


class TrendTemplateConfig(BaseModel):
    sma_200_uptrend_days: int = 20
    pct_above_52w_low: float = 25.0
    pct_within_52w_high: float = 25.0
    rs_rank_threshold: float = 70.0


class ScoringWeights(BaseModel):
    pattern_score: float = 0.50
    rs_rank: float = 0.20
    volume_score: float = 0.15
    proximity_score: float = 0.15
    ml_blend_weight: float = 0.30


class MLConfig(BaseModel):
    enabled: bool = False
    min_training_samples: int = 30


class ScoringConfig(BaseModel):
    weights: ScoringWeights = ScoringWeights()


class ScreenerConfig(BaseModel):
    universe: UniverseConfig = UniverseConfig()
    patterns: PatternsConfig = PatternsConfig()
    trend_template: TrendTemplateConfig = TrendTemplateConfig()
    scoring: ScoringConfig = ScoringConfig()
    ml: MLConfig = MLConfig()


def load_config(path: str | Path | None = None) -> ScreenerConfig:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "default_params.yaml"
    path = Path(path)
    if path.exists():
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return ScreenerConfig(**data)
    return ScreenerConfig()

from __future__ import annotations

from typing import Any

from screener.config import ScoringWeights
from screener.patterns.base import PatternResult


def compute_composite_score(
    result: PatternResult,
    indicators: dict[str, Any],
    weights: ScoringWeights | None = None,
) -> float:
    """Compute a weighted composite score for ranking detected patterns."""
    if weights is None:
        weights = ScoringWeights()

    pattern_score = result.score

    rs_rank = indicators.get("rs_rank", 50)

    # Volume score: based on relative volume on detection day
    rel_vol = indicators.get("relative_volume")
    if rel_vol is not None and len(rel_vol) > 0:
        last_rvol = float(rel_vol.iloc[-1])
        volume_score = min(100, last_rvol * 50)
    else:
        volume_score = 50

    # Proximity score: how close to the pivot (actionable)
    if result.pivot_price and result.pivot_price > 0:
        current_price = indicators.get("current_price", result.pivot_price)
        distance_pct = abs(result.pivot_price - current_price) / result.pivot_price * 100
        proximity_score = max(0, 100 - distance_pct * 20)
    else:
        proximity_score = 50

    composite = (
        pattern_score * weights.pattern_score
        + rs_rank * weights.rs_rank
        + volume_score * weights.volume_score
        + proximity_score * weights.proximity_score
    )
    return round(composite, 1)


def rank_results(
    results: list[tuple[PatternResult, dict[str, Any]]],
    weights: ScoringWeights | None = None,
) -> list[tuple[PatternResult, float]]:
    """Rank a list of (PatternResult, indicators) tuples by composite score."""
    scored = []
    for result, indicators in results:
        composite = compute_composite_score(result, indicators, weights)
        scored.append((result, composite))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

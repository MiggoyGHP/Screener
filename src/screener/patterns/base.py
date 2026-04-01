from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


@dataclass
class PatternResult:
    pattern_name: str
    ticker: str
    score: float
    detected_date: date
    pivot_price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    annotations: list[dict[str, Any]] = field(default_factory=list)


class PatternDetector(ABC):
    name: str = "base"

    def __init__(self, params: dict[str, Any]):
        self.params = params

    @abstractmethod
    def detect(
        self, ticker: str, df: pd.DataFrame, indicators: dict[str, Any]
    ) -> PatternResult | None:
        """Run pattern detection on a single stock.

        Args:
            ticker: Stock ticker symbol.
            df: OHLCV DataFrame with DatetimeIndex.
            indicators: Pre-computed indicator dict (MAs, RS, volume, ATR).

        Returns:
            PatternResult if pattern detected, None otherwise.
        """
        ...


def zigzag(high: pd.Series, low: pd.Series, pct_threshold: float = 5.0) -> list[tuple[str, int, float]]:
    """Identify swing highs and lows using a percentage threshold.

    Returns list of (type, index_position, price) tuples where type is 'high' or 'low'.
    """
    if len(high) < 3:
        return []

    pivots: list[tuple[str, int, float]] = []
    last_type: str | None = None
    last_val = float(high.iloc[0])
    last_idx = 0

    for i in range(1, len(high)):
        h = float(high.iloc[i])
        lo = float(low.iloc[i])

        if last_type != "low" and lo <= last_val * (1 - pct_threshold / 100):
            if last_type is None or last_type == "low":
                # First, record a high at the start
                if not pivots:
                    pivots.append(("high", last_idx, last_val))
                    last_type = "high"
                    last_val = float(high.iloc[last_idx])
            pivots.append(("low", i, lo))
            last_type = "low"
            last_val = lo
            last_idx = i
        elif last_type != "high" and h >= last_val * (1 + pct_threshold / 100):
            if not pivots:
                pivots.append(("low", last_idx, float(low.iloc[0])))
                last_type = "low"
                last_val = float(low.iloc[0])
            pivots.append(("high", i, h))
            last_type = "high"
            last_val = h
            last_idx = i
        else:
            # Extend current pivot if price goes further
            if last_type == "high" and h > last_val:
                last_val = h
                last_idx = i
                if pivots:
                    pivots[-1] = ("high", i, h)
            elif last_type == "low" and lo < last_val:
                last_val = lo
                last_idx = i
                if pivots:
                    pivots[-1] = ("low", i, lo)

    return pivots

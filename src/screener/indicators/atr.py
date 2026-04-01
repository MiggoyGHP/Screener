from __future__ import annotations

import pandas as pd
import numpy as np


def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def atr_percent(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as a percentage of close price."""
    return atr(df, period) / df["Close"] * 100


def compute_atr_indicators(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "atr_14": atr(df, 14),
        "atr_pct_14": atr_percent(df, 14),
    }

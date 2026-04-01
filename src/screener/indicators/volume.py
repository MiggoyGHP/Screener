from __future__ import annotations

import pandas as pd


def volume_sma(volume: pd.Series, period: int) -> pd.Series:
    return volume.rolling(window=period, min_periods=period).mean()


def relative_volume(volume: pd.Series, period: int = 50) -> pd.Series:
    """Current volume / N-period average volume."""
    avg = volume_sma(volume, period)
    return volume / avg


def compute_volume_indicators(df: pd.DataFrame) -> dict[str, pd.Series]:
    vol = df["Volume"]
    return {
        "volume_sma_10": volume_sma(vol, 10),
        "volume_sma_50": volume_sma(vol, 50),
        "relative_volume": relative_volume(vol, 50),
    }

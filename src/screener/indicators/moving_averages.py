from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_all_mas(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute all standard moving averages used by the screener."""
    close = df["Close"]
    return {
        "ema_10": ema(close, 10),
        "ema_21": ema(close, 21),
        "sma_50": sma(close, 50),
        "sma_150": sma(close, 150),
        "sma_200": sma(close, 200),
    }

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_all_mas(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute all moving averages used by the screener.

    Uses EMAs for 10, 21, 50, 200 (matching the user's charting setup).
    SMA 150 kept for Trend Template calculations only.
    """
    close = df["Close"]
    return {
        "ema_10": ema(close, 10),
        "ema_21": ema(close, 21),
        "ema_50": ema(close, 50),
        "sma_50": sma(close, 50),      # kept for Trend Template
        "sma_150": sma(close, 150),     # kept for Trend Template
        "ema_200": ema(close, 200),
        "sma_200": sma(close, 200),     # kept for Trend Template
    }


def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    close = df["Close"]
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {
        "macd_line": macd_line,
        "macd_signal": signal_line,
        "macd_histogram": histogram,
    }


def is_macd_corrected(macd_indicators: dict[str, pd.Series], lookback: int = 5) -> bool:
    """Check if MACD has fully corrected (histogram reset toward zero).

    A 'full correction' means the histogram was moving away from zero,
    then reversed and contracted back toward/through zero. This applies
    regardless of whether MACD is above, at, or below the zero line.
    """
    hist = macd_indicators["macd_histogram"]
    if len(hist.dropna()) < lookback + 5:
        return False

    recent = hist.iloc[-lookback:]
    prior = hist.iloc[-lookback * 2 : -lookback]

    if len(prior.dropna()) == 0 or len(recent.dropna()) == 0:
        return False

    # Histogram was extended (large absolute value) and is now contracting toward zero
    prior_peak = prior.abs().max()
    current = abs(float(recent.iloc[-1]))

    if prior_peak == 0:
        return False

    # Corrected = current histogram is less than 30% of the prior peak magnitude
    return current < prior_peak * 0.30

from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf


def fetch_ohlcv(
    ticker: str,
    period: str = "2y",
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker. Returns DataFrame with DatetimeIndex."""
    if start or end:
        data = yf.download(
            ticker,
            start=str(start) if start else "2000-01-01",
            end=str(end) if end else None,
            progress=False,
            auto_adjust=True,
        )
    else:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def fetch_ohlcv_batch(
    tickers: list[str],
    period: str = "2y",
    start: str | date | None = None,
    end: str | date | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data for multiple tickers in a single API call."""
    if not tickers:
        return {}
    kwargs = dict(progress=True, auto_adjust=True, group_by="ticker")
    if start or end:
        kwargs["start"] = str(start) if start else "2000-01-01"
        if end:
            kwargs["end"] = str(end)
    else:
        kwargs["period"] = period
    raw = yf.download(tickers, **kwargs)
    if raw.empty:
        return {}
    result: dict[str, pd.DataFrame] = {}
    if len(tickers) == 1:
        ticker = tickers[0]
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            result[ticker] = df.dropna(subset=["Close"])
    else:
        for ticker in tickers:
            try:
                df = raw[ticker].copy()
                df = df.dropna(subset=["Close"])
                if not df.empty:
                    result[ticker] = df
            except (KeyError, TypeError):
                continue
    return result


def fetch_spy(
    period: str = "2y",
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    """Fetch SPY data for relative strength calculations."""
    return fetch_ohlcv("SPY", period=period, start=start, end=end)

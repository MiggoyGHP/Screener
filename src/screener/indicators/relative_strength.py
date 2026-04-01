from __future__ import annotations

import numpy as np
import pandas as pd


def rs_line(stock_close: pd.Series, spy_close: pd.Series) -> pd.Series:
    """Compute RS line: stock / SPY, normalized to start at 100."""
    # Align by date index
    aligned = pd.DataFrame({"stock": stock_close, "spy": spy_close}).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    raw = aligned["stock"] / aligned["spy"]
    return raw / raw.iloc[0] * 100


def rs_new_high(rs: pd.Series, window: int = 252) -> bool:
    """Check if RS line is at or near its 52-week high."""
    if len(rs) < window:
        return False
    rs_high = rs.iloc[-window:].max()
    return float(rs.iloc[-1]) >= rs_high * 0.99


def rs_trending_up(rs: pd.Series, period: int = 21) -> bool:
    """Check if RS line is above its EMA (trending up)."""
    if len(rs) < period:
        return False
    rs_ema = rs.ewm(span=period, adjust=False).mean()
    return float(rs.iloc[-1]) > float(rs_ema.iloc[-1])


def compute_rs_score(stock_close: pd.Series) -> float:
    """Compute a weighted multi-period performance score for RS ranking.

    rs_score = 0.4 * perf_3mo + 0.2 * perf_6mo + 0.2 * perf_9mo + 0.2 * perf_12mo
    """
    periods = {63: 0.4, 126: 0.2, 189: 0.2, 252: 0.2}
    score = 0.0
    current = stock_close.iloc[-1]
    for days, weight in periods.items():
        if len(stock_close) > days:
            past = stock_close.iloc[-days]
            if past > 0:
                score += weight * ((current / past) - 1) * 100
    return score


def compute_rs_rankings(all_scores: dict[str, float]) -> dict[str, float]:
    """Given {ticker: rs_score}, return {ticker: percentile_rank (0-100)}."""
    if not all_scores:
        return {}
    scores = np.array(list(all_scores.values()))
    result = {}
    for ticker, score in all_scores.items():
        rank = np.sum(scores <= score) / len(scores) * 100
        result[ticker] = round(rank, 1)
    return result


def compute_rs_indicators(
    stock_close: pd.Series, spy_close: pd.Series
) -> dict[str, pd.Series | float | bool]:
    rs = rs_line(stock_close, spy_close)
    return {
        "rs_line": rs,
        "rs_new_high": rs_new_high(rs),
        "rs_trending_up": rs_trending_up(rs),
        "rs_score": compute_rs_score(stock_close),
    }

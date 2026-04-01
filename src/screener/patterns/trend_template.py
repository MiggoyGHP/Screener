from __future__ import annotations

from typing import Any

import pandas as pd

from screener.config import TrendTemplateConfig


def check_trend_template(
    df: pd.DataFrame,
    indicators: dict[str, Any],
    config: TrendTemplateConfig | None = None,
) -> tuple[bool, dict[str, bool]]:
    """Check if a stock passes Minervini's 8-criteria Trend Template.

    Returns (passes_all, {criterion_name: bool}).
    """
    if config is None:
        config = TrendTemplateConfig()

    close = float(df["Close"].iloc[-1])
    sma_50 = float(indicators["sma_50"].iloc[-1])
    sma_150 = float(indicators["sma_150"].iloc[-1])
    sma_200 = float(indicators["sma_200"].iloc[-1])

    # 52-week high/low (252 trading days)
    lookback = min(252, len(df))
    high_52w = float(df["High"].iloc[-lookback:].max())
    low_52w = float(df["Low"].iloc[-lookback:].min())

    # C1: Price > 150 SMA and Price > 200 SMA
    c1 = close > sma_150 and close > sma_200

    # C2: 150 SMA > 200 SMA
    c2 = sma_150 > sma_200

    # C3: 200 SMA trending up for N days
    uptrend_days = config.sma_200_uptrend_days
    sma_200_series = indicators["sma_200"]
    if len(sma_200_series.dropna()) >= uptrend_days:
        c3 = float(sma_200_series.iloc[-1]) > float(sma_200_series.iloc[-uptrend_days])
    else:
        c3 = False

    # C4: 50 SMA > 150 SMA and 50 SMA > 200 SMA
    c4 = sma_50 > sma_150 and sma_50 > sma_200

    # C5: Price > 50 SMA
    c5 = close > sma_50

    # C6: Price at least X% above 52-week low
    c6 = close >= low_52w * (1 + config.pct_above_52w_low / 100)

    # C7: Price within X% of 52-week high
    c7 = close >= high_52w * (1 - config.pct_within_52w_high / 100)

    # C8: RS rank in top percentile
    rs_rank = indicators.get("rs_rank", 0)
    c8 = rs_rank >= config.rs_rank_threshold

    criteria = {
        "price_above_150_200": c1,
        "150_above_200": c2,
        "200_trending_up": c3,
        "50_above_150_200": c4,
        "price_above_50": c5,
        "above_52w_low": c6,
        "near_52w_high": c7,
        "rs_rank_pass": c8,
    }

    return all(criteria.values()), criteria

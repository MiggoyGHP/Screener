from __future__ import annotations

import pandas as pd
import yfinance as yf

from screener.config import UniverseConfig


def get_sp500_tickers() -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia."""
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return sorted(table["Symbol"].str.replace(".", "-", regex=False).tolist())
    except Exception:
        return []


def get_nasdaq100_tickers() -> list[str]:
    """Fetch NASDAQ 100 tickers from Wikipedia."""
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100#Components")[4]
        return sorted(table["Ticker"].str.replace(".", "-", regex=False).tolist())
    except Exception:
        return []


def get_broad_universe() -> list[str]:
    """Get a broad universe by combining S&P 500, NASDAQ 100, and S&P 400 mid-cap."""
    sp500 = get_sp500_tickers()

    # S&P 400 mid-cap
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies")[0]
        sp400 = table["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        sp400 = []

    nasdaq = get_nasdaq100_tickers()

    # Combine and deduplicate
    all_tickers = sorted(set(sp500 + sp400 + nasdaq))
    return all_tickers


def filter_universe(tickers: list[str], config: UniverseConfig) -> list[str]:
    """Apply basic filters (price, volume, market cap) to a ticker list.

    Uses yfinance to check current price, volume, and market cap.
    This is meant to be run once before the full screening pipeline.
    """
    passed: list[str] = []
    # Process in batches to be efficient
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        tickers_str = " ".join(batch)
        ticker_objects = yf.Tickers(tickers_str)
        for ticker in batch:
            try:
                info = ticker_objects.tickers[ticker].fast_info
                price = getattr(info, "last_price", None) or 0
                market_cap = getattr(info, "market_cap", None) or 0
                avg_vol = getattr(info, "three_month_average_volume", None) or 0

                if price < config.min_price:
                    continue
                if market_cap < config.min_market_cap:
                    continue
                if avg_vol < config.min_avg_volume:
                    continue
                passed.append(ticker)
            except Exception:
                continue
    return passed

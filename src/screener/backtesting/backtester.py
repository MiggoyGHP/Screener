from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from screener.backtesting.metrics import BacktestResult, TradeResult
from screener.config import ScreenerConfig, load_config
from screener.data.cache import get_full_history
from screener.data.provider import fetch_ohlcv
from screener.pipeline.screener import run_screen


def _compute_trade_return(
    full_df: pd.DataFrame,
    entry_date: date,
    entry_price: float,
    hold_days: int,
    stop_loss_pct: float,
) -> TradeResult | None:
    """Compute the return for a single trade given entry and risk parameters."""
    # Find the first trading day on or after entry_date
    future = full_df.loc[full_df.index.date >= entry_date]
    if len(future) < 2:
        return None

    # Entry on the day after the scan (next open)
    entry_bar = future.iloc[0]
    actual_entry = float(entry_bar["Open"]) if not pd.isna(entry_bar["Open"]) else entry_price
    stop_price = actual_entry * (1 - stop_loss_pct / 100)

    # Walk forward day by day
    exit_price = actual_entry
    exit_day = 0
    hit_stop = False

    for i in range(1, min(hold_days + 1, len(future))):
        bar = future.iloc[i]
        low = float(bar["Low"])
        close = float(bar["Close"])

        if low <= stop_price:
            exit_price = stop_price
            exit_day = i
            hit_stop = True
            break

        exit_price = close
        exit_day = i

    return_pct = ((exit_price - actual_entry) / actual_entry) * 100
    return TradeResult(
        ticker="",  # filled by caller
        pattern="",
        scan_date=entry_date,
        entry_price=round(actual_entry, 2),
        exit_price=round(exit_price, 2),
        return_pct=round(return_pct, 2),
        hit_stop=hit_stop,
        hold_days=exit_day,
        score=0,
    )


def backtest_scan(
    tickers: list[str],
    scan_date: date,
    config: ScreenerConfig | None = None,
    hold_days: int = 20,
    stop_loss_pct: float = 7.0,
    progress_callback=None,
) -> BacktestResult:
    """Run the screener as of scan_date and compute forward returns."""
    if config is None:
        config = load_config()

    # Run screener as of scan_date
    results = run_screen(tickers, config, scan_date=scan_date, progress_callback=progress_callback)

    trades: list[TradeResult] = []
    entry_date = scan_date + timedelta(days=1)

    for result, composite, indicators in results:
        # Get full history (including future data after scan_date)
        full_df = get_full_history(result.ticker, fetch_ohlcv)
        if full_df is None or full_df.empty:
            continue

        entry = result.pivot_price or float(full_df.loc[full_df.index.date >= entry_date].iloc[0]["Open"]) if len(full_df.loc[full_df.index.date >= entry_date]) > 0 else None
        if entry is None:
            continue

        trade = _compute_trade_return(full_df, entry_date, entry, hold_days, stop_loss_pct)
        if trade is None:
            continue

        trade.ticker = result.ticker
        trade.pattern = result.pattern_name
        trade.score = composite
        trades.append(trade)

    return BacktestResult.from_trades(trades)

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from screener.backtesting.backtester import backtest_scan
from screener.backtesting.metrics import BacktestResult, TradeResult
from screener.config import ScreenerConfig, load_config


def rolling_backtest(
    tickers: list[str],
    start_date: date,
    end_date: date,
    config: ScreenerConfig | None = None,
    scan_interval_days: int = 5,
    hold_days: int = 20,
    stop_loss_pct: float = 7.0,
    progress_callback=None,
) -> BacktestResult:
    """Run the screener at regular intervals and aggregate all trades.

    Args:
        tickers: Ticker universe to screen.
        start_date: First scan date.
        end_date: Last scan date.
        config: Screener config.
        scan_interval_days: Days between scans (5 = weekly).
        hold_days: How long to hold each trade.
        stop_loss_pct: Stop loss percentage.
        progress_callback: Optional (current, total, phase) callback.

    Returns:
        Aggregated BacktestResult across all scan dates.
    """
    if config is None:
        config = load_config()

    # Generate scan dates
    scan_dates: list[date] = []
    current = start_date
    while current <= end_date:
        scan_dates.append(current)
        current += timedelta(days=scan_interval_days)

    all_trades: list[TradeResult] = []

    for i, scan_date in enumerate(scan_dates):
        try:
            result = backtest_scan(
                tickers, scan_date, config,
                hold_days=hold_days,
                stop_loss_pct=stop_loss_pct,
            )
            all_trades.extend(result.trades)
        except Exception:
            continue

        if progress_callback:
            progress_callback(i + 1, len(scan_dates), "Rolling backtest")

    return BacktestResult.from_trades(all_trades)

from __future__ import annotations

from typing import Any

import pandas as pd
from tqdm import tqdm

from screener.config import ScreenerConfig, load_config
from screener.data.cache import get_cached_or_fetch
from screener.data.provider import fetch_ohlcv, fetch_spy
from screener.indicators.moving_averages import compute_all_mas
from screener.indicators.relative_strength import (
    compute_rs_indicators,
    compute_rs_rankings,
    compute_rs_score,
)
from screener.indicators.volume import compute_volume_indicators
from screener.indicators.atr import compute_atr_indicators
from screener.patterns.base import PatternResult
from screener.patterns.vcp import VCPDetector
from screener.patterns.reversal import ReversalDetector
from screener.patterns.reset import ResetDetector
from screener.patterns.coil import CoilDetector
from screener.scoring.ranker import rank_results


def compute_indicators(
    df: pd.DataFrame,
    spy_close: pd.Series,
    rs_rank: float = 50.0,
) -> dict[str, Any]:
    """Compute all indicators for a single stock."""
    mas = compute_all_mas(df)
    vol = compute_volume_indicators(df)
    atr_ind = compute_atr_indicators(df)
    rs = compute_rs_indicators(df["Close"], spy_close)

    indicators: dict[str, Any] = {}
    indicators.update(mas)
    indicators.update(vol)
    indicators.update(atr_ind)
    indicators.update(rs)
    indicators["rs_rank"] = rs_rank
    indicators["current_price"] = float(df["Close"].iloc[-1])
    return indicators


def run_screen(
    tickers: list[str],
    config: ScreenerConfig | None = None,
    progress_callback=None,
) -> list[tuple[PatternResult, float, dict[str, Any]]]:
    """Run the full screening pipeline on a list of tickers.

    Returns list of (PatternResult, composite_score, indicators) sorted by score.
    """
    if config is None:
        config = load_config()

    # Fetch SPY data for RS calculations
    spy_df = fetch_spy(period="2y")
    if spy_df.empty:
        raise RuntimeError("Failed to fetch SPY data")
    spy_close = spy_df["Close"]

    # Phase 1: Compute RS scores for all tickers (needed for ranking)
    rs_scores: dict[str, float] = {}
    stock_data: dict[str, pd.DataFrame] = {}

    for i, ticker in enumerate(tqdm(tickers, desc="Fetching data")):
        try:
            df = get_cached_or_fetch(ticker, fetch_ohlcv, period="2y")
            if df is None or len(df) < config.universe.min_history_days:
                continue

            # Apply basic price/volume filters on actual data
            current_price = float(df["Close"].iloc[-1])
            avg_vol_period = config.universe.avg_volume_period
            avg_vol = float(df["Volume"].iloc[-avg_vol_period:].mean())

            if current_price < config.universe.min_price:
                continue
            if avg_vol < config.universe.min_avg_volume:
                continue

            stock_data[ticker] = df
            rs_scores[ticker] = compute_rs_score(df["Close"])
        except Exception:
            continue

        if progress_callback:
            progress_callback(i + 1, len(tickers), "Fetching data")

    # Compute RS rankings
    rs_rankings = compute_rs_rankings(rs_scores)

    # Phase 2: Run pattern detectors
    detectors = []
    pc = config.patterns
    if pc.vcp.enabled:
        detectors.append(VCPDetector(pc.vcp.model_dump()))
    if pc.reversal.enabled:
        detectors.append(ReversalDetector(pc.reversal.model_dump()))
    if pc.reset.enabled:
        detectors.append(ResetDetector(pc.reset.model_dump()))
    if pc.coil.enabled:
        detectors.append(CoilDetector(pc.coil.model_dump()))

    raw_results: list[tuple[PatternResult, dict[str, Any]]] = []

    for i, (ticker, df) in enumerate(tqdm(stock_data.items(), desc="Scanning patterns")):
        try:
            indicators = compute_indicators(
                df, spy_close, rs_rank=rs_rankings.get(ticker, 50.0)
            )

            for detector in detectors:
                result = detector.detect(ticker, df, indicators)
                if result is not None:
                    raw_results.append((result, indicators))
        except Exception:
            continue

        if progress_callback:
            progress_callback(i + 1, len(stock_data), "Scanning patterns")

    # Phase 3: Rank results
    ranked = rank_results(raw_results, config.scoring.weights)

    # Attach indicators for visualization
    final: list[tuple[PatternResult, float, dict[str, Any]]] = []
    for (result, composite), (_, indicators) in zip(ranked, raw_results):
        final.append((result, composite, indicators))

    # Re-sort by composite (rank_results already sorted, but zip may have mismatched)
    final.sort(key=lambda x: x[1], reverse=True)
    return final


def scan_single(
    ticker: str,
    config: ScreenerConfig | None = None,
) -> list[tuple[PatternResult, float, dict[str, Any]]]:
    """Scan a single ticker for all patterns."""
    if config is None:
        config = load_config()

    spy_df = fetch_spy(period="2y")
    spy_close = spy_df["Close"]

    df = get_cached_or_fetch(ticker, fetch_ohlcv, period="2y")
    if df is None or df.empty:
        return []

    rs_score = compute_rs_score(df["Close"])
    # For single scan, use 50th percentile as default since we don't have universe
    indicators = compute_indicators(df, spy_close, rs_rank=50.0)

    detectors = []
    pc = config.patterns
    if pc.vcp.enabled:
        detectors.append(VCPDetector(pc.vcp.model_dump()))
    if pc.reversal.enabled:
        detectors.append(ReversalDetector(pc.reversal.model_dump()))
    if pc.reset.enabled:
        detectors.append(ResetDetector(pc.reset.model_dump()))
    if pc.coil.enabled:
        detectors.append(CoilDetector(pc.coil.model_dump()))

    results = []
    for detector in detectors:
        result = detector.detect(ticker, df, indicators)
        if result is not None:
            from screener.scoring.ranker import compute_composite_score

            composite = compute_composite_score(result, indicators, config.scoring.weights)
            results.append((result, composite, indicators))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

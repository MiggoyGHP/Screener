from __future__ import annotations

from typing import Any

import pandas as pd
import mplfinance as mpf


def make_ma_addplots(indicators: dict[str, Any], df: pd.DataFrame) -> list:
    """Create mplfinance addplot objects for moving average overlays.

    Uses 10 EMA, 21 EMA, 50 EMA, 200 EMA (matching the user's charting setup).
    """
    plots = []
    ma_styles = {
        "ema_10": {"color": "#2196F3", "width": 0.8},   # blue
        "ema_21": {"color": "#F44336", "width": 0.8},   # red
        "ema_50": {"color": "#4CAF50", "width": 1.0},   # green
        "ema_200": {"color": "#212121", "width": 1.2},  # black
    }
    for name, style in ma_styles.items():
        series = indicators.get(name)
        if series is not None:
            aligned = series.reindex(df.index)
            plots.append(
                mpf.make_addplot(
                    aligned,
                    color=style["color"],
                    width=style["width"],
                    label=name.upper().replace("_", " "),
                )
            )
    return plots


def make_macd_addplots(indicators: dict[str, Any], df: pd.DataFrame) -> list:
    """Create mplfinance addplots for MACD in a separate panel."""
    macd_line = indicators.get("macd_line")
    macd_signal = indicators.get("macd_signal")
    macd_hist = indicators.get("macd_histogram")

    if macd_line is None or macd_signal is None or macd_hist is None:
        return []

    macd_aligned = macd_line.reindex(df.index)
    signal_aligned = macd_signal.reindex(df.index)
    hist_aligned = macd_hist.reindex(df.index)

    # Color histogram bars green/red
    hist_pos = hist_aligned.copy()
    hist_neg = hist_aligned.copy()
    hist_pos[hist_pos < 0] = 0
    hist_neg[hist_neg > 0] = 0

    return [
        mpf.make_addplot(macd_aligned, panel=2, color="#2196F3", width=0.8, ylabel="MACD"),
        mpf.make_addplot(signal_aligned, panel=2, color="#FF9800", width=0.8),
        mpf.make_addplot(hist_pos, panel=2, type="bar", color="#26A69A", width=0.7),
        mpf.make_addplot(hist_neg, panel=2, type="bar", color="#EF5350", width=0.7),
    ]


def make_rs_addplot(indicators: dict[str, Any], df: pd.DataFrame) -> list:
    """Create mplfinance addplot for RS line in a separate panel."""
    rs_line = indicators.get("rs_line")
    if rs_line is None or rs_line.empty:
        return []
    aligned = rs_line.reindex(df.index)
    return [
        mpf.make_addplot(
            aligned,
            panel=3,
            color="#9C27B0",
            width=1.2,
            ylabel="RS Line",
        )
    ]


def make_annotation_hlines(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract horizontal line annotations for chart rendering."""
    hlines = []
    for ann in annotations:
        if ann.get("type") == "hline":
            hlines.append({
                "price": ann["price"],
                "label": ann.get("label", ""),
            })
    return hlines

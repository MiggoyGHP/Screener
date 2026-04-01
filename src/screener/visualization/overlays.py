from __future__ import annotations

from typing import Any

import pandas as pd
import mplfinance as mpf


def make_ma_addplots(indicators: dict[str, Any], df: pd.DataFrame) -> list:
    """Create mplfinance addplot objects for moving average overlays."""
    plots = []
    ma_styles = {
        "ema_10": {"color": "#2196F3", "width": 0.8},   # blue
        "ema_21": {"color": "#F44336", "width": 0.8},   # red
        "sma_50": {"color": "#4CAF50", "width": 1.0},   # green
        "sma_150": {"color": "#FF9800", "width": 0.8},  # orange
        "sma_200": {"color": "#212121", "width": 1.2},  # black
    }
    for name, style in ma_styles.items():
        series = indicators.get(name)
        if series is not None:
            # Align with df index
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


def make_rs_addplot(indicators: dict[str, Any], df: pd.DataFrame) -> list:
    """Create mplfinance addplot for RS line in a separate panel."""
    rs_line = indicators.get("rs_line")
    if rs_line is None or rs_line.empty:
        return []
    aligned = rs_line.reindex(df.index)
    return [
        mpf.make_addplot(
            aligned,
            panel=2,
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

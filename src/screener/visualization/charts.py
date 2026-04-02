from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

from screener.patterns.base import PatternResult
from screener.visualization.overlays import (
    make_ma_addplots,
    make_macd_addplots,
    make_rs_addplot,
)


CHART_STYLE = mpf.make_mpf_style(
    base_mpf_style="yahoo",
    rc={"font.size": 9},
    marketcolors=mpf.make_marketcolors(
        up="#26A69A",
        down="#EF5350",
        edge="inherit",
        wick="inherit",
        volume="in",
    ),
)


def create_pattern_chart(
    df: pd.DataFrame,
    result: PatternResult,
    indicators: dict[str, Any],
    lookback_days: int = 130,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Create a candlestick chart with overlays for a detected pattern.

    Returns a matplotlib Figure.
    """
    # Trim to lookback window
    chart_df = df.iloc[-lookback_days:].copy()

    # Build addplots
    addplots = []
    addplots.extend(make_ma_addplots(indicators, chart_df))
    addplots.extend(make_macd_addplots(indicators, chart_df))
    addplots.extend(make_rs_addplot(indicators, chart_df))

    # Build title
    title = (
        f"{result.ticker} - {result.pattern_name} "
        f"(Score: {result.score:.0f})"
    )

    # Plot
    kwargs: dict[str, Any] = {
        "type": "candle",
        "volume": True,
        "style": CHART_STYLE,
        "addplot": addplots if addplots else None,
        "title": title,
        "figsize": (14, 7),
        "panel_ratios": (4, 1, 2, 1),
        "returnfig": True,
        "warn_too_much_data": 9999,
    }

    if not addplots:
        del kwargs["addplot"]

    fig, axes = mpf.plot(chart_df, **kwargs)

    # Add pattern metadata as text
    meta_text = _format_metadata(result)
    if meta_text:
        fig.text(
            0.01, 0.01, meta_text,
            fontsize=7, fontfamily="monospace",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )

    if output_path:
        fig.savefig(str(output_path), dpi=100, bbox_inches="tight")
        plt.close(fig)

    return fig


def _format_metadata(result: PatternResult) -> str:
    """Format pattern metadata as readable text for chart annotation."""
    meta = result.metadata
    lines = []

    if result.pattern_name == "VCP":
        lines.append(f"Contractions: {meta.get('num_contractions', '?')}")
        lines.append(f"Base depth: {meta.get('base_depth', '?')}%")
        lines.append(f"Vol declining: {meta.get('volume_declining', '?')}")
        lines.append(f"Near pivot: {meta.get('near_pivot', '?')}")
    elif result.pattern_name == "Reset":
        lines.append(f"MA touched: {meta.get('ma_touched', '?')}")
        lines.append(f"Pullback: {meta.get('pullback_depth_pct', '?')}%")
        lines.append(f"Vol contracted: {meta.get('volume_contracted', '?')}")
    elif result.pattern_name == "Coil":
        lines.append(f"Box: ${meta.get('box_low', '?')} - ${meta.get('box_high', '?')}")
        lines.append(f"Range: {meta.get('box_range_pct', '?')}%")
        lines.append(f"Days: {meta.get('box_days', '?')}")
        lines.append(f"ATR ratio: {meta.get('atr_ratio', '?')}")
    elif result.pattern_name == "Reversal":
        lines.append(f"Decline: {meta.get('decline_pct', '?')}%")
        lines.append(f"Trendline broken: {meta.get('trendline_broken', '?')}")
        lines.append(f"Stage transition: {meta.get('stage_transition', '?')}")

    # Signal quality flags (VCP, Reset, Coil only)
    if result.pattern_name in ("VCP", "Reset", "Coil"):
        signals = []
        if meta.get("ema_ordered"):
            signals.append("EMA aligned")
        if meta.get("macd_near_zero"):
            signals.append("MACD@0")
        if meta.get("macd_converging"):
            signals.append("MACD conv")
        if meta.get("atr_declining"):
            signals.append("ATR quiet")
        if signals:
            lines.append(f"Signals: {', '.join(signals)}")

    return "\n".join(lines)


def create_chart_for_streamlit(
    df: pd.DataFrame,
    result: PatternResult,
    indicators: dict[str, Any],
    lookback_days: int = 130,
) -> plt.Figure:
    """Convenience wrapper that returns a Figure for st.pyplot()."""
    return create_pattern_chart(df, result, indicators, lookback_days)

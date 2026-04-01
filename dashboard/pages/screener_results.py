from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from screener.config import load_config
from screener.data.universe import get_sp500_tickers, get_nasdaq100_tickers, get_broad_universe
from screener.data.cache import get_cached_or_fetch
from screener.data.provider import fetch_ohlcv
from screener.pipeline.screener import run_screen
from screener.visualization.charts import create_chart_for_streamlit


st.title("Screener Results")

# Sidebar controls
st.sidebar.header("Scan Settings")

universe_choice = st.sidebar.selectbox(
    "Stock Universe",
    ["S&P 500", "NASDAQ 100", "Broad (S&P 500 + 400 + NASDAQ 100)", "Custom"],
)

custom_tickers = ""
if universe_choice == "Custom":
    custom_tickers = st.sidebar.text_area(
        "Enter tickers (comma-separated)",
        "AAPL, MSFT, NVDA, AMZN, META",
    )

pattern_filter = st.sidebar.multiselect(
    "Patterns to Scan",
    ["VCP", "Reversal", "Reset", "Coil"],
    default=["VCP", "Reversal", "Reset", "Coil"],
)

min_score = st.sidebar.slider("Minimum Score", 0, 100, 30)

# Run button
if st.sidebar.button("Run Scan", type="primary", use_container_width=True):
    config = load_config()

    # Disable patterns not selected
    config.patterns.vcp.enabled = "VCP" in pattern_filter
    config.patterns.reversal.enabled = "Reversal" in pattern_filter
    config.patterns.reset.enabled = "Reset" in pattern_filter
    config.patterns.coil.enabled = "Coil" in pattern_filter

    # Get tickers
    with st.spinner("Fetching ticker universe..."):
        if universe_choice == "S&P 500":
            tickers = get_sp500_tickers()
        elif universe_choice == "NASDAQ 100":
            tickers = get_nasdaq100_tickers()
        elif universe_choice == "Broad (S&P 500 + 400 + NASDAQ 100)":
            tickers = get_broad_universe()
        else:
            tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]

    st.info(f"Scanning {len(tickers)} tickers...")

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(current: int, total: int, phase: str):
        if total > 0:
            progress_bar.progress(current / total)
            status_text.text(f"{phase}: {current}/{total}")

    # Run the screen
    results = run_screen(tickers, config, progress_callback=progress_callback)

    progress_bar.empty()
    status_text.empty()

    # Filter by min score
    results = [(r, s, ind) for r, s, ind in results if s >= min_score]

    # Store in session state
    st.session_state["results"] = results
    st.session_state["tickers_data"] = {}
    for r, s, ind in results:
        # Cache the dataframe for chart rendering
        df = get_cached_or_fetch(r.ticker, fetch_ohlcv, period="2y")
        if df is not None:
            st.session_state["tickers_data"][r.ticker] = df

    st.success(f"Found {len(results)} setups!")

# Display results
if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]

    # Summary table
    table_data = []
    for result, composite, indicators in results:
        table_data.append({
            "Ticker": result.ticker,
            "Pattern": result.pattern_name,
            "Pattern Score": result.score,
            "Composite Score": composite,
            "Pivot Price": f"${result.pivot_price:.2f}" if result.pivot_price else "-",
            "RS Rank": f"{indicators.get('rs_rank', 0):.0f}",
        })

    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    # Chart viewer
    st.markdown("---")
    st.subheader("Chart Viewer")

    selected_idx = st.selectbox(
        "Select a setup to view",
        range(len(results)),
        format_func=lambda i: (
            f"{results[i][0].ticker} - {results[i][0].pattern_name} "
            f"(Score: {results[i][1]:.0f})"
        ),
    )

    if selected_idx is not None:
        result, composite, indicators = results[selected_idx]
        ticker_df = st.session_state.get("tickers_data", {}).get(result.ticker)

        if ticker_df is not None:
            fig = create_chart_for_streamlit(ticker_df, result, indicators)
            st.pyplot(fig)
            plt.close(fig)

            # Metadata expander
            with st.expander("Pattern Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.json(result.metadata)
                with col2:
                    st.metric("Composite Score", f"{composite:.1f}")
                    st.metric("RS Rank", f"{indicators.get('rs_rank', 0):.0f}")
                    if result.pivot_price:
                        current = indicators.get("current_price", 0)
                        dist = ((result.pivot_price - current) / result.pivot_price * 100) if result.pivot_price else 0
                        st.metric("Distance to Pivot", f"{dist:.1f}%")

elif "results" in st.session_state:
    st.warning("No setups found matching your criteria. Try adjusting filters.")
else:
    st.info("Click 'Run Scan' in the sidebar to start screening.")

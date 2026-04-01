from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st
import matplotlib.pyplot as plt

from screener.config import load_config
from screener.data.cache import get_cached_or_fetch, get_cached_or_fetch_as_of
from screener.data.provider import fetch_ohlcv
from screener.pipeline.screener import scan_single
from screener.visualization.charts import create_chart_for_streamlit


st.title("Pattern Explorer")
st.markdown("Enter a ticker to scan for all patterns.")

ticker = st.text_input("Ticker Symbol", "AAPL").strip().upper()
lookback = st.slider("Chart Lookback (days)", 60, 252, 130)

use_historical = st.checkbox("Scan a past date")
scan_date = None
if use_historical:
    scan_date = st.date_input(
        "Scan Date",
        value=date(2024, 6, 15),
        min_value=date(2000, 1, 1),
        max_value=date.today(),
    )

if st.button("Scan Ticker", type="primary"):
    with st.spinner(f"Scanning {ticker}..."):
        config = load_config()
        results = scan_single(ticker, config, scan_date=scan_date)

    if results:
        st.success(f"Found {len(results)} pattern(s) for {ticker}")

        if scan_date:
            df = get_cached_or_fetch_as_of(ticker, scan_date, fetch_ohlcv)
        else:
            df = get_cached_or_fetch(ticker, fetch_ohlcv, period="2y")

        for i, (result, composite, indicators) in enumerate(results):
            st.markdown("---")
            st.subheader(f"{result.pattern_name} (Score: {composite:.1f})")

            if df is not None:
                fig = create_chart_for_streamlit(df, result, indicators, lookback_days=lookback)
                st.pyplot(fig)
                plt.close(fig)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pattern Score", f"{result.score:.0f}")
            with col2:
                st.metric("Composite Score", f"{composite:.1f}")
            with col3:
                if result.pivot_price:
                    st.metric("Pivot Price", f"${result.pivot_price:.2f}")

            with st.expander("Details"):
                st.json(result.metadata)
    else:
        st.warning(f"No patterns detected for {ticker}.")

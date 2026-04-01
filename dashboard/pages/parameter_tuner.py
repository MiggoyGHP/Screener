from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st
import matplotlib.pyplot as plt

from screener.config import load_config
from screener.data.cache import get_cached_or_fetch
from screener.data.provider import fetch_ohlcv
from screener.pipeline.screener import scan_single
from screener.visualization.charts import create_chart_for_streamlit


st.title("Parameter Tuner")
st.markdown("Adjust detection parameters and see how they affect pattern detection on a single ticker.")

ticker = st.text_input("Ticker Symbol", "AAPL").strip().upper()

st.sidebar.header("Pattern Parameters")

# VCP params
st.sidebar.subheader("VCP")
vcp_lookback = st.sidebar.slider("Lookback Days", 60, 300, 180, key="vcp_lb")
vcp_min_cont = st.sidebar.slider("Min Contractions", 1, 5, 2, key="vcp_mc")
vcp_decay = st.sidebar.slider("Contraction Decay", 0.3, 1.0, 0.65, 0.05, key="vcp_cd")
vcp_min_depth = st.sidebar.slider("Min Base Depth %", 5, 30, 10, key="vcp_mind")
vcp_max_depth = st.sidebar.slider("Max Base Depth %", 20, 70, 50, key="vcp_maxd")
vcp_proximity = st.sidebar.slider("Pivot Proximity %", 1, 10, 3, key="vcp_prox")

# Reset params
st.sidebar.subheader("Reset")
reset_tolerance = st.sidebar.slider("MA Touch Tolerance %", 0.5, 5.0, 1.5, 0.5, key="rst_tol")
reset_pb_days = st.sidebar.slider("Max Pullback Days", 5, 40, 20, key="rst_pb")

# Coil params
st.sidebar.subheader("Coil")
coil_min_days = st.sidebar.slider("Min Box Days", 5, 30, 10, key="coil_min")
coil_max_days = st.sidebar.slider("Max Box Days", 15, 60, 45, key="coil_max")
coil_atr = st.sidebar.slider("ATR Contraction Ratio", 0.3, 1.0, 0.6, 0.05, key="coil_atr")
coil_range_max = st.sidebar.slider("Max Box Range %", 5, 25, 15, key="coil_rng")

# Reversal params
st.sidebar.subheader("Reversal")
rev_decline = st.sidebar.slider("Min Decline %", -50, -10, -20, key="rev_dec")
rev_vol_exp = st.sidebar.slider("Volume Expansion x", 1.0, 3.0, 1.5, 0.1, key="rev_vol")

# Trend Template
st.sidebar.subheader("Trend Template")
tt_rs_threshold = st.sidebar.slider("RS Rank Threshold", 30, 90, 70, key="tt_rs")

if st.button("Run with Custom Parameters", type="primary"):
    config = load_config()

    # Override with slider values
    config.patterns.vcp.lookback_days = vcp_lookback
    config.patterns.vcp.min_contractions = vcp_min_cont
    config.patterns.vcp.contraction_decay = vcp_decay
    config.patterns.vcp.min_base_depth = float(vcp_min_depth)
    config.patterns.vcp.max_base_depth = float(vcp_max_depth)
    config.patterns.vcp.pivot_proximity = vcp_proximity / 100

    config.patterns.reset.touch_tolerance = reset_tolerance / 100
    config.patterns.reset.max_pullback_days = reset_pb_days

    config.patterns.coil.min_box_days = coil_min_days
    config.patterns.coil.max_box_days = coil_max_days
    config.patterns.coil.atr_contraction = coil_atr
    config.patterns.coil.box_range_max = float(coil_range_max)

    config.patterns.reversal.decline_threshold = float(rev_decline)
    config.patterns.reversal.volume_expansion = rev_vol_exp

    config.trend_template.rs_rank_threshold = float(tt_rs_threshold)

    with st.spinner(f"Scanning {ticker} with custom parameters..."):
        results = scan_single(ticker, config)

    if results:
        st.success(f"Found {len(results)} pattern(s)")

        df = get_cached_or_fetch(ticker, fetch_ohlcv, period="2y")

        for result, composite, indicators in results:
            st.markdown("---")
            st.subheader(f"{result.pattern_name} (Score: {composite:.1f})")

            if df is not None:
                fig = create_chart_for_streamlit(df, result, indicators)
                st.pyplot(fig)
                plt.close(fig)

            with st.expander("Details"):
                st.json(result.metadata)
    else:
        st.warning(f"No patterns detected for {ticker} with these parameters. Try loosening thresholds.")

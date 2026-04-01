from __future__ import annotations

import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from screener.config import load_config
from screener.data.cache import get_cached_or_fetch_as_of
from screener.data.provider import fetch_ohlcv
from screener.pipeline.screener import scan_single


# Hardcoded universe — avoids Wikipedia scraping which fails on Streamlit Cloud
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOG", "TSLA", "NFLX", "AMD", "CRM",
    "AVGO", "COST", "ADBE", "PEP", "CSCO", "INTC", "QCOM", "TXN", "AMGN", "INTU",
    "ISRG", "AMAT", "LRCX", "MU", "KLAC", "SNPS", "CDNS", "MRVL", "PANW", "CRWD",
    "ABNB", "DASH", "COIN", "SQ", "SHOP", "SNOW", "DDOG", "ZS", "NET", "TEAM",
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP",
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "DHR", "ABT", "BMY",
    "HD", "LOW", "TJX", "NKE", "SBUX", "MCD", "CMG", "YUM", "DPZ", "LULU",
    "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "EA", "TTWO", "RBLX", "SPOT",
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "VLO", "OXY", "DVN", "HAL",
    "CAT", "DE", "GE", "HON", "RTX", "LMT", "BA", "UPS", "FDX", "UNP",
    "PLTR", "UBER", "LYFT", "RIVN", "LCID", "ENPH", "FSLR", "VST", "CEG", "SO",
]
from screener.visualization.charts import create_pattern_chart
from screener.ml.labels import save_label, get_all_labels, get_label_counts
from screener.ml.features import extract_features
from screener.ml.trainer import PreferenceModel


st.title("Rate Charts")
st.markdown("Charts are shown one at a time. Rate each one 1-5 stars, then move on.")

# ─── Generate a batch of charts to rate ───

def _generate_batch(n: int = 20):
    """Scan random tickers on random historical dates to build a batch of charts."""
    config = load_config()
    batch = []
    attempts = 0
    max_attempts = n * 10

    while len(batch) < n and attempts < max_attempts:
        attempts += 1
        ticker = random.choice(TICKERS)
        # Random date between 2015 and 2025
        days_back = random.randint(100, 3650)
        scan_date = date.today() - timedelta(days=days_back)

        try:
            results = scan_single(ticker, config, scan_date=scan_date)
            if not results:
                continue

            result, composite, indicators = results[0]
            df = get_cached_or_fetch_as_of(ticker, scan_date, fetch_ohlcv)
            if df is None or df.empty:
                continue

            batch.append({
                "ticker": ticker,
                "scan_date": scan_date,
                "result": result,
                "composite": composite,
                "indicators": indicators,
                "df": df,
            })
        except Exception:
            continue

    return batch


# Sidebar
counts = get_label_counts()
st.sidebar.metric("Charts Rated", counts["total"])
if counts["total"] > 0:
    st.sidebar.metric("Avg Rating", f"{counts['avg_rating']}/5")
st.sidebar.markdown("---")

if counts["total"] >= 30:
    st.sidebar.success("Enough data to train!")
    if st.sidebar.button("Train Model", type="primary"):
        with st.spinner("Training..."):
            model = PreferenceModel()
            metrics = model.train()
        if "error" in metrics:
            st.sidebar.error(metrics["error"])
        else:
            st.sidebar.success(f"Model trained! R2: {metrics['r2_score']:.3f}")
else:
    st.sidebar.info(f"Need {30 - counts['total']} more ratings to train")

# Load or generate batch
if st.button("Load Charts to Rate", type="primary", use_container_width=True):
    with st.spinner("Finding setups across random tickers and dates..."):
        batch = _generate_batch(20)
    if batch:
        st.session_state["rate_batch"] = batch
        st.session_state["rate_idx"] = 0
        st.success(f"Found {len(batch)} charts to rate!")
    else:
        st.error("Couldn't find any setups. Try again.")

# ─── Show current chart ───

if "rate_batch" in st.session_state and st.session_state["rate_batch"]:
    batch = st.session_state["rate_batch"]
    idx = st.session_state.get("rate_idx", 0)

    if idx < len(batch):
        item = batch[idx]
        result = item["result"]
        indicators = item["indicators"]
        df = item["df"]

        st.markdown(f"**Chart {idx + 1} of {len(batch)}**  |  {result.ticker} - {result.pattern_name}  |  {item['scan_date']}")

        # Render chart
        fig = create_pattern_chart(df, result, indicators)
        st.pyplot(fig)
        plt.close(fig)

        # Star rating — 5 buttons in a row
        st.markdown("### How tradeable is this setup?")
        cols = st.columns(5)
        for star in range(1, 6):
            with cols[star - 1]:
                label = f"{'*' * star} {star}"
                if st.button(label, key=f"star_{idx}_{star}", use_container_width=True):
                    features = extract_features(result, indicators)
                    save_label(
                        ticker=result.ticker,
                        pattern_name=result.pattern_name,
                        scan_date=str(item["scan_date"]),
                        label=star,
                        features=features,
                        score=item["composite"],
                    )
                    st.session_state["rate_idx"] = idx + 1
                    st.rerun()

        # Skip button
        if st.button("Skip (don't rate)", use_container_width=True):
            st.session_state["rate_idx"] = idx + 1
            st.rerun()
    else:
        st.success(f"Done! Rated all {len(batch)} charts. Click 'Load Charts' for more.")
        st.session_state.pop("rate_batch", None)
        st.session_state.pop("rate_idx", None)

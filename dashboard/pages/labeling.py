from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from screener.config import load_config
from screener.data.cache import get_cached_or_fetch, get_cached_or_fetch_as_of
from screener.data.provider import fetch_ohlcv
from screener.pipeline.screener import scan_single
from screener.visualization.charts import create_pattern_chart
from screener.ml.labels import save_label, get_all_labels, get_label_counts, flip_label, delete_label
from screener.ml.features import extract_features
from screener.ml.trainer import PreferenceModel


st.title("ML Labeling & Training")

tab1, tab2, tab3 = st.tabs(["Label Setups", "Review Labels", "Train Model"])

# ─── Tab 1: Label setups ───
with tab1:
    st.subheader("Label Detected Setups")
    ticker = st.text_input("Ticker", "AAPL", key="label_ticker").strip().upper()

    use_hist = st.checkbox("Use historical date", key="label_hist")
    scan_date = None
    if use_hist:
        scan_date = st.date_input("Scan Date", value=date(2024, 6, 15), key="label_date")

    if st.button("Scan for Labeling", type="primary"):
        config = load_config()
        results = scan_single(ticker, config, scan_date=scan_date)

        if results:
            st.session_state["label_results"] = results
            st.session_state["label_ticker"] = ticker
            st.session_state["label_scan_date"] = str(scan_date) if scan_date else str(date.today())
            st.session_state["label_idx"] = 0

            if scan_date:
                df = get_cached_or_fetch_as_of(ticker, scan_date, fetch_ohlcv)
            else:
                df = get_cached_or_fetch(ticker, fetch_ohlcv, period="2y")
            st.session_state["label_df"] = df
        else:
            st.warning(f"No patterns found for {ticker}.")

    if "label_results" in st.session_state and st.session_state["label_results"]:
        results = st.session_state["label_results"]
        idx = st.session_state.get("label_idx", 0)
        df = st.session_state.get("label_df")

        if idx < len(results):
            result, composite, indicators = results[idx]

            st.markdown(f"**Setup {idx + 1} of {len(results)}**: {result.ticker} — {result.pattern_name} (Score: {composite:.1f})")

            if df is not None:
                fig = create_pattern_chart(df, result, indicators)
                st.pyplot(fig)
                plt.close(fig)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Good Setup", type="primary", use_container_width=True):
                    features = extract_features(result, indicators)
                    # Save chart
                    charts_dir = Path(__file__).resolve().parents[2] / "data" / "charts"
                    charts_dir.mkdir(parents=True, exist_ok=True)
                    chart_path = str(charts_dir / f"{result.ticker}_{result.pattern_name}_{result.detected_date}.png")
                    if df is not None:
                        fig = create_pattern_chart(df, result, indicators, output_path=chart_path)
                        plt.close(fig)

                    save_label(
                        ticker=result.ticker,
                        pattern_name=result.pattern_name,
                        scan_date=st.session_state["label_scan_date"],
                        label=1,
                        features=features,
                        chart_path=chart_path,
                        score=composite,
                    )
                    st.success("Labeled as GOOD")
                    st.session_state["label_idx"] = idx + 1
                    st.rerun()

            with col2:
                if st.button("Bad Setup", type="secondary", use_container_width=True):
                    features = extract_features(result, indicators)
                    save_label(
                        ticker=result.ticker,
                        pattern_name=result.pattern_name,
                        scan_date=st.session_state["label_scan_date"],
                        label=0,
                        features=features,
                        score=composite,
                    )
                    st.warning("Labeled as BAD")
                    st.session_state["label_idx"] = idx + 1
                    st.rerun()

            with col3:
                if st.button("Skip", use_container_width=True):
                    st.session_state["label_idx"] = idx + 1
                    st.rerun()

            with st.expander("Details"):
                st.json(result.metadata)
        else:
            st.info("All setups labeled! Scan another ticker or train the model.")

# ─── Tab 2: Review labels ───
with tab2:
    st.subheader("Labeled Data")
    counts = get_label_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Good", counts["good"])
    c2.metric("Bad", counts["bad"])
    c3.metric("Total", counts["total"])

    labels = get_all_labels()
    if labels:
        table_data = []
        for l in labels:
            table_data.append({
                "ID": l["id"],
                "Ticker": l["ticker"],
                "Pattern": l["pattern_name"],
                "Scan Date": l["scan_date"],
                "Label": "Good" if l["label"] == 1 else "Bad",
                "Score": f"{l['score']:.1f}" if l["score"] else "-",
                "Created": l["created_at"][:16],
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            flip_id = st.number_input("Flip label ID", min_value=1, step=1, key="flip_id")
            if st.button("Flip Label"):
                flip_label(int(flip_id))
                st.success(f"Flipped label {flip_id}")
                st.rerun()
        with col2:
            del_id = st.number_input("Delete label ID", min_value=1, step=1, key="del_id")
            if st.button("Delete Label"):
                delete_label(int(del_id))
                st.success(f"Deleted label {del_id}")
                st.rerun()

# ─── Tab 3: Train model ───
with tab3:
    st.subheader("Train Preference Model")
    counts = get_label_counts()

    if counts["total"] < 30:
        st.warning(f"Need at least 30 labels to train. Currently have {counts['total']}.")
    else:
        st.info(f"{counts['total']} labels available ({counts['good']} good, {counts['bad']} bad)")

    if st.button("Train Model", type="primary", disabled=counts["total"] < 30):
        with st.spinner("Training..."):
            model = PreferenceModel()
            metrics = model.train()

        if "error" in metrics:
            st.error(metrics["error"])
        else:
            st.success(f"Model trained! Accuracy: {metrics['accuracy']}% (+/- {metrics['accuracy_std']}%)")
            st.metric("Samples Used", metrics["n_samples"])

            # Feature importance chart
            importance = metrics.get("feature_importance", {})
            if importance:
                fig, ax = plt.subplots(figsize=(10, 6))
                names = list(importance.keys())[:15]
                values = [importance[n] for n in names]
                ax.barh(names[::-1], values[::-1], color="#58a6ff")
                ax.set_xlabel("Importance")
                ax.set_title("Top Feature Importances")
                st.pyplot(fig)
                plt.close(fig)

    # Model status
    st.markdown("---")
    model = PreferenceModel()
    if model.load():
        st.success(f"Trained model loaded ({model.metrics.get('n_samples', '?')} samples, {model.metrics.get('accuracy', '?')}% accuracy)")
    else:
        st.info("No trained model found yet.")

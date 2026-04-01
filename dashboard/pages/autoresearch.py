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
from screener.data.universe import get_sp500_tickers, get_nasdaq100_tickers
from screener.autoresearch.optimizer import run_autoresearch
from screener.autoresearch.experiment_log import get_all_experiments, get_best_experiment


st.title("Autoresearch")
st.markdown(
    "Autonomous parameter optimization. Mutates detection parameters, "
    "backtests each variant, and keeps improvements that increase expectancy."
)

# Sidebar settings
st.sidebar.header("Optimization Settings")

universe_choice = st.sidebar.selectbox(
    "Universe",
    ["Custom (fast)", "NASDAQ 100", "S&P 500"],
    key="ar_universe",
)

custom_tickers = ""
if universe_choice == "Custom (fast)":
    custom_tickers = st.sidebar.text_area(
        "Tickers",
        "AAPL, MSFT, NVDA, AMZN, META, GOOG, TSLA, NFLX, AMD, CRM, AVGO, COST",
        key="ar_custom",
    )

col1, col2 = st.sidebar.columns(2)
with col1:
    eval_start = st.date_input("Eval Start", date(2023, 6, 1), key="ar_start")
with col2:
    eval_end = st.date_input("Eval End", date(2024, 6, 1), key="ar_end")

max_iterations = st.sidebar.slider("Max Iterations", 5, 100, 20, key="ar_iter")
scan_interval = st.sidebar.slider("Scan Interval (days)", 5, 30, 10, key="ar_interval")
hold_days = st.sidebar.slider("Hold Days", 5, 60, 20, key="ar_hold")
stop_loss = st.sidebar.slider("Stop Loss %", 3.0, 15.0, 7.0, 0.5, key="ar_stop")
n_mutations = st.sidebar.slider("Mutations per Iteration", 1, 5, 2, key="ar_mut")
perturbation = st.sidebar.slider("Perturbation Size", 0.05, 0.50, 0.20, 0.05, key="ar_pert")

if st.sidebar.button("Run Optimization", type="primary", use_container_width=True):
    if universe_choice == "Custom (fast)":
        tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
    elif universe_choice == "NASDAQ 100":
        tickers = get_nasdaq100_tickers()
    else:
        tickers = get_sp500_tickers()

    progress_bar = st.progress(0)
    status_text = st.empty()
    best_text = st.empty()

    def progress_cb(current, total, best_exp, improved):
        progress_bar.progress(current / total)
        marker = " ** IMPROVED **" if improved else ""
        status_text.text(f"Iteration {current}/{total}{marker}")
        best_text.text(f"Best expectancy so far: {best_exp:.2f}%")

    with st.spinner("Running autoresearch..."):
        summary = run_autoresearch(
            tickers=tickers,
            eval_start=eval_start,
            eval_end=eval_end,
            max_iterations=max_iterations,
            scan_interval_days=scan_interval,
            hold_days=hold_days,
            stop_loss_pct=stop_loss,
            n_mutations=n_mutations,
            perturbation=perturbation,
            progress_callback=progress_cb,
        )

    progress_bar.empty()
    status_text.empty()
    best_text.empty()

    st.session_state["ar_summary"] = summary

# Results
if "ar_summary" in st.session_state:
    summary = st.session_state["ar_summary"]

    st.subheader("Optimization Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline Expectancy", f"{summary['baseline_expectancy']:.2f}%")
    c2.metric("Best Expectancy", f"{summary['best_expectancy']:.2f}%")
    c3.metric("Improvements", summary["improvements"])
    c4.metric("Iterations", summary["iterations_run"])

    if summary["best_expectancy"] > summary["baseline_expectancy"]:
        st.success("Optimized config saved to config/optimized_params.yaml")
    else:
        st.info("No improvement found over baseline. Try more iterations or wider perturbation.")

    with st.expander("Best Parameters"):
        st.json(summary["best_config"])

# Experiment history
st.markdown("---")
st.subheader("Experiment History")

experiments = get_all_experiments()
if experiments:
    exp_data = []
    for e in experiments:
        exp_data.append({
            "Iter": e["iteration"],
            "Expectancy": f"{e['expectancy']:.2f}%",
            "Win Rate": f"{e['win_rate']:.1f}%",
            "Profit Factor": f"{e['profit_factor']:.2f}",
            "Trades": e["total_trades"],
            "Improved": "Yes" if e["improved"] else "",
            "Time": e["timestamp"][:16],
        })
    st.dataframe(pd.DataFrame(exp_data), use_container_width=True, hide_index=True)

    # Expectancy over iterations chart
    fig, ax = plt.subplots(figsize=(12, 4))
    iters = [e["iteration"] for e in experiments]
    exps = [e["expectancy"] for e in experiments]
    colors = ["#26A69A" if e["improved"] else "#EF5350" for e in experiments]
    ax.bar(iters, exps, color=colors, alpha=0.7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Expectancy %")
    ax.set_title("Expectancy by Iteration (green = improvement)")
    ax.axhline(y=0, color="white", linewidth=0.5, linestyle="--")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # Best experiment
    best = get_best_experiment()
    if best:
        st.markdown(f"**Best experiment**: Iteration {best['iteration']} — Expectancy {best['expectancy']:.2f}%")
else:
    st.info("No experiments run yet. Click 'Run Optimization' to start.")

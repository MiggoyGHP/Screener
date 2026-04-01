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
from screener.backtesting.backtester import backtest_scan
from screener.backtesting.rolling import rolling_backtest


st.title("Backtester")
st.markdown("Test screener performance on historical data. Measures forward returns and expectancy.")

# Sidebar
st.sidebar.header("Backtest Settings")

mode = st.sidebar.radio("Mode", ["Single Date", "Rolling Backtest"])

universe_choice = st.sidebar.selectbox(
    "Universe",
    ["S&P 500", "NASDAQ 100", "Custom"],
    key="bt_universe",
)

custom_tickers = ""
if universe_choice == "Custom":
    custom_tickers = st.sidebar.text_area(
        "Tickers (comma-separated)",
        "AAPL, MSFT, NVDA, AMZN, META, GOOG, TSLA, NFLX",
        key="bt_custom",
    )

pattern_filter = st.sidebar.multiselect(
    "Patterns",
    ["VCP", "Reversal", "Reset", "Coil"],
    default=["VCP", "Reversal", "Reset", "Coil"],
    key="bt_patterns",
)

hold_days = st.sidebar.slider("Hold Days", 5, 60, 20, key="bt_hold")
stop_loss = st.sidebar.slider("Stop Loss %", 3.0, 15.0, 7.0, 0.5, key="bt_stop")

if mode == "Single Date":
    scan_date = st.sidebar.date_input(
        "Scan Date",
        value=date(2024, 1, 15),
        min_value=date(2000, 1, 1),
        max_value=date(2025, 12, 31),
        key="bt_date",
    )
else:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start",
            value=date(2023, 1, 1),
            min_value=date(2000, 1, 1),
            key="bt_start",
        )
    with col2:
        end_date = st.date_input(
            "End",
            value=date(2024, 1, 1),
            min_value=date(2000, 1, 1),
            key="bt_end",
        )
    scan_interval = st.sidebar.slider("Scan Interval (days)", 5, 30, 5, key="bt_interval")

if st.sidebar.button("Run Backtest", type="primary", use_container_width=True):
    config = load_config()
    config.patterns.vcp.enabled = "VCP" in pattern_filter
    config.patterns.reversal.enabled = "Reversal" in pattern_filter
    config.patterns.reset.enabled = "Reset" in pattern_filter
    config.patterns.coil.enabled = "Coil" in pattern_filter

    # Get tickers
    with st.spinner("Fetching tickers..."):
        if universe_choice == "S&P 500":
            tickers = get_sp500_tickers()
        elif universe_choice == "NASDAQ 100":
            tickers = get_nasdaq100_tickers()
        else:
            tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]

    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_cb(current, total, phase):
        if total > 0:
            progress_bar.progress(current / total)
            status_text.text(f"{phase}: {current}/{total}")

    if mode == "Single Date":
        with st.spinner("Running backtest..."):
            result = backtest_scan(
                tickers, scan_date, config,
                hold_days=hold_days, stop_loss_pct=stop_loss,
                progress_callback=progress_cb,
            )
    else:
        with st.spinner("Running rolling backtest..."):
            result = rolling_backtest(
                tickers, start_date, end_date, config,
                scan_interval_days=scan_interval,
                hold_days=hold_days, stop_loss_pct=stop_loss,
                progress_callback=progress_cb,
            )

    progress_bar.empty()
    status_text.empty()

    st.session_state["bt_result"] = result

# Display results
if "bt_result" in st.session_state:
    result = st.session_state["bt_result"]

    if result.total_setups == 0:
        st.warning("No setups detected in the selected period.")
    else:
        # Metrics
        st.subheader("Performance Metrics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Trades", result.total_setups)
            st.metric("Wins", result.wins)
        with c2:
            st.metric("Win Rate", f"{result.win_rate}%")
            st.metric("Losses", result.losses)
        with c3:
            color = "normal" if result.expectancy > 0 else "inverse"
            st.metric("Expectancy", f"{result.expectancy}%", delta_color=color)
            st.metric("Profit Factor", f"{result.profit_factor}")
        with c4:
            st.metric("Avg Return", f"{result.avg_return_pct}%")
            st.metric("Best / Worst", f"{result.best_trade}% / {result.worst_trade}%")

        # Trades table
        st.markdown("---")
        st.subheader("Individual Trades")
        trades_data = []
        for t in result.trades:
            trades_data.append({
                "Ticker": t.ticker,
                "Pattern": t.pattern,
                "Scan Date": str(t.scan_date),
                "Entry": f"${t.entry_price:.2f}",
                "Exit": f"${t.exit_price:.2f}",
                "Return": f"{t.return_pct}%",
                "Hit Stop": t.hit_stop,
                "Days Held": t.hold_days,
                "Score": f"{t.score:.1f}",
            })
        df_trades = pd.DataFrame(trades_data)
        st.dataframe(df_trades, use_container_width=True, hide_index=True)

        # Equity curve
        if len(result.trades) > 1:
            st.markdown("---")
            st.subheader("Cumulative Returns")
            cumulative = [0.0]
            for t in result.trades:
                cumulative.append(cumulative[-1] + t.return_pct)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(range(len(cumulative)), cumulative, color="#26A69A", linewidth=1.5)
            ax.axhline(y=0, color="#EF5350", linewidth=0.5, linestyle="--")
            ax.set_xlabel("Trade #")
            ax.set_ylabel("Cumulative Return %")
            ax.set_title("Equity Curve (Cumulative % Return)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

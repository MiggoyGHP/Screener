import streamlit as st

st.set_page_config(
    page_title="Stock Pattern Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Stock Pattern Screener")
st.markdown(
    "Scan for **VCPs**, **Reversals**, **Resets**, and **Coils** across the market. "
    "Select a page from the sidebar to get started."
)

st.sidebar.success("Select a page above.")

st.markdown("---")
st.subheader("Quick Start")
st.markdown("""
1. **Screener Results** - Run a full market scan and browse detected setups ranked by score.
2. **Pattern Explorer** - Enter a single ticker to see all detected patterns with charts.
3. **Parameter Tuner** - Adjust detection parameters and see how they affect results in real time.
""")

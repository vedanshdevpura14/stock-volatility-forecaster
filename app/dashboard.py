import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from predictor import predict_volatility

BASE_PATH = r"E:\ML\End-to-End Stock Volatility Forecasting System"

st.set_page_config(
    page_title = "Volatility Forecaster",
    page_icon  = "📈",
    layout     = "wide"
)

st.title("📈 Stock Volatility Forecasting Dashboard")
st.markdown("*Predicts next-day annualized volatility using LightGBM*")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

AVAILABLE_STOCKS = {
    "TCS"       : "TCS.NS",
    "Reliance"  : "RELIANCE.NS",
    "HDFC Bank" : "HDFCBANK.NS",
    "Infosys"   : "INFY.NS",
    "State Bank": "SBIN.NS"
}

selected_name   = st.sidebar.selectbox(
    "Select Stock",
    options = list(AVAILABLE_STOCKS.keys())
)
selected_ticker = AVAILABLE_STOCKS[selected_name]

lookback_days = st.sidebar.slider(
    "Lookback Period (days)",
    min_value = 30,
    max_value = 365,
    value     = 120,
    step      = 10
)

st.sidebar.divider()
st.sidebar.info("Models trained on NSE data 2019-2024")

# ── Prediction ────────────────────────────────────────────────
with st.spinner(f"Fetching prediction for {selected_name}..."):
    result = predict_volatility(selected_ticker)

if "error" in result:
    st.error(f"❌ {result['error']}")
    st.stop()

# ── Metric cards ──────────────────────────────────────────────
st.subheader(f"📊 {selected_name} ({selected_ticker})")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Predicted Volatility", result["predicted_vol_pct"])
with col2:
    st.metric("Current Volatility",   f"{result['current_vol']:.2%}")
with col3:
    st.metric("Current Price",        f"₹{result['current_price']:,.2f}")
with col4:
    risk  = result["risk_level"]
    icon  = "🔴" if risk == "High" else "🟡" if risk == "Medium" else "🟢"
    st.metric("Risk Level", f"{icon} {risk}")

st.divider()

# ── Charts ────────────────────────────────────────────────────
st.subheader("📉 Historical Price & Volatility")

raw     = yf.download(selected_ticker, period=f"{lookback_days}d",
                      auto_adjust=True, progress=False)
close   = raw["Close"].squeeze()
returns = close.pct_change().dropna()
vol     = returns.rolling(20).std() * np.sqrt(252)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=close.index, y=close.values,
        mode="lines", name="Close Price",
        line=dict(color="steelblue", width=1.5)
    ))
    fig.update_layout(title=f"{selected_name} Close Price",
                      xaxis_title="Date", yaxis_title="Price (INR)", height=350)
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=vol.index, y=vol.values,
        mode="lines", name="Realized Vol",
        line=dict(color="darkorange", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,165,0,0.1)"
    ))
    fig2.add_hline(
        y=result["predicted_vol"], line_dash="dash", line_color="red",
        annotation_text=f"Predicted: {result['predicted_vol_pct']}"
    )
    fig2.update_layout(title=f"{selected_name} Realized Volatility",
                       xaxis_title="Date", yaxis_title="Ann. Volatility", height=350)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Multi stock comparison ────────────────────────────────────
st.subheader("🔍 Multi-Stock Risk Comparison")

with st.spinner("Loading all stocks..."):
    comparison = []
    for name, ticker in AVAILABLE_STOCKS.items():
        pred = predict_volatility(ticker)
        if "error" not in pred:
            comparison.append({
                "Stock"         : name,
                "Ticker"        : ticker,
                "Predicted Vol" : pred["predicted_vol_pct"],
                "Current Price" : f"₹{pred['current_price']:,.2f}",
                "RSI"           : pred["rsi"],
                "Risk Level"    : pred["risk_level"]
            })

comp_df = pd.DataFrame(comparison)

fig3 = px.bar(
    comp_df, x="Stock",
    y=[float(v.strip("%"))/100 for v in comp_df["Predicted Vol"]],
    color="Risk Level",
    color_discrete_map={"High":"#ff4444","Medium":"#ffaa00","Low":"#44bb44"},
    title="Predicted Next-Day Volatility by Stock",
    labels={"y": "Predicted Volatility"}
)
st.plotly_chart(fig3, use_container_width=True)

st.dataframe(comp_df, use_container_width=True, hide_index=True)

st.divider()
st.caption("Built with LightGBM + GARCH | Data: Yahoo Finance | NSE Stocks")

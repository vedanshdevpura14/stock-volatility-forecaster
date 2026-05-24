import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

BASE_PATH = r"E:\ML\End-to-End Stock Volatility Forecasting System"

# ── RSI helper ────────────────────────────────────────────────
def compute_rsi(price_series, window=14):
    delta    = price_series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ── Main prediction function ──────────────────────────────────
def predict_volatility(ticker: str) -> dict:

    # Download last 120 days of data
    raw = yf.download(
        ticker,
        period      = "120d",
        auto_adjust = True,
        progress    = False
    )

    if raw.empty:
        return {"error": f"No data found for {ticker}"}

    # Base series
    close   = raw["Close"].squeeze()
    returns = close.pct_change().dropna()
    vol     = returns.rolling(20).std() * np.sqrt(252)

    # Feature dataframe
    df = pd.DataFrame(index=returns.index)
    df["close"]   = close
    df["returns"] = returns
    df["vol"]     = vol

    # Lagged volatility
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"vol_lag_{lag}"] = df["vol"].shift(lag)

    # Lagged returns
    for lag in [1, 2, 3, 5]:
        df[f"return_lag_{lag}"] = df["returns"].shift(lag)

    # Rolling stats
    for window in [5, 10, 20, 60]:
        df[f"vol_roll_mean_{window}"] = df["vol"].shift(1).rolling(window).mean()
        df[f"vol_roll_std_{window}"]  = df["vol"].shift(1).rolling(window).std()

    # RSI
    df["rsi_14"] = compute_rsi(df["close"], window=14).shift(1)

    # Volume z-score
    volume              = raw["Volume"].squeeze().reindex(df.index)
    vol_mean            = volume.shift(1).rolling(20).mean()
    vol_std             = volume.shift(1).rolling(20).std()
    df["volume_zscore"] = (volume - vol_mean) / vol_std

    # Calendar features
    df["day_of_week"]   = df.index.dayofweek
    df["month"]         = df.index.month
    df["quarter"]       = df.index.quarter
    df["week_of_month"] = (df.index.day - 1) // 7 + 1

    # Drop NaN rows
    df = df.dropna()

    if df.empty:
        return {"error": "Not enough data to generate features"}

    # Load saved model
    model_path = os.path.join(
        BASE_PATH, "models",
        f"lgb_model_{ticker.replace('.', '_')}.pkl"
    )

    if not os.path.exists(model_path):
        return {"error": f"No trained model found for {ticker}"}

    model = joblib.load(model_path)

    # Build feature row
    feature_cols = [c for c in df.columns
                    if c not in ["close", "returns", "vol"]]

    X_latest   = df[feature_cols].iloc[[-1]]
    prediction = model.predict(X_latest)[0]

    return {
        "ticker"           : ticker,
        "prediction_date"  : str(df.index[-1].date()),
        "predicted_vol"    : round(float(prediction), 4),
        "predicted_vol_pct": f"{prediction:.2%}",
        "current_vol"      : round(float(df["vol"].iloc[-1]), 4),
        "current_price"    : round(float(close.iloc[-1]), 2),
        "rsi"              : round(float(df["rsi_14"].iloc[-1]), 2),
        "risk_level"       : "High"   if prediction > 0.30 else
                             "Medium" if prediction > 0.15 else "Low"
    }

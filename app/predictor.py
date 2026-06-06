import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def compute_rsi(price_series, window=14):
    delta    = price_series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def predict_volatility(ticker: str) -> dict:

    raw = yf.download(
        ticker,
        period      = "200d",
        auto_adjust = True,
        progress    = False
    )

    if raw.empty:
        return {"error": f"No data found for {ticker}"}

    close   = raw["Close"].squeeze()
    returns = close.pct_change().dropna()
    vol     = returns.rolling(20).std() * np.sqrt(252)

    df = pd.DataFrame(index=returns.index)
    df["close"]   = close
    df["returns"] = returns
    df["vol"]     = vol

    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"vol_lag_{lag}"] = df["vol"].shift(lag)

    for lag in [1, 2, 3, 5]:
        df[f"return_lag_{lag}"] = df["returns"].shift(lag)

    for window in [5, 10, 20, 60]:
        df[f"vol_roll_mean_{window}"] = df["vol"].shift(1).rolling(window).mean()
        df[f"vol_roll_std_{window}"]  = df["vol"].shift(1).rolling(window).std()

    df["rsi_14"] = compute_rsi(df["close"], window=14).shift(1)

    volume              = raw["Volume"].squeeze().reindex(df.index)
    vol_mean            = volume.shift(1).rolling(20).mean()
    vol_std             = volume.shift(1).rolling(20).std()
    df["volume_zscore"] = (volume - vol_mean) / vol_std

    df["day_of_week"]   = df.index.dayofweek
    df["month"]         = df.index.month
    df["quarter"]       = df.index.quarter
    df["week_of_month"] = (df.index.day - 1) // 7 + 1

    df = df.dropna()

    if df.empty:
        return {"error": "Not enough data to generate features"}

    model_path = os.path.join(
        BASE_PATH,
        "models",
        f"lgb_model_{ticker.replace('.', '_')}.pkl"
    )

    if not os.path.exists(model_path):
        return {"error": f"No trained model found for {ticker}"}

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Model loading failed for {ticker}: {repr(e)}")

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
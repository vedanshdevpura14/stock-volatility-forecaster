import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from predictor import predict_volatility
import uvicorn

app = FastAPI(
    title       = "Stock Volatility Forecaster",
    description = "Predicts next-day annualized volatility using LightGBM",
    version     = "1.0.0"
)

@app.get("/")
def root():
    return {
        "status" : "running",
        "message": "Volatility Forecaster API is live"
    }

@app.get("/predict/{ticker}")
def get_prediction(ticker: str):
    result = predict_volatility(ticker.upper())
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.get("/compare")
def compare_stocks(tickers: str = "TCS.NS,INFY.NS,HDFCBANK.NS"):
    ticker_list = [t.strip() for t in tickers.split(",")]
    results = []
    for ticker in ticker_list:
        pred = predict_volatility(ticker)
        if "error" not in pred:
            results.append(pred)
    results = sorted(results, key=lambda x: x["predicted_vol"], reverse=True)
    return {"count": len(results), "stocks": results}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

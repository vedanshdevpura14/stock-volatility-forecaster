"I built a system that predicts how risky a stock will be tomorrow, and deployed it as a live web application."
"Most people try to predict stock prices — which is nearly 
impossible. I took a smarter approach and predicted volatility 
— meaning how much a stock will move, not which direction.

I took 5 major Indian stocks like TCS, Infosys, HDFC Bank, 
Reliance and State Bank. I downloaded 5 years of their 
historical data, built 20+ features from it like momentum, 
trend, volume spikes and technical indicators.

I then trained two models — GARCH which is the traditional 
finance industry standard, and LightGBM which is a modern 
machine learning model. I compared both and LightGBM 
outperformed GARCH.

Finally I deployed the whole thing as a REST API using FastAPI 
and built an interactive dashboard using Streamlit where you 
can select any stock and see its predicted volatility, risk 
level and historical charts in real time."

Why did you predict volatility instead of stock price?

"Predicting stock prices is essentially a random walk — even the best hedge funds struggle with it. Volatility on the other hand is mean reverting, meaning high volatility tends to come back down and low volatility tends to stay low. This makes it more predictable. Also volatility is directly used in the real world by options traders, risk managers and portfolio managers — so it has clear business value."


Q: What is volatility exactly?

"Volatility measures how much a stock's price moves, not which direction. I calculated it as the rolling 20-day standard deviation of daily returns, annualized by multiplying with square root of 252 — which is the number of trading days in a year. So a volatility of 25% means the stock is moving at a 25% annualized rate."


Q: What features did you engineer and why?

"I built four types of features. First, lagged volatility — yesterday's volatility is the single strongest predictor of tomorrow's, this is called volatility clustering. Second, lagged returns — to capture momentum and direction. Third, rolling statistics — 5, 10, 20 and 60 day rolling mean and standard deviation of volatility to capture short and long term trends. Fourth, technical indicators like RSI to detect overbought or oversold conditions, volume z-score to detect unusual trading activity, and calendar features like day of week because markets behave differently on Mondays versus Fridays."


Q: Why did you use LightGBM?

"LightGBM is a gradient boosting algorithm that works extremely well on tabular data. It's fast, handles missing values well, and is widely used in the finance industry for exactly this type of problem. It also gives feature importance scores which helps with explainability — important in finance where you need to justify your model's decisions."


Q: What is GARCH and why did you use it?

"GARCH stands for Generalized Autoregressive Conditional Heteroskedasticity. It's a statistical model developed in the 1980s that explicitly models volatility clustering — the idea that volatile periods tend to stay volatile. It's the industry standard baseline used by banks and hedge funds daily. I used it as my baseline to benchmark against — if my ML model couldn't beat GARCH, something was wrong."


Q: How did you validate your model?

"I used walk-forward validation instead of a random train-test split. This means I trained on past data and tested on future data — simulating exactly how the model would perform in real production. For example train on 2019 to 2022, test on 2023, then train on 2019 to 2023, test on 2024. A random split would leak future information into training which gives artificially good results but fails in production — this is called data leakage."


Q: What is data leakage and how did you prevent it?

"Data leakage is when your model accidentally sees future information during training, making it look accurate in testing but fail completely in production. I prevented it in two ways — first by always using shift(1) before computing rolling features, meaning on any given day the model only sees yesterday's data. Second by using walk-forward validation so the test set is always in the future relative to the training set."


Q: What metrics did you use to evaluate the model?

"I used three metrics. MAE — Mean Absolute Error — which tells the average error in volatility units and is the most intuitive. RMSE — Root Mean Squared Error — which penalizes large errors more heavily. And MAPE — Mean Absolute Percentage Error — which expresses error as a percentage, making it easy to explain to non-technical stakeholders. LightGBM outperformed GARCH on all three."


Q: How did you deploy it?

"I built a REST API using FastAPI with three endpoints — a health check, a single stock prediction endpoint, and a multi-stock comparison endpoint. FastAPI automatically generates interactive API documentation. I also built an interactive dashboard using Streamlit where users can select a stock, adjust the lookback period and see real-time predictions, historical price charts, volatility charts and a risk comparison across all five stocks. The dashboard is deployed publicly on Streamlit Cloud."


Q: What is FastAPI and why did you choose it?

"FastAPI is a modern Python web framework for building APIs. I chose it because it's extremely fast, automatically validates inputs using Python type hints, and auto-generates interactive API documentation at the /docs endpoint. This means anyone can test the API directly from the browser without writing any code."


Q: What would you improve if you had more time?

"Three things. First I would add SHAP values for explainability — showing exactly which features drove each prediction. Second I would implement MLflow for experiment tracking to systematically compare model versions. Third I would add more stocks and extend to multi-asset correlation forecasting — predicting how volatility spreads across sectors. I would also containerize with Docker for cleaner deployment."


Q: Who would use this in the real world?

"Options traders use volatility forecasts to price options contracts. Risk managers at banks use it to calculate Value at Risk — how much they could lose tomorrow. Portfolio managers use it to rebalance away from high-risk stocks. Fintechs use it to set dynamic margin requirements. So the end users are across the entire finance industry."


Q: What was the hardest part of this project?

"Two things. First, avoiding data leakage in time series — it's very easy to accidentally use future data in your features which makes results look great but fail in production. Second, the GARCH rolling forecast — refitting GARCH on an expanding window for every single test day was computationally expensive and required careful handling of convergence failures which produced NaN values that had to be cleaned before evaluation."


"I built an end-to-end machine learning system that predicts next-day stock volatility for five NSE stocks. I engineered over 20 features from raw price data, trained and compared a GARCH baseline against a LightGBM model using walk-forward validation to prevent data leakage, and deployed the system as a REST API with FastAPI and an interactive dashboard with Streamlit. The project covers the full ML lifecycle from data collection to production deployment."

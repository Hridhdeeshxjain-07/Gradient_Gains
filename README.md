# 📈 Gradient Gains: Machine Learning Meets Quantitative Finance

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Gradient Gains** is an end-to-end quantitative finance platform and interactive Streamlit dashboard[cite: 1, 11]. It is designed to predict stock prices for the next 3 days based on historical market data[cite: 10]. By combining modern machine learning ensembles with traditional financial momentum indicators, the platform provides actionable forecasting and trading signals[cite: 1, 16].

---

## 🚀 Development Timeline & Hackathon

This project was developed over a structured 4-week timeline, culminating in a robust financial forecasting tool[cite: 3]:
* **Week 1 (Foundations):** Set up foundational tools (Git, GitHub, Google Colab) and explored simple linear regression and gradient descent using NumPy, Pandas, and Matplotlib[cite: 4].
* **Week 2 (Core ML & Finance):** Explored linear/logistic regression, decision trees, and fundamental technical indicators (SMA, EMA, RSI, MACD)[cite: 5]. 
* **Week 3 (Advanced Models & Hackathon):** Explored Neural Networks, Random Forests, Boosting, and Bagging[cite: 7]. The week featured a competitive hackathon requiring teams to forecast the next day's closing price of a mystery stock using 4 years of historical data, evaluated by RMSE[cite: 7].
* **Week 4 (Deep Learning & UI):** Explored Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks, resolving vanishing gradient problems, and ultimately built the final Streamlit frontend[cite: 9].

---

## 🧠 Hybrid Model Architecture

After evaluating multiple architectures ranging from simple linear regression to deep sequence models (RNN/LSTM), this project utilizes a **Hybrid XGBoost & Random Forest Ensemble** as its final model[cite: 14, 15].

* **The Bias-Variance Tradeoff:** Neural networks and basic LSTMs were found to either overfit market noise or suffer from prediction lag[cite: 15]. 
* **XGBoost (Weight 2):** Highly sensitive to small but significant price trend shifts, reducing overall model bias (accuracy)[cite: 14, 15].
* **Random Forest (Weight 1):** Reduces model variance (stability), ensuring the model filters out random market noise without simply memorizing training data[cite: 14, 15].

### Model Performance Comparison (Backtest)
*Testing metric: Mean Absolute Percentage Error (MAPE) on an unseen 60-day test set[cite: 15, 17, 18].*

| Asset Ticker | Asset Name | Linear Reg. | RNN / LSTM | XGBoost | Random Forest | **XGB + RF (Final)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `HDFCBANK.NS` | HDFC Bank | 9.235% | 2.511% | 0.751% | 0.871% | **0.691%** |
| `BHARTIARTL.NS`| Airtel | 8.624% | 3.296% | 0.761% | 0.834% | **0.706%** |
| `AAPL` | Apple Inc. | 8.750% | 4.459% | 1.041% | 1.227% | **0.926%** |
| `RELIANCE.NS` | Reliance Ind.| 5.216% | 4.382% | 1.378% | 1.479% | **1.164%** |
| `BTC-USD` | Bitcoin | 12.023% | 7.090% | 3.162% | 4.359% | **2.838%** |

---

## ⚙️ Data Pipeline & Feature Engineering

The backend logic relies heavily on custom feature engineering to give the models contextual financial understanding[cite: 1, 13]:

1. **Dynamic Data Fetching:** Fetches 3 years of daily historical OHLCV (Open, High, Low, Close, Volume) data dynamically via `yfinance` to ensure sufficient training volume[cite: 12].
2. **Lagged Feature Construction:** Generates 21-day lagged features applied to OHLCV data to provide historical context[cite: 13].
3. **Log Returns:** Calculates 1, 5, 10, and 20-day log returns to capture price velocity[cite: 13].
4. **Mean Reversion (Moving Averages):** Calculates the ratio of the current price relative to its 5, 10, and 20-day moving averages[cite: 13].
5. **Volatility Tracking:** A custom `vol5` feature calculates the rolling standard deviation over a 5-day window to measure market nervousness[cite: 13].
6. **Data Integrity:** Missing values are scrubbed post-feature generation, ensuring the model trains exclusively on complete rows[cite: 12].

---

## 📊 Momentum Analysis & Interactive UI

The Streamlit UI provides deep interactive visualizations and localized metrics[cite: 11, 20]:

* **RSI (Relative Strength Index) Signals:** Calculates a 14-day RSI, plotting dynamic color-coded trading signals directly to the user: **BUY** (Green, RSI < 40), **SELL** (Red, RSI > 60), and **HOLD** (Gray)[cite: 1, 16].
* **Hybrid Charting:** Utilizes Plotly to render multi-row subplots featuring the asset's price, 50-day EMA, 200-day EMA, and a dotted "lines+markers" trace for the 3-day forecast[cite: 1, 20].
* **Contextual Metrics & Localization:** Automatically swaps currency symbols ($ or ₹) based on asset origin (US vs. NSE), displaying exact next-close predictions alongside percentage deltas[cite: 1, 20].
* **Walk-Forward Validation:** Transparently displays backtest performance (RMSE and MAPE) for the specific selected asset using the last 60 trading days as a dedicated test set[cite: 17, 18, 19].

---

## 🛠️ Technology Stack

- **Frontend Application:** Streamlit[cite: 1, 11]
- **Data Acquisition:** `yfinance`[cite: 1, 11]
- **Numerical Computation & Data Processing:** NumPy, Pandas[cite: 1, 11]
- **Machine Learning & Scaling:** Scikit-Learn (`VotingRegressor`, `RandomForestRegressor`), XGBoost[cite: 1, 11, 14]
- **Interactive Visualizations:** Plotly[cite: 1, 11]

---

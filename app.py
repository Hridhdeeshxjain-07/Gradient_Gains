import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb

st.set_page_config(page_title="Stock Predictor", layout="wide")

st.title("BCS Gradient Gains: XGBoost + Random Forest Model")
st.markdown("---")


st.sidebar.header("Configuration")


if 'ticker' not in st.session_state:
    st.session_state.ticker = "AAPL"

def set_ticker(ticker):
    st.session_state.ticker = ticker


selected_ticker = st.sidebar.text_input("Search Ticker", value=st.session_state.ticker).upper()

# US STOCKS 
st.sidebar.markdown("### 🇺🇸 US Giants")
u1, u2, u3 = st.sidebar.columns(3)
if u1.button("AAPL"): set_ticker("AAPL")
if u2.button("MSFT"): set_ticker("MSFT")
if u3.button("GOOGL"): set_ticker("GOOGL")

u4, u5, u6 = st.sidebar.columns(3)
if u4.button("AMZN"): set_ticker("AMZN")
if u5.button("META"): set_ticker("META")
if u6.button("NFLX"): set_ticker("NFLX")

st.sidebar.markdown("#### Chips & Auto")
u7, u8, u9 = st.sidebar.columns(3)
if u7.button("NVDA"): set_ticker("NVDA")
if u8.button("AMD"): set_ticker("AMD")
if u9.button("TSLA"): set_ticker("TSLA")

#  INDIAN STOCKS 
st.sidebar.markdown("### 🇮🇳 Indian Market")
st.sidebar.caption("NSE (National Stock Exchange)")

i1, i2, i3 = st.sidebar.columns(3)
if i1.button("RELIANCE"): set_ticker("RELIANCE.NS")
if i2.button("TCS"): set_ticker("TCS.NS")
if i3.button("INFY"): set_ticker("INFY.NS")

i4, i5, i6 = st.sidebar.columns(3)
if i4.button("HDFC"): set_ticker("HDFCBANK.NS")
if i5.button("ICICI"): set_ticker("ICICIBANK.NS")
if i6.button("SBI"): set_ticker("SBIN.NS")

i7, i8, i9 = st.sidebar.columns(3)
if i7.button("TATA"): set_ticker("TATAMOTORS.NS")
if i8.button("ADANI"): set_ticker("ADANIENT.NS")
if i9.button("AIRTEL"): set_ticker("BHARTIARTL.NS")

#  CRYPTO 
st.sidebar.markdown("### 🪙 Crypto")
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("BTC"): set_ticker("BTC-USD")
if c2.button("ETH"): set_ticker("ETH-USD")
if c3.button("SOL"): set_ticker("SOL-USD")

#  GRAPH SETTINGS 
st.sidebar.markdown("---")
lookback_options = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "5 Years": 1825
}
selected_lookback = st.sidebar.selectbox("Graph Timeframe", list(lookback_options.keys()))
days_to_plot = lookback_options[selected_lookback]

# 3. BACKEND LOGIC


def clean_columns(df):
    df.columns = [col.lower() for col in df.columns]
    return df

def create_lags(df):
    df = df.copy()
    target_cols = ["open", "high", "low", "close", "volume"]
    for col in target_cols:
        if col in df.columns:
            for i in range(21):
                df[f"{col}_lag_{i}"] = df[col].shift(i)
    return df

def create_features(df):
    # Log Returns
    for i in [1, 5, 10, 20]:
        if f"close_lag_{i}" in df.columns:
            df[f"logret_{i}"] = np.log(df["close_lag_0"] / df[f"close_lag_{i}"])
    # Moving Averages relative to price
    for i in [5, 10, 20]:
        cols = [f"close_lag_{j}" for j in range(i) if f"close_lag_{j}" in df.columns]
        if cols:
            ma = df[cols].mean(axis=1)
            df[f"price_ma_{i}"] = df["close_lag_0"] / ma
    # Volatility
    cols = [f"close_lag_{i}" for i in range(5) if f"close_lag_{i}" in df.columns]
    if cols:
        df["vol5"] = df[cols].std(axis=1)
    return df

def add_technical_indicators(df):
    df = df.copy()
    # EMA 50 & 200
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def train_and_predict(ticker, horizon=3):
    # 1. Fetch 3 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    
    df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), 
                     end=end_date.strftime('%Y-%m-%d'), interval="1d")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.reset_index(inplace=True)
    df = clean_columns(df)
    work_df = df.copy()

    # 2. Define Model: XGBoost + RandomForest
    xgb_model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=5, 
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )
    rf_model = RandomForestRegressor(
        n_estimators=400, max_depth=10, min_samples_leaf=5, 
        random_state=42, n_jobs=-1
    )
    model = VotingRegressor([("xgb", xgb_model), ("rf", rf_model)], weights=[2, 1])

    # 3. Prepare Dataset
    full_df = create_lags(work_df)
    full_df = create_features(full_df)
    
    full_df["target_price"] = full_df["close_lag_0"].shift(-1)
    full_df["y"] = np.log(full_df["target_price"] / full_df["close_lag_0"])
    
    features = []
    for i in range(10):
        if f"close_lag_{i}" in full_df.columns: features.append(f"close_lag_{i}")
    for i in [1, 5, 10, 20]:
        if f"logret_{i}" in full_df.columns: features.append(f"logret_{i}")
    for i in [5, 10, 20]:
        if f"price_ma_{i}" in full_df.columns: features.append(f"price_ma_{i}")
    if "vol5" in full_df.columns: features.append("vol5")

    clean_df = full_df.dropna().reset_index(drop=True)
    
    # 4. Train/Test Split (Last 60 days for validation)
    test_size = 60 
    train_df = clean_df.iloc[:-test_size]
    test_df = clean_df.iloc[-test_size:]
    
    X_train = train_df[features]
    y_train = train_df["y"]
    X_test = test_df[features]
    
    # Train Model
    model.fit(X_train, y_train)
    
    # Calculate Metrics
    y_pred = model.predict(X_test)
    test_df = test_df.copy()
    test_df["pred_price"] = test_df["close_lag_0"] * np.exp(y_pred)
    test_df["actual_price"] = test_df["target_price"]
    
    rmse = np.sqrt(mean_squared_error(test_df["actual_price"], test_df["pred_price"]))
    mape = mean_absolute_percentage_error(test_df["actual_price"], test_df["pred_price"]) * 100
    
    # 5. Retrain on FULL Data
    X_full = clean_df[features]
    y_full = clean_df["y"]
    model.fit(X_full, y_full)

    # 6. Predict Future (Recursive)
    future_predictions = []
    future_dates = []
    current_df = work_df.copy()
    
    last_actual_date = current_df['date'].iloc[-1]
    if isinstance(last_actual_date, str):
        last_actual_date = pd.to_datetime(last_actual_date)
    current_pred_date = last_actual_date
    
    for i in range(horizon):
        temp_df = create_lags(current_df)
        temp_df = create_features(temp_df)
        last_row = temp_df.iloc[[-1]][features]
        
        pred_log_ret = model.predict(last_row)[0]
        current_price = temp_df.iloc[-1]["close_lag_0"]
        next_price = current_price * np.exp(pred_log_ret)
        
        current_pred_date = current_pred_date + timedelta(days=1)
        while current_pred_date.weekday() >= 5: 
            current_pred_date = current_pred_date + timedelta(days=1)
            
        future_predictions.append(next_price)
        future_dates.append(current_pred_date)
        
        new_row = pd.DataFrame({
            "date": [current_pred_date],
            "open": [next_price], "high": [next_price], "low": [next_price], 
            "close": [next_price], "volume": [current_df.iloc[-1]["volume"]]
        })
        current_df = pd.concat([current_df, new_row], ignore_index=True)

    return df, future_dates, future_predictions, rmse, mape, len(train_df), len(test_df)


# 4. MAIN APP EXECUTION


if st.sidebar.button("Generate Prediction"):
    with st.spinner(f"Training XGBoost + Random Forest for {selected_ticker}..."):
        try:
            # 1. Run Pipeline
            hist_df, future_dates, future_preds, rmse, mape, train_s, test_s = train_and_predict(selected_ticker, horizon=3)
            
            # 2. Add Indicators (RSI, EMA)
            hist_df = add_technical_indicators(hist_df)
            
            # 3. Metrics
            currency_symbol = "₹" if selected_ticker.endswith(".NS") else "$"
            current_rsi = hist_df['RSI'].iloc[-1]
            if current_rsi < 40: signal_txt, signal_color = "BUY", "green"
            elif current_rsi > 60: signal_txt, signal_color = "SELL", "red"
            else: signal_txt, signal_color = "HOLD", "gray"
            
            # Performance Metrics
            st.subheader("📊 Model Performance (Backtest)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE", f"{currency_symbol}{rmse:.2f}")
            c2.metric("MAPE", f"{mape:.2f}%")
            c3.metric("Train Data", f"{train_s} days")
            c4.metric("Test Data", f"{test_s} days")
            st.markdown("---")
            
            # Price Prediction
            st.subheader("💰 Price Prediction")
            m1, m2, m3 = st.columns(3)
            current_price = hist_df['close'].iloc[-1]
            pred_price = future_preds[0]
            pct = ((pred_price - current_price) / current_price) * 100
            
            m1.metric("Current Price", f"{currency_symbol}{current_price:,.2f}")
            m2.metric("Next Close", f"{currency_symbol}{pred_price:,.2f}", f"{pct:.2f}%")
            m3.markdown(f"**Signal:** :{signal_color}[{signal_txt}] (RSI: {current_rsi:.1f})")
            
            # 4. PLOTTING (With Subplots)
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
                subplot_titles=(f"{selected_ticker} Price & EMAs", "RSI (14)")
            )
            
            cutoff = hist_df['date'].iloc[-1] - timedelta(days=days_to_plot)
            plot_df = hist_df[hist_df['date'] > cutoff]
            
            # --- Row 1: Price + EMA + Prediction ---
            fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['close'], mode='lines', name='Price', line=dict(color='#00f2ff', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['EMA_50'], mode='lines', name='EMA 50', line=dict(color='#ffa500', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['EMA_200'], mode='lines', name='EMA 200', line=dict(color='#9370db', width=1)), row=1, col=1)
            
            pred_x = [plot_df['date'].iloc[-1]] + future_dates
            pred_y = [plot_df['close'].iloc[-1]] + future_preds
            fig.add_trace(go.Scatter(x=pred_x, y=pred_y, mode='lines+markers', name='Forecast', line=dict(color='#ff0055', width=2, dash='dot')), row=1, col=1)
            
            # --- Row 2: RSI ---
            fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['RSI'], mode='lines', name='RSI', line=dict(color='#e0e0e0', width=1.5)), row=2, col=1)
            fig.add_hline(y=60, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=40, line_dash="dot", line_color="green", row=2, col=1)
            
            fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            # Data Table
            st.write("### Forecast Data")
            f_df = pd.DataFrame({"Date": future_dates, "Predicted Close": [f"{currency_symbol}{x:,.2f}" for x in future_preds]})
            st.dataframe(f_df)

        except Exception as e:
            st.error(f"Error: {e}")
            st.warning("Please check if the ticker is correct.")
else:
    st.info("Select a ticker from the sidebar and click 'Generate Prediction'.")
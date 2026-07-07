"""
MarketSentinel Streamlit Dashboard - Main App
6-page interactive dashboard for stock prediction and backtesting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import logging

from src.utils import load_config
from src.data_loader import get_available_tickers, load_stock_data
from src.trainer import load_model_artifacts
from src.features import engineer_features

# Configure page
st.set_page_config(
    page_title="MarketSentinel",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .metric-card {
        background-color: #1f2937;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00d9ff;
    }
    .disclaimer {
        background-color: #dc2626;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load config
config = load_config("config.yaml")

# Session state
if 'ticker' not in st.session_state:
    st.session_state.ticker = config['data']['default_ticker']

# Disclaimer
st.markdown("""
<div class="disclaimer">
⚠️ EDUCATIONAL USE ONLY — This is NOT financial advice. Do not use for actual trading decisions.
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📈 MarketSentinel")
st.sidebar.markdown("**AI-Powered Stock Prediction System**")

# Kaggle status
kaggle_status = "✅ Kaggle Connected" if Path("kaggle.json").exists() else "🔴 Using Synthetic Data"
st.sidebar.info(f"Data Status: {kaggle_status}")

# Ticker selection
available_tickers = get_available_tickers(config['paths']['data_raw'])
if not available_tickers:
    available_tickers = config['data']['available_tickers']

selected_ticker = st.sidebar.selectbox(
    "📊 Select Ticker",
    available_tickers,
    index=available_tickers.index(st.session_state.ticker) if st.session_state.ticker in available_tickers else 0
)
st.session_state.ticker = selected_ticker

# Navigation
page = st.sidebar.radio(
    "🗂️ Navigation",
    ["🏠 Dashboard", "🎯 Predict", "💰 Backtest", "📈 Analytics", "⚙️ Train", "ℹ️ About"]
)

# Load data
@st.cache_data
def load_ticker_data(ticker):
    try:
        df = load_stock_data(ticker, config['paths']['data_raw'])
        if df.empty:
            return None
        return df
    except:
        return None

# Load predictions
@st.cache_data
def load_predictions(ticker):
    pred_path = Path(config['paths']['data_processed']) / f"{ticker}_predictions.csv"
    if pred_path.exists():
        return pd.read_csv(pred_path)
    return None

# Load model
@st.cache_resource
def load_model(ticker):
    try:
        model = load_model_artifacts(ticker, config)
        return model
    except:
        return None

# Main navigation
if page == "🏠 Dashboard":
    st.title("📊 Live Dashboard")
    
    df = load_ticker_data(st.session_state.ticker)
    predictions = load_predictions(st.session_state.ticker)
    
    if df is not None:
        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        latest_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        price_change = ((latest_price - prev_price) / prev_price) * 100
        
        with col1:
            st.metric("Latest Price", f"${latest_price:.2f}", f"{price_change:+.2f}%")
        
        if predictions is not None:
            accuracy = (predictions['correct_rf'].mean()) * 100
            with col2:
                st.metric("Model Accuracy", f"{accuracy:.1f}%")
            
            precision = predictions['prediction_rf'].sum() / len(predictions) * 100
            with col3:
                st.metric("UP Predictions", f"{precision:.1f}%")
            
            avg_confidence = predictions['confidence_rf'].mean()
            with col4:
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            
            # Latest prediction
            latest_pred = predictions.iloc[-1]
            pred_text = "📈 UP" if latest_pred['prediction_rf'] == 1 else "📉 DOWN"
            with col5:
                st.metric("Latest Signal", pred_text)
        
        # Price chart
        st.subheader("💹 Price Chart (Last 365 Days)")
        
        df_year = df.tail(252)
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_year.index,
            open=df_year['open'],
            high=df_year['high'],
            low=df_year['low'],
            close=df_year['close'],
            name='Price'
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=df_year.index,
            y=df_year['close'].rolling(20).mean(),
            name='SMA 20',
            line=dict(color='orange', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df_year.index,
            y=df_year['close'].rolling(50).mean(),
            name='SMA 50',
            line=dict(color='blue', width=1)
        ))
        
        fig.update_layout(
            title=f"{st.session_state.ticker} Price Chart",
            xaxis_rangeslider_visible=False,
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Predictions table
        if predictions is not None:
            st.subheader("📋 Recent Predictions")
            
            recent = predictions.tail(20).copy()
            recent['date'] = pd.to_datetime(recent['date'])
            recent['correct'] = recent['correct_rf'].apply(lambda x: "✅" if x else "❌")
            recent['signal'] = recent['prediction_rf'].apply(lambda x: "📈" if x else "📉")
            
            display_cols = ['date', 'signal', 'probability_up_rf', 'confidence_rf', 'actual', 'correct']
            display_df = recent[display_cols].rename(columns={
                'date': 'Date',
                'signal': 'Signal',
                'probability_up_rf': 'Probability',
                'confidence_rf': 'Confidence',
                'actual': 'Actual',
                'correct': 'Result'
            })
            
            st.dataframe(display_df, use_container_width=True)
    else:
        st.error(f"❌ No data available for {st.session_state.ticker}")

elif page == "🎯 Predict":
    st.title("🎯 Next-Day Prediction")
    
    df = load_ticker_data(st.session_state.ticker)
    model = load_model(st.session_state.ticker)
    
    if df is not None and model is not None:
        # Engineer features for latest data
        df_features = engineer_features(df, config)
        
        if not df_features.empty:
            latest = df_features.iloc[-1:].drop(columns=['target'])
            
            pred, confidence = model.predict_with_confidence(latest)
            proba = model.predict_proba(latest)[0, 1]
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                signal_text = "📈 PRICE GOES UP" if pred[0] == 1 else "📉 PRICE GOES DOWN"
                st.markdown(f"# {signal_text}")
            
            with col2:
                st.metric("Confidence", f"{confidence[0]*100:.1f}%")
            
            with col3:
                st.metric("Probability UP", f"{proba*100:.1f}%")
            
            # Feature values
            st.subheader("📊 Top Features (Current Values)")
            
            feature_imp = model.get_feature_importances(top_n=10)
            
            feature_data = []
            for feat_name in feature_imp.index:
                if feat_name in latest.columns:
                    value = latest[feat_name].values[0]
                    importance = feature_imp[feat_name]
                    feature_data.append({
                        'Feature': feat_name,
                        'Value': f"{value:.4f}",
                        'Importance': importance
                    })
            
            feature_df = pd.DataFrame(feature_data)
            st.dataframe(feature_df, use_container_width=True)
    else:
        if model is None:
            st.warning(f"⚠️ Model not trained for {st.session_state.ticker}")
            st.info(f"Run the Train page to train a model first.")
        if df is None:
            st.error(f"❌ No data available for {st.session_state.ticker}")

elif page == "💰 Backtest":
    st.title("💰 Backtest Results")
    
    # Load backtest results
    backtest_path = Path(config['paths']['reports']) / f"{st.session_state.ticker}_backtest_report.txt"
    equity_curve_path = Path(config['paths']['reports']) / f"{st.session_state.ticker}_equity_curve.csv"
    
    if backtest_path.exists():
        with open(backtest_path, 'r') as f:
            report_text = f.read()
        st.markdown(f"```\n{report_text}\n```")
        
        # Plot equity curve if available
        if equity_curve_path.exists():
            equity_df = pd.read_csv(equity_curve_path, index_col=0, parse_dates=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df.index,
                y=equity_df.values,
                name='Strategy',
                fill='tozeroy'
            ))
            fig.update_layout(
                title=f"{st.session_state.ticker} Equity Curve",
                yaxis_title="Portfolio Value ($)",
                xaxis_title="Date",
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"⚠️ No backtest results available for {st.session_state.ticker}")
        st.info("Run training and backtesting from the command line first.")

elif page == "📈 Analytics":
    st.title("📈 Advanced Analytics")
    
    st.info("Detailed analysis features will be available after model training.")
    
    # Placeholder for future analytics
    st.subheader("Feature Analysis")
    st.markdown("""
    - Feature importance rankings
    - Feature correlation heatmap
    - Prediction accuracy by confidence level
    - Market regime analysis
    """)

elif page == "⚙️ Train":
    st.title("⚙️ Train Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Training Configuration")
        
        mode = st.radio("Training Mode", ["Walk-Forward", "Train-Test Split", "Full Retrain"])
        mode_map = {
            "Walk-Forward": "walk_forward",
            "Train-Test Split": "train_test_split",
            "Full Retrain": "full_retrain"
        }
        
        st.markdown("""
        **Walk-Forward:** Simulate realistic trading by retraining periodically
        
        **Train-Test Split:** Simple 80-20 chronological split
        
        **Full Retrain:** Train on all available data (for production deployment)
        """)
        
        st.info("To train a model, run from command line:\n\n```bash\npython train.py --ticker {} --mode {}\n```".format(
            st.session_state.ticker, mode_map[mode]
        ))
    
    with col2:
        st.subheader("Training Status")
        
        model = load_model(st.session_state.ticker)
        if model is not None:
            st.success("✅ Model Trained")
            metadata = model.get_model_metadata()
            st.json(metadata)
        else:
            st.error("❌ Model Not Trained")

elif page == "ℹ️ About":
    st.title("ℹ️ About MarketSentinel")
    
    st.markdown("""
    ## 📊 What is MarketSentinel?
    
    MarketSentinel is an **educational machine learning system** that predicts next-day 
    stock price direction (UP/DOWN) based on technical indicators and historical patterns.
    
    ### 🎯 Key Features
    - **Random Forest Model**: Ensemble learning with 300 decision trees
    - **30+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, and more
    - **Walk-Forward Validation**: Prevents lookahead bias for realistic performance
    - **Comprehensive Backtesting**: Trading simulation with transaction costs
    - **Interactive Dashboard**: Real-time visualizations and analytics
    
    ### ⚠️ IMPORTANT DISCLAIMER
    
    **This is an EDUCATIONAL project. NOT financial advice.**
    
    - Stock markets are inherently unpredictable
    - Past performance does not guarantee future results
    - Do NOT use for actual trading without professional guidance
    - Always consult a qualified financial advisor
    
    ### 📚 Technical Details
    
    **Data Source:** Kaggle S&P 500 dataset or synthetic fallback
    
    **Feature Engineering:**
    - Price features (returns, ranges, ratios)
    - Moving averages (SMA, EMA)
    - Momentum indicators (RSI, MACD, ROC)
    - Volatility features (ATR, Bollinger Bands)
    - Volume indicators (OBV, Volume Ratio)
    - Temporal features (day of week, month, quarter)
    
    **Model Architecture:**
    - Primary: Random Forest Classifier (300 trees)
    - Baseline: Logistic Regression for comparison
    - No lookahead bias: Features at time t use only data ≤ t
    
    **Validation Strategy:**
    - Walk-forward validation with expanding window
    - 252-day (1 year) initial training window
    - Retrain every 20 days
    - Single-day ahead predictions
    
    ### 📊 Typical Performance
    
    - Accuracy: 50-55% (better than random 50%)
    - Sharpe Ratio: 0.2-0.5 (positive risk-adjusted returns)
    - Max Drawdown: 10-20% (manageable risk)
    
    Note: Modest accuracy is realistic for stock prediction. Even 51-52% accuracy 
    can be profitable with proper risk management.
    
    ### 🔧 Technology Stack
    
    - **Python 3.10+**: Programming language
    - **Pandas**: Data manipulation
    - **scikit-learn**: Machine learning
    - **Streamlit**: Interactive dashboard
    - **Plotly**: Interactive visualizations
    - **Kaggle API**: Data source
    
    ### 📝 License
    
    MIT License - Feel free to fork and modify for educational purposes.
    
    ### 🙏 Acknowledgments
    
    - Kaggle for the S&P 500 dataset
    - scikit-learn community
    - Streamlit for the amazing framework
    """)

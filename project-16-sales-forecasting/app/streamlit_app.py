"""Project 16 — Sales Forecasting — Streamlit dashboard.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Allow `import src...` when launched as `streamlit run app/streamlit_app.py`
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils import load_config, ensure_dirs
from src.data_generator import generate_sales_data
from src.features import create_features, get_feature_columns
from src.model import build_model
from src.evaluator import compute_metrics

st.set_page_config(page_title="Sales Forecasting", layout="wide")

config = load_config()
ensure_dirs(config)

st.title("📈 Project 16 — Sales Forecasting")
st.caption("Lag features + rolling statistics · Linear Regression vs Random Forest")

# ---------------------------------------------------------------- Sidebar
st.sidebar.header("Data")
data_source = st.sidebar.radio("Data source", ["Synthetic (generated)", "Upload CSV"])

if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("CSV with 'date' and 'sales' columns", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded, parse_dates=["date"])
    else:
        st.info("Upload a CSV, or switch to synthetic data, to continue.")
        st.stop()
else:
    n_days = st.sidebar.slider("Days of history", 180, 1095, config["data"]["n_days"], step=30)
    cfg = dict(config)
    cfg["data"] = dict(config["data"])
    cfg["data"]["n_days"] = n_days
    df = generate_sales_data(cfg)

st.sidebar.header("Model")
model_type = st.sidebar.selectbox("Model type", ["rf", "linear"], format_func=lambda x: {
    "rf": "Random Forest", "linear": "Linear Regression"
}[x])
train_split = st.sidebar.slider("Train split", 0.5, 0.95, config["evaluation"]["train_split"], step=0.05)

# ---------------------------------------------------------------- Pipeline
df_feat = create_features(df, config["features"]["lags"], config["features"]["windows"])
feature_cols = get_feature_columns(df_feat)

split_idx = int(len(df_feat) * train_split)
train_df, test_df = df_feat.iloc[:split_idx], df_feat.iloc[split_idx:]

X_train, y_train = train_df[feature_cols], train_df["sales"]
X_test, y_test = test_df[feature_cols], test_df["sales"]

model = build_model(model_type, config)
model.fit(X_train, y_train)
preds = model.predict(X_test)

metrics = compute_metrics(y_test, preds)

# ---------------------------------------------------------------- Layout
col1, col2, col3 = st.columns(3)
col1.metric("MAE", metrics["MAE"])
col2.metric("RMSE", metrics["RMSE"])
col3.metric("MAPE (%)", metrics["MAPE_%"])

fig = go.Figure()
fig.add_trace(go.Scatter(x=train_df["date"], y=train_df["sales"], name="Train (actual)",
                          line=dict(color="lightgray")))
fig.add_trace(go.Scatter(x=test_df["date"], y=y_test, name="Test (actual)",
                          line=dict(color="royalblue")))
fig.add_trace(go.Scatter(x=test_df["date"], y=preds, name="Forecast",
                          line=dict(color="orangered", dash="dash")))
fig.update_layout(height=450, xaxis_title="Date", yaxis_title="Sales",
                   legend=dict(orientation="h", y=1.05))
st.plotly_chart(fig, use_container_width=True)

if hasattr(model, "feature_importances_"):
    st.subheader("Feature importance")
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    st.bar_chart(importance)

with st.expander("Preview engineered features"):
    st.dataframe(df_feat.tail(20), use_container_width=True)

st.download_button(
    "Download forecast (CSV)",
    data=test_df[["date"]].assign(actual=y_test.values, forecast=preds).to_csv(index=False),
    file_name=f"{model_type}_forecast.csv",
    mime="text/csv",
)

"""
Streamlit dashboard for the Energy Consumption Forecasting project.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features import create_features, feature_columns
from src.trainer import train_all_models, select_and_save_best
from src.evaluator import weekday_error_chart, forecast_vs_actual_chart, model_comparison_chart, write_report

st.set_page_config(page_title="Energy Forecast Dashboard", layout="wide")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
config = yaml.safe_load(open(CONFIG_PATH))


@st.cache_data
def load_default_data():
    return pd.read_csv(config["paths"]["data"], parse_dates=["date"])


st.title("⚡ Energy Consumption Forecast Dashboard")
st.caption("Daily load forecasting with seasonality-aware features and multi-model comparison.")

with st.sidebar:
    st.header("Data Source")
    uploaded = st.file_uploader("Upload your own CSV (columns: date, consumption)", type="csv")
    st.caption("Leave empty to use the bundled synthetic dataset.")
    retrain = st.button("🔁 Retrain models on this data")

if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=["date"])
    st.sidebar.success(f"Loaded {len(df)} rows from upload.")
else:
    df = load_default_data()

df_feat = create_features(df, config)

model_path = Path(config["paths"]["best_model"])

if retrain or not model_path.exists():
    with st.spinner("Training candidate models..."):
        results, feature_cols = train_all_models(df_feat, config)
        best_name, metadata = select_and_save_best(results, feature_cols, config)

        figures_dir = config["paths"]["figures"]
        Path(figures_dir).mkdir(parents=True, exist_ok=True)
        weekday_mae = weekday_error_chart(results[best_name], figures_dir)
        forecast_vs_actual_chart(results[best_name], figures_dir)
        model_comparison_chart(results, figures_dir)
        write_report(best_name, metadata, weekday_mae,
                     str(Path(config["paths"]["reports"]) / "evaluation_report.txt"))
    st.session_state["results"] = results
    st.session_state["best_name"] = best_name
    st.session_state["metadata"] = metadata

results = st.session_state.get("results")
best_name = st.session_state.get("best_name")
metadata = st.session_state.get("metadata")

if results is None:
    st.info("Click **Retrain models** in the sidebar to generate results for this dataset.")
    st.stop()

best = results[best_name]

col1, col2, col3 = st.columns(3)
col1.metric("Best Model", best_name.replace("_", " ").title())
col2.metric("Test MAE (kWh)", f"{best['mae']:.2f}")
col3.metric("Test MAPE", f"{best['mape']:.2f}%")

st.subheader("📈 Forecast vs Actual (test period)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=pd.to_datetime(best["dates_test"]), y=best["y_test"],
                          mode="lines", name="Actual", line=dict(color="#333333")))
fig.add_trace(go.Scatter(x=pd.to_datetime(best["dates_test"]), y=best["preds"],
                          mode="lines", name="Forecast", line=dict(color="#E63946", dash="dash")))
fig.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)

with c1:
    st.subheader("📊 Model Comparison")
    comp_df = pd.DataFrame({
        "model": list(metadata["metrics"].keys()),
        "MAE": [m["mae"] for m in metadata["metrics"].values()],
    })
    st.bar_chart(comp_df.set_index("model"))

with c2:
    st.subheader("🗓️ Error by Weekday")
    error_df = pd.DataFrame({
        "date": pd.to_datetime(best["dates_test"]),
        "error": abs(best["y_test"] - best["preds"]),
    })
    error_df["weekday"] = error_df["date"].dt.day_name()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_mae = error_df.groupby("weekday")["error"].mean().reindex(weekday_order)
    st.bar_chart(weekday_mae)

st.subheader("🔍 Raw Data")
st.dataframe(df.tail(30), use_container_width=True)

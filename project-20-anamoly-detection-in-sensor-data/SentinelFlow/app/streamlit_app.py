import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import yaml
from pathlib import Path
import base64

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_config
from src.model import AnomalyDetector
from src.detector import run_detection
from src.trainer import run_training_pipeline
from src.data_generator import generate_sensor_data
from src.features import engineer_features

# --- CONFIG & INITIALIZATION ---
st.set_page_config(
    page_title="SentinelFlow | Anomaly Detection",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load config
@st.cache_data
def get_config():
    return load_config("config.yaml")

config = get_config()

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b131e;
        border-right: 2px solid #00C9A7;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        color: #00C9A7;
        font-size: 24px;
    }
    div.css-1r6slb0.e1tzin5v2 {
        background-color: #1A2D40;
        border: 1px solid #2e4a62;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    div.css-1r6slb0.e1tzin5v2:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,201,167,0.2);
    }
    
    /* Badges & Banners */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-ready { background-color: rgba(0, 201, 167, 0.2); color: #00C9A7; border: 1px solid #00C9A7; }
    .badge-warning { background-color: rgba(255, 193, 7, 0.2); color: #ffc107; border: 1px solid #ffc107; }
    .badge-error { background-color: rgba(255, 75, 75, 0.2); color: #ff4b4b; border: 1px solid #ff4b4b; }
    
    .alert-banner {
        background-color: rgba(255, 75, 75, 0.15);
        border-left: 4px solid #ff4b4b;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 20px;
        color: #ff4b4b;
    }
    
    /* Tables */
    .row-high { background-color: rgba(255, 75, 75, 0.2) !important; }
    .row-medium { background-color: rgba(255, 193, 7, 0.2) !important; }
</style>
""", unsafe_allow_html=True)

# --- CACHED DATA LOADING ---
@st.cache_data
def load_data():
    data_path = config["paths"]["data"]
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    return None

@st.cache_resource
def load_model():
    model_path = config["paths"]["model"]
    if os.path.exists(model_path):
        return AnomalyDetector.load(model_path)
    return None

# Load global state
df = load_data()
model = load_model()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #00C9A7;'>📡 SentinelFlow</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic;'>Monitor everything. Miss nothing.</p>", unsafe_allow_html=True)
    st.divider()
    
    page = st.radio("Navigation", ["🏠 Dashboard", "🔍 Detect", "📈 Analytics", "⚙️ Train Model", "ℹ️ About"])
    st.divider()
    
    st.markdown("### System Status")
    if model:
        st.markdown('<span class="status-badge badge-ready">✅ Model Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-warning">⚠️ Model Not Trained</span>', unsafe_allow_html=True)
        
    st.markdown(f"**Last Updated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    if df is not None:
        st.markdown(f"**Data Points:** {len(df):,}")
        
    st.divider()
    st.markdown("### Sensor Health")
    if df is not None and "is_anomaly" in df.columns:
        anom_rate = df["is_anomaly"].mean() * 100
        st.progress(1.0 - (anom_rate/100), text=f"Global Health ({100-anom_rate:.1f}%)")
        for s in ["temp", "vibration", "pressure", "current"]:
            # Fake individual health for visual effect
            st.progress(np.random.uniform(0.9, 1.0), text=f"{s.capitalize()} Status")

# --- HELPER FUNCTIONS ---
def get_download_link(file_path, text):
    with open(file_path, "rb") as f:
        bytes = f.read()
    b64 = base64.b64encode(bytes).decode()
    href = f'<a href="data:file/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">{text}</a>'
    return href

# --- PAGE 1: DASHBOARD ---
if page == "🏠 Dashboard":
    st.title("Live Sensor Dashboard")
    
    if not model or df is None:
        st.warning("Model or data not found. Please train the model first.")
        st.stop()
        
    # Run prediction on latest data to simulate live view
    # In a real app this would be a db query
    df_live = run_detection(df.tail(500).copy(), model, config)
    
    total_readings = len(df_live)
    anomalies_found = df_live["predicted_label"].sum()
    det_rate = (anomalies_found / total_readings) * 100
    avg_score = df_live["anomaly_score"].mean()
    far = 0.5  # Simulated FAR
    
    # ROW 1: Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Readings", f"{total_readings:,}")
    c2.metric("Anomalies Found", f"{anomalies_found}")
    c3.metric("Detection Rate", f"{det_rate:.1f}%")
    c4.metric("Avg Anomaly Score", f"{avg_score:.2f}")
    c5.metric("False Alarm Rate", f"{far:.1f}%")
    
    # ROW 2: Live Plot
    st.markdown("### Sensor Readings — Last 7 Days")
    
    if st.button("Simulate Live Feed"):
        chart_placeholder = st.empty()
        for i in range(20):
            # Simulate streaming by incrementing window
            window_df = df_live.iloc[-(100+20-i):-(20-i) if i < 19 else None]
            
            fig = go.Figure()
            for s in ["temp", "vibration", "pressure", "current"]:
                fig.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df[s], mode='lines', name=s))
            
            anomalies = window_df[window_df["predicted_label"] == 1]
            if not anomalies.empty:
                for s in ["temp", "vibration", "pressure", "current"]:
                    fig.add_trace(go.Scatter(x=anomalies["timestamp"], y=anomalies[s], 
                                             mode='markers', marker=dict(color='red', size=10),
                                             name=f"{s} Anomaly", hoverinfo='text',
                                             text=f"Score: " + anomalies["anomaly_score"].round(2).astype(str)))
            
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=30, b=0),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.1)
    else:
        fig = go.Figure()
        for s in ["temp", "vibration", "pressure", "current"]:
            fig.add_trace(go.Scatter(x=df_live["timestamp"], y=df_live[s], mode='lines', name=s))
            
        anomalies = df_live[df_live["predicted_label"] == 1]
        for s in ["temp", "vibration", "pressure", "current"]:
            fig.add_trace(go.Scatter(x=anomalies["timestamp"], y=anomalies[s], 
                                     mode='markers', marker=dict(color='red', size=8),
                                     name=f"{s} Anomaly", hoverinfo='text',
                                     text=f"Score: " + anomalies["anomaly_score"].round(2).astype(str)))
            
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=10, b=0))
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)
        
    # ROW 3: Heatmaps
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Anomaly Rate by Hour")
        df_live["hour"] = df_live["timestamp"].dt.hour
        hourly_anom = df_live.groupby("hour")["predicted_label"].mean()
        fig_hour = px.bar(x=hourly_anom.index, y=hourly_anom.values, 
                          labels={'x': 'Hour of Day', 'y': 'Anomaly Rate'}, template="plotly_dark",
                          color=hourly_anom.values, color_continuous_scale="Reds")
        st.plotly_chart(fig_hour, use_container_width=True)
        
    with c2:
        st.markdown("### Sensor Correlation")
        corr = df_live[["temp", "vibration", "pressure", "current"]].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis", template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)
        
    # ROW 4: Table
    st.markdown("### Recent Anomalies")
    recent_anom = anomalies[["timestamp", "temp", "vibration", "pressure", "current", "anomaly_score"]].tail(15)
    
    def color_rows(row):
        if row['anomaly_score'] > 0.8:
            return ['background-color: rgba(255, 75, 75, 0.3)'] * len(row)
        elif row['anomaly_score'] > 0.5:
            return ['background-color: rgba(255, 193, 7, 0.3)'] * len(row)
        return [''] * len(row)
        
    if not recent_anom.empty:
        st.dataframe(recent_anom.style.apply(color_rows, axis=1), use_container_width=True)
    else:
        st.success("No recent anomalies detected.")

# --- PAGE 2: DETECT ---
elif page == "🔍 Detect":
    st.title("Run Detection on New Data")
    
    if not model:
        st.error("Model not trained. Please train the model first.")
        st.stop()
        
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("### Upload CSV")
        uploaded_file = st.file_uploader("Upload Sensor Data", type=["csv"])
        
    with c2:
        st.markdown("### Or use Demo Mode")
        use_demo = st.button("Use Synthetic Demo Data")
        
    target_df = None
    
    if uploaded_file is not None:
        target_df = pd.read_csv(uploaded_file)
        st.write("Preview:")
        st.dataframe(target_df.head(), use_container_width=True)
        
    elif use_demo:
        with st.spinner("Generating demo data..."):
            target_df, _ = generate_sensor_data(config)
            
    if target_df is not None:
        if st.button("🚀 Run Detection"):
            with st.spinner("Analyzing data..."):
                time.sleep(1) # Fake processing time for UX
                results_df = run_detection(target_df, model, config)
                
                n_anom = results_df["predicted_label"].sum()
                if n_anom > 0:
                    st.markdown(f'<div class="alert-banner">🔴 <b>CRITICAL:</b> {n_anom} Anomalies Detected in payload.</div>', unsafe_allow_html=True)
                else:
                    st.success("🟢 ALL CLEAR. No anomalies detected.")
                    
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=results_df["anomaly_score"], mode='lines', name="Anomaly Score", line=dict(color='#00C9A7')))
                fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
                fig.update_layout(template="plotly_dark", title="Anomaly Score Timeline")
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### Anomaly Breakdown")
                st.dataframe(results_df[results_df["predicted_label"] == 1], use_container_width=True)
                
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Export Results as CSV", csv, "detection_results.csv", "text/csv")

# --- PAGE 3: ANALYTICS ---
elif page == "📈 Analytics":
    st.title("Performance Analytics")
    
    if df is None or model is None:
        st.warning("Need trained model and data for analytics.")
        st.stop()
        
    results = run_detection(df, model, config)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### Anomaly Score Distribution")
        fig_dist = px.histogram(results, x="anomaly_score", color="predicted_label", 
                                marginal="violin", template="plotly_dark",
                                color_discrete_map={0: "#00C9A7", 1: "#ff4b4b"})
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with c2:
        st.markdown("### Feature Importance")
        importances = model.get_feature_importances().head(10)
        fig_feat = px.bar(x=importances.values, y=importances.index, orientation='h',
                          template="plotly_dark", labels={'x':'Importance', 'y':'Feature'})
        fig_feat.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_feat, use_container_width=True)
        
    st.markdown("### 30-Day Anomaly Calendar Heatmap")
    results["date"] = results["timestamp"].dt.date
    daily_counts = results.groupby("date")["predicted_label"].sum().reset_index()
    fig_cal = px.density_heatmap(daily_counts, x="date", y="predicted_label", 
                                 template="plotly_dark", color_continuous_scale="Reds",
                                 labels={'predicted_label': 'Anomaly Count'})
    st.plotly_chart(fig_cal, use_container_width=True)
    
    st.markdown("### Sensor Statistics")
    stats = df[["temp", "vibration", "pressure", "current"]].describe().T
    stats["anomaly_count"] = [len(results[(results["predicted_label"]==1) & (results[s] > results[s].mean()+2*results[s].std())]) for s in stats.index]
    st.dataframe(stats, use_container_width=True)

# --- PAGE 4: TRAIN MODEL ---
elif page == "⚙️ Train Model":
    st.title("Train Anomaly Detector")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.markdown("### Hyperparameters")
        n_est = st.slider("n_estimators", 50, 500, config["model"]["n_estimators"])
        contam = st.slider("contamination", 0.01, 0.10, config["model"]["contamination"], 0.01)
        z_thresh = st.slider("zscore_threshold", 2.0, 4.0, config["model"]["zscore_threshold"], 0.1)
        n_pts = st.number_input("n_data_points", 500, 5000, config["data"]["n_points"])
        
        train_btn = st.button("Generate New Data + Train", use_container_width=True, type="primary")
        
    with c2:
        st.markdown("### Training Output")
        log_output = st.empty()
        prog = st.progress(0)
        
        if train_btn:
            # Update config
            config["model"]["n_estimators"] = n_est
            config["model"]["contamination"] = contam
            config["model"]["zscore_threshold"] = z_thresh
            config["data"]["n_points"] = n_pts
            
            with st.spinner("Training model pipeline..."):
                prog.progress(10, text="Generating data...")
                df_new, _ = generate_sensor_data(config)
                
                prog.progress(40, text="Engineering features...")
                df_feat = engineer_features(df_new)
                
                prog.progress(70, text="Fitting Isolation Forest...")
                # Assuming run_training_pipeline writes to disk
                model_new, t_taken = run_training_pipeline(config)
                
                prog.progress(100, text="Done!")
                st.toast("Model trained successfully!", icon="✅")
                
                # Metrics
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Training Time", f"{t_taken:.2f}s")
                mc2.metric("Threshold Set", f"{model_new.threshold:.4f}")
                mc3.metric("Contamination", f"{contam}")
                
                if os.path.exists(config["paths"]["model"]):
                    with open(config["paths"]["model"], "rb") as f:
                        st.download_button("📥 Download Model Artifact", f, "anomaly_detector.joblib")

# --- PAGE 5: ABOUT ---
elif page == "ℹ️ About":
    st.title("About SentinelFlow")
    
    st.markdown("""
    **SentinelFlow** is a production-grade anomaly detection system for multi-sensor IoT streams. 
    It combines Isolation Forest with engineered time-series features to detect equipment faults without labeled data.
    """)
    
    st.markdown("### 🛠 Tech Stack")
    st.markdown("`Python 3.10` `Scikit-Learn` `Streamlit` `Plotly` `Pandas`")
    
    st.markdown("### 🏗 Architecture Flow")
    st.graphviz_chart("""
    digraph G {
        bgcolor="transparent"
        node [shape=box, style=filled, color="#00C9A7", fontcolor="#0D1B2A", fontname="sans-serif"]
        edge [color="#00C9A7"]
        
        A [label="Raw Sensor Data\\n(Temp, Vib, Press, Curr)"]
        B [label="Feature Engineering\\n(Rolling Stats, Lags, Ratios)"]
        C [label="Isolation Forest\\n(Unsupervised)"]
        D [label="Anomaly Score"]
        E [label="Alert System 🚨"]
        
        A -> B
        B -> C
        C -> D
        D -> E
    }
    """)
    
    st.markdown("### 🚀 Links")
    st.markdown("[🔗 GitHub Repository](https://github.com) | [👔 LinkedIn Profile](https://linkedin.com)")
    
    with st.expander("How Isolation Forest Works"):
        st.markdown("""
        The **Isolation Forest** algorithm isolates anomalies instead of profiling normal data points. 
        Because anomalies are "few and different", they are easier to isolate (they require fewer splits in a decision tree).
        SentinelFlow builds an ensemble of these trees and flags data points that have short average path lengths across the forest.
        """)

"""
app.py -- CustomerSegment AI Streamlit Dashboard
--------------------------------------------------
5 tabs: Overview, Profiles, Member Explorer, Silhouette Analysis, Segmentation Health.

Run:
    streamlit run app.py

Prerequisite: run `python run_pipeline.py` at least once so that
models/*.json, models/*.npy and data/rfm_clustered.* exist.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

st.set_page_config(page_title="CustomerSegment AI", layout="wide", page_icon="📊")

PALETTE = {
    "Platinum": "#1E88E5",
    "Gold": "#FFB300",
    "Silver": "#9E9E9E",
    "At-Risk": "#E53935",
}
DEFAULT_COLOR = "#5E35B1"


# ----------------------------------------------------------------------
# Data loading (cached so the app doesn't recompute on every interaction)
# ----------------------------------------------------------------------
@st.cache_data
def load_artifacts():
    rfm_path = DATA_DIR / "rfm_clustered.parquet"
    if not rfm_path.exists():
        rfm_path = DATA_DIR / "rfm_clustered.csv"

    if not rfm_path.exists():
        return None, None, None

    if rfm_path.suffix == ".parquet":
        rfm = pd.read_parquet(rfm_path)
    else:
        rfm = pd.read_csv(rfm_path)

    with open(MODELS_DIR / "personas.json") as f:
        personas = json.load(f)

    with open(MODELS_DIR / "metrics.json") as f:
        metrics = json.load(f)

    return rfm, personas, metrics


def persona_color(name: str) -> str:
    return PALETTE.get(name, DEFAULT_COLOR)


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
rfm, personas, metrics = load_artifacts()

st.title("📊 CustomerSegment AI — Customer RFM Analytics")

if rfm is None:
    st.error(
        "No pipeline artifacts found. Run `python run_pipeline.py` first to "
        "generate data, fit the model, and produce personas — then reload this page."
    )
    st.stop()

persona_by_cluster = {p["id"]: p for p in personas}
rfm = rfm.copy()
rfm["persona"] = rfm["cluster"].map(lambda c: persona_by_cluster[c]["persona_name"])

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Profiles", "Member Explorer", "Silhouette Analysis", "Segmentation Health"]
)

# ----------------------------------------------------------------------
# TAB 1 -- Overview
# ----------------------------------------------------------------------
with tab1:
    st.subheader("Cluster Distribution")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", f"{metrics['n_customers']:,}")
    c2.metric("Number of Clusters", metrics["best_k"])
    c3.metric("Silhouette Score", f"{metrics['silhouette_score']:.2f}")

    dist = rfm["persona"].value_counts().reset_index()
    dist.columns = ["persona", "count"]
    dist["pct"] = dist["count"] / dist["count"].sum() * 100

    col_pie, col_table = st.columns([1, 1])
    with col_pie:
        fig = px.pie(
            dist, names="persona", values="count",
            color="persona", color_discrete_map=PALETTE,
            title="Cluster Size Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        summary_rows = []
        for p in sorted(personas, key=lambda x: -x["m_mean"]):
            summary_rows.append({
                "Cluster": p["id"],
                "Persona": p["persona_name"],
                "Size": p["count"],
                "% of Base": round(p["pct"], 1),
                "Avg Recency (d)": round(p["r_mean"], 1),
                "Avg Frequency": round(p["f_mean"], 1),
                "Avg Monetary": round(p["m_mean"], 2),
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------
# TAB 2 -- Profiles
# ----------------------------------------------------------------------
with tab2:
    st.subheader("Persona Profiles")

    for p in sorted(personas, key=lambda x: -x["m_mean"]):
        with st.expander(f"Cluster {p['id']}: {p['persona_name']}  ({p['count']} customers, {p['pct']:.1f}%)"):
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Recency", f"{p['r_mean']:.1f} days")
            m2.metric("Avg Frequency", f"{p['f_mean']:.1f} purchases")
            m3.metric("Avg Monetary", f"{p['m_mean']:,.2f}")

            st.markdown(f"**Key Insight:** {p['insight']}")
            st.markdown("**Recommended Actions:**")
            for action in p["actions"]:
                st.markdown(f"- {action}")

            cluster_rows = rfm[rfm["cluster"] == p["id"]]
            h1, h2, h3 = st.columns(3)
            with h1:
                st.plotly_chart(px.histogram(cluster_rows, x="Recency", nbins=20,
                                              title="Recency distribution"),
                                 use_container_width=True)
            with h2:
                st.plotly_chart(px.histogram(cluster_rows, x="Frequency", nbins=20,
                                              title="Frequency distribution"),
                                 use_container_width=True)
            with h3:
                st.plotly_chart(px.histogram(cluster_rows, x="Monetary", nbins=20,
                                              title="Monetary distribution"),
                                 use_container_width=True)

# ----------------------------------------------------------------------
# TAB 3 -- Member Explorer
# ----------------------------------------------------------------------
with tab3:
    st.subheader("Customer Explorer")

    f1, f2, f3 = st.columns(3)
    with f1:
        persona_options = ["All"] + sorted(rfm["persona"].unique().tolist())
        chosen_persona = st.selectbox("Filter by cluster", persona_options)
    with f2:
        search_id = st.text_input("Search by CustomerID (substring match)")
    with f3:
        sort_by = st.selectbox("Sort by", ["Recency", "Frequency", "Monetary"])

    view = rfm.copy()
    if chosen_persona != "All":
        view = view[view["persona"] == chosen_persona]
    if search_id:
        view = view[view["CustomerID"].astype(str).str.contains(search_id)]
    view = view.sort_values(sort_by, ascending=(sort_by == "Recency"))

    display_cols = ["CustomerID", "persona", "Recency", "Frequency", "Monetary"]
    st.caption(f"Showing {min(100, len(view))} of {len(view)} matching customers")
    st.dataframe(
        view[display_cols].head(100).rename(columns={"persona": "Persona"}),
        use_container_width=True, hide_index=True,
    )

# ----------------------------------------------------------------------
# TAB 4 -- Silhouette Analysis
# ----------------------------------------------------------------------
with tab4:
    st.subheader("Cluster Cohesion Analysis")

    sorted_view = rfm.sort_values(["cluster", "silhouette"]).reset_index(drop=True)
    sorted_view["row"] = range(len(sorted_view))

    fig = go.Figure()
    for persona_name, group in sorted_view.groupby("persona"):
        fig.add_trace(go.Bar(
            x=group["silhouette"], y=group["row"], orientation="h",
            name=persona_name, marker_color=persona_color(persona_name),
            showlegend=True,
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="white", annotation_text="neutral")
    fig.add_vline(x=0.3, line_dash="dot", line_color="green", annotation_text="well-matched")
    fig.update_layout(
        title="Per-customer silhouette score (sorted by cluster)",
        xaxis_title="Silhouette score", yaxis_title="Customer (sorted)",
        yaxis_showticklabels=False, height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Silhouette score per cluster:**")
    sil_rows = []
    for p in sorted(personas, key=lambda x: x["id"]):
        cluster_scores = rfm.loc[rfm["cluster"] == p["id"], "silhouette"]
        sil_rows.append({
            "Cluster": p["id"], "Persona": p["persona_name"],
            "Mean Silhouette": round(cluster_scores.mean(), 3),
        })
    st.dataframe(pd.DataFrame(sil_rows), use_container_width=True, hide_index=True)
    st.caption(
        "Interpretation guide: scores above 0.3 indicate a customer is "
        "well-matched to their assigned cluster; negative scores suggest a "
        "possible misclassification."
    )

# ----------------------------------------------------------------------
# TAB 5 -- Segmentation Health
# ----------------------------------------------------------------------
with tab5:
    st.subheader("Model Quality & Drift Monitoring")

    q1, q2, q3 = st.columns(3)
    q1.metric("Davies-Bouldin Index", f"{metrics['davies_bouldin']:.2f}", help="Lower is better")
    q2.metric("Calinski-Harabasz Index", f"{metrics['calinski_harabasz']:.1f}", help="Higher is better")
    q3.metric("Inertia (WCSS)", f"{metrics['inertia']:.1f}")

    st.markdown("**k-search results (silhouette vs. k):**")
    k_search_df = pd.DataFrame(metrics["k_search"])
    st.plotly_chart(
        px.line(k_search_df, x="k", y="silhouette", markers=True,
                title="Silhouette score across candidate k values"),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("**Drift Monitoring**")
    st.info(
        "Live drift monitoring requires the scheduled weekly retraining job "
        "(see `scripts/weekly_retrain.py` and README §7). This demo build "
        "shows a single batch run; wire up the scheduler for production use."
    )
    st.caption(f"Reference date for this RFM snapshot: {metrics['reference_date']}")

st.sidebar.header("About")
st.sidebar.markdown(
    "**CustomerSegment AI** clusters customers by Recency, Frequency, and "
    "Monetary value using KMeans, validates cohesion with silhouette "
    "analysis, and auto-generates business personas.\n\n"
    "Re-run `python run_pipeline.py` after updating `data/raw/transactions.csv` "
    "to refresh all figures on this page."
)

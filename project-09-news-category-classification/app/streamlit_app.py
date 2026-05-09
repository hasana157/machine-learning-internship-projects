"""
app/streamlit_app.py — Professional NewsLens dashboard.
Run: streamlit run app/streamlit_app.py
"""
import json, re, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib, numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="NewsLens — AI News Classifier",
    page_icon="\U0001f4f0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@300;400;600&family=JetBrains+Mono:wght@500&display=swap');
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:1.2rem;padding-bottom:2rem}
.masthead{border-top:3px solid #0f0f0f;border-bottom:1px solid #e5e0d8;padding:.5rem 0;
    margin-bottom:1.6rem;display:flex;align-items:baseline;gap:1rem}
.masthead-title{font-family:"Playfair Display",serif;font-size:2.4rem;
    font-weight:900;color:#0f0f0f;letter-spacing:-.02em;line-height:1;margin:0}
.masthead-badge{font-family:"JetBrains Mono",monospace;font-size:.6rem;font-weight:600;
    color:#faf9f7;background:#c0392b;padding:.15rem .45rem;border-radius:2px;
    letter-spacing:.08em;text-transform:uppercase;align-self:center}
.masthead-sub{font-family:"Source Sans 3",sans-serif;font-size:.82rem;
    color:#6b6560;margin-left:auto;align-self:center}
.pred-card{background:#0f0f0f;color:#faf9f7;border-radius:4px;padding:1.3rem 1.5rem;margin:1rem 0}
.pred-lbl{font-family:"JetBrains Mono",monospace;font-size:.58rem;letter-spacing:.14em;
    text-transform:uppercase;color:#aaa;margin-bottom:.3rem}
.pred-cat{font-family:"Playfair Display",serif;font-size:2.1rem;font-weight:900;line-height:1;margin:0}
.pred-conf{font-family:"JetBrains Mono",monospace;font-size:.82rem;color:#d4a017;margin-top:.35rem}
.metric-tile{background:#fff;border:1px solid #e5e0d8;border-top:3px solid #0f0f0f;
    padding:.9rem 1.1rem;text-align:center}
.metric-val{font-family:"Playfair Display",serif;font-size:2.2rem;font-weight:700;
    color:#0f0f0f;line-height:1}
.metric-lbl{font-family:"Source Sans 3",sans-serif;font-size:.72rem;color:#6b6560;
    margin-top:.25rem;letter-spacing:.04em;text-transform:uppercase}
.insight-card{border-left:3px solid #c0392b;background:#fff8f8;padding:.7rem .95rem;
    margin-bottom:.7rem;font-family:"Source Sans 3",sans-serif;font-size:.88rem;line-height:1.55;color:#333}
.error-card{background:#fffbf0;border:1px solid #f0e0a0;border-left:3px solid #D97706;
    padding:.65rem .85rem;margin-bottom:.55rem;border-radius:2px;
    font-family:"Source Sans 3",sans-serif;font-size:.83rem}
.etag{font-family:"JetBrains Mono",monospace;font-size:.62rem;font-weight:600;
    padding:.08rem .3rem;border-radius:2px;display:inline-block;margin-bottom:.25rem}
.tag-t{background:#dcfce7;color:#166534}
.tag-p{background:#fee2e2;color:#991b1b}
[data-testid="stSidebar"]{background:#0f0f0f !important;padding-top:1.2rem}
[data-testid="stSidebar"] *{color:#e0dcd6 !important}
[data-testid="stSidebar"] hr{border-color:#333 !important}
.stButton>button{font-family:"JetBrains Mono",monospace;font-size:.78rem;
    letter-spacing:.06em;text-transform:uppercase;background:#0f0f0f;color:#faf9f7;
    border:none;border-radius:2px;padding:.5rem 1.3rem}
.stButton>button:hover{background:#333}
.stTabs [data-baseweb="tab"]{font-family:"JetBrains Mono",monospace;font-size:.7rem;
    letter-spacing:.08em;text-transform:uppercase}
hr.sr{border:none;border-top:2px solid #0f0f0f;margin:1.4rem 0 .7rem}
</style>
""", unsafe_allow_html=True)

# ─── Loaders ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    mp = ROOT/"models"/"best_model.joblib"
    mm = ROOT/"models"/"model_meta.json"
    if not mp.exists(): return None, None
    pipe = joblib.load(str(mp))
    with open(str(mm)) as f: meta = json.load(f)
    return pipe, meta

@st.cache_data
def load_metrics():
    p = ROOT/"models"/"metrics.json"
    return json.load(open(p)) if p.exists() else {}

@st.cache_data
def load_error():
    p = ROOT/"reports"/"error_analysis.json"
    return json.load(open(p)) if p.exists() else {}

def clean(text):
    if not isinstance(text,str): return ""
    text=text.lower()
    text=re.sub(r"#\d+;"," ",text)
    text=re.sub(r"<[^>]+>"," ",text)
    text=re.sub(r"https?://\S+|www\.\S+"," ",text)
    text=re.sub(r"^\s*(AP|AFP|Reuters)\s*[-–]?\s*","",text,flags=re.I)
    text=re.sub(r"[^\w\s\.\,\!\?\;\:\'\"-]"," ",text)
    text=re.sub(r"\s+"," ",text)
    return text.strip()

# ─── Config ───────────────────────────────────────────────────────────────────
CAT = {
    "World":    {"color":"#2563EB","icon":"\U0001f30d","desc":"Global news & geopolitics"},
    "Sports":   {"color":"#16A34A","icon":"\u26bd","desc":"Athletics & competitions"},
    "Business": {"color":"#DC2626","icon":"\U0001f4c8","desc":"Markets & corporate news"},
    "Sci/Tech": {"color":"#D97706","icon":"\U0001f52c","desc":"Science & technology"},
}

SAMPLES = {
    "\U0001f30d World":    "The United Nations Security Council held an emergency session addressing escalating tensions in Eastern Europe, with world leaders calling for an immediate ceasefire and diplomatic negotiations to prevent further conflict.",
    "\u26bd Sports":      "Manchester City clinched the Premier League title after a thrilling 3-2 victory against Arsenal at the Etihad Stadium, with Erling Haaland scoring a decisive hat-trick in the final 20 minutes.",
    "\U0001f4c8 Business": "The Federal Reserve raised interest rates by 25 basis points amid persistent inflation. Markets reacted sharply, with the S&P 500 dropping 1.8% and Treasury yields climbing to multi-year highs.",
    "\U0001f52c Sci/Tech": "Google DeepMind unveiled a new protein structure prediction model that significantly outperforms existing benchmarks, potentially revolutionizing drug discovery and our understanding of diseases.",
    "\U0001f914 Ambiguous":"Apple reported record quarterly revenue of $94.8 billion, driven by strong iPhone 15 Pro sales and growing services, despite headwinds in the Chinese market amid geopolitical tensions.",
}

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="font-family:'Playfair Display',serif;font-size:1.35rem;
    font-weight:900;margin-bottom:.15rem;">\U0001f4f0 NewsLens</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#888;
    letter-spacing:.1em;text-transform:uppercase;margin-bottom:1.4rem;">AI News Intelligence</div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Nav", ["\U0001f50d Classify Article","\U0001f4ca Model Performance",
                             "\U0001f52c Error Analysis","\U0001f4d6 Feature Insights"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""<div style="font-family:'Source Sans 3',sans-serif;font-size:.76rem;color:#888;">
    <b style="color:#ccc;">Model</b><br>TF-IDF (1–3 grams)<br>Logistic Regression<br><br>
    <b style="color:#ccc;">Dataset</b><br>AG News · 120k train<br><br>
    <b style="color:#ccc;">Categories</b><br>World · Sports<br>Business · Sci/Tech
    </div>""", unsafe_allow_html=True)

# ─── Load ─────────────────────────────────────────────────────────────────────
pipeline, meta = load_model()
metrics        = load_metrics()
err_report     = load_error()

# ─── Masthead ─────────────────────────────────────────────────────────────────
st.markdown("""<div class="masthead">
  <span class="masthead-title">NewsLens</span>
  <span class="masthead-badge">AI</span>
  <span class="masthead-sub">Multi-Class News Classification · v1.0</span>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Classify
# ══════════════════════════════════════════════════════════════════════════════
if page == "\U0001f50d Classify Article":
    col_in, col_out = st.columns([1.1, 0.9], gap="large")

    with col_in:
        st.markdown("**Paste a news article**")
        sel = st.selectbox("Sample", ["— type your own —"]+list(SAMPLES.keys()),
                           label_visibility="collapsed")
        default = SAMPLES.get(sel,"") if sel != "— type your own —" else ""
        text = st.text_area("Article", value=default, height=230,
                            placeholder="Paste any news headline or article…",
                            label_visibility="collapsed")
        btn = st.button("Classify →")

    with col_out:
        st.markdown("**Classification result**")
        if pipeline is None:
            st.error("Model not found. Run `python train.py` first.")
        elif btn or text:
            if not text.strip():
                st.info("Enter text on the left.")
            else:
                cleaned  = clean(text)
                proba    = pipeline.predict_proba([cleaned])[0]
                cnames   = meta["class_names"]
                idx      = int(np.argmax(proba))
                label    = cnames[idx]
                conf     = float(proba[idx])
                cfg      = CAT[label]

                st.markdown(f"""<div class="pred-card">
                  <div class="pred-lbl">Predicted Category</div>
                  <div class="pred-cat">{cfg["icon"]} {label}</div>
                  <div class="pred-conf">Confidence: {conf:.1%}</div>
                </div>""", unsafe_allow_html=True)

                pdf = pd.DataFrame({"Category":cnames,"P":[float(p) for p in proba]}
                                   ).sort_values("P")
                colors = [CAT[c]["color"] for c in pdf["Category"]]
                fig = go.Figure(go.Bar(x=pdf["P"], y=pdf["Category"], orientation="h",
                    marker_color=colors, text=[f"{p:.1%}" for p in pdf["P"]],
                    textposition="outside", cliponaxis=False))
                fig.update_layout(height=175, margin=dict(l=0,r=45,t=0,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False,showticklabels=False,range=[0,1.15]),
                    yaxis=dict(showgrid=False),
                    font=dict(family="Source Sans 3",size=12), showlegend=False)
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar":False})
                st.caption(f"**{label}** — {cfg['desc']}")

    st.markdown("<hr class=\"sr\">", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col,val,lbl in [
        (c1, f"{metrics.get('accuracy',0):.2%}", "Test Accuracy"),
        (c2, f"{metrics.get('macro_f1',0):.2%}", "Macro F1"),
        (c3, "4", "Categories"),
        (c4, f"{meta.get('training_samples',0):,}" if meta else "—", "Training Samples"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-tile">
              <div class="metric-val">{val}</div>
              <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "\U0001f4ca Model Performance":
    st.markdown("### Model Performance Report")
    pcf = metrics.get("per_class_f1", {})
    c1,c2,c3,c4 = st.columns(4)
    best_cls = max(pcf, key=pcf.get) if pcf else "—"
    for col,key,lbl in [(c1,"accuracy","Accuracy"),(c2,"macro_f1","Macro F1"),
                         (c3,"weighted_f1","Weighted F1")]:
        with col:
            st.markdown(f"""<div class="metric-tile">
              <div class="metric-val">{metrics.get(key,0):.2%}</div>
              <div class="metric-lbl">{lbl}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-tile">
          <div class="metric-val">{pcf.get(best_cls,0):.2%}</div>
          <div class="metric-lbl">Best Class ({best_cls})</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_cm, col_f1 = st.columns([1.2, 0.8], gap="large")
    with col_cm:
        st.markdown("##### Confusion Matrix")
        p = ROOT/"reports"/"figures"/"confusion_matrix.png"
        if p.exists(): st.image(str(p), use_container_width=True)
        else: st.info("Run training to generate.")
    with col_f1:
        st.markdown("##### Per-Class F1-Score")
        if pcf:
            cnames = list(pcf.keys()); vals = list(pcf.values())
            colors = [CAT.get(c,{}).get("color","#333") for c in cnames]
            fig = go.Figure(go.Bar(x=cnames, y=vals, marker_color=colors,
                text=[f"{v:.2%}" for v in vals], textposition="outside"))
            fig.update_layout(height=300, margin=dict(l=0,r=0,t=5,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0,1.12], tickformat=".0%", gridcolor="#eee"),
                xaxis=dict(showgrid=False),
                font=dict(family="Source Sans 3"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown("""<div class="insight-card"><b>Why Macro F1?</b><br>
        Weights all classes equally regardless of size — penalises models that ignore
        minority classes, unlike accuracy alone.</div>""", unsafe_allow_html=True)

    st.markdown("<hr class=\"sr\">", unsafe_allow_html=True)
    st.markdown("##### Training Class Distribution")
    dp = ROOT/"reports"/"figures"/"train_class_distribution.png"
    if dp.exists(): st.image(str(dp), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Error Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "\U0001f52c Error Analysis":
    st.markdown("### Error Analysis Report")
    if not err_report:
        st.info("Run training to generate the error analysis report.")
    else:
        s = err_report.get("summary",{})
        c1,c2,c3 = st.columns(3)
        for col,v,l in [(c1,f"{s.get('error_rate',0):.2%}","Error Rate"),
                        (c2,f"{s.get('total_errors',0):,}","Total Errors"),
                        (c3,f"{s.get('total_samples',0):,}","Test Samples")]:
            with col:
                st.markdown(f"""<div class="metric-tile">
                  <div class="metric-val">{v}</div>
                  <div class="metric-lbl">{l}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        pce = err_report.get("per_class_errors",{})
        if pce:
            st.markdown("##### Error Rate by Category")
            df = pd.DataFrame([{"Category":k,"Error Rate":v["error_rate"]}
                                for k,v in pce.items()]).sort_values("Error Rate",ascending=False)
            colors = [CAT.get(c,{}).get("color","#555") for c in df["Category"]]
            fig = go.Figure(go.Bar(x=df["Category"], y=df["Error Rate"],
                marker_color=colors, text=[f"{r:.1%}" for r in df["Error Rate"]],
                textposition="outside"))
            fig.update_layout(height=260, margin=dict(l=0,r=0,t=5,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0, df["Error Rate"].max()*1.35],
                           tickformat=".0%", gridcolor="#eee"),
                xaxis=dict(showgrid=False),
                font=dict(family="Source Sans 3"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

        confused = err_report.get("top_confused_pairs",{})
        if confused:
            st.markdown("<hr class=\"sr\">", unsafe_allow_html=True)
            st.markdown("##### Most Confused Class Pairs")
            pairs = list(confused.items())
            cl, cr = st.columns(2, gap="large")
            for i,(key,data) in enumerate(pairs[:6]):
                col = cl if i%2==0 else cr
                true_l,pred_l = key.split(" → ")
                with col:
                    st.markdown(f"**{key}** — {data['count']} cases")
                    for ex in data.get("examples",[])[:2]:
                        conf_s = f"Confidence: {ex['model_confidence']:.1%}" if ex.get("model_confidence") else ""
                        st.markdown(f"""<div class="error-card">
                          <span class="etag tag-t">True: {true_l}</span>&nbsp;
                          <span class="etag tag-p">Pred: {pred_l}</span><br>
                          {ex['text'][:240]}…<br>
                          <span style="font-size:.73rem;color:#999;">{conf_s}</span>
                        </div>""", unsafe_allow_html=True)

        hce = err_report.get("high_confidence_errors",[])
        if hce:
            st.markdown("<hr class=\"sr\">", unsafe_allow_html=True)
            st.markdown(f"##### High-Confidence Errors ({len(hce)} cases ≥80%)")
            st.caption("The model is wrong but certain — most dangerous failure mode.")
            for e in hce[:5]:
                st.markdown(f"""<div class="error-card">
                  <span class="etag tag-t">True: {e['true']}</span>&nbsp;
                  <span class="etag tag-p">Pred: {e['pred']}</span>&nbsp;
                  <span style="font-family:'JetBrains Mono',monospace;font-size:.62rem;
                        color:#c0392b;">\u26a0 {e['confidence']:.1%} confident</span><br>
                  {e['text'][:270]}…
                </div>""", unsafe_allow_html=True)

        insights = err_report.get("insights",[])
        if insights:
            st.markdown("<hr class=\"sr\">", unsafe_allow_html=True)
            st.markdown("##### ML Engineer Insights")
            for ins in insights:
                st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Feature Insights
# ══════════════════════════════════════════════════════════════════════════════
elif page == "\U0001f4d6 Feature Insights":
    st.markdown("### Top Discriminative Features per Class")
    fp = ROOT/"reports"/"figures"/"feature_importance.png"
    if fp.exists(): st.image(str(fp), use_container_width=True)
    else: st.info("Run training to generate feature importance chart.")

    st.markdown("<hr class=\"sr\">", unsafe_allow_html=True)
    if meta and meta.get("top_features"):
        tf = meta["top_features"]
        cls = st.selectbox("Explore category:", list(tf.keys()))
        n   = st.slider("Features to show", 5, 20, 15)
        feats = tf.get(cls,[])[:n][::-1]
        names = [x[0] for x in feats]; vals = [x[1] for x in feats]
        color = CAT.get(cls,{}).get("color","#333")
        fig = go.Figure(go.Bar(x=vals, y=names, orientation="h",
            marker_color=color, marker_opacity=0.85,
            text=[f"{v:.3f}" for v in vals], textposition="outside", cliponaxis=False))
        fig.update_layout(height=max(340, 22*n), margin=dict(l=0,r=60,t=5,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="#eee", title="TF-IDF Coefficient"),
            yaxis=dict(showgrid=False),
            font=dict(family="Source Sans 3", size=12), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown("""<div class="insight-card"><b>How to read:</b>
        Each bar = a TF-IDF feature (unigram, bigram, or trigram).
        The Logistic Regression coefficient shows how strongly that feature
        pushes the model toward this category. Higher = stronger discriminative signal.
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Train the model to see interactive feature insights.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""<hr style="border:none;border-top:1px solid #e5e0d8;margin-top:2.5rem;">
<div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#bbb;
     text-align:center;padding:.4rem 0 1rem;letter-spacing:.06em;">
NEWSLENS &middot; TF-IDF + LOGISTIC REGRESSION &middot; AG NEWS &middot; 92% MACRO F1
</div>""", unsafe_allow_html=True)

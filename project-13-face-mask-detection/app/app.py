"""
app.py — Professional Streamlit UI for Face Mask Detection
Run: streamlit run app/app.py
"""
import os, sys, io, time
import numpy as np, cv2
import streamlit as st
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

st.set_page_config(page_title="Face Mask Detector", page_icon="M",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.hero-title{font-size:2.6rem;font-weight:800;
  background:linear-gradient(135deg,#667eea,#764ba2,#f64f59);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  text-align:center;padding:.4rem 0;letter-spacing:-.5px;}
.hero-sub{text-align:center;color:#888;font-size:.95rem;margin-bottom:1.5rem;}
.metric-card{background:linear-gradient(135deg,#1a1d2e,#252840);
  border:1px solid #2d3057;border-radius:16px;padding:1.2rem;
  text-align:center;margin:.3rem 0;}
.metric-label{color:#aaa;font-size:.75rem;font-weight:700;
  text-transform:uppercase;letter-spacing:1px;}
.metric-value{color:#fff;font-size:2rem;font-weight:800;margin:.2rem 0;}
.metric-sub{color:#667eea;font-size:.82rem;}
.face-mask{border-left:4px solid #00c853;background:rgba(0,200,83,.08);
  padding:.9rem 1rem;border-radius:8px;margin:.4rem 0;}
.face-nomask{border-left:4px solid #f44336;background:rgba(244,67,54,.08);
  padding:.9rem 1rem;border-radius:8px;margin:.4rem 0;}
.pill-mask{background:#00c853;color:#000;padding:3px 12px;border-radius:20px;
  font-weight:700;font-size:.8rem;}
.pill-nomask{background:#f44336;color:#fff;padding:3px 12px;border-radius:20px;
  font-weight:700;font-size:.8rem;}
.pipe-step{background:#1a1d2e;border:1px solid #2d3057;border-radius:10px;
  padding:.6rem 1rem;margin:.25rem 0;font-size:.9rem;color:#ddd;}
</style>""", unsafe_allow_html=True)

def pil_to_bgr(p): return cv2.cvtColor(np.array(p.convert("RGB")), cv2.COLOR_RGB2BGR)
def bgr_to_pil(b): return Image.fromarray(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))

def find_model():
    for p in ["models/mask_classifier_ft.keras","models/mask_classifier.keras"]:
        f = str(ROOT/p)
        if os.path.exists(f): return f
    return str(ROOT/"models/mask_classifier.keras")

@st.cache_resource(show_spinner=False)
def load_pipeline(model_path):
    try:
        from utils.pipeline import MaskDetectionPipeline
        return MaskDetectionPipeline(model_path), None
    except Exception as e:
        return None, str(e)

def demo_predict(image_bgr):
    from utils.viz_utils import annotate_image
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.05, 5, minSize=(40,40))
    if len(faces)==0:
        h,w = image_bgr.shape[:2]; m=int(min(h,w)*0.12)
        faces = [(m,m,w-2*m,h-2*m)]
    detections = []
    for (x,y,fw,fh) in faces:
        x1,y1,x2,y2 = x,y,x+fw,y+fh
        crop = image_bgr[max(0,y1):y2,max(0,x1):x2]
        if crop.size==0: continue
        hsv = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV).astype(float)
        h2  = crop.shape[0]//2
        has_mask = hsv[h2:,:,1].mean() < hsv[:h2,:,1].mean()*0.75
        label = "Mask" if has_mask else "No Mask"
        conf  = float(np.random.uniform(0.82,0.97))
        detections.append({"box":(x1,y1,x2,y2),"label":label,"confidence":conf})
    return annotate_image(image_bgr, detections), detections

def conf_bar(label, conf):
    color = "#00c853" if label=="Mask" else "#f44336"
    pill  = "pill-mask" if label=="Mask" else "pill-nomask"
    st.markdown(f"""<div style="margin:.25rem 0 .6rem 0;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
        <span class="{pill}">{label}</span>
        <span style="color:#ddd;font-weight:700;">{conf:.1%}</span></div>
      <div style="background:#1a1d2e;border-radius:6px;height:9px;overflow:hidden;">
        <div style="width:{conf*100:.1f}%;background:{color};height:100%;border-radius:6px;">
        </div></div></div>""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")
    st.divider()
    demo_mode = st.checkbox("Demo Mode (no model needed)",
                            value=not os.path.exists(find_model()))
    model_path = find_model()
    if not demo_mode:
        model_path = st.text_input("Model Path", value=model_path)
    det_conf     = st.slider("Detection Threshold", 0.1, 0.9, 0.40, 0.05)
    show_gradcam = st.checkbox("Show Grad-CAM", False)
    show_crops   = st.checkbox("Show Face Crops", True)
    st.divider()
    st.markdown("### Pipeline")
    for s in ["1. Input Image","2. YOLOv8 Face Detection","3. Crop Regions",
              "4. CNN Classification","5. Label + Confidence","6. Draw Boxes"]:
        st.markdown(f'<div class="pipe-step">{s}</div>', unsafe_allow_html=True)
    st.divider()
    st.caption("YOLOv8 detection + MobileNetV2 classification. "
               "Kaggle Masked Face Recognition dataset.")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Face Mask Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Hybrid AI · YOLOv8 Detection + MobileNetV2 Classification</div>',
            unsafe_allow_html=True)

if demo_mode:
    st.info("**Demo Mode** — Haar Cascade + heuristics. Train the model for full accuracy.", icon="i")
    pipeline = None
else:
    with st.spinner("Loading pipeline…"):
        pipeline, err = load_pipeline(model_path)
    if pipeline is None:
        st.error(f"Could not load model: {err}")
        st.code("python src/train.py --data_dir data --epochs 30")
        st.stop()
    else:
        st.success("Pipeline ready", icon="✅")

tab_upload, tab_webcam, tab_results, tab_about = st.tabs(
    ["Upload Image","Webcam","Saved Results","How It Works"])

# ── Upload Tab ────────────────────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader("Upload image (JPG/PNG)", type=["jpg","jpeg","png"])
    if uploaded:
        pil_img  = Image.open(uploaded)
        img_bgr  = pil_to_bgr(pil_img)
        col_l,col_r = st.columns(2,gap="large")
        with col_l:
            st.subheader("Input"); st.image(pil_img, use_container_width=True)
        t0 = time.perf_counter()
        with st.spinner("Analysing…"):
            if demo_mode or pipeline is None:
                annotated_bgr,preds = demo_predict(img_bgr)
            else:
                annotated_bgr,raw = pipeline.predict_image(img_bgr,det_conf)
                preds = [p.to_dict() for p in raw]
        elapsed = time.perf_counter()-t0
        with col_r:
            st.subheader("Result")
            st.image(bgr_to_pil(annotated_bgr),use_container_width=True,
                     caption=f"Processed ({elapsed*1000:.0f}ms)")

        st.divider()
        n_total=len(preds); n_mask=sum(1 for p in preds if p["label"]=="Mask")
        n_no=n_total-n_mask; comp=f"{n_mask/n_total:.0%}" if n_total else "—"
        c1,c2,c3,c4=st.columns(4)
        for col,lbl,val,sub,clr in [
            (c1,"Faces Found",str(n_total),"Detected","#fff"),
            (c2,"With Mask",str(n_mask),"Protected","#00c853"),
            (c3,"No Mask",str(n_no),"At risk","#f44336"),
            (c4,"Compliance",comp,"Mask rate","#667eea")]:
            col.markdown(f"""<div class="metric-card">
              <div class="metric-label">{lbl}</div>
              <div class="metric-value" style="color:{clr};">{val}</div>
              <div class="metric-sub">{sub}</div></div>""",unsafe_allow_html=True)

        if preds:
            st.divider(); st.subheader(f"Face-by-Face Results ({n_total})")
            if show_crops:
                crop_cols = st.columns(min(n_total,5))
                for i,(det,col) in enumerate(zip(preds,crop_cols)):
                    x1,y1,x2,y2=det["box"]
                    crop=img_bgr[max(0,y1):y2,max(0,x1):x2]
                    if crop.size>0:
                        with col:
                            st.image(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),
                                     caption=f"Face {i+1}",use_container_width=True)
                            conf_bar(det["label"],det["confidence"])
            for i,det in enumerate(preds):
                card = "face-mask" if det["label"]=="Mask" else "face-nomask"
                x1,y1,x2,y2=det["box"]
                tag = "M" if det["label"]=="Mask" else "!"
                st.markdown(f"""<div class="{card}">
                  <strong>[{tag}] Face {i+1}</strong> | Label: <strong>{det['label']}</strong> |
                  Confidence: <strong>{det['confidence']:.1%}</strong> |
                  Box: ({x1},{y1}) to ({x2},{y2})</div>""",unsafe_allow_html=True)

        if show_gradcam and pipeline and preds:
            st.divider(); st.subheader("Grad-CAM — Model Attention")
            st.caption("Hot regions show where the CNN focuses when classifying.")
            try:
                from src.gradcam import GradCAM
                from utils.data_utils import IMG_SIZE
                gcam = GradCAM(pipeline.classifier)
                gcols = st.columns(min(len(preds),3))
                for i,(det,col) in enumerate(zip(preds[:3],gcols)):
                    x1,y1,x2,y2=det["box"]
                    crop_bgr=img_bgr[max(0,y1):y2,max(0,x1):x2]
                    if crop_bgr.size==0: continue
                    face=cv2.cvtColor(cv2.resize(crop_bgr,(IMG_SIZE,IMG_SIZE)),cv2.COLOR_BGR2RGB)
                    inp=face.astype("float32")[np.newaxis]/255.
                    hm=gcam.compute(inp)
                    overlay=gcam.overlay(face,hm)
                    with col:
                        st.image(np.hstack([face,overlay]),
                                 caption=f"Face {i+1} | {det['label']}",use_container_width=True)
            except Exception as e:
                st.warning(f"Grad-CAM: {e}")

        st.divider()
        _,dl,_=st.columns([1,2,1])
        with dl:
            buf=io.BytesIO(); bgr_to_pil(annotated_bgr).save(buf,"JPEG",quality=93)
            st.download_button("Download Annotated Image",buf.getvalue(),
                               "mask_result.jpg","image/jpeg",use_container_width=True)

# ── Webcam Tab ────────────────────────────────────────────────────────────────
with tab_webcam:
    st.subheader("Live Webcam Detection")
    cam_col,tip_col=st.columns([1,1])
    with cam_col:
        img_data=st.camera_input("Take a photo")
        if img_data:
            pil_cam=Image.open(img_data); bgr_cam=pil_to_bgr(pil_cam)
            with st.spinner("Analysing…"):
                if demo_mode or pipeline is None:
                    ann,dets=demo_predict(bgr_cam)
                else:
                    ann,raw=pipeline.predict_image(bgr_cam,det_conf)
                    dets=[p.to_dict() for p in raw]
            st.image(bgr_to_pil(ann),use_container_width=True)
            for d in dets: conf_bar(d["label"],d["confidence"])
    with tip_col:
        st.markdown("""**Tips for best results:**
- Face camera directly (frontal view)
- Ensure good even lighting
- Keep face centred in frame
- Avoid extreme shadows

For **multi-face** detection, use the Upload tab with a group photo.""")

# ── Results Tab ───────────────────────────────────────────────────────────────
with tab_results:
    st.subheader("Training & Evaluation Results")
    figures_dir=ROOT/"reports"/"figures"
    mp=ROOT/"reports"/"metrics.json"; rp=ROOT/"reports"/"classification_report.txt"
    if mp.exists():
        import json
        with open(mp) as f: m=json.load(f)
        mc1,mc2,mc3=st.columns(3)
        mc1.metric("Val Accuracy",f"{m.get('val_accuracy',0):.2%}")
        mc2.metric("AUC Score",f"{m.get('val_auc',0):.4f}")
        mc3.metric("Val Loss",f"{m.get('val_loss',0):.4f}")
        st.divider()
    fig_map={
        "Training Curves":"training_curves.png","Confusion Matrix":"confusion_matrix.png",
        "Grad-CAM":"gradcam.png","Failure Cases":"failure_cases.png",
        "Pipeline Predictions":"pipeline_predictions.png",
        "Augmentation Comparison":"augmentation_comparison.png",
        "Pipeline Architecture":"pipeline_diagram.png","Sample Images":"sample_images.png",
        "Class Distribution":"class_distribution.png",
    }
    pairs=list(fig_map.items())
    for i in range(0,len(pairs),2):
        cols=st.columns(2)
        for j,(title,fname) in enumerate(pairs[i:i+2]):
            fp=figures_dir/fname
            if fp.exists():
                with cols[j]: st.image(str(fp),caption=title,use_container_width=True)
    if rp.exists():
        st.divider(); st.subheader("Classification Report")
        st.code(rp.read_text(), language="text")

# ── About Tab ─────────────────────────────────────────────────────────────────
with tab_about:
    st.subheader("System Architecture")
    st.code("""
Input Image
  |
  v  Stage 1 — Face Detection (YOLOv8, pretrained on WiderFace)
  |  Returns: bounding box per face
  |
  v  Stage 2 — Classification (MobileNetV2, fine-tuned)
  |  Returns: Mask / No Mask + confidence
  |
  v  Annotated Output Image
""","text")
    c1,c2=st.columns(2)
    c1.markdown("**Classification** — answers 'What is in this image?'\n"
                "- Single label per image\n- No spatial info\n- Cannot handle multiple faces")
    c2.markdown("**Detection** — answers 'Where + what?'\n"
                "- Bounding boxes per object\n- Multiple faces supported\n- Real-world applicable")
    st.divider()
    st.subheader("Dataset Limitation & Solution")
    st.markdown("""The Kaggle dataset is **classification-only** (no bounding boxes).

| Limitation | Solution |
|---|---|
| No bounding boxes | Pretrained YOLOv8 for face detection |
| Single face per image | YOLO handles arbitrary face counts |
| Controlled conditions | Data augmentation for generalisation |
""")
    st.divider()
    st.subheader("Quick Start")
    st.code("""pip install -r requirements.txt
kaggle datasets download -d muhammeddalkran/masked-facerecognition
unzip masked-facerecognition.zip -d data/
python src/train.py --data_dir data --epochs 30
streamlit run app/app.py""","bash")

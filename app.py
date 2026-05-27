import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from collections import Counter
import tempfile
import os
import time

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Background */
.stApp {
    background-color: #0d0d0d;
    color: #f0f0f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #2a2a2a;
}

/* Header */
.main-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    letter-spacing: -1px;
    color: #ffffff;
    margin: 0;
}
.main-header .accent {
    color: #f5a623;
}
.main-header p {
    color: #888;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    flex: 1;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover {
    border-color: #f5a623;
}
.metric-card .icon {
    font-size: 2rem;
    margin-bottom: 0.4rem;
}
.metric-card .count {
    font-family: 'Space Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    color: #f5a623;
    line-height: 1;
}
.metric-card .label {
    font-size: 0.8rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 0.3rem;
}

/* Total card */
.total-card {
    background: linear-gradient(135deg, #f5a623 0%, #f7c26b 100%);
    border-radius: 12px;
    padding: 1.2rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 1rem 0;
}
.total-card .total-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #1a1a1a;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.total-card .total-count {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    color: #1a1a1a;
}

/* Upload area */
.upload-area {
    border: 2px dashed #2a2a2a;
    border-radius: 16px;
    padding: 3rem;
    text-align: center;
    background: #111;
    transition: border-color 0.2s;
}
.upload-area:hover {
    border-color: #f5a623;
}

/* Detection tag */
.detection-tag {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 0.3rem 0.8rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    margin: 0.2rem;
    color: #f0f0f0;
}

/* Confidence bar */
.conf-bar-bg {
    background: #2a2a2a;
    border-radius: 4px;
    height: 6px;
    margin-top: 4px;
}
.conf-bar-fill {
    background: #f5a623;
    border-radius: 4px;
    height: 6px;
}

/* Section title */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #555;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid #1e1e1e;
    padding-bottom: 0.5rem;
}

/* Streamlit overrides */
.stButton > button {
    background: #f5a623 !important;
    color: #0d0d0d !important;
    border: none !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    width: 100%;
}
.stButton > button:hover {
    background: #f7c26b !important;
}

div[data-testid="stSlider"] label {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #888;
}

/* Image captions */
.stImage > div > div > p {
    color: #555;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #f5a623 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)

# ─── Constants ─────────────────────────────────────────────────────────────────
CLASS_CONFIG = {
    "bus": {"icon": "🚌", "color": (239, 68,  68)},   # red
    "car": {"icon": "🚗", "color": (59,  130, 246)},   # blue
    "van": {"icon": "🚐", "color": (34,  197, 94)},    # green
}

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    model_path = st.text_input(
        "Model Path (.pt)",
        value="best_vehicle.pt",
        help="Path ke file model YOLOv12 yang sudah ditraining"
    )

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="Minimum confidence score untuk deteksi"
    )

    iou_threshold = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="IoU threshold untuk Non-Maximum Suppression"
    )

    show_conf = st.checkbox("Tampilkan confidence score", value=True)
    show_labels = st.checkbox("Tampilkan label", value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family: Space Mono, monospace; font-size: 0.7rem; color: #444;'>
    Capstone Project Module 4<br>
    Vehicle Detection · YOLOv12<br>
    Classes: bus · car · van
    </div>
    """, unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>Vehicle <span class='accent'>Detection</span></h1>
    <p>// YOLOv12 · Real-time object detection · bus · car · van</p>
</div>
""", unsafe_allow_html=True)

# ─── Load model ────────────────────────────────────────────────────────────────
if not os.path.exists(model_path):
    st.error(f"❌ Model tidak ditemukan: `{model_path}`\n\nPastikan file `.pt` ada di folder yang sama dengan `app.py`.")
    st.stop()

model = load_model(model_path)

# ─── Upload ────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Input</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload gambar kendaraan",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown("""
    <div class='upload-area'>
        <div style='font-size: 3rem;'>📸</div>
        <div style='font-family: Space Mono, monospace; color: #555; font-size: 0.85rem; margin-top: 1rem;'>
            Upload gambar JPG / PNG / WEBP<br>
            <span style='color: #333;'>untuk mulai deteksi kendaraan</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Process ───────────────────────────────────────────────────────────────────
image = Image.open(uploaded_file).convert("RGB")
img_array = np.array(image)

col_img, col_result = st.columns(2, gap="large")

with col_img:
    st.markdown("<div class='section-title'>Original Image</div>", unsafe_allow_html=True)
    st.image(image, use_container_width=True, caption=f"📁 {uploaded_file.name}")

with st.spinner("🔍 Mendeteksi kendaraan..."):
    start_time = time.time()
    results = model(
        img_array,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )[0]
    inference_time = (time.time() - start_time) * 1000  # ms

# ─── Draw results ──────────────────────────────────────────────────────────────
annotated = img_array.copy()
detections = []

if results.boxes is not None and len(results.boxes) > 0:
    for box in results.boxes:
        cls_id   = int(box.cls.item())
        cls_name = model.names[cls_id]
        conf     = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        color = CLASS_CONFIG.get(cls_name, {}).get("color", (255, 255, 0))
        icon  = CLASS_CONFIG.get(cls_name, {}).get("icon", "🚗")

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        if show_labels:
            label = f"{cls_name} {conf:.0%}" if show_conf else cls_name
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        detections.append({"class": cls_name, "conf": conf, "icon": icon,
                            "bbox": (x1, y1, x2, y2)})

with col_result:
    st.markdown("<div class='section-title'>Detection Result</div>", unsafe_allow_html=True)
    st.image(annotated, use_container_width=True,
             caption=f"⚡ Inference: {inference_time:.1f}ms · {len(detections)} objek terdeteksi")

# ─── Summary ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>Detection Summary</div>", unsafe_allow_html=True)

if not detections:
    st.warning("⚠️ Tidak ada kendaraan yang terdeteksi. Coba turunkan **Confidence Threshold** di sidebar.")
else:
    counts = Counter(d["class"] for d in detections)
    total  = sum(counts.values())

    # Total card
    st.markdown(f"""
    <div class='total-card'>
        <div class='total-label'>Total Kendaraan Terdeteksi</div>
        <div class='total-count'>{total}</div>
    </div>
    """, unsafe_allow_html=True)

    # Per-class metric cards
    cards_html = "<div class='metric-row'>"
    for cls_name, cfg in CLASS_CONFIG.items():
        count = counts.get(cls_name, 0)
        opacity = "1" if count > 0 else "0.3"
        cards_html += f"""
        <div class='metric-card' style='opacity:{opacity};'>
            <div class='icon'>{cfg['icon']}</div>
            <div class='count'>{count}</div>
            <div class='label'>{cls_name}</div>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    # Hasil dalam format teks
    result_text = "  ·  ".join(
        [f"{cls}: {cnt}" for cls, cnt in sorted(counts.items())]
    )
    st.markdown(f"""
    <div style='background:#111; border:1px solid #2a2a2a; border-radius:10px;
                padding:1rem 1.5rem; font-family: Space Mono, monospace;
                font-size:1.1rem; color:#f5a623; margin:1rem 0;'>
        🎯 {result_text}
    </div>
    """, unsafe_allow_html=True)

    # Detail per detection
    st.markdown("<div class='section-title' style='margin-top:1.5rem;'>Detail Deteksi</div>", unsafe_allow_html=True)
    det_cols = st.columns(min(len(detections), 4))
    for i, det in enumerate(detections):
        with det_cols[i % len(det_cols)]:
            conf_pct = int(det["conf"] * 100)
            st.markdown(f"""
            <div style='background:#111; border:1px solid #1e1e1e; border-radius:10px;
                        padding:0.8rem; margin-bottom:0.5rem;'>
                <div style='font-size:1.4rem;'>{det['icon']}</div>
                <div style='font-family:Space Mono,monospace; font-size:0.8rem;
                            color:#f0f0f0; font-weight:700;'>{det['class'].upper()}</div>
                <div style='font-family:Space Mono,monospace; font-size:0.7rem; color:#888;'>
                    conf: {conf_pct}%</div>
                <div class='conf-bar-bg'>
                    <div class='conf-bar-fill' style='width:{conf_pct}%;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Info tambahan
    st.markdown(f"""
    <div style='font-family:Space Mono,monospace; font-size:0.7rem; color:#333;
                margin-top:1rem; text-align:right;'>
        conf≥{conf_threshold:.0%} · iou={iou_threshold:.2f} · {inference_time:.1f}ms
    </div>
    """, unsafe_allow_html=True)
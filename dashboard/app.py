import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tempfile
import cv2
import numpy as np
import streamlit as st
from src.detector import AegisDetector

st.set_page_config(page_title="AEGIS-CUAS Tactical Console", layout="wide")
st.title("AEGIS-CUAS — Counter-UAS Aerial Detection Console")

# Sidebar Controls
st.sidebar.header("Model & Sensor Configuration")
models_dir = ROOT_DIR / "models"
available_weights = (
    [str(p) for p in sorted(models_dir.glob("*.pt"))] if models_dir.exists() else []
)
default_weight = str(ROOT_DIR / "runs/detect/aegis_runs/aegis_v01/weights/best.pt")
if Path(default_weight).exists() and default_weight not in available_weights:
    available_weights.insert(0, default_weight)

selected_weight = st.sidebar.selectbox(
    "Model Weights", available_weights or [default_weight]
)
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.10, 0.90, 0.30, 0.05)
mode = st.sidebar.radio("Input Source", ["Image Upload", "Video File", "Live Webcam"])


@st.cache_resource
def load_detector(weight_path: str):
    det = AegisDetector(config_path=str(ROOT_DIR / "configs/default.yaml"))
    det.model = det.model.__class__(weight_path)
    return det


detector = load_detector(selected_weight)

if mode == "Image Upload":
    uploaded = st.file_uploader(
        "Upload Aerial Image", type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated, detections, fps = detector.process_frame(
            frame, conf_override=conf_thresh
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
        with col2:
            st.metric("Inference Speed", f"{fps:.1f} FPS")
            st.metric("Detected Targets", len(detections))
            st.dataframe(detections, width="stretch")

elif mode == "Video File":
    uploaded_vid = st.file_uploader(
        "Upload Surveillance Video", type=["mp4", "avi", "mov", "mkv"]
    )
    if uploaded_vid and st.button("Run Detection"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_vid.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        stats_box = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated, detections, fps = detector.process_frame(
                frame, conf_override=conf_thresh
            )
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
            stats_box.caption(
                f"Speed: {fps:.1f} FPS | Active Tracks: {len(detections)}"
            )
        cap.release()

elif mode == "Live Webcam":
    run_cam = st.toggle("Start Camera Feed")
    stframe = st.empty()
    if run_cam:
        cap = cv2.VideoCapture(0)
        while run_cam and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to read from camera index 0.")
                break
            annotated, _, _ = detector.process_frame(frame, conf_override=conf_thresh)
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
        cap.release()